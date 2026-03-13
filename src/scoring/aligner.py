"""
Alignment Engine — computes all signals needed for scoring.

Skill matching strategy — LLM-based (replaces sentence-transformers):
    A single structured LLM call per resume determines which JD required
    and preferred skills the candidate genuinely possesses.

    Why LLM over embeddings:
    - Embeddings at any threshold produce false positives across adjacent
      domains ("microservices" → "agent-based systems") or depth differences
      ("PyTorch for inference" ≠ "PyTorch for training"). These require
      threshold tuning that is JD-specific and fragile.
    - The LLM understands context: "Elasticsearch for search queries only"
      does NOT match "retrieval systems" as an ML skill. "NLP" matches
      "Natural Language Processing". "RL" matches "reinforcement learning".
    - Cost: 1 LLM call per resume. For a batch of 20 resumes this is
      20 calls — comparable to what extraction already costs.

    Coverage is binary — a skill is either matched or missing.
    No partial match: a partial match is ambiguous and dilutes the signal.
    If a skill is genuinely uncertain, it is marked missing.

Penalty signals:
    Only genuine negative signals count — employment gaps, inconsistencies,
    unsupported claims, job hopping, declining trajectory.
    Absence of bonus credentials (no OSS, no publications, no certs) is NOT
    a penalty. Those signals belong in the bonus component only.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.extraction.jd_extraction.schemas import JDProfile
from src.extraction.resume_extraction.schemas import ResumeProfile
from src.scoring.schemas import AlignmentResult
from src.utils.llm_client import extract_structured
from config import config

logger = logging.getLogger(__name__)


# ── Seniority ordering ─────────────────────────────────────────────────────────

SENIORITY_ORDER: dict[str, int] = {
    "intern":    0,
    "junior":    1,
    "mid":       2,
    "senior":    3,
    "lead":      4,
    "staff":     4,
    "manager":   4,
    "principal": 5,
    "director":  6,
    "vp":        7,
    "executive": 8,
    "unknown":   2,   # neutral — don't penalise when unknown
}

# ── Penalty signal filter keywords ────────────────────────────────────────────
# Red flags containing these phrases are "absence of bonus", not genuine negatives.
# They should NOT contribute to the penalty score.

_ABSENT_BONUS_PHRASES = (
    "no open source",
    "no oss",
    "no publication",
    "no conference",
    "no certification",
    "no patent",
    "limited open source",
    "lacks open source",
    "no github",
    "no contributions",
    "no community",
    "no blog",
    "no speaking",
    "no award",
)


# ── LLM skill matching schema ──────────────────────────────────────────────────

class _SkillDecision(BaseModel):
    skill: str = Field(description="Skill name copied exactly from the JD list.")
    matched: bool = Field(description="True if the candidate genuinely has this skill.")
    reasoning: str = Field(
        description=(
            "One sentence explaining the decision. Cite the specific evidence "
            "(e.g. 'Explicit: PyTorch listed + LLaMA-2 fine-tuning in work history') "
            "or the gap (e.g. 'Implicit: only inference/serving, no training evidence'). "
            "This is required — it makes the output auditable and ensures consistent decisions."
        )
    )


class _SkillMatchResult(BaseModel):
    """
    Both required and preferred skills assessed in a single LLM call.
    Keeping them together avoids duplicate context and ensures the model
    makes consistent decisions across both lists.
    """
    required: list[_SkillDecision] = Field(
        description="One entry per required JD skill."
    )
    preferred: list[_SkillDecision] = Field(
        description="One entry per preferred JD skill."
    )


_SKILL_MATCH_SYSTEM = """
You are a technical recruiter assessing whether a candidate has specific skills
required for a job opening.

You will be given:
  1. A list of required JD skills
  2. A list of preferred JD skills
  3. The candidate's explicit skills (listed directly on resume — higher confidence)
  4. The candidate's implicit skills (inferred from work history — lower confidence,
     treat with more scrutiny)

Your task: for EVERY skill in both lists, determine whether the candidate genuinely has it.
Provide a brief reasoning for each decision.

CONFIDENCE RULE:
- Explicit skills: accept if plausible given work history
- Implicit skills: only accept if the work history provides STRONG, SPECIFIC evidence.
  Vague adjacency ("worked in ML" → "reinforcement learning") is NOT sufficient.

MATCHING RULES — be strict:
- Semantic synonyms OK: "NLP" = "Natural Language Processing", "RL" = "reinforcement learning"
- Framework aliases OK: "Hugging Face" implies "PyTorch" IF work history confirms model training
- Tool-to-concept: match only at the conceptual level required.
  "Elasticsearch for search queries" does NOT match "retrieval systems" as an ML skill.
- Depth matters: "PyTorch for inference optimization" does NOT match "PyTorch" for a role
  requiring model training or fine-tuning.
- Deployment ≠ design: serving/deploying LLMs does NOT imply agent-based system design.
- Adjacent domain ≠ match: "microservices" does NOT match "agent-based systems".
- Classical ML ≠ deep learning: "scikit-learn, XGBoost" does NOT match "TensorFlow" or "PyTorch".
- Basic NLP ≠ NLP: "TF-IDF, sentiment analysis" does NOT match "NLP" for a role requiring
  transformers, LLMs, or neural NLP.

When uncertain → matched: false. False negatives are safer than false positives.
"""


def _llm_match_skills(
    required_skills: list[str],
    preferred_skills: list[str],
    candidate_explicit: list[str],
    candidate_implicit: list[str],
    candidate_name: str,
) -> tuple[list[str], list[str], float, list[str], list[str], float]:
    """
    Single LLM call assessing both required and preferred skills together.

    Merging the two lists into one call:
    - Eliminates duplicate context (same resume sent twice previously)
    - Prevents the model making inconsistent decisions on the same evidence
    - Halves LLM calls in the alignment layer

    Returns:
        matched_req, missing_req, req_coverage,
        matched_pref, missing_pref, pref_coverage
    """
    if not required_skills and not preferred_skills:
        return [], [], 1.0, [], [], 1.0

    if not candidate_explicit and not candidate_implicit:
        return [], list(required_skills), 0.0, [], list(preferred_skills), 0.0

    user_prompt = f"""
Candidate: {candidate_name}

Required JD Skills (assess each):
{required_skills}

Preferred JD Skills (assess each):
{preferred_skills}

Candidate's Explicit Skills (listed on resume — higher confidence):
{candidate_explicit}

Candidate's Implicit Skills (inferred from work history — lower confidence, be more critical):
{candidate_implicit}

For every skill in both lists, return a decision with matched (true/false) and one-sentence reasoning.
Copy skill names exactly as they appear in the JD lists.
"""

    try:
        result = extract_structured(
            system_prompt=_SKILL_MATCH_SYSTEM,
            user_prompt=user_prompt,
            schema=_SkillMatchResult,
        )

        def _resolve(
            decisions: list[_SkillDecision],
            jd_skills: list[str],
            label: str,
        ) -> tuple[list[str], list[str], float]:
            """Split decisions into matched/missing, validate coverage, log reasoning."""
            returned = {d.skill: d for d in decisions}
            matched, missing = [], []

            for skill in jd_skills:
                decision = returned.get(skill)
                if decision is None:
                    logger.warning(
                        f"[{candidate_name}] {label} skill '{skill}' not classified — marking missing"
                    )
                    missing.append(skill)
                elif decision.matched:
                    logger.debug(
                        f"[{candidate_name}] ✓ {label} '{skill}': {decision.reasoning}"
                    )
                    matched.append(skill)
                else:
                    logger.debug(
                        f"[{candidate_name}] ✗ {label} '{skill}': {decision.reasoning}"
                    )
                    missing.append(skill)

            coverage = len(matched) / len(jd_skills) if jd_skills else 1.0
            logger.info(
                f"[{candidate_name}] {label}: {len(matched)}/{len(jd_skills)} matched ({coverage:.0%})"
            )
            return matched, missing, coverage

        matched_req, missing_req, req_cov = _resolve(result.required, required_skills, "required")
        matched_pref, missing_pref, pref_cov = _resolve(result.preferred, preferred_skills, "preferred")
        return matched_req, missing_req, req_cov, matched_pref, missing_pref, pref_cov

    except Exception as e:
        logger.error(
            f"[{candidate_name}] LLM skill matching failed: {e}. Marking all skills as missing."
        )
        return [], list(required_skills), 0.0, [], list(preferred_skills), 0.0


# ── Seniority scoring ──────────────────────────────────────────────────────────

def _compute_seniority(
    candidate_level: str,
    required_level: str,
) -> tuple[str, float]:
    """
    Returns (human-readable description, 0→1 score).
    Symmetric — over-qualified is penalised similarly to under-qualified.
    """
    c = SENIORITY_ORDER.get(candidate_level.lower(), 2)
    r = SENIORITY_ORDER.get(required_level.lower(), 2)
    delta = abs(c - r)

    score_map = {0: 1.0, 1: 0.7, 2: 0.4}
    score = score_map.get(delta, 0.1)

    if delta == 0:
        label = "exact match"
    else:
        direction = "above" if c > r else "below"
        label = f"{delta} level{'s' if delta > 1 else ''} {direction}"

    return label, score


# ── YoE scoring ────────────────────────────────────────────────────────────────

def _compute_yoe(
    candidate_yoe: float,
    required_yoe: Optional[float],
) -> tuple[float, float]:
    """
    Returns (gap_years, 0→1 score).
    Neutral 0.8 when no requirement stated.
    Linear penalty when under, capped bonus when over.
    """
    if required_yoe is None or required_yoe <= 0:
        return 0.0, 0.8

    gap = candidate_yoe - required_yoe
    ratio = candidate_yoe / required_yoe

    if ratio >= 1.0:
        score = min(1.0, 0.8 + 0.2 * min(ratio - 1.0, 1.0))
    else:
        score = ratio * 0.8

    return gap, score


# ── Domain scoring ─────────────────────────────────────────────────────────────

def _compute_domain_overlap(
    candidate_domains: list[str],
    jd_domains: list[str],
) -> float:
    """Simple case-insensitive overlap. Neutral 0.8 if no JD domain preference."""
    if not jd_domains:
        return 0.8
    candidate_set = {d.lower().strip() for d in candidate_domains}
    jd_set = {d.lower().strip() for d in jd_domains}
    return len(candidate_set & jd_set) / len(jd_set)


# ── Quality signals ────────────────────────────────────────────────────────────

def _is_absent_bonus(flag: str) -> bool:
    """
    Returns True if the red flag is describing absence of a bonus credential
    rather than a genuine negative signal.

    Absence of bonus (ignore): "No open source contributions", "No publications"
    Genuine negative (keep):   "Employment gap 2021-2022", "Claims without evidence",
                               "Job hopping — 4 roles in 3 years"
    """
    flag_lower = flag.lower()
    return any(phrase in flag_lower for phrase in _ABSENT_BONUS_PHRASES)


def _compute_quality(
    resume: ResumeProfile,
) -> tuple[float, float, float, list[str], list[str]]:
    """
    Returns:
        depth_score      — 0→1 from explicit + implicit depth signals (3+ = full)
        red_flag_penalty — 0→0.4 capped; only genuine negatives, not absent bonuses
        bonus            — 0→0.3 capped (publications, OSS, certs)
        bonus_signals    — human-readable list for display
        penalty_signals  — filtered genuine red flags only

    Penalty filtering:
        Red flags describing missing credentials (no OSS, no publications, no certs)
        are excluded from penalty calculation. A candidate shouldn't be penalised
        for not having something — only for having genuine negative signals.
        Those absent credentials are simply not awarded the bonus.
    """
    all_depth_signals = (
        resume.skills.explicit_skill_depth_signals
        + resume.skills.implicit_skill_depth_signals
    )
    depth_score = min(1.0, len(all_depth_signals) / 3.0)

    # Filter: keep only genuine negatives, drop absent-bonus flags
    genuine_red_flags = [
        f for f in resume.red_flags
        if f and not _is_absent_bonus(f)
    ]
    red_flag_penalty = min(0.4, len(genuine_red_flags) * 0.1)
    penalty_signals = genuine_red_flags  # only show genuine flags in output

    # Bonus from concrete credentials — independent of depth and penalty.
    # Publications/OSS/certs are verifiable facts that can't be "cancelled" by
    # red flags the way work-history depth can. They're additive on top.
    bonus_signals: list[str] = []
    bonus = 0.0

    if resume.education.has_publications:
        bonus += 0.3
        pub_count = len(resume.education.publications)
        bonus_signals.append(f"{pub_count} publication{'s' if pub_count != 1 else ''}")

    if resume.education.has_open_source:
        bonus += 0.3
        bonus_signals.append("open source contributions")

    if resume.education.certifications:
        bonus += 0.2
        certs = resume.education.certifications
        bonus_signals.append(
            f"certifications: {', '.join(certs[:2])}{'...' if len(certs) > 2 else ''}"
        )

    bonus = min(0.3, bonus)

    return depth_score, red_flag_penalty, bonus, bonus_signals, penalty_signals


# ── Main alignment function ────────────────────────────────────────────────────

def align(jd: JDProfile, resume: ResumeProfile) -> AlignmentResult:
    """
    Compute full alignment between a JD and a resume.

    Skill matching uses a single LLM call per resume — more accurate than
    embedding cosine similarity for distinguishing adjacent domains and
    depth differences.
    """
    candidate_name = resume.identity.full_name or "Unknown"

    required_skills  = jd.hard_requirements.required_skills
    preferred_skills = jd.soft_requirements.preferred_skills

    # ── Skill matching — single merged LLM call ────────────────────────────────
    # Required + preferred assessed together: same context, no duplicate prompts,
    # consistent decisions across both lists.
    matched_req, missing_req, req_coverage, matched_pref, _missing_pref, pref_coverage = (
        _llm_match_skills(
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            candidate_explicit=resume.skills.explicit_skills,
            candidate_implicit=resume.skills.implicit_skills,
            candidate_name=candidate_name,
        )
    )

    # Preferred skills add a bonus capped at +0.1
    skill_score = min(1.0, req_coverage + 0.1 * pref_coverage)

    # ── Experience ─────────────────────────────────────────────────────────────
    gap_years, yoe_sc = _compute_yoe(
        resume.total_years_experience,
        jd.hard_requirements.required_years_of_experience,
    )
    seniority_label, sen_sc = _compute_seniority(
        resume.current_seniority.value,
        jd.role_identity.seniority_level.value,
    )
    jd_domains = jd.soft_requirements.preferred_domain_experience
    dom_sc = _compute_domain_overlap(resume.domains_worked_in, jd_domains)

    experience_score = 0.5 * yoe_sc + 0.3 * sen_sc + 0.2 * dom_sc

    # ── Quality ────────────────────────────────────────────────────────────────
    depth_score, red_flag_penalty, bonus, bonus_signals, penalty_signals = _compute_quality(resume)

    # depth is penalized by red flags; bonus (publications/OSS/certs) is not.
    # Credentials are verifiable facts — a job-hopping flag shouldn't cancel publications.
    # Old: (depth + bonus) * (1 - penalty)  ← bonus unfairly penalized, depth+bonus can exceed 1.0
    # New: depth * (1 - penalty) + bonus     ← clean separation, bonus is additive on top
    quality_score = min(1.0, depth_score * (1.0 - red_flag_penalty) + bonus)

    logger.debug(
        f"{candidate_name}: skill={skill_score:.3f} (req_cov={req_coverage:.0%}) "
        f"exp={experience_score:.3f} qual={quality_score:.3f} "
        f"depth={depth_score:.2f} penalty={red_flag_penalty:.2f} bonus={bonus:.2f} "
        f"seniority={seniority_label} gap={gap_years:+.1f}yr flags={len(penalty_signals)}"
    )

    return AlignmentResult(
        matched_required_skills=matched_req,
        missing_required_skills=missing_req,
        matched_preferred_skills=matched_pref,
        required_skill_coverage=req_coverage,
        preferred_skill_coverage=pref_coverage,
        experience_gap_years=gap_years,
        yoe_score=yoe_sc,
        seniority_match=seniority_label,
        seniority_score=sen_sc,
        domain_overlap_score=dom_sc,
        depth_score=depth_score,
        red_flag_penalty=red_flag_penalty,
        bonus=bonus,
        bonus_signals=bonus_signals,
        penalty_signals=penalty_signals,
        skill_score=skill_score,
        experience_score=experience_score,
        quality_score=quality_score,
    )