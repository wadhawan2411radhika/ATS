"""
Alignment Engine — computes all signals needed for scoring.

Responsibilities:
    - Semantic skill matching (cosine similarity via sentence-transformers)
    - Experience / YoE / seniority / domain alignment
    - Quality signals (depth, red flags, bonus)
    - Assembles AlignmentResult (no weights applied here)

Caching:
    JD skill embeddings are cached at module level by content hash.
    The embedding model is lazy-loaded once on first use.
    Resume skills are encoded fresh per call (no cross-resume caching).
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.extraction.jd_extraction.schemas import JDProfile
from src.extraction.resume_extraction.schemas import ResumeProfile
from src.scoring.schemas import AlignmentResult
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

# ── Matching thresholds ────────────────────────────────────────────────────────

_FULL_MATCH_THRESHOLD    = 0.82   # cosine sim → full match (weight 1.0)
_PARTIAL_MATCH_THRESHOLD = 0.65   # cosine sim → partial match (weight 0.6)
_PARTIAL_WEIGHT          = 0.6

_RECENCY_WEIGHTS: dict[str, float] = {
    "recent":      1.0,
    "established": 0.85,
    "dated":       0.60,
}
_RECENCY_DEFAULT = 0.75   # when recency is unknown for a skill


# ── Embedding cache ────────────────────────────────────────────────────────────

class _EmbeddingCache:
    """
    Lazy-loads the sentence-transformer model once.
    Caches encoded skill lists by content hash — JD skills are encoded
    once per session and reused across all resumes.
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        # key: hash(tuple(skills)) → np.ndarray of shape (n_skills, embed_dim)
        self._cache: dict[int, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {config.model.embedding_model}")
            self._model = SentenceTransformer(
                config.model.embedding_model,
                device=config.model.embedding_device,
            )
        return self._model

    def get_cached(self, skills: list[str]) -> np.ndarray:
        """
        Return embeddings for a skill list, using cache when available.
        Designed for JD skills — encode once, reuse for every resume.
        """
        if not skills:
            return np.zeros((0, 384))   # bge-small-en-v1.5 dim = 384

        key = hash(tuple(skills))
        if key not in self._cache:
            embs = self.model.encode(
                skills,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )
            self._cache[key] = embs
            logger.debug(f"Cached {len(skills)} skill embeddings (key={key})")
        return self._cache[key]

    def encode(self, skills: list[str]) -> np.ndarray:
        """
        Encode without caching — used for per-resume candidate skills.
        Not cached because each resume has a unique skill set.
        """
        if not skills:
            return np.zeros((0, 384))
        return self.model.encode(
            skills,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )


# Module-level singleton — shared across all align() calls in a session
_cache = _EmbeddingCache()


# ── Skill matching ─────────────────────────────────────────────────────────────

def _match_skills(
    jd_skills: list[str],
    jd_embeddings: np.ndarray,
    candidate_skills: list[str],
    candidate_embeddings: np.ndarray,
    recency_map: dict[str, str],   # lowercased skill → recency string
) -> tuple[list[str], list[str], float]:
    """
    For each JD skill, find the best-matching candidate skill via cosine similarity.

    Returns:
        matched_skills  — JD skills that found a match at or above partial threshold
        missing_skills  — JD skills with no sufficient match
        weighted_coverage — float 0→1 accounting for match strength and recency

    Coverage formula per skill:
        match_weight (1.0 full / 0.6 partial) × recency_weight (0.6–1.0)
    Total coverage = sum(per-skill scores) / len(jd_skills)
    """
    if not jd_skills:
        return [], [], 1.0   # no requirements → full coverage by definition

    if candidate_skills == [] or candidate_embeddings.size == 0 or jd_embeddings.size == 0:
        return [], list(jd_skills), 0.0

    # Cosine similarity matrix: (n_jd, n_candidate)
    # Both embedding matrices are already L2-normalised → dot product = cosine sim
    sim_matrix: np.ndarray = jd_embeddings @ candidate_embeddings.T

    matched: list[str] = []
    missing: list[str] = []
    total_score: float = 0.0

    for i, jd_skill in enumerate(jd_skills):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_idx])
        best_candidate_skill = candidate_skills[best_idx]

        recency = recency_map.get(best_candidate_skill.lower())
        recency_weight = (
            _RECENCY_WEIGHTS.get(recency, _RECENCY_DEFAULT)
            if recency else _RECENCY_DEFAULT
        )

        if best_sim >= _FULL_MATCH_THRESHOLD:
            matched.append(jd_skill)
            total_score += 1.0 * recency_weight
        elif best_sim >= _PARTIAL_MATCH_THRESHOLD:
            matched.append(jd_skill)   # still shown as matched in output
            total_score += _PARTIAL_WEIGHT * recency_weight
        else:
            missing.append(jd_skill)

    coverage = total_score / len(jd_skills)
    return matched, missing, coverage


# ── Seniority scoring ──────────────────────────────────────────────────────────

def _compute_seniority(
    candidate_level: str,
    required_level: str,
) -> tuple[str, float]:
    """
    Returns (human-readable description, 0→1 score).
    Scoring is symmetric — being over-qualified is treated similarly to under-qualified
    (over-qualified candidates are often a flight risk or misaligned on scope).
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

    If no requirement stated → neutral score of 0.8 (don't penalise or reward).
    Under-qualified: linear penalty proportional to gap.
    Over-qualified: slight bonus, capped at 1.0.
    """
    if required_yoe is None or required_yoe <= 0:
        return 0.0, 0.8

    gap = candidate_yoe - required_yoe
    ratio = candidate_yoe / required_yoe

    if ratio >= 1.0:
        # Over by up to 100% of requirement → bonus from 0.8 to 1.0
        score = min(1.0, 0.8 + 0.2 * min(ratio - 1.0, 1.0))
    else:
        # Under: linear — 0 YoE scores 0.0, meeting requirement scores 0.8
        score = ratio * 0.8

    return gap, score


# ── Domain scoring ─────────────────────────────────────────────────────────────

def _compute_domain_overlap(
    candidate_domains: list[str],
    jd_domains: list[str],
) -> float:
    """
    Simple case-insensitive overlap ratio.
    Returns neutral 0.8 if no JD domain preference stated.
    """
    if not jd_domains:
        return 0.8

    candidate_set = {d.lower().strip() for d in candidate_domains}
    jd_set = {d.lower().strip() for d in jd_domains}
    overlap = len(candidate_set & jd_set) / len(jd_set)
    return overlap


# ── Quality signals ────────────────────────────────────────────────────────────

def _compute_quality(
    resume: ResumeProfile,
) -> tuple[float, float, float, list[str], list[str]]:
    """
    Returns:
        depth_score      — 0→1 from skill_depth_signals (3+ = full)
        red_flag_penalty — 0→0.4 capped (each red flag costs 0.1)
        bonus            — 0→0.3 capped (publications, open source, certs)
        bonus_signals    — human-readable list for display
        penalty_signals  — human-readable list for display (the red flags)

    These are the extraction signals that went completely unused in the
    original scoring layer. Now they directly affect the quality component.
    """
    # Depth: skill_depth_signals are mastery-evidence strings like
    # "PyTorch: fine-tuned 13B model with FSDP" — 3+ is a strong signal
    n_depth = len(resume.skills.skill_depth_signals)
    depth_score = min(1.0, n_depth / 3.0)

    # Penalty from red flags surfaced by the LLM synthesizer
    penalty_signals = [f for f in resume.red_flags if f]
    red_flag_penalty = min(0.4, len(penalty_signals) * 0.1)

    # Bonus from concrete credentials
    bonus_signals: list[str] = []
    bonus = 0.0

    if resume.education.has_publications:
        bonus += 0.3
        pub_count = len(resume.education.publications)
        bonus_signals.append(
            f"{pub_count} publication{'s' if pub_count != 1 else ''}"
        )

    if resume.education.has_open_source:
        bonus += 0.3
        bonus_signals.append("open source contributions")

    if resume.education.certifications:
        bonus += 0.2
        certs = resume.education.certifications
        bonus_signals.append(
            f"certifications: {', '.join(certs[:2])}{'...' if len(certs) > 2 else ''}"
        )

    bonus = min(0.3, bonus)   # cap so credentials don't dominate

    return depth_score, red_flag_penalty, bonus, bonus_signals, penalty_signals


# ── Main alignment function ────────────────────────────────────────────────────

def align(jd: JDProfile, resume: ResumeProfile) -> AlignmentResult:
    """
    Compute full alignment between a JD and a resume.

    JD skill embeddings are cached — encoded once per unique skill list,
    reused across all resumes in the same session.

    Args:
        jd:     Extracted JD profile (frozen extraction layer output).
        resume: Extracted resume profile (frozen extraction layer output).

    Returns:
        AlignmentResult with all raw signals. Weights are applied in scorer.py.
    """

    # ── Skill alignment ────────────────────────────────────────────────────────
    required_skills  = jd.hard_requirements.required_skills
    preferred_skills = jd.soft_requirements.preferred_skills

    # Deduplicate candidate skills, preserving insertion order
    # explicit first (listed), then implicit (demonstrated)
    seen: set[str] = set()
    candidate_skills: list[str] = []
    for s in (resume.skills.explicit_skills + resume.skills.implicit_skills):
        if s.lower() not in seen:
            seen.add(s.lower())
            candidate_skills.append(s)

    recency_map = {k.lower(): v for k, v in resume.skills.skill_recency_map.items()}

    # JD embeddings are cached — same required/preferred skills → same embedding
    req_embs  = _cache.get_cached(required_skills)
    pref_embs = _cache.get_cached(preferred_skills)

    # Candidate embeddings are NOT cached (unique per resume)
    candidate_embs = _cache.encode(candidate_skills)

    matched_req, missing_req, req_coverage = _match_skills(
        required_skills, req_embs,
        candidate_skills, candidate_embs,
        recency_map,
    )
    matched_pref, _, pref_coverage = _match_skills(
        preferred_skills, pref_embs,
        candidate_skills, candidate_embs,
        recency_map,
    )

    # Preferred skills give a bonus capped at +0.1 on top of required coverage
    skill_score = min(1.0, req_coverage + 0.1 * pref_coverage)

    # ── Experience alignment ───────────────────────────────────────────────────
    gap_years, yoe_sc = _compute_yoe(
        resume.total_years_experience,
        jd.hard_requirements.required_years_of_experience,
    )
    seniority_label, sen_sc = _compute_seniority(
        resume.current_seniority.value,
        jd.role_identity.seniority_level.value,
    )

    # Use JD's preferred domain experience as the domain target
    # (this is where domain signals live in the JD schema)
    jd_domains = jd.soft_requirements.preferred_domain_experience
    dom_sc = _compute_domain_overlap(resume.domains_worked_in, jd_domains)

    experience_score = 0.5 * yoe_sc + 0.3 * sen_sc + 0.2 * dom_sc

    # ── Quality signals ────────────────────────────────────────────────────────
    depth_score, red_flag_penalty, bonus, bonus_signals, penalty_signals = _compute_quality(resume)

    quality_score = min(1.0, (depth_score + bonus) * (1.0 - red_flag_penalty))

    logger.debug(
        f"{resume.identity.full_name or 'Unknown'}: "
        f"skill={skill_score:.3f} exp={experience_score:.3f} qual={quality_score:.3f} "
        f"req_cov={req_coverage:.2%} gap={gap_years:+.1f}yr seniority={seniority_label}"
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