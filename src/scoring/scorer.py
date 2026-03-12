"""
Scoring Engine.

Takes an AlignmentResult + JDExtracted and produces a final score (0.0 - 1.0)
with full per-tier breakdown for explainability.

Architecture:
  1. Gate check (hard filters → score 0.0 if failed)
  2. Tier scoring (T1 must-haves, T2 good-to-haves, T3 quality, T4 bonus)
  3. Penalty deduction
  4. Final weighted aggregation
"""

import logging
from dataclasses import dataclass

from src.extraction.jd_extraction import JDProfile
from src.extraction.resume_extraction import ResumeProfile as ResumeExtracted
from src.scoring.schemas import AlignmentResult
from config import config

logger = logging.getLogger(__name__)

SENIORITY_ORDER = {
    "intern": 0, "junior": 1, "mid": 2,
    "senior": 3, "lead": 4, "staff": 4,
    "principal": 5, "manager": 3, "director": 5, "unknown": 2,
}


@dataclass
class ScoringBreakdown:
    """Full score breakdown — the explainability artifact."""
    candidate_name: str

    # Gate
    passed_gate: bool
    gate_failure_reason: str | None

    # Per-tier raw scores (0.0 - 1.0)
    tier1_score: float
    tier2_score: float
    tier3_score: float
    tier4_score: float

    # Penalty
    penalty_score: float  # Amount subtracted

    # Final
    final_score: float

    # Tier sub-scores for deep explainability
    required_skill_score: float
    experience_score: float
    seniority_score: float
    domain_score: float
    preferred_skill_score: float
    semantic_score: float
    quality_score: float

    # Human-readable signals
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    seniority_match: str
    experience_gap_years: float
    bonus_signals: list[str]
    penalty_signals: list[str]

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate_name,
            "final_score": round(self.final_score, 3),
            "passed_gate": self.passed_gate,
            "gate_failure_reason": self.gate_failure_reason,
            "tier_breakdown": {
                "tier1_must_haves": round(self.tier1_score, 3),
                "tier2_good_to_haves": round(self.tier2_score, 3),
                "tier3_quality_signals": round(self.tier3_score, 3),
                "tier4_bonus": round(self.tier4_score, 3),
                "penalty_deducted": round(self.penalty_score, 3),
            },
            "sub_scores": {
                "required_skill_coverage": round(self.required_skill_score, 3),
                "experience_alignment": round(self.experience_score, 3),
                "seniority_alignment": round(self.seniority_score, 3),
                "domain_alignment": round(self.domain_score, 3),
                "preferred_skill_coverage": round(self.preferred_skill_score, 3),
                "semantic_similarity": round(self.semantic_score, 3),
                "candidate_quality": round(self.quality_score, 3),
            },
            "signals": {
                "matched_required_skills": self.matched_required_skills,
                "missing_required_skills": self.missing_required_skills,
                "matched_preferred_skills": self.matched_preferred_skills,
                "seniority_match": self.seniority_match,
                "experience_gap_years": self.experience_gap_years,
                "bonus_signals": self.bonus_signals,
                "penalty_signals": self.penalty_signals,
            },
        }


def _check_gate(jd: JDProfile, alignment: AlignmentResult) -> tuple[bool, str | None]:
    """
    Hard gate filters. Return (passed, failure_reason).
    If failed, resume is scored 0.0 without further processing.
    """
    cfg = config.gate

    # Gate 1: Minimum experience
    if cfg.enforce_min_experience and jd.hard_requirements.required_years_of_experience:
        if alignment.experience_gap_years < -2.0:
            return False, (
                f"Under-experienced by {abs(alignment.experience_gap_years):.1f} years "
                f"(required {jd.hard_requirements.required_years_of_experience}yr)"
            )

    # Gate 2: Required skill floor (if configured)
    if cfg.enforce_required_skills_gate:
        if alignment.skill_coverage_required < cfg.min_required_skill_coverage:
            return False, (
                f"Required skill coverage {alignment.skill_coverage_required:.0%} "
                f"below gate threshold {cfg.min_required_skill_coverage:.0%}"
            )

    return True, None


def _score_experience_alignment(gap: float) -> float:
    """
    Convert experience gap to a 0-1 score.
    - Perfect: gap between -1 and +3 years
    - Slight mismatch: penalized gradually
    - Severely under-experienced: near 0
    """
    if gap >= -1 and gap <= 3:
        return 1.0
    elif gap > 3:
        # Overqualified — small penalty
        return max(0.7, 1.0 - (gap - 3) * 0.05)
    elif gap >= -2:
        # Slightly under
        return 0.5 + (gap + 2) * 0.25  # linear from 0.5 → 0.75
    else:
        # Severely under
        return max(0.0, 0.5 + gap * 0.15)


def _score_seniority(seniority_match: str) -> float:
    return {
        "aligned": 1.0,
        "overqualified": 0.75,
        "slightly underqualified": 0.5,
        "underqualified": 0.2,
    }.get(seniority_match, 0.5)


def _score_tier1(alignment: AlignmentResult, jd: JDProfile) -> tuple[float, dict]:
    """Must-haves tier scoring with skill coverage floor enforcement."""
    w = config.scoring

    required_skill_score = alignment.skill_coverage_required
    experience_score = _score_experience_alignment(alignment.experience_gap_years)
    seniority_score = _score_seniority(alignment.seniority_match)
    domain_score = alignment.domain_match_score

    tier1 = (
        w.t1_required_skills * required_skill_score +
        w.t1_experience_years * experience_score +
        w.t1_seniority_match * seniority_score +
        w.t1_domain_relevance * domain_score
    )

    # Skill coverage floor: if required skills are defined and coverage is very low,
    # cap T1 so that strong experience/seniority can't rescue a fundamentally mismatched candidate.
    # Thresholds: <20% coverage → cap T1 at 0.35 | <40% → cap at 0.55
    if jd.hard_requirements.required_skills:
        if required_skill_score < 0.20:
            tier1 = min(tier1, 0.40)   # Near-zero skill match: hard cap
        elif required_skill_score < 0.40:
            tier1 = min(tier1, 0.62)   # Below half: soft cap — experience can partially rescue
        # >40% coverage: no cap — let the full score express itself

    sub = {
        "required_skill_score": required_skill_score,
        "experience_score": experience_score,
        "seniority_score": seniority_score,
        "domain_score": domain_score,
    }
    return min(tier1, 1.0), sub


def _score_tier2(alignment: AlignmentResult) -> float:
    """Good-to-haves tier scoring."""
    w = config.scoring

    # Company brand: map tier to 0-1 score
    tier_scores = {
        "tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
        "tier_4": 0.25, "academic": 0.6, None: 0.4
    }
    company_brand_score = tier_scores.get(alignment.highest_company_tier, 0.4)

    return min(
        w.t2_preferred_skills * alignment.skill_coverage_preferred +
        w.t2_company_brand * company_brand_score +
        w.t2_semantic_similarity * alignment.semantic_similarity_score,
        1.0
    )


def _score_tier3(alignment: AlignmentResult) -> float:
    """Quality signals tier scoring."""
    w = config.scoring
    return min(
        w.t3_impact_quality * alignment.candidate_quality_score +
        w.t3_project_quality * alignment.candidate_quality_score +  # same composite for now
        w.t3_career_trajectory * alignment.candidate_quality_score,
        1.0
    )


def _score_tier4(alignment: AlignmentResult) -> float:
    """Bonus tier scoring."""
    w = config.scoring
    has_oss = any("open source" in s.lower() for s in alignment.bonus_signals)
    has_pub = any("publication" in s.lower() for s in alignment.bonus_signals)
    has_rare = any("additional" in s.lower() for s in alignment.bonus_signals)

    return min(
        w.t4_open_source * (1.0 if has_oss else 0.0) +
        w.t4_publications * (1.0 if has_pub else 0.0) +
        w.t4_rare_skills * (1.0 if has_rare else 0.0),
        1.0
    )


def _compute_penalty(alignment: AlignmentResult) -> float:
    """
    Compute total penalty from red flags.
    Cap at penalty_cap to prevent catastrophic scoring from single signals.
    """
    penalty = 0.0
    for signal in alignment.penalty_signals:
        if "exaggeration" in signal.lower():
            penalty += 0.05
        elif "buzzword" in signal.lower():
            penalty += 0.04
        elif "specificity" in signal.lower():
            penalty += 0.04
        elif "gap" in signal.lower():
            penalty += 0.03
        elif "plateau" in signal.lower():
            penalty += 0.03

    return min(penalty, config.scoring.penalty_cap)


def score(
    jd: JDProfile,
    resume: ResumeExtracted,
    alignment: AlignmentResult,
) -> ScoringBreakdown:
    """
    Compute final relevance score for a resume against a JD.

    Returns a ScoringBreakdown with the final score and full explainability trail.
    """
    w = config.scoring
    candidate_name = resume.candidate_name or "Unknown Candidate"

    # ── Gate check ─────────────────────────────────────────────────────────────
    passed, gate_reason = _check_gate(jd, alignment)
    if not passed:
        logger.info(f"{candidate_name}: FAILED gate — {gate_reason}")
        return ScoringBreakdown(
            candidate_name=candidate_name,
            passed_gate=False,
            gate_failure_reason=gate_reason,
            tier1_score=0.0, tier2_score=0.0,
            tier3_score=0.0, tier4_score=0.0,
            penalty_score=0.0, final_score=0.0,
            required_skill_score=0.0, experience_score=0.0,
            seniority_score=0.0, domain_score=0.0,
            preferred_skill_score=0.0, semantic_score=0.0,
            quality_score=0.0,
            matched_required_skills=alignment.matched_required_skills,
            missing_required_skills=alignment.missing_required_skills,
            matched_preferred_skills=alignment.matched_preferred_skills,
            seniority_match=alignment.seniority_match,
            experience_gap_years=alignment.experience_gap_years,
            bonus_signals=alignment.bonus_signals,
            penalty_signals=alignment.penalty_signals,
        )

    # ── Tier scoring ───────────────────────────────────────────────────────────
    tier1, sub = _score_tier1(alignment, jd)
    tier2 = _score_tier2(alignment)
    tier3 = _score_tier3(alignment)
    tier4 = _score_tier4(alignment)
    penalty = _compute_penalty(alignment)

    # ── Weighted aggregation ───────────────────────────────────────────────────
    raw_score = (
        w.tier1_must_haves * tier1 +
        w.tier2_good_to_haves * tier2 +
        w.tier3_quality_signals * tier3 +
        w.tier4_bonus * tier4
    )
    final_score = max(0.0, min(raw_score - penalty, 1.0))

    logger.info(
        f"{candidate_name}: score={final_score:.3f} "
        f"(T1={tier1:.2f}, T2={tier2:.2f}, T3={tier3:.2f}, T4={tier4:.2f}, penalty={penalty:.2f})"
    )

    return ScoringBreakdown(
        candidate_name=candidate_name,
        passed_gate=True,
        gate_failure_reason=None,
        tier1_score=tier1,
        tier2_score=tier2,
        tier3_score=tier3,
        tier4_score=tier4,
        penalty_score=penalty,
        final_score=final_score,
        required_skill_score=sub["required_skill_score"],
        experience_score=sub["experience_score"],
        seniority_score=sub["seniority_score"],
        domain_score=sub["domain_score"],
        preferred_skill_score=alignment.skill_coverage_preferred,
        semantic_score=alignment.semantic_similarity_score,
        quality_score=alignment.candidate_quality_score,
        matched_required_skills=alignment.matched_required_skills,
        missing_required_skills=alignment.missing_required_skills,
        matched_preferred_skills=alignment.matched_preferred_skills,
        seniority_match=alignment.seniority_match,
        experience_gap_years=alignment.experience_gap_years,
        bonus_signals=alignment.bonus_signals,
        penalty_signals=alignment.penalty_signals,
    )