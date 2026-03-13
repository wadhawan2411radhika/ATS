"""
Scoring Engine — applies weights to AlignmentResult, enforces gate checks.

Three-component scoring:

    final_score = skill_score      × 0.55
                + experience_score × 0.25
                + quality_score    × 0.20

Why these weights:
    Skill match is the dominant signal — a candidate who can't do the work
    shouldn't rank highly regardless of trajectory or credentials.
    Experience is a meaningful but secondary filter.
    Quality (depth, red flags, bonus credentials) differentiates candidates
    who are otherwise similarly skilled.

Tier naming (main.py-compatible):
    tier1 → skill_score
    tier2 → experience_score
    tier3 → quality_score
    tier4 → 0.0  (no separate bonus tier; bonus folded into quality_score)

ScoringBreakdown is defined in schemas.py and re-exported here so that
engine.py's existing import `from src.scoring.scorer import score, ScoringBreakdown`
continues to work without modification.
"""

import logging

from src.extraction.jd_extraction.schemas import JDProfile
from src.extraction.resume_extraction.schemas import ResumeProfile
from src.scoring.schemas import AlignmentResult, ScoringBreakdown   
from config import config

logger = logging.getLogger(__name__)

# ── Weights ────────────────────────────────────────────────────────────────────
# In production these would be learned from recruiter feedback data.

_SKILL_WEIGHT      = 0.55
_EXPERIENCE_WEIGHT = 0.25
_QUALITY_WEIGHT    = 0.20

assert abs(_SKILL_WEIGHT + _EXPERIENCE_WEIGHT + _QUALITY_WEIGHT - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"


# ── Public API ─────────────────────────────────────────────────────────────────

def score(
    jd: JDProfile,
    resume: ResumeProfile,
    alignment: AlignmentResult,
) -> ScoringBreakdown:
    """
    Apply weights and gate checks to produce the final ScoringBreakdown.

    Gate logic (score = 0.0 immediately if failed):
        1. Required skill coverage < config.gate.min_required_skill_coverage (default 25%)
        2. Candidate YoE < 50% of required YoE

    Args:
        jd:        Extracted JD profile.
        resume:    Extracted resume profile.
        alignment: Raw alignment signals from aligner.align().

    Returns:
        ScoringBreakdown with final_score in [0.0, 1.0] and full explainability.
    """
    required_yoe = jd.hard_requirements.required_years_of_experience
    candidate_name = resume.candidate_name  # may be None if extraction missed it

    # ── Gate 1: skill coverage ─────────────────────────────────────────────────
    if (
        config.gate.enforce_required_skills_gate
        and alignment.required_skill_coverage < config.gate.min_required_skill_coverage
    ):
        reason = (
            f"Skill coverage {alignment.required_skill_coverage:.0%} is below "
            f"the minimum threshold of {config.gate.min_required_skill_coverage:.0%}"
        )
        logger.debug(f"Gate FAILED (skill): {candidate_name or 'Unknown'} — {reason}")
        return _gated_out(candidate_name, alignment, reason)

    # ── Gate 2: experience floor ───────────────────────────────────────────────
    if (
        config.gate.enforce_min_experience
        and required_yoe is not None
        and required_yoe > 0
        and resume.total_years_experience < required_yoe * 0.5
    ):
        reason = (
            f"Candidate YoE ({resume.total_years_experience:.1f}y) is less than 50% "
            f"of required ({required_yoe:.1f}y)"
        )
        logger.debug(f"Gate FAILED (YoE): {candidate_name or 'Unknown'} — {reason}")
        return _gated_out(candidate_name, alignment, reason)

    # ── Weighted final score ───────────────────────────────────────────────────
    raw = (
        alignment.skill_score      * _SKILL_WEIGHT
        + alignment.experience_score * _EXPERIENCE_WEIGHT
        + alignment.quality_score    * _QUALITY_WEIGHT
    )
    final = round(min(1.0, max(0.0, raw)), 4)

    logger.debug(
        f"Score: {candidate_name or 'Unknown'} → {final:.4f} "
        f"(T1={alignment.skill_score:.3f} × {_SKILL_WEIGHT} + "
        f"T2={alignment.experience_score:.3f} × {_EXPERIENCE_WEIGHT} + "
        f"T3={alignment.quality_score:.3f} × {_QUALITY_WEIGHT})"
    )

    return ScoringBreakdown(
        candidate_name=candidate_name,
        passed_gate=True,
        gate_failure_reason=None,
        matched_required_skills=alignment.matched_required_skills,
        missing_required_skills=alignment.missing_required_skills,
        matched_preferred_skills=alignment.matched_preferred_skills,
        experience_gap_years=alignment.experience_gap_years,
        seniority_match=alignment.seniority_match,
        tier1_score=round(alignment.skill_score, 4),
        tier2_score=round(alignment.experience_score, 4),
        tier3_score=round(alignment.quality_score, 4),
        bonus_signals=alignment.bonus_signals,
        penalty_signals=alignment.penalty_signals,
        final_score=final,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _gated_out(
    candidate_name: str | None,
    alignment: AlignmentResult,
    reason: str,
) -> ScoringBreakdown:
    """
    Zero-score breakdown for a candidate that failed a gate check.
    Preserves matched/missing skills and seniority for the display table
    so the recruiter can see *why* the candidate was filtered.
    """
    return ScoringBreakdown(
        candidate_name=candidate_name,
        passed_gate=False,
        gate_failure_reason=reason,
        matched_required_skills=alignment.matched_required_skills,
        missing_required_skills=alignment.missing_required_skills,
        matched_preferred_skills=[],
        experience_gap_years=alignment.experience_gap_years,
        seniority_match=alignment.seniority_match,
        tier1_score=0.0,
        tier2_score=0.0,
        tier3_score=0.0,
        bonus_signals=[],
        penalty_signals=alignment.penalty_signals,
        final_score=0.0,
    )