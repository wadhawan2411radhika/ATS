"""
Scoring layer schemas.
AlignmentResult lives here — it's produced by the aligner and consumed by the scorer.
"""

from pydantic import BaseModel, Field


class AlignmentResult(BaseModel):
    """Explicit mapping of how a resume aligns to a JD. Input to scoring layer."""

    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    skill_coverage_required: float = Field(ge=0.0, le=1.0)
    skill_coverage_preferred: float = Field(ge=0.0, le=1.0)

    experience_gap_years: float
    seniority_match: str

    domain_overlap: list[str]
    domain_match_score: float = Field(ge=0.0, le=1.0)

    ownership_style_match: bool
    work_style_match: bool

    semantic_similarity_score: float = Field(ge=0.0, le=1.0)
    candidate_quality_score: float = Field(ge=0.0, le=1.0)

    bonus_signals: list[str] = Field(default_factory=list)
    penalty_signals: list[str] = Field(default_factory=list)

    # Passed through from ResumeProfile for scorer use
    highest_company_tier: str | None = Field(default=None, description="e.g. 'tier_1', 'tier_2', 'academic'")