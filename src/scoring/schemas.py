"""
Scoring layer data containers.

AlignmentResult  — raw signals computed by the aligner (pre-weighting)
ScoringBreakdown — final weighted scores + explainability fields for main.py

Kept as plain dataclasses (not Pydantic) — these are internal objects,
not LLM outputs, so schema enforcement overhead is unnecessary.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AlignmentResult:
    """
    Raw alignment signals between a JD and a resume.
    Computed entirely by aligner.py — no weights applied yet.
    """

    # ── Skill alignment ────────────────────────────────────────────────────────
    matched_required_skills: list[str]       # JD required skills that matched
    missing_required_skills: list[str]       # JD required skills with no match
    matched_preferred_skills: list[str]      # JD preferred skills that matched
    required_skill_coverage: float           # 0→1, recency-weighted coverage of required skills
    preferred_skill_coverage: float          # 0→1, recency-weighted coverage of preferred skills

    # ── Experience alignment ───────────────────────────────────────────────────
    experience_gap_years: float              # candidate_yoe - required_yoe (negative = under)
    yoe_score: float                         # 0→1 score for years of experience
    seniority_match: str                     # human-readable: "exact match", "1 level above", etc.
    seniority_score: float                   # 0→1 score for seniority alignment
    domain_overlap_score: float              # 0→1 overlap of candidate domains vs JD domains

    # ── Quality signals ────────────────────────────────────────────────────────
    depth_score: float                       # 0→1 from skill_depth_signals (mastery evidence)
    red_flag_penalty: float                  # 0→0.4 penalty from red_flags
    bonus: float                             # 0→0.3 from publications, OSS, certifications
    bonus_signals: list[str]                 # human-readable bonus descriptions
    penalty_signals: list[str]               # human-readable penalty descriptions (red flags)

    # ── Component scores (pre-weighting) ──────────────────────────────────────
    skill_score: float                       # 0→1
    experience_score: float                  # 0→1
    quality_score: float                     # 0→1


@dataclass
class ScoringBreakdown:
    """
    Final scoring output — weighted scores + gate result + explainability.

    Tier mapping (preserves main.py compatibility):
        tier1 → skill_score        (55% of final)
        tier2 → experience_score   (25% of final)
        tier3 → quality_score      (20% of final)
        tier4 → 0.0                (no separate tier; bonus folded into quality)
    """

    # ── Gate ──────────────────────────────────────────────────────────────────
    passed_gate: bool
    gate_failure_reason: Optional[str]

    # ── Skill explainability ───────────────────────────────────────────────────
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]

    # ── Experience explainability ──────────────────────────────────────────────
    experience_gap_years: float
    seniority_match: str

    # ── Tier scores (main.py-compatible naming) ────────────────────────────────
    tier1_score: float   # skill
    tier2_score: float   # experience
    tier3_score: float   # quality
    tier4_score: float   # always 0.0

    # ── Bonus / penalty labels ─────────────────────────────────────────────────
    bonus_signals: list[str]
    penalty_signals: list[str]

    # ── Final ──────────────────────────────────────────────────────────────────
    final_score: float

    def to_dict(self) -> dict:
        return asdict(self)