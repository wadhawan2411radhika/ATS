"""
Explainability Layer.

Takes a ScoringBreakdown and generates a recruiter-facing natural language summary.
Only called for top-N candidates to control LLM cost.
"""

import logging
from src.utils.llm_client import extract_structured
from src.scoring.scorer import ScoringBreakdown
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RecruiterExplanation(BaseModel):
    headline: str = Field(description="One sentence verdict on this candidate — punchy and honest")
    why_strong: list[str] = Field(description="Top 3 reasons this candidate is a good match")
    why_weak: list[str] = Field(description="Top 2-3 specific concerns or gaps")
    interview_focus: list[str] = Field(description="2-3 things to probe in the interview given the gaps")
    recommendation: str = Field(description="One of: 'Strong Yes', 'Lean Yes', 'Maybe', 'Lean No', 'No'")


SYSTEM_PROMPT = """
You are an expert technical recruiter making hiring recommendations.
Your job is to synthesize a candidate's scoring breakdown into a clear, honest, actionable summary for a hiring manager.
Be direct. Do not be a cheerleader. If there are concerns, state them plainly.
"""


def explain(breakdown: ScoringBreakdown, jd_title: str) -> RecruiterExplanation:
    """
    Generate a recruiter-facing natural language explanation for a candidate's score.

    Args:
        breakdown: The full scoring breakdown for a candidate.
        jd_title: The job title they're being evaluated for.

    Returns:
        RecruiterExplanation with headline, pros, cons, and recommendation.
    """
    user_prompt = f"""
Generate a recruiter summary for the following candidate evaluation for the role: {jd_title}

CANDIDATE: {breakdown.candidate_name}
FINAL SCORE: {breakdown.final_score:.2f} / 1.0

SKILL SIGNALS:
- Matched required skills: {', '.join(breakdown.matched_required_skills) or 'None'}
- Missing required skills: {', '.join(breakdown.missing_required_skills) or 'None'}
- Matched preferred skills: {', '.join(breakdown.matched_preferred_skills) or 'None'}

EXPERIENCE:
- Seniority match: {breakdown.seniority_match}
- Experience gap: {breakdown.experience_gap_years:+.1f} years vs requirement

QUALITY:
- Candidate quality score: {breakdown.quality_score:.2f}
- Semantic fit: {breakdown.semantic_score:.2f}

TIER SCORES:
- Must-haves (50%): {breakdown.tier1_score:.2f}
- Good-to-haves (25%): {breakdown.tier2_score:.2f}
- Quality signals (15%): {breakdown.tier3_score:.2f}
- Bonus (5%): {breakdown.tier4_score:.2f}

POSITIVE SIGNALS: {', '.join(breakdown.bonus_signals) or 'None'}
CONCERNS: {', '.join(breakdown.penalty_signals) or 'None'}

Write an honest, specific, actionable recruiter summary.
"""

    return extract_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=RecruiterExplanation,
    )
