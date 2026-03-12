"""
JD Extraction Pipeline — deterministic sequential tool calls.

Replaces the ReAct agent with a fixed, pre-determined call sequence:
    1. extract_role_identity
    2. extract_hard_requirements
    3. extract_soft_requirements
    4. extract_role_character
    5. synthesize → JDProfile

Why deterministic over ReAct:
- Tool order is always the same — the reasoner loop added cost with no real benefit
- Faster: no extra LLM call per step to decide what to call next
- Cheaper: N tools = N LLM calls, not 2N
- Easier to debug: execution path is always predictable
- Failures are isolated: one tool failing doesn't corrupt the rest
"""

import logging
from pydantic import BaseModel

from src.extraction.jd_extraction.schemas import (
    JDProfile, RoleIdentity, HardRequirements, SoftRequirements,
    RoleCharacter, OwnershipStyle, WorkStyle
)
from src.extraction.jd_extraction.agent_tools import (
    tool_extract_role_identity,
    tool_extract_hard_requirements,
    tool_extract_soft_requirements,
    tool_extract_role_character,
)
from src.utils.llm_client import extract_structured

logger = logging.getLogger(__name__)


# ── Synthesizer ────────────────────────────────────────────────────────────────

def _synthesize(
    jd_text: str,
    role_identity: RoleIdentity,
    hard_requirements: HardRequirements,
    soft_requirements: SoftRequirements,
    role_character: RoleCharacter,
) -> JDProfile:
    """Assemble sub-schema outputs into a final JDProfile."""

    class SynthMeta(BaseModel):
        ideal_candidate_persona: str
        role_in_one_line: str
        # extraction_confidence: float
        # extraction_notes: list[str]

    system = """
Synthesize the final summary fields for a JD extraction.

ideal_candidate_persona: 2-3 concrete sentences about what the ideal hire looks like.
  Be specific — not generic HR language.
role_in_one_line: One punchy sentence summarizing the role for a potential candidate.
"""
    user = f"""
Job: {role_identity.job_title} ({role_identity.seniority_level})
Required skills: {hard_requirements.required_skills}
Required YoE: {hard_requirements.required_years_of_experience}
Preferred skills: {soft_requirements.preferred_skills}
Ownership: {role_character.ownership_style} | Work style: {role_character.work_style}

JD excerpt: {jd_text[:600]}
"""
    meta = extract_structured(system, user, SynthMeta)

    return JDProfile(
        role_identity=role_identity,
        hard_requirements=hard_requirements,
        soft_requirements=soft_requirements,
        role_character=role_character,
        ideal_candidate_persona=meta.ideal_candidate_persona,
        role_in_one_line=meta.role_in_one_line,
        # extraction_confidence=meta.extraction_confidence,
        # extraction_notes=meta.extraction_notes,
    )

# extraction_confidence: 0.0-1.0. How complete and unambiguous was the JD?
#   1.0 = detailed and specific. 0.5 = vague in places. 0.2 = very sparse.
# extraction_notes: List any ambiguities or low-confidence fields.

# ── Safe tool runner ───────────────────────────────────────────────────────────

def _run_tool(name: str, fn, jd_text: str, fallback):
    """Run a single extraction tool. Returns fallback value on failure."""
    try:
        result = fn(jd_text)
        logger.info(f"[JD] {name}: OK")
        return result
    except Exception as e:
        logger.warning(f"[JD] {name}: FAILED — {e}. Using fallback.")
        return fallback


# ── Main extraction function ───────────────────────────────────────────────────

def extract_jd(jd_text: str) -> JDProfile:
    """
    Extract structured JD profile via deterministic sequential tool calls.

    Call order:
        1. extract_role_identity      → who/what/where
        2. extract_hard_requirements  → must-haves
        3. extract_soft_requirements  → nice-to-haves
        4. extract_role_character     → ownership + work style
        5. synthesize                 → persona, confidence, notes

    Args:
        jd_text: Raw job description text (any domain).

    Returns:
        JDProfile: Fully structured, domain-agnostic JD representation.
    """
    logger.info("Starting JD extraction...")

    role_identity = _run_tool(
        "extract_role_identity", tool_extract_role_identity, jd_text,
        fallback=RoleIdentity(job_title="Unknown", seniority_level="unknown")
    )
    hard_requirements = _run_tool(
        "extract_hard_requirements", tool_extract_hard_requirements, jd_text,
        fallback=HardRequirements(required_skills=[])
    )
    soft_requirements = _run_tool(
        "extract_soft_requirements", tool_extract_soft_requirements, jd_text,
        fallback=SoftRequirements()
    )
    role_character = _run_tool(
        "extract_role_character", tool_extract_role_character, jd_text,
        fallback=RoleCharacter(
            ownership_style=OwnershipStyle.UNKNOWN,
            work_style=WorkStyle.UNKNOWN,
        )
    )

    profile = _synthesize(
        jd_text, role_identity, hard_requirements, soft_requirements, role_character
    )

    logger.info(
        f"JD extracted: '{profile.role_identity.job_title}' | "
        f"Seniority: {profile.role_identity.seniority_level} | "
        f"Required skills: {len(profile.hard_requirements.required_skills)} | "
        # f"Confidence: {profile.extraction_confidence:.2f}"
    )
    return profile