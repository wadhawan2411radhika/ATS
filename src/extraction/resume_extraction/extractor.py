"""
Resume Extraction Pipeline — deterministic sequential tool calls.

Fixed call sequence:
    1. extract_identity
    2. extract_work_history
    3. extract_skills
    4. extract_education
    5. assess_quality_signals
    6. synthesize → ResumeProfile
"""

import logging
from pydantic import BaseModel

from src.extraction.resume_extraction.schemas import (
    ResumeProfile, CandidateIdentity, WorkHistory, SkillsProfile,
    EducationAndCredentials, CompanyTier, ExplicitSkillsResult, ImplicitSkillsResult
)
from src.extraction.resume_extraction.agent_tools import (
    tool_extract_identity,
    tool_extract_work_history,
    tool_extract_explicit_skills,
    tool_extract_implicit_skills,
    tool_extract_education,
)
from src.utils.llm_client import extract_structured

logger = logging.getLogger(__name__)


# ── Synthesizer ────────────────────────────────────────────────────────────────

def _synthesize(
    resume_text: str,
    identity: CandidateIdentity,
    work_history: WorkHistory,
    skills: SkillsProfile,
    education: EducationAndCredentials,
) -> ResumeProfile:
    """Assemble sub-schema outputs into a final ResumeProfile."""

    class SynthMeta(BaseModel):
        career_archetype: str
        career_narrative: str
        green_flags: list[str]
        red_flags: list[str]
        extraction_confidence: float
        extraction_notes: list[str]

    system = """
Synthesize the final narrative fields for a resume extraction.

career_archetype: One concise professional identity label. Be specific.
  e.g. "ML Platform Engineer", "Researcher-Practitioner", "Enterprise Sales Lead"

career_narrative: 2-3 specific sentences about this person's career arc.

green_flags: 2-5 genuine standout signals only. Be selective.
red_flags: Real concerns — gaps, stagnation, all buzz no substance, inconsistencies.

extraction_confidence: 0.0-1.0. How complete was the resume?
extraction_notes: Ambiguities or low-confidence fields to flag.
"""
    user = f"""
Name: {identity.full_name} | Title: {identity.current_title} ({identity.current_seniority})
YoE: {work_history.total_years_experience}
Highest company tier: {work_history.highest_company_tier}
Domains: {work_history.domains_worked_in}
Explicit skills: {skills.explicit_skills}
Implicit skills: {skills.implicit_skills}
Education: {education.highest_degree}
Publications: {education.has_publications} | OSS: {education.has_open_source}

Resume excerpt: {resume_text[:600]}
"""
    meta = extract_structured(system, user, SynthMeta)

    return ResumeProfile(
        identity=identity,
        work_history=work_history,
        skills=skills,
        education=education,
        career_archetype=meta.career_archetype,
        career_narrative=meta.career_narrative,
        green_flags=meta.green_flags,
        red_flags=meta.red_flags,
        extraction_confidence=meta.extraction_confidence,
        extraction_notes=meta.extraction_notes,
    )


# ── Safe tool runner ───────────────────────────────────────────────────────────

def _run_tool(name: str, fn, resume_text: str, fallback):
    """Run a single extraction tool. Returns fallback value on failure."""
    try:
        result = fn(resume_text)
        logger.info(f"[Resume] {name}: OK")
        return result
    except Exception as e:
        logger.warning(f"[Resume] {name}: FAILED — {e}. Using fallback.")
        return fallback


# ── Main extraction function ───────────────────────────────────────────────────

def extract_resume(resume_text: str) -> ResumeProfile:
    """
    Extract structured resume profile via deterministic sequential tool calls.

    Call order:
        1. extract_identity           → name, title, seniority, contact
        2. extract_work_history       → roles, YoE, trajectory, leadership
        3. extract_skills             → explicit + implicit, recency, depth
        4. extract_education          → degrees, certs, publications, OSS
        5. assess_quality_signals     → exaggeration, specificity, consistency
        6. synthesize                 → archetype, narrative, flags

    Args:
        resume_text: Raw resume text (parsed from PDF/DOCX/TXT).

    Returns:
        ResumeProfile: Fully structured, domain-agnostic resume representation.
    """
    logger.info("Starting resume extraction...")

    identity = _run_tool(
        "extract_identity", tool_extract_identity, resume_text,
        fallback=CandidateIdentity()
    )

    education = _run_tool(
        "extract_education", tool_extract_education, resume_text,
        fallback=EducationAndCredentials()
    )

    work_history = _run_tool(
        "extract_work_history", tool_extract_work_history, resume_text,
        fallback=WorkHistory(
            total_years_experience=0.0,
            has_leadership_experience=False,
            has_people_management=False,
            highest_company_tier=CompanyTier.TIER_4,
            domains_worked_in=[],
        )
    )
    explicit_skills = _run_tool(
        "extract_explicit_skills", tool_extract_explicit_skills, resume_text,
        fallback=ExplicitSkillsResult(explicit_skills=[])
    )
    implicit_skills = _run_tool(
        "extract_implicit_skills", tool_extract_implicit_skills, resume_text,
        fallback=ImplicitSkillsResult(implicit_skills=[])
    )
    skills = SkillsProfile.from_results(explicit_skills, implicit_skills)
    

    profile = _synthesize(resume_text, identity, work_history, skills, education)

    logger.info(
        f"Resume extracted: {profile.identity.full_name or 'Unknown'} | "
        f"{profile.career_archetype} | "
        f"YoE: {profile.total_years_experience} | "
        f"Confidence: {profile.extraction_confidence:.2f}"
    )
    return profile