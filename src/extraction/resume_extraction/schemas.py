"""
Domain-agnostic resume extraction schemas.

Designed to work across any candidate background — engineering, finance,
legal, marketing, healthcare, etc. No domain-specific assumptions.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"


class CompanyTier(str, Enum):
    TIER_1 = "tier_1"   # FAANG, top AI labs, elite firms
    TIER_2 = "tier_2"   # Strong known companies, funded unicorns
    TIER_3 = "tier_3"   # Mid-market, consulting, lesser-known
    TIER_4 = "tier_4"   # Unknown / very small / stealth
    ACADEMIC = "academic"


# ── Sub-schemas (assembled by agent tools) ─────────────────────────────────────

class CandidateIdentity(BaseModel):
    """Basic candidate metadata extracted from header/summary."""
    full_name: Optional[str] = None
    current_title: Optional[str] = None
    current_seniority: SeniorityLevel = SeniorityLevel.UNKNOWN
    email: Optional[str] = None


# class WorkRole(BaseModel):
#     """A single job/role entry."""
#     company: str
#     title: str
#     duration_years: float = Field(description="Computed duration. Use 0.5 for <6 months.")
#     company_tier: CompanyTier
#     domain: Optional[str] = Field(None, description="Industry/domain of the company")
#     seniority_at_role: SeniorityLevel = SeniorityLevel.UNKNOWN
#     key_responsibilities: list[str] = Field(
#         default_factory=list,
#         description="1-2 sentence summaries of what they actually did. NOT copied bullets."
#     )
#     impact_highlights: list[str] = Field(
#         default_factory=list,
#         description="Quantified or clearly significant achievements only."
#     )
#     has_quantified_impact: bool = Field(
#         description="True if any bullet contains a number, %, $, or measurable metric."
#     )
#     skills_demonstrated: list[str] = Field(
#         default_factory=list,
#         description="Atomic skill tokens clearly used in this role."
#     )
# ROLES:
# - key_responsibilities: Summarize in your own words (1-2 sentences). Do NOT copy bullet points verbatim.
# - impact_highlights: Only clearly quantified or objectively significant achievements.
#   e.g. "$2M cost savings", "reduced latency by 40%", "led team of 8". Not "improved processes".
# - has_quantified_impact: True if ANY bullet has a number/metric/dollar amount.
# - skills_demonstrated: Atomic skill tokens (1-4 words) actually used in this role. Evidence-based only.



class WorkHistory(BaseModel):
    """Full work history analysis."""
    highest_company_tier: CompanyTier

    # roles: list[WorkRole] = Field(default_factory=list)
    total_years_experience: float = Field(
        description=(
            "Computed by summing non-overlapping durations. "
            "Do not include internship experience. "
            "Use 0.5 for short stints without clear dates."
        )
    )
    has_leadership_experience: bool = Field(
        description="True if they led a team, project, or people at any point."
    )
    has_people_management: bool = Field(
        description="True if they had direct reports or hiring responsibility."
    )
    
    domains_worked_in: list[str] = Field(
        description="All industry domains across career. e.g. ['fintech', 'healthcare', 'e-commerce']"
    )
 
class ExplicitSkillsResult(BaseModel):
    """Output of tool_extract_explicit_skills. Skills section only."""
    explicit_skills: list[str] = Field(
        description=(
            "Atomic tokens (1-4 words) copied EXACTLY from a dedicated Skills, "
            "Technical Skills, or Core Competencies section. "
            "No inference. No additions. If no skills section exists, return []."
        )
    )
    explicit_skill_depth_signals: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 skills with clear mastery evidence. Short phrase per skill. "
            "e.g. 'PyTorch: fine-tuned 13B model on 4xA100s with FSDP'. "
            "Only include genuinely strong signals — be selective."
        )
    )
 
 
class ImplicitSkillsResult(BaseModel):
    """Output of tool_extract_implicit_skills. Work history + education only."""
    implicit_skills: list[str] = Field(
        description=(
            "Atomic tokens (1-4 words) clearly demonstrated in work history, "
            "projects, or education — but NOT listed in a skills section. "
            "Evidence-based only. Include domain-level concepts when overwhelmingly demonstrated."
        )
    )
    implicit_skill_depth_signals: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 skills with clear mastery evidence. Short phrase per skill. "
            "e.g. 'PyTorch: fine-tuned 13B model on 4xA100s with FSDP'. "
            "Only include genuinely strong signals — be selective."
        )
    )
 
 
class SkillsProfile(BaseModel):
    """Merged skills profile assembled from explicit + implicit extraction results."""
    explicit_skills: list[str] = Field(default_factory=list)
    implicit_skills: list[str] = Field(default_factory=list)
    explicit_skill_depth_signals: list[str] = Field(default_factory=list)
    implicit_skill_depth_signals: list[str] = Field(default_factory=list)
 
    @classmethod
    def from_results(
        cls,
        explicit: ExplicitSkillsResult,
        implicit: ImplicitSkillsResult,
    ) -> "SkillsProfile":
        """Merge the two focused extraction results into a single profile."""
        return cls(
            explicit_skills=explicit.explicit_skills,
            implicit_skills=implicit.implicit_skills,
            explicit_skill_depth_signals=explicit.explicit_skill_depth_signals,
            implicit_skill_depth_signals=implicit.implicit_skill_depth_signals,
        )

class EducationAndCredentials(BaseModel):
    """Education, certifications, publications, open source."""
    highest_degree: Optional[str] = Field(None, description="e.g. 'PhD Computer Science', 'MBA', 'BS Mechanical Engineering'")
    institution_tier: Optional[str] = Field(
        None,
        description="'tier_1' (top 20 global), 'tier_2' (strong regional), 'tier_3' (unknown), 'online_only'"
    )

    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications. e.g. ['CPA', 'AWS Solutions Architect', 'PMP']"
    )
    publications: list[str] = Field(
        default_factory=list,
        description="Published papers, articles, or research. Include venue if available."
    )
    open_source_contributions: list[str] = Field(
        default_factory=list,
        description="OSS projects, GitHub repos with stars, notable contributions."
    )

    has_publications: bool = False
    has_open_source: bool = False





# ── Final assembled profile ────────────────────────────────────────────────────

class ResumeProfile(BaseModel):
    """
    Complete structured profile of a resume.
    Assembled from sub-schemas extracted by the ReAct agent.
    Domain-agnostic — works for any candidate type.
    """
    identity: CandidateIdentity
    work_history: WorkHistory
    skills: SkillsProfile
    education: EducationAndCredentials

    # Synthesized by agent
    career_archetype: str = Field(
        description=(
            "One concise label for the candidate's professional identity. "
            "e.g. 'ML Platform Engineer', 'Generalist Builder', 'Researcher-Practitioner', "
            "'Enterprise Sales Leader', 'Full-Stack Product Engineer', 'Clinical Data Scientist'"
        )
    )
    career_narrative: str = Field(
        description="2-3 sentences telling the story of this person's career arc. Be specific."
    )
    green_flags: list[str] = Field(
        default_factory=list,
        description="Genuine standout signals. Be selective — only real differentiators."
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Real concerns: gaps, stagnation, all buzz no substance, "
            "claims without evidence, inconsistencies."
        )
    )
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Agent self-assessed confidence. "
            "1.0 = resume was detailed and clear. "
            "0.5 = some sections sparse or ambiguous. "
            "0.2 = very sparse resume, many fields inferred."
        )
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Ambiguities, missing sections, or low-confidence extractions to flag."
    )

    # Convenience properties for backward compatibility with scoring layer
    @property
    def candidate_name(self) -> Optional[str]:
        return self.identity.full_name

    @property
    def current_seniority(self) -> SeniorityLevel:
        return self.identity.current_seniority

    @property
    def total_years_experience(self) -> float:
        return self.work_history.total_years_experience

    @property
    def explicit_skills(self) -> list[str]:
        return self.skills.explicit_skills

    @property
    def implicit_skills(self) -> list[str]:
        return self.skills.implicit_skills

    @property
    def domains_worked_in(self) -> list[str]:
        return self.work_history.domains_worked_in

    @property
    def highest_company_tier(self) -> CompanyTier:
        return self.work_history.highest_company_tier

    @property
    def has_leadership_experience(self) -> bool:
        return self.work_history.has_leadership_experience

    @property
    def has_open_source(self) -> bool:
        return self.education.has_open_source

    @property
    def has_publications(self) -> bool:
        return self.education.has_publications