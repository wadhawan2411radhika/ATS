"""
Resume Extraction Agent Tools.

Five focused, single-responsibility extraction tools.
Each handles one natural section of a resume.

Design principles:
- Atomic skill tokens only — never sentence-length skill entries
- Evidence-based — only extract what is actually present
- Domain-agnostic — works for any professional background
"""

import logging
from src.utils.llm_client import extract_structured
from src.extraction.resume_extraction.schemas import (
    CandidateIdentity, WorkHistory, SkillsProfile,
    EducationAndCredentials, ExplicitSkillsResult, ImplicitSkillsResult
)

logger = logging.getLogger(__name__)


# ── Tool 1: Identity ───────────────────────────────────────────────────────────

def tool_extract_identity(resume_text: str) -> CandidateIdentity:
    """
    Extract candidate identity from resume header and summary.
    """
    system = """
You extract candidate identity information from the resume.

Rules:
- full_name: Exactly as written. null if not found.
- current_title: Their most recent or stated job title. null if absent.
- current_seniority: Infer from title AND from the responsibilities described.
  "Staff", "Principal", "Director" in title → staff/principal/director.
  "Lead X" or "owned end-to-end" → lead/senior.
  "Junior", "Associate", "Graduate" → junior.
  PhD postdocs with no industry → treat as mid unless they have clear seniority signals.
- email: Exactly as written. null if not found 
- If a field is absent, return null — never invent contact details.
"""
    user = f"Extract identity information from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, CandidateIdentity)


# ── Tool 2: Work History ───────────────────────────────────────────────────────

def tool_extract_work_history(resume_text: str) -> WorkHistory:
    """
    Extract structured work history: all roles, durations, impact, trajectory.
    The most information-dense tool — handles the experience section.
    """
    system = """
You extract work history from resumes with high precision.

HIGHEST COMPANY TIER:
- tier_1: FAANG + elite peers: Google, Meta, Apple, Amazon, Microsoft, Netflix, OpenAI,
  Anthropic, DeepMind, Cohere, Stripe, Palantir, Databricks, Snowflake, Goldman Sachs (top banks),
  McKinsey/BCG/Bain (MBB), top law firms (Cravath, Skadden), top hospitals (Mayo, Cleveland Clinic).
- tier_2: Strong known companies, funded unicorns ($1B+ valuation), well-known regional firms.
- tier_3: Mid-market, consulting firms not in MBB, lesser-known companies.
- tier_4: Unknown, very small, or stealth companies.
- academic: Universities, research labs, postdocs, fellowships.

TOTAL YEARS EXPERIENCE:
- Compute from date ranges. Sum non-overlapping periods for total_years_experience.
- "Present" = assume up to today. Use 0.5 for stints under 6 months or unclear.
- Take value if explicitly mentioned: eg "10+ years experience".
- Do not add internship experience into account.

LEADERSHIP:
- has_leadership_experience: Led a team, project, or initiative (not just senior IC work).
- has_people_management: Had direct reports, conducted performance reviews, or hired.
"""
    user = f"Extract complete work history from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, WorkHistory)

# ── Tool 3a: Explicit Skills ──────────────────────────────────────────────────
 
def tool_extract_explicit_skills(resume_text: str) -> ExplicitSkillsResult:
    """
    Extract ONLY skills listed in a dedicated skills section.
    Single responsibility: read the skills section, copy it exactly.
    """
    system = """
You extract skills from a dedicated Skills / Technical Skills / Core Competencies section.
 
RULES:
1. ONLY look at a dedicated skills section — ignore work history and education entirely.
2. Each skill must be an ATOMIC TOKEN: 1-4 words.
   ✓ "Python", "PyTorch", "RAG", "Kubernetes", "A/B testing", "GAAP accounting"
   ✗ "experience with Python", "building scalable systems"
3. Copy EXACTLY what is listed. No inference, no additions, no cleanup.
4. If the resume has no dedicated skills section, return explicit_skills: [].
5. For skill_depth_signals: identify depth for the given skill
Format: "SkillName: one-line evidence"
e.g. "PyTorch: fine-tuned 13B LLaMA on 32 A100s with FSDP, 12% ROUGE improvement"
"""
    user = f"Extract explicit skills from the Skills section of this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, ExplicitSkillsResult)
 
 
# ── Tool 3b: Implicit Skills ───────────────────────────────────────────────────
 
def tool_extract_implicit_skills(resume_text: str) -> ImplicitSkillsResult:
    """
    Extract skills demonstrated in work history and education — NOT from a skills section.
    Single responsibility: infer what the candidate can do from what they've done.
    """
    system = """
You extract implicit skills from a resume's work history, projects, and education.
Implicit = clearly demonstrated through experience, but NOT listed in a skills section.
 
RULES:
1. Ignore any dedicated Skills / Technical Skills section entirely.
2. Each skill must be an ATOMIC TOKEN: 1-4 words.
   ✓ "machine learning", "NLP", "ETL", "MLOps", "cloud platforms", "stakeholder management"
   ✗ "experience building scalable ML systems"
3. Evidence threshold — only include if clearly demonstrated:
   - STRONG evidence: skill appears across 2+ roles, OR is central to a job title, OR has quantified outcomes
   - WEAK evidence (exclude): mentioned once in passing, or vaguely implied
4. Include DOMAIN-LEVEL CONCEPTS when the evidence is overwhelming:
   - 8 years of ML engineering roles → "machine learning" is implicit even if not listed
   - NLP publications + NLP job titles → "Natural Language Processing" is implicit
   - ETL pipelines built in every role → "ETL", "data processing" are implicit
   - Deployed to AWS/GCP across multiple roles → "cloud platforms" is implicit
   - MLOps job title or owned infra → "MLOps" is implicit
5. Include functional skills if clearly evidenced:
   "stakeholder management", "P&L ownership", "performance reviews", "hiring"
6. Do NOT hallucinate. If uncertain → exclude.
 
For skill_depth_signals: identify 2-5 skills with clear mastery evidence.
Format: "SkillName: one-line evidence"
e.g. "PyTorch: fine-tuned 13B LLaMA on 32 A100s with FSDP, 12% ROUGE improvement"
Be selective — only genuine standouts, not every skill used.
"""
    user = f"Extract implicit skills from the work history and education of this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, ImplicitSkillsResult)


# ── Tool 4: Education & Credentials ───────────────────────────────────────────

def tool_extract_education(resume_text: str) -> EducationAndCredentials:
    """
    Extract education, certifications, publications, OSS, patents, talks.
    """
    system = """
You extract education and credentials from resumes.

DEGREES:
- highest_degree: Full degree name. e.g. "PhD Computer Science", "MBA", "LLB", "MD".
- List postdocs under highest_degree if they represent the terminal qualification.
- additional_degrees: All other completed degrees in full.

INSTITUTION TIER:
- "tier_1": Top 20 globally — MIT, Stanford, Harvard, Oxford, Cambridge, ETH Zurich,
  IIT (top campuses), NUS, Caltech, UCL, Imperial, CMU, Princeton, Yale, etc.
- "tier_2": Strong regional universities, well-regarded state schools, top programs
  (even if not globally elite — e.g. UT Austin CS, UWaterloo, UIUC, Purdue).
- "tier_3": Lesser-known or unranked institutions.

CERTIFICATIONS:
- Professional certs only: CPA, CFA, PMP, AWS Solutions Architect, GCP Professional ML Engineer,
  Bar License, Medical License, CKA, CISSP, etc.
- Do NOT include online course certificates like "Coursera ML Specialization" as certifications —
  these belong in education notes, not certifications.

PUBLICATIONS:
- Include paper title and venue if mentioned. e.g. "RAG-Dialogue — ACL 2023".
- Conference abbreviations are fine: ACL, NeurIPS, EMNLP, ICLR, ICML, CVPR, SIGIR.
- Set has_publications = True if ANY publications are present.

OPEN SOURCE:
- Include project name + any signal of adoption (stars, downloads, PyPI).
- Set has_open_source = True if ANY OSS contributions are present.


"""
    user = f"Extract education and credentials from this resume:\n\n---\n{resume_text}\n---"
    return extract_structured(system, user, EducationAndCredentials)
