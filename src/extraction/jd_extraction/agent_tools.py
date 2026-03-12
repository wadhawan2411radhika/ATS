"""
JD Extraction Agent Tools.

Each tool is a focused, single-responsibility LLM call.
The ReAct agent decides which to call and in what order.

Design principles:
- Domain-agnostic: no ML/AI assumptions
- Each tool returns a validated Pydantic sub-schema
- Failures are surfaced as structured errors, not exceptions
- Tools can be re-invoked if the agent detects low-quality output
"""

import logging
from src.utils.llm_client import extract_structured
from src.extraction.jd_extraction.schemas import (
    RoleIdentity, HardRequirements, SoftRequirements,
    RoleCharacter
)

logger = logging.getLogger(__name__)


# ── Tool 1: Role Identity ──────────────────────────────────────────────────────

def tool_extract_role_identity(jd_text: str) -> RoleIdentity:
    """
    Extract basic role metadata: title, seniority, location, employment type.

    Agent use: always call this first — establishes the frame for all other extractions.
    """
    system = """
You extract factual role metadata from job descriptions. Be precise and literal.

Rules:
- job_title: Use the exact title from the JD. Do not paraphrase.
- seniority_level: Infer from title AND from responsibility language.
  "Lead", "define strategy", "mentor", "drive roadmap" = senior/lead/staff.
  "Support", "assist", "learn", "under guidance" = junior/intern.
  If unclear, use "unknown".
- company_stage: Infer from context if not stated.
  Mentions of "Series A/B/C", "post-IPO", "Fortune 500", "early stage", "stealth" are signals.
- remote_policy: Extract exactly. "3 days in office" → "hybrid 3 days/week onsite".
- employment_type: Infer from context if not stated.
  Mentions of "full-time", "contract", "part-time"
- If a field is genuinely absent, return null — do not invent.
"""
    user = f"Extract role identity metadata from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, RoleIdentity)


# ── Tool 2: Hard Requirements ──────────────────────────────────────────────────

def tool_extract_hard_requirements(jd_text: str) -> HardRequirements:
    """
    Extract non-negotiable requirements: required skills, YoE, education, certifications.

    Agent use: call after role identity. This is the highest-stakes extraction.
    """
    system = """
You extract hard (non-negotiable and clearly stated must have) requirements from job descriptions.

CRITICAL — SKILLS MUST BE ATOMIC TOKENS:
- Each skill must be 1-4 words. Never copy JD sentences into skill lists.
- ✓ Correct: ["Python", "SQL", "financial modeling", "clinical trial design", "Kubernetes", "GAAP accounting"]
- ✗ Wrong: ["experience building scalable systems", "strong communication skills"]
- Extract skills from ALL sections — "requirements", "qualifications", "you will need", "must have".

- For each skill ask: "Is this explicitly stated as required or clearly non-negotiable?"
  If yes → required_skills. If uncertain → leave for soft requirements.
- Include both technical skills (tools, languages, frameworks) AND functional skills
  (e.g. "stakeholder management", "P&L ownership", "clinical documentation") AND domain knowledge
  (e.g. "options trading", "HIPAA compliance", "IFRS accounting").

- required_years_of_experience: Extract the minimum number. If "5-8 years" → 5.0. If not stated → null.
- required_education: Be specific. "BS/MS in CS" not just "degree required".
- required_certifications: Only hard requirements. e.g. CPA, PMP, bar license, medical license.
"""
    user = f"Extract hard (non-negotiable) requirements from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, HardRequirements)


# ── Tool 3: Soft Requirements ──────────────────────────────────────────────────

def tool_extract_soft_requirements(jd_text: str) -> SoftRequirements:
    """
    Extract preferred (nice-to-have) requirements.

    Agent use: call after hard requirements. Differentiates candidates who exceed the bar.
    """
    system = """
You extract soft (preferred, nice-to-have) requirements from job descriptions.

Rules:
- Only include what is explicitly framed as optional: "nice to have", "bonus", "preferred",
  "plus", "ideally", "we'd love if", "familiarity with".
- Each skill must be 1-4 words. Never copy JD sentences into skill lists.
- ✓ Correct: ["Python", "SQL", "financial modeling", "clinical trial design", "Kubernetes", "GAAP accounting"]
- ✗ Wrong: ["experience building scalable systems", "strong communication skills"]
- Extract skills from ALL sections — "requirements", "qualifications", "you will need", "must have".

Fields to be extracted:
- preferred_skills: Atomic skill tokens from nice to have, bonus, preferred sections.
- preferred_domain_experience: Domains that would strengthen the application but aren't mandatory.
- preferred_education: Preferred but not mandatory education level.
- preferred_company_background: Signals about where the candidate came from.
  e.g. "startup experience preferred", "Big 4 background a plus", "prior agency experience helpful".
- If the JD has no nice-to-haves section, return empty lists — do not invent.
"""
    user = f"Extract soft (preferred/nice-to-have) requirements from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, SoftRequirements)


# ── Tool 4: Role Character ─────────────────────────────────────────────────────

def tool_extract_role_character(jd_text: str) -> RoleCharacter:
    """
    Extract ownership style and work style from the JD.
    These two signals drive candidate-role fit scoring.
    """
    system = """
You extract two role character signals from a job description.
 
ownership_style — pick exactly one:
- "executor": clear tasks handed down. Language: "implement", "support", "maintain", "execute".
- "ic_owner": owns outcomes end-to-end with autonomy. Language: "own", "drive", "define", "lead the work".
- "tech_lead": sets technical direction for a team. Language: "lead the team", "set standards", "architect for others".
- "player_coach": leads a small team AND still does hands-on work. Language: "lead while still coding/building".
- "manager": primarily people management. Language: "hire", "performance reviews", "grow the team", "headcount".
- "unknown": genuinely unclear from the JD.
 
work_style — pick exactly one:
- "research": exploration, experimentation, prototyping. Language: "investigate", "experiment", "publish", "discover".
- "execution": build and ship to production. Language: "deploy", "deliver", "launch", "production", "scale".
- "hybrid": clear evidence of both research AND execution expectations.
- "unknown": genuinely unclear.
"""
    user = f"Extract ownership_style and work_style from this JD:\n\n---\n{jd_text}\n---"
    return extract_structured(system, user, RoleCharacter)
