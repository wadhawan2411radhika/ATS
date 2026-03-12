"""
Alignment Engine.

Takes a JDProfile (from ReAct agent) + ResumeExtracted and produces an AlignmentResult.
This is the core "comparison" layer — deterministic logic, no LLM needed.
The AlignmentResult is the input to the scoring layer.
"""

import re
import logging
from sentence_transformers import SentenceTransformer, util

from src.extraction.jd_extraction import JDProfile
from src.extraction.resume_extraction import (
    ResumeProfile as ResumeExtracted,
    SeniorityLevel,
    CompanyTier,
)
from src.scoring.schemas import AlignmentResult
from config import config

logger = logging.getLogger(__name__)

# Seniority ordering for comparison arithmetic
SENIORITY_ORDER = {
    SeniorityLevel.INTERN: 0,
    SeniorityLevel.JUNIOR: 1,
    SeniorityLevel.MID: 2,
    SeniorityLevel.SENIOR: 3,
    SeniorityLevel.LEAD: 4,
    SeniorityLevel.STAFF: 4,
    SeniorityLevel.PRINCIPAL: 5,
    SeniorityLevel.MANAGER: 3,
    SeniorityLevel.DIRECTOR: 5,
    SeniorityLevel.UNKNOWN: 2,
}

# Company tier → prestige score
COMPANY_TIER_SCORE = {
    CompanyTier.TIER_1: 1.0,
    CompanyTier.TIER_2: 0.75,
    CompanyTier.TIER_3: 0.5,
    CompanyTier.TIER_4: 0.25,
    CompanyTier.ACADEMIC: 0.5,
}


def _normalize_skill(skill: str) -> str:
    """Lowercase, strip, remove punctuation (keep hyphens)."""
    s = skill.lower().strip()
    s = re.sub(r"[^\w\s\-]", "", s)
    return s

# Common abbreviation aliases — applied to BOTH JD skills and resume skills before matching
_SKILL_ALIASES = {
    # ML/AI
    "ml": "machine learning", "nlp": "natural language processing",
    "natural language processing": "nlp",  # reverse — normalize to short form for matching
    "llm": "large language model", "llms": "large language model",
    "rl": "reinforcement learning", "dl": "deep learning",
    "genai": "generative ai", "gen ai": "generative ai",
    "ir": "information retrieval",
    # Frameworks
    "tf": "tensorflow", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "hf": "hugging face",
    # Cloud
    "gcp": "google cloud", "aws": "amazon web services",
    "azure": "microsoft azure", "k8s": "kubernetes",
    # Data
    "etl": "data pipeline", "data pipeline": "etl",
    "mlops": "ml operations", "ml operations": "mlops",
    "cloud platforms": "cloud", "cloud": "cloud platforms",
}

def _apply_aliases(skill: str) -> str:
    return _SKILL_ALIASES.get(skill, skill)

def _skill_matches(jd_skill_norm: str, resume_skills_norm: set[str]) -> bool:
    """
    Multi-strategy matching (in order of strictness):
    1. Exact match after normalization + alias expansion (both sides)
    2. JD skill contained in resume skill (only if JD skill >= 4 chars)
    3. Resume skill contained in JD skill (only if resume skill >= 5 chars)
    4. All significant words (>3 chars) of JD skill appear in a resume skill token
    """
    jd = _apply_aliases(jd_skill_norm)
    # Build alias-expanded resume set for strategy 1
    resume_aliased = {_apply_aliases(rs) for rs in resume_skills_norm}

    if jd in resume_skills_norm or jd in resume_aliased:
        return True
    for rs in resume_skills_norm:
        rs_a = _apply_aliases(rs)
        # Strategy 2: jd skill is a substring of a resume skill token
        if len(jd) >= 4 and jd in rs_a:
            return True
        # Strategy 3: resume skill is a substring of jd skill token
        if len(rs_a) >= 5 and rs_a in jd:
            return True
        # Strategy 4: all significant words of JD skill present in resume skill
        jd_words = {w for w in jd.split() if len(w) > 3}
        rs_words = {w for w in rs_a.split() if len(w) > 3}
        if len(jd_words) >= 2 and jd_words.issubset(rs_words):
            return True
    return False

def _skill_overlap(jd_skills: list[str], resume_skills: list[str]) -> tuple[list[str], list[str]]:
    """
    Compute matched and missing skills using fuzzy multi-strategy matching.
    Far more robust than exact match — handles aliases, substrings, and word overlap.
    """
    resume_norm = {_normalize_skill(s) for s in resume_skills}
    matched, missing = [], []
    for jd_skill in jd_skills:
        jd_norm = _normalize_skill(jd_skill)
        if _skill_matches(jd_norm, resume_norm):
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)
    return matched, missing


def _seniority_match_label(jd_level: SeniorityLevel, resume_level: SeniorityLevel) -> str:
    jd_score = SENIORITY_ORDER.get(jd_level, 2)
    resume_score = SENIORITY_ORDER.get(resume_level, 2)
    delta = resume_score - jd_score

    if delta == 0:
        return "aligned"
    elif delta > 0:
        return "overqualified"
    elif delta == -1:
        return "slightly underqualified"
    else:
        return "underqualified"


def _domain_match_score(jd_domains: list[str], resume_domains: list[str]) -> tuple[list[str], float]:
    """
    Compute domain overlap with fuzzy matching.
    Handles cases like 'AI' matching 'artificial intelligence', 'fintech' matching 'finance'.
    """
    if not jd_domains:
        return [], 1.0  # No domain requirement = full score
    if not resume_domains:
        return [], 0.0

    # Domain synonym expansions
    _DOMAIN_ALIASES = {
        "ai": ["artificial intelligence", "machine learning", "ml"],
        "ml": ["machine learning", "artificial intelligence", "ai"],
        "nlp": ["natural language processing", "language models"],
        "fintech": ["finance", "financial", "banking", "payments"],
        "healthtech": ["healthcare", "health", "medical", "clinical"],
        "e-commerce": ["ecommerce", "retail", "marketplace"],
        "b2b saas": ["saas", "enterprise software", "b2b"],
    }

    jd_norm = [d.lower().strip() for d in jd_domains]
    resume_norm = [d.lower().strip() for d in resume_domains]
    resume_text = " ".join(resume_norm)  # for substring search

    matched = []
    for jd_d in jd_norm:
        hit = False
        # Exact match
        if jd_d in resume_norm:
            hit = True
        # Substring: "fintech" matches "financial technology" or vice versa
        elif any(jd_d in r or r in jd_d for r in resume_norm):
            hit = True
        # Word overlap: "machine learning" matches resume domain "ml engineering"
        elif any(w in resume_text for w in jd_d.split() if len(w) > 3):
            hit = True
        # Alias expansion
        elif jd_d in _DOMAIN_ALIASES and any(
            alias in resume_text for alias in _DOMAIN_ALIASES[jd_d]
        ):
            hit = True
        if hit:
            matched.append(jd_d)

    score = len(matched) / len(jd_norm)
    return matched, min(score, 1.0)


def _candidate_quality_score(resume: ResumeExtracted) -> float:
    """
    Composite quality score from impact + specificity + project signals.
    Range: 0.0 - 1.0
    """
    # Impact quality
    impact_score = (
        0.5 * resume.quantified_impact_ratio +
        0.3 * resume.specificity_score +
        0.2 * (1.0 - resume.exaggeration_index)
    )

    # Project quality — use skill depth signals as proxy
    depth_signals = len(resume.skills.skill_depth_signals) if hasattr(resume, 'skills') else 0
    project_score = min(1.0, 0.3 + depth_signals * 0.15)  # 0.3 base, up to 1.0 with depth signals

    
    quality = (
        0.50 * impact_score +
        0.30 * project_score +
        0.20 * trajectory_score
    )
    return round(min(quality, 1.0), 3)


def _semantic_similarity(jd: JDProfile, resume: ResumeExtracted, model: SentenceTransformer) -> float:
    """
    Compute semantic similarity between JD persona description and resume narrative.
    Also includes skill-level semantic matching for implicit skill coverage.
    """
    # Text representations for embedding
    jd_text = (
        f"{jd.role_identity.job_title}. {jd.ideal_candidate_persona}. "
        f"Required: {', '.join(jd.hard_requirements.required_skills)}. "
        f"Preferred: {', '.join(jd.soft_requirements.preferred_skills[:5])}"
    )
    resume_text = (
        f"{resume.identity.current_title or ''}. {resume.career_narrative}. "
        f"Skills: {', '.join(resume.explicit_skills + resume.implicit_skills[:10])}. "
        f"Archetype: {resume.career_archetype}"
    )

    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    similarity = float(util.cos_sim(jd_emb, resume_emb))

    # Normalize from [-1,1] to [0,1]
    return round((similarity + 1) / 2, 3)


def _compute_bonus_signals(jd: JDProfile, resume: ResumeExtracted) -> list[str]:
    signals = []
    if resume.has_open_source:
        signals.append("Has open source contributions")
    if resume.has_publications:
        signals.append("Has publications / research output")
    if resume.highest_company_tier == CompanyTier.TIER_1:
        signals.append("Tier-1 company experience")
    if resume.has_leadership_experience:
        signals.append("Leadership experience")

    # Rare/niche skills the JD didn't explicitly ask for but are relevant
    all_jd_skills_norm = {_normalize_skill(s) for s in jd.hard_requirements.required_skills + jd.soft_requirements.preferred_skills}
    all_resume_skills_norm = {_normalize_skill(s) for s in resume.explicit_skills + resume.implicit_skills}
    extra_skills = all_resume_skills_norm - all_jd_skills_norm
    if len(extra_skills) > 3:
        signals.append(f"Additional relevant skills: {', '.join(list(extra_skills)[:4])}")

    return signals


# ── Load embedding model once ──────────────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {config.model.embedding_model}")
        _embedding_model = SentenceTransformer(
            config.model.embedding_model,
            device=config.model.embedding_device
        )
    return _embedding_model


# ── Main alignment function ────────────────────────────────────────────────────

def align(jd: JDProfile, resume: ResumeExtracted) -> AlignmentResult:
    """
    Produce a structured alignment object comparing a resume against a JD.

    This is fully deterministic — no LLM calls.
    """
    logger.info(f"Aligning resume for: {resume.candidate_name or 'Unknown'}")
    model = _get_embedding_model()

    # All candidate skills (explicit + implicit)
    all_resume_skills = resume.explicit_skills + resume.implicit_skills

    # Skill alignment
    matched_req, missing_req = _skill_overlap(jd.hard_requirements.required_skills, all_resume_skills)
    matched_pref, _ = _skill_overlap(jd.soft_requirements.preferred_skills, all_resume_skills)

    skill_coverage_required = len(matched_req) / len(jd.hard_requirements.required_skills) if jd.hard_requirements.required_skills else 1.0
    skill_coverage_preferred = len(matched_pref) / len(jd.soft_requirements.preferred_skills) if jd.soft_requirements.preferred_skills else 0.0

    # Experience alignment
    required_yoe = jd.hard_requirements.required_years_of_experience or 0.0
    exp_gap = round(resume.total_years_experience - required_yoe, 1)
    seniority_match = _seniority_match_label(jd.role_identity.seniority_level, resume.current_seniority)

    # Domain alignment
    domain_overlap, domain_match_score = _domain_match_score(
        jd.hard_requirements.required_domain_experience, resume.domains_worked_in
    )

    # Character alignment
    # ownership_style: compare JD expectation vs candidate's most recent role seniority
    # IC_OWNER/TECH_LEAD JDs want senior+ ICs; MANAGER JDs want people managers
    from src.extraction.jd_extraction.schemas import OwnershipStyle
    jd_ownership = jd.role_character.ownership_style
    # most_recent_seniority = resume.work_experiences[0].seniority_at_role if resume.work_experiences else None
    manager_jd = jd_ownership in (OwnershipStyle.MANAGER, OwnershipStyle.PLAYER_COACH)
    candidate_is_manager = resume.work_history.has_people_management
    ownership_style_match = (manager_jd == candidate_is_manager) or jd_ownership == OwnershipStyle.UNKNOWN

    work_style_match = True  # Placeholder — extend from WorkStyle enum comparison

    # Semantic similarity
    semantic_sim = _semantic_similarity(jd, resume, model)

    # Quality score
    quality_score = _candidate_quality_score(resume)

    # Bonus + penalty
    bonus = _compute_bonus_signals(jd, resume)

    return AlignmentResult(
        matched_required_skills=matched_req,
        missing_required_skills=missing_req,
        matched_preferred_skills=matched_pref,
        skill_coverage_required=round(skill_coverage_required, 3),
        skill_coverage_preferred=round(skill_coverage_preferred, 3),
        experience_gap_years=exp_gap,
        seniority_match=seniority_match,
        domain_overlap=domain_overlap,
        domain_match_score=round(domain_match_score, 3),
        ownership_style_match=ownership_style_match,
        work_style_match=work_style_match,
        semantic_similarity_score=semantic_sim,
        candidate_quality_score=quality_score,
        bonus_signals=bonus,
        highest_company_tier=resume.highest_company_tier.value if resume.highest_company_tier else None,
    )