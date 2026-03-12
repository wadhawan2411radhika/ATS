"""
Central configuration for the resume matching system.
Weights are documented and tunable — this is intentional.
In production, these would be learned from recruiter feedback data.
"""

from dataclasses import dataclass, field


@dataclass
class ScoringWeights:
    """
    Tier weights must sum to 1.0.
    Within each tier, sub-weights must also sum to 1.0.
    """

    # ── Tier weights ──────────────────────────────────────────────────────────
    tier1_must_haves: float = 0.50       # Hard requirements
    tier2_good_to_haves: float = 0.25    # Preferred signals
    tier3_quality_signals: float = 0.15  # Impact, trajectory
    tier4_bonus: float = 0.05            # Open source, rare skills
    penalty_cap: float = 0.15           # Max penalty deduction

    # ── Tier 1 sub-weights (must sum to 1.0) ─────────────────────────────────
    t1_required_skills: float = 0.50    # Dominant signal — skill match is the gating factor
    t1_experience_years: float = 0.20
    t1_seniority_match: float = 0.15
    t1_domain_relevance: float = 0.15

    # ── Tier 2 sub-weights (must sum to 1.0) ─────────────────────────────────
    t2_preferred_skills: float = 0.40     # Nice-to-have skill match
    t2_company_brand: float = 0.25         # Company tier / prestige signal
    t2_semantic_similarity: float = 0.35  # Semantic fit of resume to JD persona

    # ── Tier 3 sub-weights (must sum to 1.0) ─────────────────────────────────
    t3_impact_quality: float = 0.40
    t3_project_quality: float = 0.30
    t3_career_trajectory: float = 0.30

    # ── Tier 4 sub-weights (must sum to 1.0) ─────────────────────────────────
    t4_open_source: float = 0.40
    t4_publications: float = 0.30
    t4_rare_skills: float = 0.30


@dataclass
class ModelConfig:
    # LLM for extraction
    llm_provider: str = "openai"                    # "openai" | "groq" | "anthropic"
    llm_model: str = "gpt-4.1"                   # Supports Structured Outputs, 3x higher RPM than gpt-4o
    llm_temperature: float = 0.0                    # Deterministic extraction
    llm_max_tokens: int = 4096

    # Embedding model for semantic similarity
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Local, no API cost
    embedding_device: str = "cpu"                    # "cuda" if available


@dataclass
class GateConfig:
    """Hard filters — if failed, resume scores 0.0 without further processing."""
    enforce_min_experience: bool = True
    enforce_required_skills_gate: bool = True    # Gate candidates below minimum skill coverage
    min_required_skill_coverage: float = 0.25    # Must match at least 25% of required skills


@dataclass
class AppConfig:
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    model: ModelConfig = field(default_factory=ModelConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    # Output
    top_n_results: int = 10
    explain_top_n: int = 5     # Generate LLM explanations only for top N (cost control)


# Singleton config — import this everywhere
config = AppConfig()