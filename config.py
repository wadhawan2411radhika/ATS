"""
Central configuration for the resume matching system.
Weights are documented and tunable — this is intentional.
In production, these would be learned from recruiter feedback data.
"""

from dataclasses import dataclass, field


@dataclass
class ScoringWeights:
    # In production, replace with weights learned from recruiter feedback
    # via logistic regression on [skill_score, experience_score, quality_score] → hire_outcome
    skill: float = 0.55
    experience: float = 0.25
    quality: float = 0.20


@dataclass
class ModelConfig:
    # LLM for extraction AND skill matching
    llm_provider: str = "openai"       # "openai" | "groq"
    llm_model: str = "gpt-4.1"         # Supports Structured Outputs
    llm_temperature: float = 0.0       # Deterministic extraction
    llm_max_tokens: int = 4096


@dataclass
class GateConfig:
    """
    Hard filters — candidates failing these score 0.0 immediately.

    With LLM-based binary skill matching, 0.40 means the candidate must
    genuinely match at least 7 of 17 required skills (for the synthetic JD).
    A backend engineer with zero ML will not clear this. A classical DS
    without deep learning will likely not clear it either.

    Set lower (0.25) if the JD has fewer required skills or the role is
    more generalist. Set higher (0.50) for highly specialised roles.
    """
    enforce_min_experience: bool = True
    enforce_required_skills_gate: bool = True
    min_required_skill_coverage: float = 0.40    # binary match: 40% of required skills


@dataclass
class AppConfig:
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    model: ModelConfig = field(default_factory=ModelConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    # Output
    top_n_results: int = 10
    explain_top_n: int = 3     # LLM explanations only for top N (cost control)


# Singleton — import this everywhere
config = AppConfig()