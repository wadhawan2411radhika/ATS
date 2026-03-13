"""
Deterministic Skill Matcher for Evaluation.

Uses normalization + alias lookup — no LLM, no embeddings.
Keeping eval independent of the system being evaluated prevents circular validation.

Matching strategy (in order):
    1. Exact match after normalization (lowercase, strip punctuation)
    2. Alias resolution (NLP → natural language processing)
    3. Substring containment (handles abbreviations within longer forms)
"""

import re

# ── Alias map ─────────────────────────────────────────────────────────────────
# Maps short forms / abbreviations to canonical long form.
# Both sides are normalized before comparison.

ALIASES: dict[str, str] = {
    # ML / AI concepts
    "ml":                       "machine learning",
    "dl":                       "deep learning",
    "nlp":                      "natural language processing",
    "rl":                       "reinforcement learning",
    "cv":                       "computer vision",
    "llm":                      "large language models",
    "llms":                     "large language models",
    "rag":                      "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
    "genai":                    "generative ai",
    "gen ai":                   "generative ai",
    "generative ai":            "generative ai",
    "fsdp":                     "fully sharded data parallel",
    "peft":                     "parameter efficient fine-tuning",
    "lora":                     "low rank adaptation",
    "rlhf":                     "reinforcement learning from human feedback",
    "sft":                      "supervised fine-tuning",

    # Frameworks / libraries
    "hf":                       "hugging face",
    "huggingface":              "hugging face",
    "sk-learn":                 "scikit-learn",
    "sklearn":                  "scikit-learn",
    "tf":                       "tensorflow",
    "xgb":                      "xgboost",
    "lgbm":                     "lightgbm",

    # Infrastructure
    "k8s":                      "kubernetes",
    "kube":                     "kubernetes",
    "gcp":                      "google cloud platform",
    "google cloud":             "google cloud platform",
    "aws":                      "amazon web services",
    "azure":                    "microsoft azure",

    # Data
    "postgres":                 "postgresql",
    "pg":                       "postgresql",
    "nosql":                    "nosql databases",
    "es":                       "elasticsearch",

    # MLOps / dev
    "cicd":                     "ci/cd",
    "ci cd":                    "ci/cd",
    "mlops":                    "mlops",
    "lcel":                     "langchain expression language",
}


def _normalize(skill: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, resolve aliases."""
    skill = skill.lower().strip()
    skill = re.sub(r"[^\w\s/]", "", skill)   # keep / for things like ci/cd
    skill = re.sub(r"\s+", " ", skill)
    return ALIASES.get(skill, skill)


def skill_matches(predicted: str, ground_truth: str) -> bool:
    """
    Returns True if a predicted skill matches a ground truth skill.

    Conservative — avoids false positives. Two unrelated skills that
    happen to share a word will NOT match unless one contains the other
    as a complete substring.
    """
    p = _normalize(predicted)
    g = _normalize(ground_truth)

    # 1. Exact match after normalization
    if p == g:
        return True

    # 2. Substring containment — one is fully contained in the other.
    #    e.g. "pytorch" ↔ "pytorch training", "sql" ↔ "sql databases"
    #    Guard: the shorter must be at least 3 chars to avoid false positives like "ml" ↔ "xml"
    shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
    if len(shorter) >= 3 and shorter in longer:
        # Extra guard: must be a word boundary match, not just substring
        # e.g. "sql" should match "sql databases" but not "nosql"
        pattern = r"\b" + re.escape(shorter) + r"\b"
        if re.search(pattern, longer):
            return True

    return False


def match_skill_to_gt(predicted: str, gt_skills: list[str]) -> bool:
    """Returns True if predicted skill matches ANY ground truth skill."""
    return any(skill_matches(predicted, gt) for gt in gt_skills)


def find_matched_skills(
    predicted: list[str],
    ground_truth: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (true_positives, false_positives, false_negatives).

    true_positives:  predicted skills that matched a GT skill
    false_positives: predicted skills with no GT match
    false_negatives: GT skills that no predicted skill matched
    """
    tp = [p for p in predicted if match_skill_to_gt(p, ground_truth)]
    fp = [p for p in predicted if not match_skill_to_gt(p, ground_truth)]
    fn = [g for g in ground_truth if not match_skill_to_gt(g, predicted)]
    return tp, fp, fn


def compute_precision_recall(
    predicted: list[str],
    ground_truth: list[str],
) -> dict:
    """
    Compute precision, recall, F1 for a single skill list comparison.

    Precision: of what we extracted, what fraction is correct?
               → measures noise / hallucination
    Recall:    of what exists in GT, what fraction did we find?
               → measures coverage / missed skills

    For a resume matcher, recall matters more than precision:
    a missed required skill (false negative) causes a candidate to
    be underscored. A hallucinated skill (false positive) has less
    impact because the aligner checks against the resume anyway.
    """
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0,
                "matched": [], "false_positives": [], "missed": []}

    tp_list, fp_list, fn_list = find_matched_skills(predicted, ground_truth)

    tp = len(tp_list)
    fp = len(fp_list)
    fn = len(fn_list)

    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(ground_truth) if ground_truth else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "tp":             tp,
        "fp":             fp,
        "fn":             fn,
        "matched":        tp_list,
        "false_positives": fp_list,
        "missed":         fn_list,
    }