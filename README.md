# Resume Matching Engine

An AI-powered resume screening system that scores and ranks candidates (0.0–1.0) against a job description — replacing slow, inconsistent manual review with explainable, structured output.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Architecture Overview](#2-architecture-overview)
3. [Model Selection & Justification](#3-model-selection--justification)
4. [Feature Engineering & Preprocessing](#4-feature-engineering--preprocessing)
5. [Performance Evaluation](#5-performance-evaluation)
6. [Quick Start](#6-quick-start)
7. [Repository Structure](#7-repository-structure)
8. [Next Steps for Production](#8-next-steps-for-production)

---

## 1. The Problem

Talent Acquisition teams screening resumes manually face three compounding problems:

- **Speed**: Reviewing 100+ resumes per role creates bottlenecks that delay time-to-hire.
- **Inconsistency**: Different reviewers apply different criteria to the same resume.
- **Missed candidates**: Qualified candidates are overlooked when skills don't surface in a quick skim.

This system produces a calibrated 0.0–1.0 relevance score per resume with a recruiter-facing explanation — reducing a multi-hour screening task to minutes.

---

## 2. Architecture Overview

```
JD Text ──► JD Extraction ──────────────────────────────────────────┐
                                                                     ▼
Resume Texts ──► Resume Extraction (parallel) ──► Alignment Engine ──► Scoring Engine ──► Ranked Output
                                                                                                │
                                                                                                ▼
                                                                                    Explainability Layer (Top-N)
```

The system runs in five deterministic stages:

| Stage | What it does | LLM calls |
|---|---|---|
| **1 — JD Extraction** | Parses JD into structured `JDProfile`: required skills, preferred skills, seniority, YoE, role character | 4 (sequential) |
| **2 — Resume Extraction** | Parses each resume into `ResumeProfile`: identity, work history, explicit skills, implicit skills, education | 5 per resume (parallel) |
| **3 — Alignment Engine** | Determines which JD skills each candidate genuinely has, with per-skill reasoning | 1 per resume |
| **4 — Scoring Engine** | Applies weighted formula (55% skills / 25% experience / 20% quality) with hard gate | 0 (deterministic) |
| **5 — Explainability** | Generates recruiter-facing summary: verdict, strengths, concerns, interview questions | 1 (top-N only) |

**Hard gate:** Candidates with < 40% required skill coverage OR < 50% of required YoE score 0.0 immediately and are not ranked.

**Parallel processing:** Resume extraction runs all 5 calls per resume in parallel via `ThreadPoolExecutor`. JD extraction runs once and is cached for all resumes.

---

## 3. Model Selection & Justification

### Core model: GPT-4.1 via OpenAI Structured Outputs

All extraction, alignment, and explanation calls use GPT-4.1 via `beta.chat.completions.parse`, which constrains the model to return valid JSON matching a Pydantic schema at the token level — no post-hoc parsing, no validation failures.

**Why an LLM over a rule-based parser:**

Resumes have no consistent format. Section headers, date formats, and bullet styles vary arbitrarily across candidates. An LLM handles this gracefully; a regex-based parser requires hundreds of rules and still fails on edge cases. More importantly, the system needs to extract *implicit* skills — things a candidate demonstrably knows from their work history but didn't explicitly list. No rule-based system can do this.

**Alternative considered: spaCy + rule-based NER**

| | LLM (chosen) | spaCy + rules (rejected) |
|---|---|---|
| Format robustness | Handles arbitrary formatting | Brittle on format variation |
| Implicit skill inference | Yes | No — keyword matching only |
| Schema enforcement | Structured Outputs (token-level) | Manual validation required |
| Cost | ~$0.01–0.05/resume | Free |
| Speed | ~3–5s/resume | <0.1s/resume |

**Decision:** Implicit skill inference and format robustness outweigh the cost and latency difference at POC and low-to-mid volume. At very high volume (10,000+ resumes/day), a hybrid approach — rules for initial filtering, LLM for top candidates — would be appropriate.

---

### Skill Matching: LLM binary matching (alignment stage)

A single structured LLM call per resume determines which JD skills the candidate genuinely has — binary (matched / missing) with a one-sentence reasoning string per skill for auditability.

**Why LLM over embedding cosine similarity:**

Embeddings at any fixed threshold produce systematic errors in two directions:
- `"PyTorch for inference"` and `"PyTorch for model training"` embed nearly identically — but are different skills for a training-focused role.
- `"microservices"` and `"agent-based systems"` have high cosine similarity — but are different domains.

The LLM understands depth and context. No threshold calibration required.

**Alternative considered: `BAAI/bge-small-en-v1.5` + cosine similarity**

| | LLM matching (chosen) | Embedding similarity (rejected) |
|---|---|---|
| Synonym handling | Prompt-level rules | Alias lookup table required |
| Depth sensitivity | Yes (`"PyTorch inference" ≠ "PyTorch training"`) | No — vectors nearly identical |
| Adjacent domain exclusion | Yes | Unreliable at any threshold |
| Auditability | Per-skill reasoning string | Score only |
| Cost | 1 LLM call per resume | Free, fully local |

**Decision:** Removed embedding-based matching after finding systematic false positives on ML skill pairs in early testing.

**Matching rules encoded in the prompt:**
- Semantic synonyms accepted: `"NLP"` = `"natural language processing"`, `"RL"` = `"reinforcement learning"`
- Depth must match role requirements: `"PyTorch for inference"` does NOT match `"PyTorch"` for a training-focused role
- Adjacent domains excluded: `"Elasticsearch for search queries"` does NOT match `"retrieval systems"` as an ML skill
- Classical ML ≠ deep learning: `"scikit-learn/XGBoost"` does NOT match `"PyTorch"` or `"TensorFlow"`
- When uncertain → `matched: false` (false negatives are safer than false positives in a gate-enforcing system)

---

## 4. Feature Engineering & Preprocessing

### Explicit vs. Implicit skill split

The most important preprocessing decision: skills are extracted into two separate pools.

- **Explicit skills**: copied verbatim from the resume's Skills section only. High precision, direct evidence.
- **Implicit skills**: inferred from work history — included only if the skill appears across 2+ roles, is central to a job title, or has quantified outcomes attached.

This prevents buzzword-stuffed resumes from scoring well. A candidate listing `"LLMs"` in their skills section with no work history evidence scores identically to one who doesn't list it — the gate requires verified coverage.

### Atomic skill tokenisation

Extraction prompts enforce 1–4 word atomic tokens: `"experience building scalable ML systems"` → `["MLOps", "distributed training", "Kubernetes"]`. This is the prerequisite for reliable LLM-based skill matching.

### No aggressive text cleaning

LLMs handle messy formatting and inconsistent whitespace well. Aggressive cleaning risks losing signal — e.g. stripping parenthetical qualifiers like `"PyTorch (inference only)"` which are meaningful to the alignment engine.

### YoE derived independently

Candidate-stated values like `"10+ years"` are not trusted. The extractor derives YoE from date ranges, summing non-overlapping tenures.

### Seniority from responsibility language, not title

`"own", "drive", "define strategy", "mentor"` → senior/lead. `"support", "assist", "under guidance"` → junior. Title alone is unreliable.

### Company tier excluded from scoring

`highest_company_tier` is extracted but excluded from the scoring formula to avoid prestige bias — it would penalise strong candidates from less-known companies. It surfaces in the explainability layer as context only.

---

## 5. Performance Evaluation

### Evaluation set

9 resumes were manually constructed and labelled against the **Ema Software Engineering Lead (ML)** job description before running the system — not post-hoc rationalisation of scores.

| Candidate | Manual Label | Score | Gate | Result |
|---|---|---|---|---|
| Arjun Mehta | Good Match (1.0) | 0.87 | Pass | ✓ Rank 1 |
| Dr. Priya Venkataraman | Good Match (1.0) | 0.80 | Pass | ✓ Rank 2 |
| Marcus Chen | Partial Match (0.5) | 0.65 | Pass | ✓ Rank 3 |
| Sneha Iyer | Partial Match (0.5) | 0.64 | Pass | ✓ Rank 4 |
| Jake Thompson | Poor Match (0.0) | 0.0 | **Gated** — 18% coverage | ✓ |
| Ram Sharma | Poor Match (0.0) | 0.0 | **Gated** — 18% coverage | ✓ |
| Shaym Narayan | Poor Match (0.0) | 0.0 | **Gated** — 6% coverage | ✓ |
| Rahul Soni | Poor Match (0.0) | 0.0 | **Gated** — 6% coverage | ✓ |
| Kamal Anand | Poor Match (0.0) | 0.0 | **Gated** — 0% coverage | ✓ |

All Good Match candidates ranked in the correct order. All Poor Match candidates correctly gated out.

---

### Quantitative metrics (three-stage eval pipeline)

Evaluation reads from saved run artifacts — no re-running the pipeline, no API cost.

```bash
python evaluation/run_eval.py
```

**Stage 3 — Scoring (primary)**

| Metric | Score | Interpretation |
|---|---|---|
| **nDCG@3** | **1.0000** | Top-3 ranked in perfect order |
| **nDCG@5** | **1.0000** | Full shortlist in perfect order |
| **Spearman ρ** | **0.9950** (p≈0) | Near-perfect rank correlation across all 9 candidates |

**Stage 1 — JD Extraction**

| Metric | Score | Notes |
|---|---|---|
| Required skill Precision | **1.000** | Zero hallucinated required skills |
| Required skill Recall | **0.680** | 32% of required skills missed — see known limitation below |
| Preferred skill P / R | 0.250 / 0.333 | Low-stakes; preferred skills carry 20% weight only |

**Stage 2 — Resume Extraction (macro avg, 9 candidates)**

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Explicit skills | 0.180 | 0.391 | 0.237 |
| Implicit skills | 0.042 | 0.166 | 0.063 |

The low explicit precision reflects the extractor being more inclusive than conservative GT annotations. The ranking result (nDCG@3=1.0) confirms this over-inclusion does not harm scoring — the skill taxonomy (Next Steps, Step 3) will calibrate this.

---

### Known limitation: JD recall on soft-framed JDs

The Ema JD uses `"Ideally, you'd have:"` framing for all requirements including core technical skills. The extractor conservatively treats soft framing as preferred, causing 32% of required skills to be missed.

**Impact on this run:** Contained — the same extraction applies equally to all candidates, so relative ranking is preserved. Absolute scores are slightly depressed.

**Fix:** A single prompt rule: *"For JDs that soft-frame all requirements, treat core ML technical skills as required regardless of framing."* No retraining required.

---

### Formal success metrics at production scale

| Metric | Why |
|---|---|
| **nDCG@K** | Primary. Recruitment is a ranking problem — recruiters read down a list and stop. nDCG penalises wrong ordering near the top exponentially more than errors at the bottom. |
| **Precision@K** | What fraction of the top-K shortlist are genuinely good candidates? Maps to recruiter time savings. |
| **Recall@K** | What fraction of all good candidates appear in the top-K? Maps to missed hires — the cost the system exists to solve. |
| **Spearman ρ** | Overall rank correlation — interpretable sanity check across the full pool. |
| **Calibration** | Are scores comparable across roles? A 0.7 for an ML role should represent the same quality bar as a 0.7 for a data science role. |

Binary accuracy is excluded — it doesn't capture the cost asymmetry between ranking errors (wrong #2 vs #1 is less harmful than wrong #1 vs #5).

**Weak supervision path:** Recruiter shortlist decisions (Y/N) and interview invites can serve as labels. With ~300 examples, logistic regression on `[skill_score, experience_score, quality_score]` replaces the current heuristic weights (55/25/20). The `ScoringBreakdown` already preserves all three component scores — no pipeline changes required to enable this.

---

## 6. Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Add OPENAI_API_KEY to .env

# 3. Validate parsing (no API calls)
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --parse-only

# 4. Full run with explanations for top 3
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --explain-top 3

# 5. Save results to JSON
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --output results.json

# 6. Run evaluation pipeline
python evaluation/run_eval.py
```

**Supported resume formats:** `.pdf`, `.docx`, `.txt`

**Optional: Groq instead of OpenAI** — set `GROQ_API_KEY` in `.env` and `llm_provider = "groq"` in `config.py`.

---

## 7. Repository Structure

```
resume-matcher/
├── main.py                               # CLI entrypoint
├── config.py                             # Weights, model, gate thresholds
├── requirements.txt
├── .env.example
├── src/
│   ├── engine.py                         # Orchestrator
│   ├── run_saver.py                      # Persists all artifacts to runs/
│   ├── extraction/
│   │   ├── jd_extraction/
│   │   │   ├── schemas.py                # JDProfile, HardRequirements, SoftRequirements
│   │   │   ├── agent_tools.py            # 4 LLM extraction calls
│   │   │   └── extractor.py             # Sequential pipeline
│   │   └── resume_extraction/
│   │       ├── schemas.py                # ResumeProfile, SkillsProfile
│   │       ├── agent_tools.py            # 5 LLM extraction calls (explicit/implicit split)
│   │       └── extractor.py             # Parallel pipeline
│   ├── scoring/
│   │   ├── schemas.py                    # AlignmentResult, ScoringBreakdown
│   │   ├── aligner.py                    # LLM skill matching + experience/quality signals
│   │   ├── scorer.py                     # Weight application + gate
│   │   └── explainer.py                 # Recruiter-facing summary (top-N)
│   └── utils/
│       ├── file_parser.py                # PDF, DOCX, TXT extraction
│       └── llm_client.py                 # OpenAI/Groq abstraction + retry
├── data/
│   ├── sample_jd.txt
│   └── resumes/                          # 9 evaluation resumes
├── evaluation/
│   ├── run_eval.py                       # Three-stage eval runner (reads from runs/)
│   ├── skill_matcher.py                  # Deterministic alias-based skill matcher (no LLM)
│   ├── eval_dataset.json                 # Ground truth relevance labels (Stage 3)
│   └── ground_truth/
│       ├── jd_gt.json                    # Required/preferred skill GT (Stage 1)
│       └── resume_gt.json                # Explicit/implicit skill GT per candidate (Stage 2)
├── runs/                                 # Auto-generated (gitignored)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── jd_profile.json
│       ├── candidates/{slug}/
│       │   ├── resume_profile.json
│       │   ├── alignment.json
│       │   └── scoring.json
│       ├── ranked_results.json
│       └── eval_results.json             # Written by run_eval.py
└── notebooks/
    └── evaluation.ipynb
```

---

## 8. Next Steps for Production

Three steps derived directly from the evaluation results, ordered by impact-to-effort ratio.

### Step 1 — Close the feedback loop (Week 1–4)

**Signal:** nDCG@3 = 1.0. Ranking is correct. The right time to capture labels is now, before edge cases accumulate.

Collect ~300 recruiter shortlist decisions, then fit logistic regression on `[skill_score, experience_score, quality_score] → shortlist`. The `ScoringBreakdown` already preserves all three component scores — no pipeline changes needed.

**Outcome:** Weights adapt to each client's actual hiring bar rather than staying fixed at the heuristic (55/25/20).

---

### Step 2 — Fix JD extraction recall (Week 2–3)

**Signal:** Required skill recall = 0.68. Every missed required skill silently underscores every candidate for that role.

One prompt rule addition, tested against 5 real JDs with varied framing using the Stage 1 eval. Target: recall ≥ 0.90.

**Outcome:** Highest downstream impact per unit of effort — all candidates scored against an accurate skill list.

---

### Step 3 — Build a skill taxonomy (Month 2)

**Signal:** Resume explicit precision = 0.18, driven by surface-form mismatches between extractor output and GT.

Normalise all extracted skills to canonical forms before comparison and scoring. Start from `skill_matcher.py` alias dict; long-term, map to ESCO or O*NET ontology.

**Outcome:** Extraction precision targets 0.80+, enabling Stage 2 eval to serve as a regression gate for prompt changes in production.