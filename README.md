# Resume Matching Engine

An AI-powered resume screening system that automatically scores and ranks candidates against a job description — replacing a slow, inconsistent manual process with explainable, structured scoring.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Architecture Overview](#2-architecture-overview)
3. [Technical Approach & Justification](#3-technical-approach--justification)
4. [Key Challenges & How We Solved Them](#4-key-challenges--how-we-solved-them)
5. [Performance Evaluation](#5-performance-evaluation)
6. [Quick Start](#6-quick-start)
7. [Repository Structure](#7-repository-structure)
8. [Next Steps for Production](#8-next-steps-for-production)

---

## 1. The Problem

Talent Acquisition teams screening resumes manually face three compounding problems:

- **Speed**: Reviewing 100+ resumes per role is slow. High-volume pipelines create bottlenecks that delay hiring.
- **Inconsistency**: Different reviewers apply different criteria. The same resume may be shortlisted or rejected depending on who reads it.
- **Missed candidates**: Without a structured framework, qualified candidates are overlooked — particularly those whose skills don't surface in a quick skim.

The goal of this system is to produce a calibrated relevance score (0.0–1.0) for each resume against a given job description, ranked and explained in terms a recruiter can act on.

---

## 2. Architecture Overview

```
JD Text ──► JD Extraction Pipeline ──────────────────────────┐
                                                              ▼
Resume Texts ──► Resume Extraction Pipeline (parallel) ──► Alignment Engine ──► Scoring Engine ──► Ranked Output
                                                                                                          │
                                                                                                          ▼
                                                                                               Explainability Layer (Top-N)
```

The system has five sequential stages:

### Stage 1 — JD Extraction
Raw job description text is parsed into a structured `JDProfile` via four focused LLM calls:
- **Role identity**: title, seniority level, remote policy, employment type
- **Hard requirements**: required skills (atomic tokens), minimum YoE, education
- **Soft requirements**: preferred skills, domain experience, company background preferences
- **Role character**: ownership style (`ic_owner` / `tech_lead` / `player_coach`), work style (`research` / `execution` / `hybrid`)

A final synthesis call produces an `ideal_candidate_persona` — a 2–3 sentence description of the ideal hire used for context in the explainability layer.

### Stage 2 — Resume Extraction (Parallelized)
Each resume is parsed into a `ResumeProfile` via five focused LLM calls running in parallel across all resumes:
- **Identity**: name, current title, inferred seniority
- **Work history**: total YoE, highest company tier, domains worked in, leadership signals
- **Explicit skills**: skills copied verbatim from the skills section
- **Implicit skills**: skills inferred from work history (e.g. 8 years of ML roles → `machine learning` is implicit even if unlisted)
- **Education & credentials**: degrees, certifications, publications, open source contributions

A synthesis call produces `career_archetype`, `career_narrative`, `green_flags`, and `red_flags`.

### Stage 3 — Alignment Engine
One LLM call per resume + deterministic signal computation. Produces all raw signals needed for scoring:
- **Skill matching via LLM**: a single structured LLM call determines which JD required and preferred skills the candidate genuinely has, with per-skill reasoning logged for auditability
- YoE gap calculation and scoring
- Seniority level comparison
- Domain overlap ratio
- Mastery depth from `skill_depth_signals`
- Red flag penalty and credential bonus

### Stage 4 — Scoring Engine
Applies weights to alignment signals and enforces hard gate checks:

| Component | Weight | Signals |
|---|---|---|
| Skill match | 55% | Required + preferred skill coverage via semantic similarity |
| Experience | 25% | YoE gap, seniority alignment, domain overlap |
| Quality | 20% | Mastery depth signals, red flags, publications/OSS/certs |

Candidates failing the gate (< 40% required skill coverage or < 50% of required YoE) score 0.0 immediately.

### Stage 5 — Explainability Layer
LLM-generated recruiter-facing summary for the top-N candidates only (cost control):
- One-sentence headline verdict
- Top 3 strengths
- Top 2–3 concerns
- Interview probing questions
- Recommendation: `Strong Yes` / `Lean Yes` / `Maybe` / `Lean No` / `No`

---

## 3. Technical Approach & Justification

### LLM: GPT-4.1 (structured extraction)

Used for all extraction and explanation calls via OpenAI's Structured Outputs API (`beta.chat.completions.parse`). The model is constrained to return valid JSON matching a Pydantic schema — no parsing fragility.

**Why LLM-based extraction over rule-based parsing:**
- Resumes have no consistent format. Section headers vary ("Work Experience", "Employment History", "Where I've Worked"), bullets vary in style, dates vary in format. An LLM handles this gracefully; a regex-based parser would need hundreds of rules and still fail on edge cases.
- Implicit skills — things a candidate demonstrably knows from their work history but didn't explicitly list — are invisible to keyword matching. An LLM can infer that a candidate with 6 years of ML engineering roles knows `machine learning` even if it doesn't appear in their skills section.
- Structured Outputs enforces the schema at the token level — the model cannot return malformed JSON.

**Alternative considered: spaCy + rule-based NER**
- Advantage: fully local, fast, no API cost
- Disadvantage: requires labelled training data per schema field, brittle on format variation, cannot infer implicit skills or synthesize narrative signals like `green_flags`

**Why deterministic sequential tool calls over a ReAct agent:**
Each extraction step (role identity → hard requirements → soft requirements → role character → synthesis) is a fixed, pre-determined call. A ReAct agent would add an extra LLM call per step to decide what to call next — doubling cost with no real benefit, since the call order is always the same.

---

### Skill Matching: LLM-based (alignment engine)

A single structured LLM call per resume determines which JD required and preferred skills the candidate genuinely has. Coverage is binary — matched or missing. Partial matches are intentionally excluded: they are ambiguous and dilute the signal. When uncertain, the model marks a skill as missing (false negatives are safer than false positives in a gate-enforcing system).

**Why LLM over embedding cosine similarity:**

Embeddings at any fixed threshold produce systematic errors in two directions:
- **False positives across adjacent domains**: `"microservices"` and `"agent-based systems"` have high cosine similarity but are not the same skill.
- **Depth insensitivity**: `"PyTorch for inference"` and `"PyTorch for model training"` embed nearly identically. A role requiring training experience should not match a candidate who has only served models.

The LLM understands context: it can see that a candidate has `"Elasticsearch for search queries only"` and correctly decide this does NOT match `"retrieval systems"` as an ML skill. No threshold tuning required.

**Why not exact string matching:**
`NLP` ≠ `Natural Language Processing`. `k8s` ≠ `Kubernetes`. `ML` ≠ `Machine Learning`. The LLM handles all synonym and alias resolution correctly without a hand-crafted lookup table.

**Matching rules encoded in the prompt:**
- Semantic synonyms accepted: `"NLP"` = `"Natural Language Processing"`, `"RL"` = `"reinforcement learning"`
- Depth must be appropriate to the role: `"PyTorch for inference"` does NOT match `"PyTorch"` for a training-focused role
- Adjacent domains excluded: `"microservices"` does NOT match `"agent-based systems"`
- Classical ML ≠ deep learning: `"scikit-learn, XGBoost"` does NOT match `"PyTorch"` or `"TensorFlow"`
- When uncertain → `matched: false`

**Per-skill reasoning:** Every match decision includes a one-sentence reasoning string logged to the run artifacts. This makes the alignment layer fully auditable — a recruiter or engineer can inspect exactly why a skill was matched or missed.

**Alternative considered: `BAAI/bge-small-en-v1.5` embeddings**
- Advantage: fully local, no API cost per resume, no PII sent externally
- Disadvantage: requires threshold calibration that is JD-domain-specific and fragile; cannot handle depth differences or adjacent-domain false positives reliably. Removed after finding systematic errors on ML skill pairs.

---

### Text Preprocessing

1. **No aggressive cleaning**: LLMs handle messy formatting, inconsistent whitespace, and mixed encodings well. Stripping text aggressively risks losing signal.
2. **Section-aware extraction**: Explicit and implicit skills are extracted by separate tools, each with focused instructions. Explicit = what the candidate listed; implicit = what they demonstrably know from their history. Conflating them degrades signal quality.
3. **Atomic skill tokens**: Extraction prompts enforce 1–4 word skill tokens. `"experience building scalable ML systems"` is not a skill token — `"MLOps"`, `"distributed training"`, `"Kubernetes"` are. This is the prerequisite for reliable LLM-based skill matching — the model compares atomic tokens, not sentences.
4. **Date normalisation**: The LLM computes YoE from date ranges in context. Candidate-stated values like "10+ years" are not trusted — the extractor derives the number independently.
5. **Deduplication**: Explicit and implicit skills are merged with deduplication before alignment, with explicit skills taking precedence (higher confidence, directly stated).

---

## 4. Key Challenges & How We Solved Them

### Challenge 1: Skill synonym problem
**Problem**: Exact string matching produces false negatives on synonymous skill tokens. A JD requiring `NLP` would miss a candidate listing `Natural Language Processing`.

**Solution**: LLM-based skill matching with explicit synonym resolution rules encoded in the prompt. The model sees both the JD skill list and the candidate's explicit and implicit skills, and determines matches with a one-sentence reasoning per decision. `NLP` correctly matches `Natural Language Processing`; `microservices` correctly does not match `agent-based systems`. No threshold calibration required.

---

### Challenge 2: Buzzword-stuffed resumes
**Problem**: A naive TF-IDF or keyword-matching system scores a buzzword-stuffed resume highly if it contains all the right terms — regardless of whether the candidate can actually do the work.

**Solution**: The `implicit skills` extractor requires evidence threshold — a skill is only included if it appears across 2+ roles, is central to a job title, or has quantified outcomes. The `skill_depth_signals` field captures mastery evidence strings (e.g. `"PyTorch: fine-tuned 13B model with FSDP on 4×A100s"`). These feed directly into the quality score. The LLM synthesizer also surfaces `red_flags` such as "all buzz no substance" which apply a penalty multiplier. A candidate who lists `LLM` in their skills section but has no evidence of using it in their work history will have it absent from implicit skills, reducing coverage.

---

### Challenge 3: No labelled dataset for evaluation
**Problem**: Without ground truth labels, we cannot compute standard classification metrics.

**Solution**: A synthetic evaluation set of 8 resumes was manually constructed and labelled (`Good Match` / `Partial Match` / `Poor Match`) before running the system. Labels were set based on an independent reading of each resume against the JD — not post-hoc rationalisation of system output. This allows comparison of system scores against human judgement. See [Section 5](#5-performance-evaluation).

---

### Challenge 4: Cost at scale
**Problem**: Running 5–6 LLM calls per resume at $X/1M tokens becomes significant at hundreds of resumes.

**Solutions applied**:
- JD is extracted once and cached — not re-extracted per resume
- Skill matching (1 LLM call per resume) is merged with required and preferred skills in a single call — not two separate calls
- Explainability layer (most expensive call) only runs for top-N candidates, not the full set
- `--parse-only` flag allows dry-run validation before any API calls are made
- Parallel resume extraction via `ThreadPoolExecutor` reduces wall-clock time

---

### Challenge 5: Seniority signal reliability
**Problem**: Candidates inflate titles. "Lead Engineer" at a 3-person startup is not the same as "Lead Engineer" at Google.

**Solution**: Seniority is inferred from both title AND responsibility language in the extraction prompt — not title alone. `"own", "drive", "define strategy", "mentor"` → senior/lead. `"support", "assist", "under guidance"` → junior. The `highest_company_tier` signal is extracted but deliberately excluded from the scoring formula to avoid prestige bias — it would penalise strong candidates from less-known companies.

---

## 5. Performance Evaluation

### Synthetic Evaluation Set

8 resumes were manually constructed to cover the full score range against a sample ML Engineer JD. Labels were assigned before running the system.

| Candidate | Manual Label | Expected Score | System Score | Result |
|---|---|---|---|---|
| Priya Sharma | Good Match | ≥ 0.80 | 0.884 | ✓ |
| Chen Li | Good Match | ≥ 0.80 | 0.830 | ✓ |
| James Wu | Good Match (gap) | 0.70–0.80 | 0.768 | ✓ |
| Sarah Okonkwo | Partial Match | 0.45–0.65 | 0.760 | △ slight overrank |
| Alex Nguyen | Partial Match (junior) | 0.30–0.50 | 0.742 | △ slight overrank |
| Michael Patel | Poor Match | ≤ 0.25 | 0.000 | ✓ (gated out) |
| Ryan Taylor | Poor Match (adversarial) | ≤ 0.20 | 0.000 | ✓ (gated out) |
| Jessica Brown | Poor Match | 0.0 | 0.000 | ✓ (gated out) |

**Key observations:**
- All `Good Match` candidates rank in the top 3 ✓
- All `Poor Match` candidates are correctly gated out ✓
- The mid-tier candidates (Sarah, Alex) are ranked correctly relative to each other but score higher than expected — likely because all required skills matched via LLM skill matching and the quality component rewarded their depth signals. Root cause analysis: pull `alignment.json` for these two candidates from `runs/` and inspect which skills matched and why.

**Ryan Taylor (adversarial case):** This resume was deliberately crafted to contain all required skill keywords with no work history evidence. The gate correctly filtered this candidate because explicit skills with no corroborating work history did not appear in `implicit_skills`, driving LLM-assessed coverage below the 40% gate threshold.

---

### Three-Stage Evaluation Framework

Evaluation runs against saved run artifacts — no re-running the pipeline, no extra API cost.

```bash
python evaluation/run_eval.py                     # all three stages, latest run
python evaluation/run_eval.py --stage jd          # Stage 1 only
python evaluation/run_eval.py --stage resume      # Stage 2 only
python evaluation/run_eval.py --stage scoring     # Stage 3 only
python evaluation/run_eval.py --run-id run_XXX    # specific run
```

**Stage 1 — JD Extraction quality**

Reads `runs/{run_id}/jd_profile.json` and compares against `evaluation/ground_truth/jd_gt.json`.

| Metric | What it catches |
|---|---|
| Required skill Precision | Hallucinated required skills — inflates all candidate scores |
| Required skill Recall | Missed required skills — silently underscores good candidates |
| Preferred skill Precision / Recall | Same, with lower stakes |

Required skill recall is the higher-stakes metric: a missed required skill propagates to every resume scored against this JD.

**Stage 2 — Resume Extraction quality**

Reads `runs/{run_id}/candidates/{slug}/resume_profile.json` and compares against `evaluation/ground_truth/resume_gt.json`. Evaluated separately for explicit and implicit skills, per candidate and as macro average.

| Metric | What it catches |
|---|---|
| Explicit Precision | Hallucinated skills from the skills section |
| Explicit Recall | Skills present in skills section but missed |
| Implicit Precision | Skills inferred from work history without evidence — inflates scores for weak candidates |
| Implicit Recall | Skills demonstrable from work history but missed — underscores good candidates |

Implicit skill extraction is the highest-risk component. The evaluation tool prints missed and hallucinated implicit skills per candidate so failures are directly actionable.

**Stage 3 — Scoring quality**

Reads `runs/{run_id}/ranked_results.json` and compares against `evaluation/eval_dataset.json`.

| Metric | Why |
|---|---|
| **nDCG@3** (primary) | Did the right people land in the top 3? Logarithmic discount penalises wrong ordering at the top more than at the bottom — matches recruiter behaviour |
| **nDCG@5** (secondary) | Shortlist quality across the broader pool |
| **Spearman ρ** | Overall rank correlation — interpretable sanity check across all 8 candidates |

nDCG uses graded relevance labels (1.0 / 0.75 / 0.5 / 0.0) — not binary. This correctly penalises ranking a Partial Match above a Good Match, which binary accuracy would not catch.

**Production labelling strategy:** Recruiter thumbs up/down, interview invite, and hire outcome can serve as weak supervision labels. These can be used to tune the tier weights (currently heuristic) via a simple logistic regression on the three component scores: `[skill_score, experience_score, quality_score] → hire_outcome`.

---

## 6. Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env: add OPENAI_API_KEY

# 3. Validate file parsing (no API calls)
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --parse-only

# 4. Full run with explanations for top 3
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --explain-top 3

# 5. Save results to JSON
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --output results.json
```

**Supported resume formats:** `.pdf`, `.docx`, `.txt`

**Optional: Groq instead of OpenAI**
```bash
# In .env:
GROQ_API_KEY=gsk_...

# In config.py:
llm_provider = "groq"
llm_model = "llama-3.3-70b-versatile"
```

---

## 7. Repository Structure

```
resume-matcher/
├── main.py                               # CLI entrypoint
├── config.py                             # Weights, model config, gate thresholds
├── requirements.txt
├── .env.example
├── src/
│   ├── engine.py                         # Main orchestrator
│   ├── run_saver.py                      # Persists run artifacts (jd_profile, resume_profile, alignment, scoring)
│   ├── extraction/
│   │   ├── jd_extraction/
│   │   │   ├── schemas.py                # JDProfile, HardRequirements, etc.
│   │   │   ├── agent_tools.py            # 4 focused LLM extraction tools
│   │   │   └── extractor.py              # Deterministic sequential pipeline
│   │   └── resume_extraction/
│   │       ├── schemas.py                # ResumeProfile, SkillsProfile, etc.
│   │       ├── agent_tools.py            # 5 focused LLM extraction tools (incl. explicit/implicit split)
│   │       └── extractor.py              # Deterministic sequential pipeline
│   ├── scoring/
│   │   ├── schemas.py                    # AlignmentResult, ScoringBreakdown
│   │   ├── aligner.py                    # LLM skill matching, experience, quality signals
│   │   ├── scorer.py                     # Weight application, gate checks
│   │   └── explainer.py                  # Recruiter-facing LLM explanation (top-N)
│   └── utils/
│       ├── file_parser.py                # PDF, DOCX, TXT text extraction
│       └── llm_client.py                 # OpenAI / Groq abstraction, backoff
├── data/
│   ├── sample_jd.txt                     # Sample job description
│   └── resumes/                          # 8 synthetic evaluation resumes
├── evaluation/
│   ├── __init__.py
│   ├── run_eval.py                       # Three-stage evaluation runner (reads from runs/)
│   ├── skill_matcher.py                  # Deterministic skill matcher for eval (no LLM)
│   ├── eval_dataset.json                 # Ground truth relevance labels for scoring eval
│   └── ground_truth/
│       ├── jd_gt.json                    # Ground truth required/preferred skills for JD eval
│       └── resume_gt.json                # Ground truth explicit/implicit skills per candidate
├── runs/                                 # Auto-generated run artifacts (gitignored)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_meta.json
│       ├── jd_raw.txt
│       ├── jd_profile.json
│       ├── candidates/{slug}/
│       │   ├── resume_raw.txt
│       │   ├── resume_profile.json
│       │   ├── alignment.json
│       │   └── scoring.json
│       ├── ranked_results.json
│       └── eval_results.json             # Written by run_eval.py
└── notebooks/
    └── evaluation.ipynb                  # Visual evaluation (run after main.py + run_eval.py)
```

---

## 8. Next Steps for Production

### Immediate (before first production deployment)

**Weight learning from recruiter feedback**
The current tier weights (55/25/20) are heuristic. Once recruiter feedback is collected (thumbs up/down, interview invite, hire outcome), these can be replaced with learned weights via logistic regression on `[skill_score, experience_score, quality_score]`. This requires ~200–300 labelled examples to be reliable.

**Skill taxonomy**
The LLM skill matcher handles synonym resolution contextually, but display labels in output can be noisy (`"ML"` and `"machine learning"` may both appear in results as separate entries). A curated taxonomy (`ML` → `machine learning`, `k8s` → `Kubernetes`) would clean display labels. The `skill_matcher.py` alias map in `evaluation/` is a starting point — it can be promoted to a shared normalisation utility.

**Bias audit**
Scoring distributions should be checked across demographic proxies (name origin, institution tier, location) before any production deployment. This is a legal and ethical requirement in regulated hiring environments. The `highest_company_tier` signal was deliberately excluded from scoring for this reason — prestige bias would systematically penalise strong candidates from less-known companies.

---

### Medium-term (scaling to production volume)

**ATS integration**
Connect to Workday or Greenhouse via API for real-time scoring on inbound applications. The engine is stateless — JD extraction runs once per role opening, resume extraction runs per applicant. This maps cleanly to webhook-driven ATS architectures.

**Async job queue**
The current `ThreadPoolExecutor` approach works well up to ~50 resumes. For batch processing thousands of resumes (e.g. end-of-posting-period bulk review), move to a Celery + Redis job queue with async LLM calls. The extraction pipeline is already designed with per-resume isolation — one failure doesn't block others.

**LLM skill matching prompt refinement**
The skill matching prompt encodes domain-agnostic rules (depth sensitivity, adjacent domain exclusion). For a specific domain (e.g. ML engineering roles exclusively), few-shot examples of correct and incorrect decisions added to the prompt would improve match accuracy without any model training. Disagreement cases from `eval_results.json` are the natural source for these examples.

---

### Longer-term

**Structured feedback loop**
Interview outcomes and hire decisions fed back as weak supervision labels, enabling continuous weight recalibration without manual re-labelling. The scoring architecture is designed for this — component scores are preserved in `ScoringBreakdown` and map directly to regression features.

**Multi-JD batch scoring**
For organisations running many simultaneous roles, a candidate pool could be scored against multiple JDs in a single pass — resume extraction runs once per candidate, alignment and scoring run once per (candidate, JD) pair.

**Explainability improvements**
Current explainability is LLM-generated natural language. For regulated environments, a rule-based explanation layer (directly from `matched_required_skills`, `missing_required_skills`, `seniority_match`) would be more auditable and legally defensible than free-text generation.