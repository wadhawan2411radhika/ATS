# Resume Matching Engine — Technical Documentation

A production-grade AI-powered resume screening system using structured LLM extraction, multi-signal scoring, and explainable ranking.

---

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run matching
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --explain-top 3

# 4. Save results to JSON
python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --output results.json
```

---

## Architecture Overview

```
JD Text ──► JD Extraction Pipeline ──────────────────────┐
                                                          ▼
Resume Texts ──► Resume Extraction Pipeline (parallel) ──► Alignment Engine ──► Scoring Engine ──► Ranked Output
                                                                                        │
                                                                                        ▼
                                                                               Explainability Layer (Top-N)
```

### 1. JD Information Extraction Pipeline (`src/extractors/jd_extractor.py`)

Extracts structured fields from raw JD text via LLM with JSON schema enforcement:

- **Hard requirements**: required skills, minimum YoE, education, domain experience
- **Soft requirements**: preferred skills, company/university background signals
- **Tech stack**: core vs secondary technologies
- **Role character**: ownership style (IC owner / tech lead / player-coach), work style (research vs engineering), pace signal
- **Synthesized**: `ideal_candidate_persona` — LLM-written 2-3 sentence description of the ideal hire

### 2. Resume Information Extraction Pipeline (`src/extractors/resume_extractor.py`)

Extracts structured fields from raw resume text:

- **Skills**: explicit (listed) + implicit (demonstrated but unlisted) + recency map
- **Experience quality**: company tier, tenure patterns, career trajectory
- **Impact signals**: quantified impact ratio, specificity score, ownership language
- **Quality meta-signals**:
  - `exaggeration_index` (0–1): vague superlatives without evidence
  - `specificity_score` (0–1): how concrete are the claims
  - `buzzword_density` (0–1): trendy terms vs substantive content
- **Synthesized**: career archetype, career narrative, green/red flags

### 3. Alignment Engine (`src/scoring/aligner.py`)

Deterministic (no LLM) comparison layer that produces an `AlignmentResult`:

- Skill set intersection (required + preferred)
- Experience gap calculation
- Seniority ordering comparison
- Domain overlap scoring
- Semantic similarity via `sentence-transformers` embeddings
- Composite candidate quality score

### 4. Scoring Engine (`src/scoring/scorer.py`)

Weighted tiered scoring with full explainability:

| Tier | Weight | Signals |
|------|--------|---------|
| T1 Must-Haves | 50% | Required skills, YoE, seniority, domain |
| T2 Good-to-Haves | 25% | Preferred skills, tech stack, semantic similarity |
| T3 Quality Signals | 15% | Impact quality, project complexity, career trajectory |
| T4 Bonus | 5% | Open source, publications, rare skills |
| Penalty | up to -15% | Exaggeration, buzzwords, gaps |

**Gate check**: Hard filters applied before scoring. Resumes failing the gate score 0.0 immediately.

### 5. Explainability Layer (`src/scoring/explainer.py`)

LLM-generated recruiter-facing summary for top-N candidates:
- Headline verdict
- Top 3 strengths
- Top 2-3 concerns
- Interview probing questions
- Recommendation (Strong Yes / Lean Yes / Maybe / Lean No / No)

---

## Model Selection & Justification

### LLM: GPT-4o-mini (default) or Groq

**Why GPT-4o-mini:**
- Strong at structured JSON extraction with schema enforcement
- Low cost (~$0.15/1M input tokens) — important for batch processing
- Fast enough for interactive use (~3s per resume)
- Deterministic at temperature=0 for reproducibility

**Alternative considered: TF-IDF + cosine similarity**
- Advantage: No API cost, fully local, fast
- Disadvantage: Cannot handle synonyms ("built" vs "developed"), misses implicit skills, no concept of seniority or ownership style. A TF-IDF matcher would score a buzzword-stuffed resume (R6) as high as a substantive one.

**Alternative considered: Cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`)**
- Advantage: No API cost, captures semantic relevance well
- Disadvantage: No structured extraction — you get a similarity score but no explanation of *why*. Can't extract seniority, exaggeration index, or career trajectory.

### Embeddings: `BAAI/bge-small-en-v1.5` (local)

- Runs fully locally — no API cost or data privacy concern
- Strong performance on retrieval tasks (MTEB benchmark)
- Used only as one signal (T2 semantic similarity), not the sole score

---

## Text Preprocessing

1. **No aggressive cleaning**: LLMs handle messy formatting well. We pass raw text.
2. **Section-awareness**: Prompts instruct the LLM to distinguish skills sections from body text — implicit vs explicit skills.
3. **Date normalisation**: LLM infers duration in years from date ranges; we don't trust candidate's own "10+ years" claims.
4. **Skill normalisation**: Set intersection uses lowercased, stripped skill strings. Can be extended with embedding-based fuzzy matching for synonyms.

---

## Performance Evaluation

### Synthetic Test Set

| Candidate | Expected Label | Expected Score |
|-----------|---------------|----------------|
| Priya Sharma | Good Match | 1.0 |
| Chen Li | Good Match | 0.85 |
| James Wu | Good Match (slight gap) | 0.75 |
| Sarah Okonkwo | Partial Match | 0.5 |
| Alex Nguyen | Partial Match (junior) | 0.25 |
| Michael Patel | Poor Match | 0.15 |
| Ryan Taylor | Poor Match (adversarial) | 0.1 |
| Jessica Brown | Poor Match | 0.0 |

**Note on R6 (Ryan Taylor):** This is the adversarial test case — a buzzword-stuffed resume that would fool a naive TF-IDF or embedding-only system. Our exaggeration_index and specificity_score signals should correctly penalise this candidate.

### Metrics for Larger Labeled Datasets

**Primary metric: nDCG@K** (Normalized Discounted Cumulative Gain)
- Why: This is a *ranking* problem, not classification. nDCG rewards getting the *order* right, with heavier penalty for misranking the top positions. A recruiter reviews the top 5 — getting rank 1 wrong is worse than getting rank 8 wrong.

**Secondary metrics:**
- **Spearman ρ**: Overall rank correlation — intuitive and interpretable
- **Precision@3 and @5**: Fraction of the top shortlist that are genuine matches — direct business metric (recruiter time)
- **Pearson r**: Score calibration — are our 0.8s actually better than our 0.5s?

**Not just Accuracy**: Binary accuracy is wrong here. A "partial match" scored 0.48 vs 0.52 is not a meaningful error. Ranking quality metrics capture the real use case.

**For production labeling**: Recruiter feedback (thumbs up/down, interview outcome, hire outcome) can be used as weak supervision labels to tune the tier weights via a simple regression.

---

## Next Steps for Production

1. **Weight learning**: Replace heuristic tier weights with learned weights from recruiter feedback data (logistic regression or simple MLP on alignment features)
2. **Skill taxonomy**: Replace exact string match with a curated skill taxonomy + embedding fuzzy match for synonyms (ML vs Machine Learning vs ml-engineer)
3. **ATS integration**: Connect to Workday/Greenhouse via API for real-time scoring on inbound applications
4. **Bias audit**: Audit scoring distribution across demographic proxies (name, school, location) — critical for HR compliance in regulated environments
5. **Scalability**: Move to async extraction with a job queue (Celery + Redis) for batch processing 1000s of resumes
6. **Fine-tuning**: Fine-tune the embedding model on domain-specific resume-JD pairs for better semantic similarity

---

## Repository Structure

```
resume-matcher/
├── main.py                        # CLI entrypoint
├── config.py                      # Weights, model config (all tunable here)
├── requirements.txt
├── .env.example
├── src/
│   ├── engine.py                  # Main orchestrator
│   ├── extractors/
│   │   ├── schemas.py             # Pydantic data models
│   │   ├── jd_extractor.py        # JD extraction pipeline
│   │   └── resume_extractor.py    # Resume extraction pipeline
│   ├── scoring/
│   │   ├── aligner.py             # Deterministic alignment
│   │   ├── scorer.py              # Tiered scoring engine
│   │   └── explainer.py          # Recruiter-facing explanation
│   └── utils/
│       └── llm_client.py          # LLM abstraction (OpenAI / Groq)
├── data/
│   ├── sample_jd.txt
│   └── resumes/                   # 8 synthetic test resumes
├── evaluation/
│   ├── eval_dataset.json          # Ground truth labels
│   └── metrics.py                 # nDCG, Spearman, Precision@K
└── notebooks/
    └── evaluation.ipynb           # Visual evaluation (run after main.py)
```
