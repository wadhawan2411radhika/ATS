"""
Resume Matching Engine — Main Orchestrator.

Ties together:
  1. JD extraction
  2. Resume extraction (parallelized)
  3. Alignment
  4. Scoring
  5. Explanation (top-N only)
  6. RunSaver — persists intermediates after each stage for eval layer

Usage:
    engine = MatchingEngine()
    results = engine.match(jd_text="...", resumes={"Alice": "...", "Bob": "..."})
"""

import logging
import concurrent.futures
from dataclasses import dataclass

from src.extraction.jd_extraction import extract_jd
from src.extraction.resume_extraction import extract_resume
from src.extraction.jd_extraction import JDProfile
from src.extraction.resume_extraction import ResumeProfile as ResumeExtracted
from src.scoring.schemas import AlignmentResult
from src.scoring.aligner import align
from src.scoring.scorer import score, ScoringBreakdown
from src.scoring.explainer import explain, RecruiterExplanation
from src.run_saver import RunSaver
from config import config

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Complete result for a single resume."""
    candidate_name: str
    resume_text: str
    extracted_resume: ResumeExtracted
    alignment: AlignmentResult
    scoring: ScoringBreakdown
    explanation: RecruiterExplanation | None = None  # Only for top-N

    @property
    def score(self) -> float:
        return self.scoring.final_score


class MatchingEngine:
    """
    End-to-end resume matching engine.

    Designed to be reusable across different JDs.
    JD is extracted once and reused for all resumes.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._jd_cache: dict[str, JDProfile] = {}

    def _extract_resume_safe(
        self, name: str, text: str
    ) -> tuple[str, ResumeExtracted | None]:
        """Extract resume with error handling — one failure shouldn't block others."""
        try:
            return name, extract_resume(text)
        except Exception as e:
            logger.error(f"Failed to extract resume for '{name}': {e}")
            return name, None

    def match(
        self,
        jd_text: str,
        resumes: dict[str, str],  # {candidate_id: resume_text}  (key = filename stem)
        explain_top_n: int | None = None,
        run_id: str | None = None,
    ) -> list[MatchResult]:
        """
        Run the full matching pipeline.

        Args:
            jd_text:      Raw job description text.
            resumes:      Dict mapping candidate_id (filename stem) → raw resume text.
            explain_top_n: Generate LLM explanations for top N results.
                           Defaults to config.explain_top_n.
            run_id:       Optional fixed run ID. Auto-generated if None.

        Returns:
            List of MatchResult sorted by score descending.
        """
        n_explain = explain_top_n if explain_top_n is not None else config.explain_top_n

        # ── RunSaver: one per run ─────────────────────────────────────────────
        saver = RunSaver(run_id=run_id)

        # ── Step 1: Extract JD ────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting Job Description")
        logger.info("=" * 60)

        saver.save_meta(jd_path="<text input>", resume_count=len(resumes))
        saver.save_jd_raw(jd_text)

        jd_hash = hash(jd_text)
        if jd_hash not in self._jd_cache:
            self._jd_cache[jd_hash] = extract_jd(jd_text)
        jd = self._jd_cache[jd_hash]

        saver.save_jd_profile(jd_profile=jd)
        logger.info(f"JD: {jd.role_identity.job_title} | {jd.role_identity.seniority_level}")

        # ── Step 2: Extract Resumes (parallelized) ────────────────────────────
        logger.info("=" * 60)
        logger.info(f"STEP 2: Extracting {len(resumes)} Resumes (parallel)")
        logger.info("=" * 60)
        extracted: dict[str, ResumeExtracted] = {}
        texts: dict[str, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_resume_safe, name, text): name
                for name, text in resumes.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                if result is not None:
                    extracted[name] = result
                    texts[name] = resumes[name]
                else:
                    logger.warning(f"Skipping '{name}' due to extraction failure.")

        logger.info(f"Successfully extracted {len(extracted)}/{len(resumes)} resumes")

        # Save resume intermediates (sequential — disk I/O is fast)
        for candidate_id, resume_profile in extracted.items():
            saver.save_resume_raw(candidate_id, texts[candidate_id])
            saver.save_resume_profile(candidate_id, resume_profile)

        # ── Step 3: Align + Score ─────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3: Aligning and Scoring")
        logger.info("=" * 60)
        results: list[MatchResult] = []

        for candidate_id, resume in extracted.items():
            alignment = align(jd, resume)
            saver.save_alignment(candidate_id, alignment)

            scoring = score(jd, resume, alignment)
            saver.save_scoring(candidate_id, scoring)

            results.append(MatchResult(
                candidate_name=candidate_id,
                resume_text=texts[candidate_id],
                extracted_resume=resume,
                alignment=alignment,
                scoring=scoring,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # ── Step 4: Explain Top N ─────────────────────────────────────────────
        if n_explain > 0:
            logger.info("=" * 60)
            logger.info(f"STEP 4: Generating explanations for top {n_explain}")
            logger.info("=" * 60)
            for result in results[:n_explain]:
                if result.scoring.passed_gate:
                    try:
                        result.explanation = explain(result.scoring, jd.role_identity.job_title)
                    except Exception as e:
                        logger.warning(f"Explanation failed for {result.candidate_name}: {e}")

        # ── Save final ranked output ──────────────────────────────────────────
        ranked_dicts = [_to_output_dict(r) for r in results]
        saver.save_ranked_results(ranked_dicts)
        logger.info(f"Run complete → {saver.path}")

        logger.info(f"Matching complete. Top candidate: {results[0].candidate_name} ({results[0].score:.3f})")
        return results


# ── Output serialisation ───────────────────────────────────────────────────────

def _to_output_dict(result: MatchResult) -> dict:
    """Serialise a MatchResult to a flat dict for ranked_results.json."""
    d = result.scoring.to_dict()
    d["candidate_id"] = result.candidate_name   # filename stem
    d["explanation"]  = {
        "headline":        result.explanation.headline,
        "why_strong":      result.explanation.why_strong,
        "why_weak":        result.explanation.why_weak,
        "interview_focus": result.explanation.interview_focus,
        "recommendation":  result.explanation.recommendation,
    } if result.explanation else None
    return d