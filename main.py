"""
CLI Entrypoint for the Resume Matching Engine.

Usage:
    # Validate parsing (no API calls)
    python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --parse-only

    # Full run
    python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --explain-top 3

    # Save results to JSON
    python main.py --jd data/sample_jd.txt --resumes-dir data/resumes/ --output results.json

Supported resume formats: .pdf, .docx, .txt
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_resumes_from_dir(directory: str) -> dict[str, str]:
    """Load all supported resume files (.pdf, .docx, .txt) from a directory."""
    from src.utils.file_parser import load_resumes_from_dir as _load
    return _load(directory)


def print_results_table(results) -> None:
    print("\n" + "=" * 72)
    print(f"{'RANK':<6} {'CANDIDATE':<30} {'SCORE':<8} {'SENIORITY':<15} {'GATE'}")
    print("=" * 72)
    for rank, result in enumerate(results, 1):
        gate_str = "✓" if result.scoring.passed_gate else f"✗ {result.scoring.gate_failure_reason}"
        print(
            f"{rank:<6} {result.candidate_name:<30} "
            f"{result.score:<8.3f} "
            f"{result.extracted_resume.current_seniority.value:<15} "
            f"{gate_str}"
        )
    print("=" * 72)

    print("\n── TOP 3 DETAILED BREAKDOWN ─────────────────────────────────────────\n")
    for result in results[:3]:
        b = result.scoring
        print(f"  {result.candidate_name}  (score: {result.score:.3f})")
        print(f"    Required skills matched : {', '.join(b.matched_required_skills) or 'None'}")
        print(f"    Missing required skills : {', '.join(b.missing_required_skills) or 'None'}")
        print(f"    Preferred skills matched: {', '.join(b.matched_preferred_skills) or 'None'}")
        print(f"    Experience gap          : {b.experience_gap_years:+.1f} years")
        print(f"    Seniority               : {b.seniority_match}")
        print(f"    Tier scores             : T1={b.tier1_score:.2f} | T2={b.tier2_score:.2f} | T3={b.tier3_score:.2f} | T4={b.tier4_score:.2f}")
        if b.bonus_signals:
            print(f"    ✓ Bonus : {', '.join(b.bonus_signals)}")
        if b.penalty_signals:
            print(f"    ✗ Flags : {', '.join(b.penalty_signals)}")
        if result.explanation:
            e = result.explanation
            print(f"\n    RECRUITER VERDICT: {e.headline}")
            print(f"    Recommendation: {e.recommendation}")
            print(f"    Strengths: {'; '.join(e.why_strong)}")
            print(f"    Concerns : {'; '.join(e.why_weak)}")
            print(f"    Probe in interview: {'; '.join(e.interview_focus)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Resume Matching Engine")
    parser.add_argument("--jd", required=True, help="Path to job description (.txt)")
    parser.add_argument("--resumes-dir", required=True, help="Directory of resumes (.pdf, .docx, .txt)")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--explain-top", type=int, default=3, help="LLM explanations for top N (default 3)")
    parser.add_argument("--parse-only", action="store_true",
                        help="Dry run — parse files, skip LLM calls. Good for validating inputs.")
    args = parser.parse_args()

    if not os.path.exists(args.jd):
        logger.error(f"JD file not found: {args.jd}"); sys.exit(1)
    if not os.path.isdir(args.resumes_dir):
        logger.error(f"Resumes dir not found: {args.resumes_dir}"); sys.exit(1)

    jd_text = load_text_file(args.jd)
    resumes = load_resumes_from_dir(args.resumes_dir)

    if not resumes:
        logger.error(f"No resume files found in {args.resumes_dir}"); sys.exit(1)

    # ── Parse-only dry run ──────────────────────────────────────────────────
    if args.parse_only:
        print("\n── PARSE-ONLY MODE (no LLM calls) ──────────────────────────────────\n")
        print(f"JD loaded: {len(jd_text.split())} words")
        print(f"JD preview: {jd_text[:200].strip()}\n")
        print(f"  {'FILE':<45} {'WORDS':>6}")
        print("  " + "-" * 55)
        for name, text in resumes.items():
            print(f"  {name:<45} {len(text.split()):>6}")
        print(f"\n  ✓ {len(resumes)} resume(s) parsed successfully.")
        print("  → Set OPENAI_API_KEY and remove --parse-only for full run.\n")
        return

    # ── Full pipeline ───────────────────────────────────────────────────────
    logger.info(f"Loaded JD + {len(resumes)} resumes. Starting engine...")

    from src.engine import MatchingEngine
    engine = MatchingEngine()
    results = engine.match(jd_text=jd_text, resumes=resumes, explain_top_n=args.explain_top)

    print_results_table(results)

    if args.output:
        output_data = {
            "ranked_results": [
                {
                    **r.scoring.to_dict(),
                    "explanation": {
                        "headline": r.explanation.headline,
                        "why_strong": r.explanation.why_strong,
                        "why_weak": r.explanation.why_weak,
                        "interview_focus": r.explanation.interview_focus,
                        "recommendation": r.explanation.recommendation,
                    } if r.explanation else None,
                }
                for r in results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
