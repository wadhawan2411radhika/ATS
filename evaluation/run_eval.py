"""
Three-Stage Evaluation Pipeline.

Reads from saved run artifacts — no re-running the pipeline.

Flow:
    Stage 1: JD Extraction    → Precision / Recall on required + preferred skills
    Stage 2: Resume Extraction → Precision / Recall on explicit + implicit skills (per candidate + macro avg)
    Stage 3: Scoring           → nDCG@3, nDCG@5, Spearman ρ

Usage:
    # Evaluate latest run
    python evaluation/run_eval.py

    # Evaluate specific run
    python evaluation/run_eval.py --run-id run_20250313_143022

    # Run a single stage
    python evaluation/run_eval.py --stage jd
    python evaluation/run_eval.py --stage resume
    python evaluation/run_eval.py --stage scoring

Reads from:
    runs/{run_id}/jd_profile.json                          ← Stage 1
    runs/{run_id}/candidates/{slug}/resume_profile.json    ← Stage 2
    runs/{run_id}/ranked_results.json                      ← Stage 3
    evaluation/ground_truth/jd_gt.json
    evaluation/ground_truth/resume_gt.json
    evaluation/eval_dataset.json

Writes:
    runs/{run_id}/eval_results.json
"""

import json
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running from repo root OR from evaluation/ directory
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from evaluation.skill_matcher import compute_precision_recall, match_skill_to_gt

logging.basicConfig(
    level=logging.WARNING,   # suppress engine logs during eval
    format="%(levelname)s: %(message)s",
)

RUNS_DIR = _ROOT / "runs"
EVAL_DIR = _HERE
GT_DIR   = EVAL_DIR / "ground_truth"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}\n  → Check paths in README.")
    with open(path) as f:
        return json.load(f)


def get_latest_run() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(
            f"No runs/ directory found at {RUNS_DIR}.\n"
            "  → Run main.py first to generate a run artifact."
        )
    runs = sorted([r for r in RUNS_DIR.iterdir() if r.is_dir()], reverse=True)
    if not runs:
        raise FileNotFoundError("No runs found. Run main.py first.")
    return runs[0]


def _header(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _find_candidate_dir(candidates_dir: Path, candidate_id: str) -> Path | None:
    """
    Locate the candidate's folder under runs/{run_id}/candidates/.

    RunSaver slugifies the candidate_id (filename stem) when saving.
    Tries exact match first, then partial match on the first token.
    """
    # Exact match
    exact = candidates_dir / candidate_id
    if exact.exists():
        return exact

    # Slug of candidate_id (in case it contains spaces etc.)
    slug = candidate_id.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    slugged = candidates_dir / slug
    if slugged.exists():
        return slugged

    # Partial: scan all subdirs for one containing the candidate_id
    if candidates_dir.exists():
        for d in candidates_dir.iterdir():
            if candidate_id.replace("_resume", "") in d.name:
                return d

    return None


# ── Stage 1: JD Extraction ─────────────────────────────────────────────────────

def eval_jd_extraction(run_dir: Path) -> dict:
    _header("STAGE 1 — JD Extraction: Skill Precision / Recall")

    jd_profile = load_json(run_dir / "jd_profile.json")
    jd_gt      = load_json(GT_DIR / "jd_gt.json")

    pred_required  = jd_profile["hard_requirements"]["required_skills"]
    pred_preferred = jd_profile["soft_requirements"]["preferred_skills"]
    gt_required    = jd_gt["required_skills"]
    gt_preferred   = jd_gt["preferred_skills"]

    req_m  = compute_precision_recall(pred_required,  gt_required)
    pref_m = compute_precision_recall(pred_preferred, gt_preferred)

    # ── Required skills ────────────────────────────────────────────────────────
    print(f"\n  Required Skills  (GT: {len(gt_required)} | Predicted: {len(pred_required)})")
    print(f"    Precision : {req_m['precision']:.3f}   "
          f"({req_m['tp']} correct, {req_m['fp']} false positives)")
    print(f"    Recall    : {req_m['recall']:.3f}   "
          f"({req_m['fn']} missed)")
    print(f"    F1        : {req_m['f1']:.3f}")

    if req_m["false_positives"]:
        print(f"    ✗ Hallucinated : {req_m['false_positives']}")
    if req_m["missed"]:
        print(f"    ✗ Missed       : {req_m['missed']}")

    # ── Preferred skills ───────────────────────────────────────────────────────
    print(f"\n  Preferred Skills (GT: {len(gt_preferred)} | Predicted: {len(pred_preferred)})")
    print(f"    Precision : {pref_m['precision']:.3f}   "
          f"({pref_m['tp']} correct, {pref_m['fp']} false positives)")
    print(f"    Recall    : {pref_m['recall']:.3f}   "
          f"({pref_m['fn']} missed)")
    print(f"    F1        : {pref_m['f1']:.3f}")

    if pref_m["false_positives"]:
        print(f"    ✗ Hallucinated : {pref_m['false_positives']}")
    if pref_m["missed"]:
        print(f"    ✗ Missed       : {pref_m['missed']}")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print()
    if req_m["recall"] < 0.80:
        print("  ⚠  Required skill recall < 0.80 — missed skills will cause downstream underscoring.")
    if req_m["precision"] < 0.70:
        print("  ⚠  Required skill precision < 0.70 — hallucinated skills inflate candidate scores.")
    if req_m["recall"] >= 0.85 and req_m["precision"] >= 0.80:
        print("  ✓  JD extraction quality is acceptable for v1.")

    return {"required": req_m, "preferred": pref_m}


# ── Stage 2: Resume Extraction ─────────────────────────────────────────────────

def eval_resume_extraction(run_dir: Path) -> dict:
    _header("STAGE 2 — Resume Extraction: Skill Precision / Recall")

    resume_gt      = load_json(GT_DIR / "resume_gt.json")
    candidates_dir = run_dir / "candidates"

    col_w = 26

    print(f"\n  {'CANDIDATE':<{col_w}} {'EXPL_P':>7} {'EXPL_R':>7} {'EXPL_F1':>8}  "
          f"{'IMPL_P':>7} {'IMPL_R':>7} {'IMPL_F1':>8}")
    print("  " + "-" * 76)

    all_expl, all_impl = [], []
    skipped = []

    for candidate_id, gt in resume_gt["candidates"].items():
        if candidate_id.startswith("_"):
            continue

        cand_dir = _find_candidate_dir(candidates_dir, candidate_id)
        if cand_dir is None:
            skipped.append(candidate_id)
            continue

        profile_path = cand_dir / "resume_profile.json"
        if not profile_path.exists():
            skipped.append(candidate_id)
            continue

        profile = load_json(profile_path)
        pred_explicit = profile["skills"]["explicit_skills"]
        pred_implicit = profile["skills"]["implicit_skills"]
        gt_explicit   = gt.get("explicit_skills", [])
        gt_implicit   = gt.get("implicit_skills", [])

        expl_m = compute_precision_recall(pred_explicit, gt_explicit)
        impl_m = compute_precision_recall(pred_implicit, gt_implicit)

        all_expl.append(expl_m)
        all_impl.append(impl_m)

        label = gt.get("_label", "")
        display = candidate_id.replace("_", " ").title()
        if label:
            display = f"{display} [{label[:8]}]"

        print(
            f"  {display:<{col_w}} "
            f"{expl_m['precision']:>7.3f} {expl_m['recall']:>7.3f} {expl_m['f1']:>8.3f}  "
            f"{impl_m['precision']:>7.3f} {impl_m['recall']:>7.3f} {impl_m['f1']:>8.3f}"
        )

        # Show missed implicit skills (highest-risk errors)
        if impl_m["missed"]:
            print(f"    ✗ Implicit missed : {impl_m['missed']}")
        if impl_m["false_positives"]:
            print(f"    ✗ Implicit halluc : {impl_m['false_positives']}")

    # ── Macro averages ─────────────────────────────────────────────────────────
    def macro(lst, key):
        return round(sum(m[key] for m in lst) / len(lst), 4) if lst else 0.0

    print("  " + "-" * 76)
    print(
        f"  {'MACRO AVERAGE':<{col_w}} "
        f"{macro(all_expl,'precision'):>7.3f} {macro(all_expl,'recall'):>7.3f} {macro(all_expl,'f1'):>8.3f}  "
        f"{macro(all_impl,'precision'):>7.3f} {macro(all_impl,'recall'):>7.3f} {macro(all_impl,'f1'):>8.3f}"
    )

    if skipped:
        print(f"\n  ⚠  Skipped (no run artifact): {skipped}")
        print("     → These candidates may not have been in the evaluated run.")

    # ── Interpretation ─────────────────────────────────────────────────────────
    impl_r = macro(all_impl, "recall")
    impl_p = macro(all_impl, "precision")
    print()
    if impl_r < 0.70:
        print("  ⚠  Implicit recall < 0.70 — significant skill coverage gaps.")
    if impl_p < 0.65:
        print("  ⚠  Implicit precision < 0.65 — risk of inflating scores for weak candidates.")
    if impl_r >= 0.75 and impl_p >= 0.70:
        print("  ✓  Implicit skill extraction quality acceptable for v1.")

    return {
        "explicit": {
            "macro_precision": macro(all_expl, "precision"),
            "macro_recall":    macro(all_expl, "recall"),
            "macro_f1":        macro(all_expl, "f1"),
        },
        "implicit": {
            "macro_precision": macro(all_impl, "precision"),
            "macro_recall":    macro(all_impl, "recall"),
            "macro_f1":        macro(all_impl, "f1"),
        },
        "evaluated": len(all_expl),
        "skipped":   skipped,
    }


# ── Stage 3: Scoring ──────────────────────────────────────────────────────────

def _dcg(relevances: list[float]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(
    predicted_order: list[str],
    gt_labels: dict[str, float],
    k: int,
) -> float:
    """
    nDCG@K — Normalised Discounted Cumulative Gain at cutoff K.

    Rewards correct ordering at the top of the ranked list.
    A Good Match ranked 5th is penalised more than one ranked 2nd.
    Directly maps to recruiter behaviour: they review down a list and stop.
    """
    top_k = predicted_order[:k]
    dcg   = _dcg([gt_labels.get(cid, 0.0) for cid in top_k])

    ideal_relevances = sorted(gt_labels.values(), reverse=True)[:k]
    idcg  = _dcg(ideal_relevances)

    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def eval_scoring(run_dir: Path) -> dict:
    _header("STAGE 3 — Scoring: nDCG & Rank Correlation")

    ranked_results = load_json(run_dir / "ranked_results.json")
    eval_dataset   = load_json(EVAL_DIR / "eval_dataset.json")

    gt_labels = {c["id"]: c["label"] for c in eval_dataset["candidates"]}

    ranked           = ranked_results["ranked_results"]
    predicted_order  = [r["candidate_name"] for r in ranked]
    predicted_scores = [r["final_score"] for r in ranked]
    gt_scores        = [gt_labels.get(cid, 0.0) for cid in predicted_order]

    # ── Metrics ────────────────────────────────────────────────────────────────
    ndcg3       = _ndcg_at_k(predicted_order, gt_labels, k=3)
    ndcg5       = _ndcg_at_k(predicted_order, gt_labels, k=5)
    spearman_r, spearman_p = spearmanr(predicted_scores, gt_scores)

    print(f"\n  nDCG@3     : {ndcg3:.4f}   (primary — did the right people land in top 3?)")
    print(f"  nDCG@5     : {ndcg5:.4f}   (secondary — shortlist quality)")
    print(f"  Spearman ρ : {spearman_r:.4f}   (p={spearman_p:.4f})  "
          f"{'significant' if spearman_p < 0.05 else 'not significant at α=0.05'}")

    # ── Per-candidate table ────────────────────────────────────────────────────
    print(f"\n  {'RNK':<5} {'CANDIDATE':<28} {'PRED':>7} {'GT':>7} {'STATUS'}")
    print("  " + "-" * 60)

    for rank, (cid, pred, gt) in enumerate(zip(predicted_order, predicted_scores, gt_scores), 1):
        if pred > 0.5 and gt >= 0.75:
            status = "✓ correct"
        elif pred <= 0.3 and gt == 0.0:
            status = "✓ correctly gated"
        elif pred > 0.5 and gt < 0.5:
            status = "△ overranked"
        elif pred <= 0.5 and gt >= 0.75:
            status = "✗ underranked"
        else:
            status = "~ acceptable"

        gate_marker = " [GATED]" if pred == 0.0 else ""
        print(f"  {rank:<5} {cid:<28} {pred:>7.3f} {gt:>7.3f}   {status}{gate_marker}")

    # ── Overrank / underrank analysis ─────────────────────────────────────────
    overranked  = [(p, g, c) for c, p, g in zip(predicted_order, predicted_scores, gt_scores)
                   if p > 0.5 and g < 0.5]
    underranked = [(p, g, c) for c, p, g in zip(predicted_order, predicted_scores, gt_scores)
                   if p <= 0.5 and g >= 0.75]

    print()
    if overranked:
        print(f"  ⚠  Overranked candidates: {[c for _, _, c in overranked]}")
        print("     → Check alignment.json for these candidates: which signals are inflating the score?")
    if underranked:
        print(f"  ⚠  Underranked candidates: {[c for _, _, c in underranked]}")
        print("     → Check missing_required_skills in scoring.json: likely a skill extraction miss.")
    if not overranked and not underranked:
        print("  ✓  No significant ranking errors detected.")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print()
    if ndcg3 >= 0.95:
        print("  ✓  nDCG@3 ≥ 0.95 — top-3 ranking is strong.")
    elif ndcg3 >= 0.85:
        print("  ~  nDCG@3 0.85-0.95 — acceptable for v1, room to improve.")
    else:
        print("  ✗  nDCG@3 < 0.85 — ranking quality needs investigation.")

    return {
        "ndcg_at_3":   ndcg3,
        "ndcg_at_5":   ndcg5,
        "spearman_rho": round(float(spearman_r), 4),
        "spearman_p":   round(float(spearman_p), 4),
        "overranked":   [c for _, _, c in overranked],
        "underranked":  [c for _, _, c in underranked],
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    _header("EVALUATION SUMMARY")
    print()

    if "jd_extraction" in results:
        jd = results["jd_extraction"]
        r  = jd["required"]
        p  = jd["preferred"]
        print(f"  JD  Required Skills   P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
        print(f"  JD  Preferred Skills  P={p['precision']:.3f}  R={p['recall']:.3f}  F1={p['f1']:.3f}")

    if "resume_extraction" in results:
        re = results["resume_extraction"]
        e  = re["explicit"]
        i  = re["implicit"]
        print(f"  Resume  Explicit  macro P={e['macro_precision']:.3f}  R={e['macro_recall']:.3f}  F1={e['macro_f1']:.3f}")
        print(f"  Resume  Implicit  macro P={i['macro_precision']:.3f}  R={i['macro_recall']:.3f}  F1={i['macro_f1']:.3f}")

    if "scoring" in results:
        sc = results["scoring"]
        print(f"  Scoring  nDCG@3={sc['ndcg_at_3']:.4f}  nDCG@5={sc['ndcg_at_5']:.4f}  Spearman ρ={sc['spearman_rho']:.4f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resume Matcher — Three-Stage Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_eval.py                              # evaluate latest run, all stages
  python evaluation/run_eval.py --run-id run_20250313_143022 # specific run
  python evaluation/run_eval.py --stage jd                   # JD extraction only
  python evaluation/run_eval.py --stage resume               # resume extraction only
  python evaluation/run_eval.py --stage scoring              # scoring only
        """,
    )
    parser.add_argument("--run-id", default=None,
                        help="Run ID to evaluate (default: most recent run)")
    parser.add_argument("--stage",
                        choices=["jd", "resume", "scoring", "all"],
                        default="all",
                        help="Which evaluation stage to run (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed per-skill breakdowns")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id if args.run_id else get_latest_run()
    print(f"\nRun : {run_dir.name}")
    print(f"Path: {run_dir}")

    results: dict = {}

    try:
        if args.stage in ("jd", "all"):
            results["jd_extraction"] = eval_jd_extraction(run_dir)

        if args.stage in ("resume", "all"):
            results["resume_extraction"] = eval_resume_extraction(run_dir)

        if args.stage in ("scoring", "all"):
            results["scoring"] = eval_scoring(run_dir)

    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        sys.exit(1)

    _print_summary(results)

    # ── Save results alongside run artifacts ───────────────────────────────────
    output_path = run_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_path}\n")


if __name__ == "__main__":
    main()