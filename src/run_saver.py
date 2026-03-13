"""
RunSaver
========
Persists intermediate pipeline outputs after each stage so the eval
layer can inspect every step independently.

Called by the engine after each stage. Each pipeline run gets its own
timestamped folder under `runs/`:

    runs/
      run_20250313_143022/
        run_meta.json              ← run timestamp, JD file, resume count
        jd_raw.txt                 ← raw JD text (source for eval)
        jd_profile.json            ← JD extraction output
        candidates/
          arjun_mehta/
            resume_raw.txt         ← raw resume text (source for eval)
            resume_profile.json    ← resume extraction output
            alignment.json         ← aligner output
            scoring.json           ← scorer output
        ranked_results.json        ← final pipeline output

The eval script reads this folder structure — it never re-runs the pipeline.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_RUNS_DIR = Path("runs")


def _slug(name: str) -> str:
    """'Dr. Priya Venkataraman' → 'dr_priya_venkataraman'"""
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def _write(path: Path, data: dict | list | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


class RunSaver:
    """
    Stateful saver for a single pipeline run.
    Instantiate once per run, call save_* after each stage.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id  = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = _RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run artifacts → {self.run_dir}")

    # ── Stage savers ──────────────────────────────────────────────────────────

    def save_meta(self, jd_path: str, resume_count: int) -> None:
        _write(self.run_dir / "run_meta.json", {
            "run_id":       self.run_id,
            "timestamp":    datetime.now().isoformat(),
            "jd_path":      str(jd_path),
            "resume_count": resume_count,
        })

    def save_jd_raw(self, raw_text: str) -> None:
        _write(self.run_dir / "jd_raw.txt", raw_text)

    def save_jd_profile(self, jd_profile) -> None:
        _write(self.run_dir / "jd_profile.json", jd_profile.model_dump())

    def save_resume_raw(self, candidate_id: str, raw_text: str) -> None:
        """candidate_id should be the resume filename stem, e.g. 'arjun_mehta_resume'."""
        path = self.run_dir / "candidates" / _slug(candidate_id) / "resume_raw.txt"
        _write(path, raw_text)

    def save_resume_profile(self, candidate_id: str, resume_profile) -> None:
        path = self.run_dir / "candidates" / _slug(candidate_id) / "resume_profile.json"
        _write(path, resume_profile.model_dump())

    def save_alignment(self, candidate_id: str, alignment) -> None:
        path = self.run_dir / "candidates" / _slug(candidate_id) / "alignment.json"
        _write(path, asdict(alignment))

    def save_scoring(self, candidate_id: str, scoring) -> None:
        path = self.run_dir / "candidates" / _slug(candidate_id) / "scoring.json"
        _write(path, scoring.to_dict())

    def save_ranked_results(self, ranked_results: list[dict]) -> None:
        _write(self.run_dir / "ranked_results.json", {"ranked_results": ranked_results})
        logger.info(f"Run complete → {self.run_dir}")

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self.run_dir