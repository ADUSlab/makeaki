#!/usr/bin/env python
"""Label MAKE-H and its components for the MIMIC-IV AKI cohort.

Requires the cohort produced by 01_build_cohort_mimic.py.

Usage:
    python scripts/02_label_outcomes_mimic.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from makeaki.config import load_config  # noqa: E402
from makeaki.outcomes_mimic import build_make_h  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    cohort_path = cfg.output_dir / "mimic_aki_cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(
            f"Cohort not found at {cohort_path}. Run 01_build_cohort_mimic.py first."
        )
    cohort = pd.read_parquet(cohort_path)

    labeled = build_make_h(cohort, cfg)
    out_path = cfg.output_dir / "mimic_aki_cohort_outcomes.parquet"
    labeled.to_parquet(out_path, index=False)

    n = len(labeled)
    classifiable = int(labeled["make_h_classifiable"].sum())

    def pct(x: int) -> str:
        return f"{x:,d} ({100.0 * x / n:.1f}%)"

    ratio_arm = (labeled["scr_last"] / labeled["baseline_creatinine"] >= 2.0)
    egfr_arm = labeled["egfr_last"] <= 0.75 * labeled["baseline_egfr"]

    print("\nMAKE-H summary (MIMIC-IV AKI cohort):")
    print(f"  cohort rows                 {n:,d}")
    print(f"  classifiable MAKE-H         {pct(classifiable)}")
    print(f"  MAKE-H positive             {pct(int(labeled['make_h'].sum()))}")
    print(f"  - death-H                   {pct(int(labeled['death_h'].sum()))}")
    print(f"  - RRT-H (dependence)        {pct(int(labeled['rrt_h'].sum()))}")
    print(f"  - ever-RRT (any time)       {pct(int(labeled['ever_rrt'].sum()))}")
    print(f"  - PRD-H (True)              {pct(int(labeled['prd_h'].fillna(False).sum()))}")
    print(f"    - PRD via SCr>=2.0        {pct(int(ratio_arm.fillna(False).sum()))}")
    print(f"    - PRD via eGFR<=0.75x     {pct(int(egfr_arm.fillna(False).sum()))}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
