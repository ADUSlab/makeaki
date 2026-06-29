#!/usr/bin/env python
"""Label MAKE-H and its components for the eICU-CRD AKI cohort.

Requires the cohort from 06_build_cohort_eicu.py.

Usage:
    python scripts/07_label_outcomes_eicu.py --config config/config.yaml
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
from makeaki.eicu_outcomes import build_make_h_eicu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config(args.config)
    cohort_path = cfg.output_dir / "eicu_aki_cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(f"Cohort not found at {cohort_path}. Run stage 6 first.")
    cohort = pd.read_parquet(cohort_path)

    labeled = build_make_h_eicu(cohort, cfg)
    labeled.to_parquet(cfg.output_dir / "eicu_aki_cohort_outcomes.parquet", index=False)

    n = len(labeled)

    def pct(x: int) -> str:
        return f"{x:,d} ({100.0 * x / n:.1f}%)"

    print("\nMAKE-H summary (eICU-CRD AKI cohort):")
    print(f"  cohort rows                 {n:,d}")
    print(f"  classifiable MAKE-H         {pct(int(labeled['make_h_classifiable'].sum()))}")
    print(f"  MAKE-H positive             {pct(int(labeled['make_h'].sum()))}")
    print(f"  - death-H                   {pct(int(labeled['death_h'].sum()))}")
    print(f"  - RRT-H (dependence)        {pct(int(labeled['rrt_h'].sum()))}")
    print(f"  - ever-RRT (any time)       {pct(int(labeled['ever_rrt'].sum()))}")
    print(f"  - PRD-H (True)              {pct(int(labeled['prd_h'].fillna(False).sum()))}")


if __name__ == "__main__":
    main()
