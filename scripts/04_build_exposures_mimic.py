#!/usr/bin/env python
"""Build the 72-hour modifiable exposures for the MIMIC-IV AKI cohort.

Requires the cohort from 01_build_cohort_mimic.py and the chartevents cache from
03_extract_chartevents.py.

Usage:
    python scripts/04_build_exposures_mimic.py --config config/config.yaml
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
from makeaki.exposures_mimic import build_exposures  # noqa: E402


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
        raise SystemExit(f"Cohort not found at {cohort_path}. Run stage 1 first.")
    cohort = pd.read_parquet(cohort_path)

    exposures = build_exposures(cohort, cfg)
    out_path = cfg.output_dir / "mimic_aki_exposures.parquet"
    exposures.to_parquet(out_path, index=False)

    n = len(exposures)

    def summ(col: str) -> str:
        s = exposures[col].dropna()
        if s.empty:
            return "no data"
        return (
            f"median {s.median():.2f}  IQR [{s.quantile(0.25):.2f}, "
            f"{s.quantile(0.75):.2f}]  n={len(s):,d}"
        )

    def rate(col: str) -> str:
        s = exposures[col].dropna().astype(bool)
        if s.empty:
            return "no data"
        return f"{int(s.sum()):,d} ({100.0 * s.mean():.1f}%) of n={len(s):,d}"

    print("\nExposure summary (MIMIC-IV AKI cohort, first 72h):")
    print(f"  cohort rows                       {n:,d}")
    print(f"  fluid accumulation %              {summ('fa_pct')}")
    print(f"  A_fluid (>5% accumulation)        {rate('a_fluid')}")
    print(f"  hypotension burden (mmHg-h)       {summ('hb_mmhg_h')}")
    print(f"  nephrotoxin drug-days             {summ('nephrotoxin_drug_days')}")
    print(f"  A_nephrotoxin (>=2 concurrent)    {rate('a_nephrotoxin')}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
