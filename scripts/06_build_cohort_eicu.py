#!/usr/bin/env python
"""Build the eICU-CRD adult ICU AKI cohort and write the screening log.

Usage:
    python scripts/06_build_cohort_eicu.py --config config/config.yaml
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
from makeaki.eicu_cohort import build_eicu_cohort  # noqa: E402


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
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cohort, screening = build_eicu_cohort(cfg)

    cohort.to_parquet(cfg.output_dir / "eicu_aki_cohort.parquet", index=False)
    pd.DataFrame({"step": list(screening), "n": list(screening.values())}).to_csv(
        cfg.output_dir / "eicu_screening_log.csv", index=False
    )

    print("\nScreening funnel (eICU-CRD):")
    for step, n in screening.items():
        print(f"  {step:40s} {n:>10,d}")
    print(f"\nFinal AKI cohort rows: {len(cohort):,d}")


if __name__ == "__main__":
    main()
