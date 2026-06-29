#!/usr/bin/env python
"""Build the MIMIC-IV adult ICU AKI cohort and write the screening log.

Usage:
    python scripts/01_build_cohort_mimic.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the src layout importable when run as a plain script.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from makeaki.cohort_mimic import build_mimic_cohort  # noqa: E402
from makeaki.config import load_config  # noqa: E402


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
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, screening = build_mimic_cohort(cfg)

    cohort_path = out_dir / "mimic_aki_cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)

    screening_df = pd.DataFrame(
        {"step": list(screening.keys()), "n": list(screening.values())}
    )
    screening_path = out_dir / "mimic_screening_log.csv"
    screening_df.to_csv(screening_path, index=False)

    logging.getLogger("makeaki").info("Cohort written to %s", cohort_path)
    logging.getLogger("makeaki").info("Screening log written to %s", screening_path)
    print("\nScreening funnel (MIMIC-IV):")
    for step, n in screening.items():
        print(f"  {step:40s} {n:>10,d}")
    print(f"\nFinal AKI cohort rows: {len(cohort):,d}")


if __name__ == "__main__":
    main()
