#!/usr/bin/env python
"""Build the lmtp-ready longitudinal panel for the eICU-CRD AKI cohort.

Requires the eICU cohort (stage 6) and outcomes (stage 7).

Usage:
    python scripts/08_build_panel_eicu.py --config config/config.yaml
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
from makeaki.eicu_panel import build_panel_eicu  # noqa: E402


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
    cohort = pd.read_parquet(cfg.output_dir / "eicu_aki_cohort.parquet")
    outcomes = pd.read_parquet(cfg.output_dir / "eicu_aki_cohort_outcomes.parquet")

    panel = build_panel_eicu(cohort, outcomes, cfg)
    panel.to_parquet(cfg.output_dir / "eicu_aki_panel.parquet", index=False)

    print(f"\nPanel shape: {panel.shape[0]:,d} rows x {panel.shape[1]} columns")
    print(f"Written to {cfg.output_dir / 'eicu_aki_panel.parquet'}")


if __name__ == "__main__":
    main()
