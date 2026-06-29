#!/usr/bin/env python
"""Build the lmtp-ready longitudinal panel for the MIMIC-IV AKI cohort.

Requires the cohort (stage 1), outcomes (stage 2), chartevents cache (stage 3),
and exposures (stage 4).

Usage:
    python scripts/05_build_panel_mimic.py --config config/config.yaml
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
from makeaki.panel_mimic import build_panel  # noqa: E402


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
    out = cfg.output_dir
    cohort = pd.read_parquet(out / "mimic_aki_cohort.parquet")
    outcomes = pd.read_parquet(out / "mimic_aki_cohort_outcomes.parquet")
    exposures = pd.read_parquet(out / "mimic_aki_exposures.parquet")

    panel = build_panel(cohort, exposures, outcomes, cfg)
    panel_path = out / "mimic_aki_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    print(f"\nPanel shape: {panel.shape[0]:,d} rows x {panel.shape[1]} columns")
    print("Columns:")
    print("  " + ", ".join(panel.columns))
    print(f"\nWritten to {panel_path}")


if __name__ == "__main__":
    main()
