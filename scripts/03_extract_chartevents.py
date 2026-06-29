#!/usr/bin/env python
"""Stream chartevents once and cache the pipeline item ids.

chartevents is the largest table. This script extracts only the item ids needed
for outcomes (RRT markers) and exposures (MAP, weight, height) and writes a
compact parquet cache reused by later stages. Run this before re-running the
outcome and exposure stages.

Usage:
    python scripts/03_extract_chartevents.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from makeaki.config import load_config  # noqa: E402
from makeaki.signals import chartevents_itemids, get_chartevents  # noqa: E402


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
    logging.getLogger("makeaki").info(
        "Extracting chartevents item ids: %s", chartevents_itemids(cfg)
    )
    chart = get_chartevents(cfg)
    print(f"\nCached chartevents rows: {len(chart):,d}")
    print("Rows per item id:")
    counts = chart["itemid"].value_counts().sort_index()
    for itemid, n in counts.items():
        print(f"  {itemid:>8d} {n:>12,d}")


if __name__ == "__main__":
    main()
