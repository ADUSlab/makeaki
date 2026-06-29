"""Cached extraction of large longitudinal signals for MIMIC-IV.

Streaming labevents and chartevents is expensive, so the filtered signals are
cached as parquet under the output directory and reused by downstream stages.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from . import io
from .config import Config
from .kdigo import clean_creatinine

logger = logging.getLogger(__name__)

_CREAT_COLS = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
_CHART_COLS = ["hadm_id", "itemid", "charttime", "valuenum"]


def chartevents_itemids(cfg: Config) -> list[int]:
    """Union of every chartevents item id needed across the pipeline."""
    keys = (
        "map_chart", "weight_chart", "height_chart", "rrt_chart",
        "fio2_chart", "spo2_chart", "gcs_chart",
    )
    items: set[int] = set()
    for key in keys:
        items.update(int(i) for i in cfg["mimic_itemids"].get(key, []))
    return sorted(items)


def get_lab(cfg: Config, itemids: list[int], cache_name: str) -> pd.DataFrame:
    """Return cached in-hospital lab values for the given item ids.

    Columns: subject_id, hadm_id, charttime, itemid, valuenum.
    """
    cache = cfg.output_dir / f"_cache_{cache_name}_mimic.parquet"
    if cache.exists():
        logger.info("Loading cached %s from %s", cache_name, cache)
        return pd.read_parquet(cache)

    lab = io.read_filtered_by_itemid(
        cfg.mimic_hosp(),
        "labevents",
        itemids=itemids,
        usecols=_CREAT_COLS,
        parse_dates=["charttime"],
    )
    lab = lab.dropna(subset=["valuenum", "hadm_id"]).copy()
    lab["hadm_id"] = lab["hadm_id"].astype("int64")
    lab = lab[["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(cache, index=False)
    logger.info("Cached %s to %s (%d rows)", cache_name, cache, len(lab))
    return lab


def get_chartevents(cfg: Config) -> pd.DataFrame:
    """Return cached chartevents filtered to the pipeline item ids.

    chartevents is streamed once (it is the largest table); the filtered result
    is cached as ``_cache_chartevents_mimic.parquet`` and reused by the outcome
    and exposure stages.
    """
    items = chartevents_itemids(cfg)
    tag = hashlib.md5(",".join(str(i) for i in items).encode()).hexdigest()[:8]
    cache = cfg.output_dir / f"_cache_chartevents_mimic_{tag}.parquet"
    if cache.exists():
        logger.info("Loading cached chartevents from %s", cache)
        return pd.read_parquet(cache)

    chart = io.read_filtered_by_itemid(
        cfg.mimic_icu(),
        "chartevents",
        itemids=items,
        usecols=_CHART_COLS,
        parse_dates=["charttime"],
    )
    chart = chart.dropna(subset=["hadm_id"]).copy()
    chart["hadm_id"] = chart["hadm_id"].astype("int64")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    chart.to_parquet(cache, index=False)
    logger.info("Cached chartevents to %s (%d rows)", cache, len(chart))
    return chart


def get_inhospital_creatinine(cfg: Config) -> pd.DataFrame:
    """Return cleaned in-hospital creatinine (subject_id, hadm_id, charttime, valuenum).

    The result is cached as ``_cache_creatinine_mimic.parquet`` in the output
    directory after the first extraction.
    """
    cache = cfg.output_dir / "_cache_creatinine_mimic.parquet"
    if cache.exists():
        logger.info("Loading cached creatinine from %s", cache)
        return pd.read_parquet(cache)

    creat = io.read_filtered_by_itemid(
        cfg.mimic_hosp(),
        "labevents",
        itemids=cfg["mimic_itemids"]["creatinine_lab"],
        usecols=_CREAT_COLS,
        parse_dates=["charttime"],
    )
    creat = clean_creatinine(creat)
    creat = creat[creat["hadm_id"].notna()].copy()
    creat["hadm_id"] = creat["hadm_id"].astype("int64")
    creat = creat[["subject_id", "hadm_id", "charttime", "valuenum"]]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    creat.to_parquet(cache, index=False)
    logger.info("Cached creatinine to %s (%d rows)", cache, len(creat))
    return creat
