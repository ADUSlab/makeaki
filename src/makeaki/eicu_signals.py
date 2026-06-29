"""Cached eICU lab and creatinine extraction with synthetic timestamps."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .kdigo import clean_creatinine

logger = logging.getLogger(__name__)


def _ref(cfg: Config) -> pd.Timestamp:
    return pd.Timestamp(cfg["eicu"]["reference_date"])


def get_eicu_lab(cfg: Config, labnames: list[str], cache_name: str) -> pd.DataFrame:
    """Return cached eICU lab values (hadm_id, charttime, valuenum) for labnames.

    charttime is the synthetic timestamp reference_date + labresultoffset minutes.
    """
    cache = cfg.output_dir / f"_cache_eicu_{cache_name}.parquet"
    if cache.exists():
        logger.info("Loading cached eICU %s from %s", cache_name, cache)
        return pd.read_parquet(cache)

    lab = io.read_table(
        cfg.eicu_dir(),
        "lab",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
    )
    wanted = [n.lower() for n in labnames]
    lab = lab[lab["labname"].astype(str).str.lower().isin(wanted)].copy()
    lab = lab.rename(columns={"patientunitstayid": "hadm_id", "labresult": "valuenum"})
    lab["valuenum"] = pd.to_numeric(lab["valuenum"], errors="coerce")
    lab["charttime"] = _ref(cfg) + pd.to_timedelta(
        lab["labresultoffset"].astype(float), unit="m"
    )
    lab = lab.dropna(subset=["valuenum"])[["hadm_id", "charttime", "valuenum"]]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(cache, index=False)
    logger.info("Cached eICU %s to %s (%d rows)", cache_name, cache, len(lab))
    return lab


def get_eicu_creatinine(cfg: Config) -> pd.DataFrame:
    """Cleaned eICU creatinine (hadm_id, charttime, valuenum)."""
    creat = get_eicu_lab(cfg, cfg["eicu"]["labnames"]["creatinine"], "creatinine")
    return clean_creatinine(creat)


def get_eicu_vitals(cfg: Config) -> pd.DataFrame:
    """Return cached MAP and SpO2 with synthetic timestamps.

    MAP combines invasive (vitalPeriodic.systemicmean) and non-invasive
    (vitalAperiodic.noninvasivemean) values; SpO2 comes from vitalPeriodic.sao2.
    Columns: hadm_id, charttime, map, sao2. vitalPeriodic is streamed in chunks.
    """
    cache = cfg.output_dir / "_cache_eicu_vitals.parquet"
    if cache.exists():
        logger.info("Loading cached eICU vitals from %s", cache)
        return pd.read_parquet(cache)

    ref = _ref(cfg)
    parts: list[pd.DataFrame] = []

    vp_path = io.resolve_table(cfg.eicu_dir(), "vitalPeriodic")
    logger.info("Streaming %s for systemicmean and sao2", vp_path)
    reader = pd.read_csv(
        vp_path,
        usecols=["patientunitstayid", "observationoffset", "systemicmean", "sao2"],
        chunksize=2_000_000,
        low_memory=False,
    )
    for chunk in reader:
        keep = chunk[chunk["systemicmean"].notna() | chunk["sao2"].notna()]
        if not keep.empty:
            parts.append(keep)
    vp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["patientunitstayid", "observationoffset", "systemicmean", "sao2"]
    )
    vp = vp.rename(
        columns={"patientunitstayid": "hadm_id", "systemicmean": "map", "sao2": "sao2"}
    )
    vp["charttime"] = ref + pd.to_timedelta(vp["observationoffset"].astype(float), unit="m")
    vp = vp[["hadm_id", "charttime", "map", "sao2"]]

    # Non-invasive MAP as additional MAP observations.
    try:
        va = io.read_table(
            cfg.eicu_dir(),
            "vitalAperiodic",
            usecols=["patientunitstayid", "observationoffset", "noninvasivemean"],
        )
        va = va[va["noninvasivemean"].notna()].rename(
            columns={"patientunitstayid": "hadm_id", "noninvasivemean": "map"}
        )
        va["charttime"] = ref + pd.to_timedelta(va["observationoffset"].astype(float), unit="m")
        va["sao2"] = np.nan
        vp = pd.concat([vp, va[["hadm_id", "charttime", "map", "sao2"]]], ignore_index=True)
    except (FileNotFoundError, ValueError):
        pass

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    vp.to_parquet(cache, index=False)
    logger.info("Cached eICU vitals to %s (%d rows)", cache, len(vp))
    return vp
