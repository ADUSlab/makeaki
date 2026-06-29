"""MAKE-H outcome labeling for the eICU-CRD AKI cohort.

Mirrors the MIMIC definitions: Death-H from in-hospital death, RRT-H from renal
replacement therapy active within the recent window before discharge (eICU
treatment table), and PRD-H from the last post-t0 creatinine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .egfr import ckd_epi_2021
from .eicu_signals import get_eicu_creatinine

logger = logging.getLogger(__name__)


def _rrt_timestamps(cfg: Config, hadms: set[int]) -> pd.DataFrame:
    """RRT activity timestamps from the eICU treatment table (synthetic time)."""
    ref = pd.Timestamp(cfg["eicu"]["reference_date"])
    treat = io.read_table(
        cfg.eicu_dir(), "treatment", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"]
    )
    pattern = "|".join(cfg["eicu"]["rrt_treatment_patterns"])
    treat = treat[
        treat["treatmentstring"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    ]
    treat = treat[treat["patientunitstayid"].isin(hadms)]
    out = treat.rename(columns={"patientunitstayid": "hadm_id"})[["hadm_id", "treatmentoffset"]].copy()
    out["rrt_time"] = ref + pd.to_timedelta(out["treatmentoffset"].astype(float), unit="m")
    return out[["hadm_id", "rrt_time"]]


def build_make_h_eicu(cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Label MAKE-H and its components for the eICU AKI cohort."""
    out = cohort.copy()
    out["obs_end"] = out["dischtime"].fillna(out["deathtime"])
    out["death_h"] = out["hospital_expire_flag"].astype("Int64").fillna(0).astype(bool)

    # RRT-H.
    rrt = _rrt_timestamps(cfg, set(out["hadm_id"]))
    window = pd.Timedelta(hours=cfg["outcome"]["rrt_recent_window_hours"])
    last_rrt = rrt.groupby("hadm_id")["rrt_time"].max().rename("last_rrt").reset_index()
    n_rrt = rrt.groupby("hadm_id").size().rename("rrt_events").reset_index()
    out = out.merge(last_rrt, on="hadm_id", how="left").merge(n_rrt, on="hadm_id", how="left")
    out["ever_rrt"] = out["rrt_events"].fillna(0) > 0
    out["rrt_h"] = out["last_rrt"].notna() & (out["last_rrt"] >= (out["obs_end"] - window))

    # PRD-H from the last post-t0 creatinine.
    creat = get_eicu_creatinine(cfg)
    creat = creat[creat["hadm_id"].isin(set(out["hadm_id"]))]
    creat = creat.merge(out[["hadm_id", "aki_t0"]], on="hadm_id", how="left")
    post = creat[creat["charttime"] > creat["aki_t0"]]
    last_post = (
        post.sort_values("charttime").groupby("hadm_id")["valuenum"].last()
        .rename("scr_last").reset_index()
    )
    out = out.merge(last_post, on="hadm_id", how="left")
    out["egfr_last"] = ckd_epi_2021(out["scr_last"], out["anchor_age"], out["female"])
    ratio = out["scr_last"] / out["baseline_creatinine"]
    egfr_drop = out["egfr_last"] <= cfg["outcome"]["prd_egfr_fraction"] * out["baseline_egfr"]
    prd = (ratio >= cfg["outcome"]["prd_scr_ratio"]) | egfr_drop
    out["prd_h"] = np.where(out["scr_last"].isna(), pd.NA, prd)
    out["prd_h"] = out["prd_h"].astype("boolean")

    out["make_h"] = (
        out["death_h"].astype(bool)
        | out["rrt_h"].astype(bool)
        | out["prd_h"].fillna(False).astype(bool)
    )
    out["make_h_classifiable"] = (
        out["death_h"].astype(bool) | out["rrt_h"].astype(bool) | out["prd_h"].notna()
    )
    return out
