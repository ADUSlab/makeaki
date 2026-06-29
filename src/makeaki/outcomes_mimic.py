"""MAKE-H outcome labeling for the MIMIC-IV AKI cohort.

MAKE-H is the composite of in-hospital death, renal replacement therapy (RRT)
dependence, or persistent renal dysfunction (PRD), evaluated at hospital
discharge or the last available in-hospital observation after AKI onset.

Component definitions (prespecified):
- Death-H: in-hospital death during the index admission.
- RRT-H:   RRT delivered within the last ``rrt_recent_window_hours`` of the
           in-hospital observation (proxy for RRT dependence at discharge).
- PRD-H:   last post-t0 creatinine with ratio to baseline >= ``prd_scr_ratio``
           (2.0, the standard MAKE doubling) OR last eGFR <= ``prd_egfr_fraction``
           times baseline eGFR.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .egfr import ckd_epi_2021
from .signals import get_chartevents, get_inhospital_creatinine

logger = logging.getLogger(__name__)


def _rrt_timestamps(cfg: Config, hadms: set[int]) -> pd.DataFrame:
    """Return all RRT activity timestamps per admission.

    Combines procedureevents (start and end of each RRT session) with
    chartevents active-RRT markers (CRRT mode, dialysate rate, effluent
    pressure, ultrafiltrate output, hemodialysis output). Using both tables and
    both procedure endpoints avoids undercounting RRT near discharge, which a
    starttime-only or procedureevents-only definition produces.
    """
    proc = io.read_table(
        cfg.mimic_icu(),
        "procedureevents",
        usecols=["hadm_id", "starttime", "endtime", "itemid"],
        parse_dates=["starttime", "endtime"],
    )
    proc = proc[
        proc["itemid"].isin(cfg["mimic_itemids"]["rrt_procedure"])
        & proc["hadm_id"].isin(hadms)
    ]
    starts = proc[["hadm_id", "starttime"]].rename(columns={"starttime": "rrt_time"})
    ends = proc[["hadm_id", "endtime"]].rename(columns={"endtime": "rrt_time"})

    chart = get_chartevents(cfg)
    chart_rrt = chart[
        chart["itemid"].isin(cfg["mimic_itemids"]["rrt_chart"])
        & chart["hadm_id"].isin(hadms)
    ][["hadm_id", "charttime"]].rename(columns={"charttime": "rrt_time"})

    ts = pd.concat([starts, ends, chart_rrt], ignore_index=True)
    return ts.dropna(subset=["rrt_time"])


def build_make_h(cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Label MAKE-H and its components for each admission in the cohort."""
    out = cohort.copy()
    end_time = out["dischtime"].fillna(out["deathtime"])
    out["obs_end"] = end_time

    # Death-H
    out["death_h"] = out["hospital_expire_flag"].astype("Int64").fillna(0).astype(bool)
    out.loc[out["deathtime"].notna(), "death_h"] = True

    # RRT-H: RRT active within the recent window before observation end.
    rrt = _rrt_timestamps(cfg, set(out["hadm_id"]))
    window = pd.Timedelta(hours=cfg["outcome"]["rrt_recent_window_hours"])
    last_rrt = (
        rrt.groupby("hadm_id")["rrt_time"].max().rename("last_rrt").reset_index()
    )
    ever_rrt = (
        rrt.groupby("hadm_id").size().rename("rrt_events").reset_index()
    )
    out = out.merge(last_rrt, on="hadm_id", how="left").merge(
        ever_rrt, on="hadm_id", how="left"
    )
    out["ever_rrt"] = out["rrt_events"].fillna(0) > 0
    out["rrt_h"] = out["last_rrt"].notna() & (
        out["last_rrt"] >= (out["obs_end"] - window)
    )

    # PRD-H: last post-t0 creatinine.
    creat = get_inhospital_creatinine(cfg)
    creat = creat[creat["hadm_id"].isin(set(out["hadm_id"]))]
    creat = creat.merge(out[["hadm_id", "aki_t0"]], on="hadm_id", how="left")
    post = creat[creat["charttime"] > creat["aki_t0"]]
    last_post = (
        post.sort_values("charttime")
        .groupby("hadm_id")["valuenum"]
        .last()
        .rename("scr_last")
        .reset_index()
    )
    out = out.merge(last_post, on="hadm_id", how="left")

    out["egfr_last"] = ckd_epi_2021(
        out["scr_last"], out["anchor_age"], out["female"]
    )
    ratio = out["scr_last"] / out["baseline_creatinine"]
    egfr_drop = out["egfr_last"] <= cfg["outcome"]["prd_egfr_fraction"] * out["baseline_egfr"]
    prd = (ratio >= cfg["outcome"]["prd_scr_ratio"]) | egfr_drop
    out["prd_h"] = np.where(out["scr_last"].isna(), pd.NA, prd)
    out["prd_h"] = out["prd_h"].astype("boolean")

    # Composite and classifiability.
    out["make_h"] = (
        out["death_h"].astype(bool)
        | out["rrt_h"].astype(bool)
        | out["prd_h"].fillna(False).astype(bool)
    )
    out["make_h_classifiable"] = (
        out["death_h"].astype(bool)
        | out["rrt_h"].astype(bool)
        | out["prd_h"].notna()
    )
    return out
