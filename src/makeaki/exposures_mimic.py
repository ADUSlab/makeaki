"""Modifiable exposure construction for the MIMIC-IV AKI cohort (first 72 hours).

Implements the three main exposures defined in the analysis plan, evaluated over
the first ``exposure_window_hours`` after AKI onset (t0):

- Fluid accumulation: cumulative net fluid balance (inputs minus outputs) as a
  percentage of body weight; the exposure A_F is fluid accumulation above the
  primary threshold.
- Hypotension burden: time-weighted mean arterial pressure deficit below the
  threshold (discrete estimator).
- Nephrotoxin burden: nephrotoxic medication-days and concurrent exposure to at
  least two nephrotoxins.

These are the policy-level exposures used by the confirmatory analysis. The
fully time-resolved per-window panel for longitudinal estimation is built in a
later stage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .signals import get_chartevents

logger = logging.getLogger(__name__)

_WEIGHT_BOUNDS = (20.0, 400.0)   # kg
_MAP_BOUNDS = (10.0, 250.0)      # mmHg
_MAX_GAP_HOURS = 4.0             # cap inter-measurement interval for HB integration


def _window_end(cohort: pd.DataFrame, hours: int) -> pd.Series:
    return cohort["aki_t0"] + pd.Timedelta(hours=hours)


def admission_weights(cfg: Config, hadms: set[int]) -> pd.Series:
    """Body weight (kg) per admission: admission weight if present else daily median."""
    chart = get_chartevents(cfg)
    w = chart[chart["itemid"].isin(cfg["mimic_itemids"]["weight_chart"])].copy()
    w = w[w["hadm_id"].isin(hadms)]
    lo, hi = _WEIGHT_BOUNDS
    w = w[(w["valuenum"] >= lo) & (w["valuenum"] <= hi)]

    admit_item = cfg["mimic_itemids"]["weight_chart"][0]
    admit_w = (
        w[w["itemid"] == admit_item]
        .sort_values("charttime")
        .groupby("hadm_id")["valuenum"]
        .first()
    )
    daily_w = w.groupby("hadm_id")["valuenum"].median()
    weight = admit_w.reindex(daily_w.index).fillna(daily_w)
    weight = weight.combine_first(admit_w)
    return weight.rename("weight_kg")


def fluid_accumulation(cfg: Config, cohort: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Cumulative net fluid balance and accumulation percentage over the window."""
    hours = cfg["exposures"]["exposure_window_hours"]
    hadms = set(cohort["hadm_id"])
    t0 = cohort.set_index("hadm_id")["aki_t0"]
    t_end = _window_end(cohort, hours).rename("t_end")
    win = cohort[["hadm_id"]].assign(t0=t0.values, t_end=t_end.values)

    inputs = io.read_table(
        cfg.mimic_icu(),
        "inputevents",
        usecols=["hadm_id", "starttime", "amount", "amountuom"],
        parse_dates=["starttime"],
    )
    inputs = inputs[inputs["hadm_id"].isin(hadms)]
    uom = inputs["amountuom"].str.lower()
    factor = np.where(uom == "l", 1000.0, np.where(uom == "ml", 1.0, np.nan))
    inputs = inputs.assign(ml=inputs["amount"] * factor).dropna(subset=["ml"])
    inputs = inputs.merge(win, on="hadm_id", how="inner")
    in_win = (inputs["starttime"] >= inputs["t0"]) & (inputs["starttime"] <= inputs["t_end"])
    fluid_in = inputs.loc[in_win].groupby("hadm_id")["ml"].sum().rename("fluid_in_ml")

    outputs = io.read_table(
        cfg.mimic_icu(),
        "outputevents",
        usecols=["hadm_id", "charttime", "value"],
        parse_dates=["charttime"],
    )
    outputs = outputs[outputs["hadm_id"].isin(hadms)].merge(win, on="hadm_id", how="inner")
    out_win = (outputs["charttime"] >= outputs["t0"]) & (outputs["charttime"] <= outputs["t_end"])
    fluid_out = outputs.loc[out_win].groupby("hadm_id")["value"].sum().rename("fluid_out_ml")

    res = pd.DataFrame(index=sorted(hadms))
    res.index.name = "hadm_id"
    res["fluid_in_ml"] = fluid_in
    res["fluid_out_ml"] = fluid_out
    res["cfb_ml"] = res["fluid_in_ml"].fillna(0) - res["fluid_out_ml"].fillna(0)
    res["weight_kg"] = weights.reindex(res.index)
    res["fa_pct"] = res["cfb_ml"] / (res["weight_kg"] * 1000.0) * 100.0
    threshold = cfg["exposures"]["fluid_overload_pct_primary"]
    res["a_fluid"] = (res["fa_pct"] > threshold).astype("boolean")
    has_fluid_data = res["fluid_in_ml"].notna() | res["fluid_out_ml"].notna()
    res.loc[~has_fluid_data, "a_fluid"] = pd.NA
    return res.reset_index()


def hypotension_burden(cfg: Config, cohort: pd.DataFrame) -> pd.DataFrame:
    """Time-weighted MAP deficit below the threshold over the window (mmHg-hours)."""
    hours = cfg["exposures"]["exposure_window_hours"]
    threshold = cfg["exposures"]["map_threshold_primary"]
    map_items = cfg["mimic_itemids"]["map_chart"]
    hadms = set(cohort["hadm_id"])

    chart = get_chartevents(cfg)
    m = chart[chart["itemid"].isin(map_items) & chart["hadm_id"].isin(hadms)].copy()
    lo, hi = _MAP_BOUNDS
    m = m[(m["valuenum"] >= lo) & (m["valuenum"] <= hi)]

    win = cohort[["hadm_id", "aki_t0"]].copy()
    win["t_end"] = win["aki_t0"] + pd.Timedelta(hours=hours)
    m = m.merge(win, on="hadm_id", how="inner")
    m = m[(m["charttime"] >= m["aki_t0"]) & (m["charttime"] <= m["t_end"])]

    # Prefer a single MAP source per admission to avoid double counting.
    priority = {itemid: rank for rank, itemid in enumerate(map_items)}
    m["src_rank"] = m["itemid"].map(priority)
    chosen = m.sort_values(["hadm_id", "src_rank"]).groupby("hadm_id")["itemid"].first()
    m = m.merge(chosen.rename("chosen_item"), on="hadm_id", how="left")
    m = m[m["itemid"] == m["chosen_item"]]

    records: list[dict[str, object]] = []
    cap = pd.Timedelta(hours=_MAX_GAP_HOURS)
    for hadm_id, grp in m.sort_values("charttime").groupby("hadm_id"):
        times = grp["charttime"].to_numpy()
        vals = grp["valuenum"].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        dt = np.diff(times).astype("timedelta64[s]").astype(float) / 3600.0
        dt = np.minimum(dt, _MAX_GAP_HOURS)
        deficit = np.maximum(0.0, threshold - vals[:-1])
        hb = float(np.sum(deficit * dt)) if len(dt) else 0.0
        records.append({"hadm_id": hadm_id, "hb_mmhg_h": hb, "n_map": len(vals)})

    return pd.DataFrame.from_records(records)


def matched_nephrotoxin_prescriptions(cfg: Config, hadms: set[int]) -> pd.DataFrame:
    """Prescriptions matched to the prespecified nephrotoxin list.

    Returns columns hadm_id, starttime, stoptime, nephrotoxin (generic name).
    """
    drugs = [d.lower() for d in cfg["nephrotoxins"]]
    pres = io.read_table(
        cfg.mimic_hosp(),
        "prescriptions",
        usecols=["hadm_id", "starttime", "stoptime", "drug"],
        parse_dates=["starttime", "stoptime"],
    )
    pres = pres[pres["hadm_id"].isin(hadms)].copy()
    name = pres["drug"].astype(str).str.lower()
    pattern = "|".join(d.replace("-", ".?") for d in drugs)
    pres = pres[name.str.contains(pattern, regex=True, na=False)]

    def _match(drug: str) -> str:
        d = drug.lower()
        for nx in drugs:
            if all(part in d for part in nx.split("-")):
                return nx
        return "other"

    pres["nephrotoxin"] = pres["drug"].map(_match)
    return pres[["hadm_id", "starttime", "stoptime", "nephrotoxin"]]


def nephrotoxin_burden(cfg: Config, cohort: pd.DataFrame) -> pd.DataFrame:
    """Nephrotoxic medication-days and concurrent (>=2) exposure over the window."""
    hours = cfg["exposures"]["exposure_window_hours"]
    concurrent_min = cfg["exposures"]["nephrotoxin_concurrent_min"]
    hadms = set(cohort["hadm_id"])

    pres = matched_nephrotoxin_prescriptions(cfg, hadms)

    win = cohort[["hadm_id", "aki_t0"]].copy()
    win["t_end"] = win["aki_t0"] + pd.Timedelta(hours=hours)
    pres = pres.merge(win, on="hadm_id", how="inner")

    # Overlap of each prescription with the exposure window.
    start = pres[["starttime", "aki_t0"]].max(axis=1)
    stop = pres[["stoptime", "t_end"]].min(axis=1)
    pres["overlap_days"] = (
        (stop - start).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    pres = pres[pres["overlap_days"] > 0]

    drug_days = pres.groupby("hadm_id")["overlap_days"].sum().rename("nephrotoxin_drug_days")
    distinct = (
        pres.groupby("hadm_id")["nephrotoxin"].nunique().rename("n_distinct_nephrotoxins")
    )
    res = pd.DataFrame(index=sorted(hadms))
    res.index.name = "hadm_id"
    res["nephrotoxin_drug_days"] = drug_days.reindex(res.index).fillna(0.0)
    res["n_distinct_nephrotoxins"] = distinct.reindex(res.index).fillna(0).astype(int)
    res["a_nephrotoxin"] = res["n_distinct_nephrotoxins"] >= concurrent_min
    return res.reset_index()


def build_exposures(cohort: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assemble the three 72-hour exposures for the cohort."""
    hadms = set(cohort["hadm_id"])
    weights = admission_weights(cfg, hadms)
    fluid = fluid_accumulation(cfg, cohort, weights)
    hb = hypotension_burden(cfg, cohort)
    ntx = nephrotoxin_burden(cfg, cohort)

    out = cohort[["subject_id", "hadm_id", "aki_t0", "aki_max_stage"]].copy()
    out = out.merge(fluid, on="hadm_id", how="left")
    out = out.merge(hb, on="hadm_id", how="left")
    out = out.merge(ntx, on="hadm_id", how="left")
    return out
