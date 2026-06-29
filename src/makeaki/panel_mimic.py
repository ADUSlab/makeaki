"""Longitudinal panel construction for LTMLE (lmtp-ready wide format).

For each admission we build one row with baseline covariates, and for each
post-t0 window k (default 0-6, 6-12, 12-24, 24-48, 48-72 hours) a set of
time-varying confounders L_k and observed treatment indicators A_k.

Per-window observed treatment encodings (matching the analysis plan; the
counterfactual policies are applied as shift functions in the R/lmtp stage):

- a_fluid_k = 1 if net fluid balance within window k is positive (input minus
  output > 0). Policy g_F sets a_fluid_k = 0 when fluid accumulation is already
  high and refractory shock is absent.
- a_map_k = 1 if the minimum MAP within window k is below the threshold
  (observed hypotension exposure). Policy g_MAP sets a_map_k = 0.
- a_ntx_k = 1 if any prespecified nephrotoxin is active within window k. Policy
  g_NTX sets a_ntx_k = 0 for substitutable nephrotoxins.

Time-varying confounders L_k: minimum MAP, last creatinine, urine-output rate,
last lactate, vasopressor use, and mechanical ventilation within the window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .exposures_mimic import matched_nephrotoxin_prescriptions
from .signals import get_chartevents, get_inhospital_creatinine, get_lab

logger = logging.getLogger(__name__)


def _bins_labels(cfg: Config) -> tuple[list[int], list[int], dict[int, float]]:
    windows = cfg["windows_hours"]
    edges = [w[0] for w in windows] + [windows[-1][1]]
    labels = list(range(1, len(windows) + 1))
    hours = {k: float(hi - lo) for k, (lo, hi) in zip(labels, windows)}
    return edges, labels, hours


def _assign_window(
    events: pd.DataFrame, time_col: str, t0: pd.DataFrame, edges: list[int], labels: list[int]
) -> pd.DataFrame:
    df = events.merge(t0, on="hadm_id", how="inner")
    delta = (df[time_col] - df["aki_t0"]).dt.total_seconds() / 3600.0
    df["win"] = pd.cut(delta, bins=edges, labels=labels, right=False, include_lowest=True)
    df = df[df["win"].notna()].copy()
    df["win"] = df["win"].astype(int)
    return df


def _pivot(grouped: pd.Series, name: str, labels: list[int]) -> pd.DataFrame:
    wide = grouped.unstack("win")
    wide.columns = [f"{name}_{int(k)}" for k in wide.columns]
    for k in labels:
        col = f"{name}_{k}"
        if col not in wide.columns:
            wide[col] = np.nan
    return wide[[f"{name}_{k}" for k in labels]]


def _locf_at_window_ends(
    events: pd.DataFrame,
    time_col: str,
    t0: pd.DataFrame,
    labels: list[int],
    his: dict[int, float],
    name: str,
) -> pd.DataFrame:
    """Last-observation-carried-forward value at each window boundary.

    For each admission and window k, returns the most recent value observed at or
    before the end of window k. This represents the patient state used as a
    time-varying confounder, rather than only values re-measured within the
    window.
    """
    ev = (
        events.rename(columns={time_col: "t"})[["hadm_id", "t", "valuenum"]]
        .dropna(subset=["t", "valuenum"])
        .sort_values("t")
    )
    end_frames = []
    for k in labels:
        e = t0[["hadm_id", "aki_t0"]].copy()
        e["k"] = k
        e["end"] = e["aki_t0"] + pd.to_timedelta(his[k], unit="h")
        end_frames.append(e)
    ends = pd.concat(end_frames, ignore_index=True).sort_values("end")
    merged = pd.merge_asof(
        ends, ev, by="hadm_id", left_on="end", right_on="t", direction="backward"
    )
    wide = merged.pivot(index="hadm_id", columns="k", values="valuenum")
    wide.columns = [f"{name}_{int(k)}" for k in wide.columns]
    for k in labels:
        col = f"{name}_{k}"
        if col not in wide.columns:
            wide[col] = np.nan
    return wide[[f"{name}_{k}" for k in labels]]


def build_panel(
    cohort: pd.DataFrame, exposures: pd.DataFrame, outcomes: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Build the lmtp-ready wide longitudinal panel."""
    edges, labels, win_hours = _bins_labels(cfg)
    his = {k: float(hi) for k, (lo, hi) in zip(labels, cfg["windows_hours"])}
    hadms = set(cohort["hadm_id"])
    t0 = cohort[["hadm_id", "aki_t0"]].copy()
    weight = exposures.set_index("hadm_id")["weight_kg"]

    # MAP: within-window minimum drives the observed hypotension treatment a_map;
    # the carried-forward last MAP is the time-varying confounder map_last.
    chart = get_chartevents(cfg)
    mp_all = chart[
        chart["itemid"].isin(cfg["mimic_itemids"]["map_chart"]) & chart["hadm_id"].isin(hadms)
    ][["hadm_id", "charttime", "valuenum"]]
    mpw = _assign_window(mp_all, "charttime", t0, edges, labels)
    map_min = _pivot(mpw.groupby(["hadm_id", "win"])["valuenum"].min(), "map_min", labels)
    map_last = _locf_at_window_ends(mp_all, "charttime", t0, labels, his, "map_last")

    # Creatinine and lactate confounders: last value carried forward to window end.
    creat = get_inhospital_creatinine(cfg)
    creat = creat[creat["hadm_id"].isin(hadms)][["hadm_id", "charttime", "valuenum"]]
    creat_last = _locf_at_window_ends(creat, "charttime", t0, labels, his, "creat_last")

    lact = get_lab(cfg, cfg["mimic_itemids"]["lactate_lab"], "lactate")
    lact = lact[lact["hadm_id"].isin(hadms)][["hadm_id", "charttime", "valuenum"]]
    lact_last = _locf_at_window_ends(lact, "charttime", t0, labels, his, "lact_last")

    # SOFA-component confounders (last value carried forward).
    def _lab_locf(item_key: str, name: str) -> pd.DataFrame:
        lab = get_lab(cfg, cfg["mimic_itemids"][item_key], name)
        lab = lab[lab["hadm_id"].isin(hadms)][["hadm_id", "charttime", "valuenum"]]
        return _locf_at_window_ends(lab, "charttime", t0, labels, his, f"{name}_last")

    plt_last = _lab_locf("platelet_lab", "platelet")
    bili_last = _lab_locf("bilirubin_lab", "bilirubin")
    pao2_last = _lab_locf("pao2_lab", "pao2")

    def _chart_locf(item_key: str, name: str) -> pd.DataFrame:
        ev = chart[
            chart["itemid"].isin(cfg["mimic_itemids"][item_key]) & chart["hadm_id"].isin(hadms)
        ][["hadm_id", "charttime", "valuenum"]]
        return _locf_at_window_ends(ev, "charttime", t0, labels, his, name)

    fio2_last = _chart_locf("fio2_chart", "fio2_last")
    spo2_last = _chart_locf("spo2_chart", "spo2_last")

    # GCS total = sum of carried-forward eye, verbal, and motor components.
    gcs_parts = []
    for itemid, nm in zip(cfg["mimic_itemids"]["gcs_chart"], ["gcse", "gcsv", "gcsm"]):
        ev = chart[
            (chart["itemid"] == itemid) & chart["hadm_id"].isin(hadms)
        ][["hadm_id", "charttime", "valuenum"]]
        gcs_parts.append(_locf_at_window_ends(ev, "charttime", t0, labels, his, nm))
    gcs = pd.concat(gcs_parts, axis=1)
    gcs_total = pd.DataFrame(index=gcs.index)
    for k in labels:
        gcs_total[f"gcs_total_{k}"] = (
            gcs[f"gcse_{k}"] + gcs[f"gcsv_{k}"] + gcs[f"gcsm_{k}"]
        )

    # Inputevents: fluids (volume) and vasopressors.
    inputs = io.read_table(
        cfg.mimic_icu(),
        "inputevents",
        usecols=["hadm_id", "starttime", "amount", "amountuom", "itemid"],
        parse_dates=["starttime"],
    )
    inputs = inputs[inputs["hadm_id"].isin(hadms)]
    uom = inputs["amountuom"].str.lower()
    inputs["ml"] = inputs["amount"] * np.where(uom == "l", 1000.0, np.where(uom == "ml", 1.0, np.nan))
    fin = _assign_window(inputs.dropna(subset=["ml"]), "starttime", t0, edges, labels)
    fluid_in = fin.groupby(["hadm_id", "win"])["ml"].sum()

    vaso = inputs[inputs["itemid"].isin(cfg["mimic_itemids"]["vasopressor_input"])]
    vw = _assign_window(vaso[["hadm_id", "starttime"]], "starttime", t0, edges, labels)
    vaso_any = _pivot(
        (vw.groupby(["hadm_id", "win"]).size() > 0).astype(int), "vaso", labels
    ).fillna(0).astype(int)

    # Outputevents: urine output.
    outputs = io.read_table(
        cfg.mimic_icu(),
        "outputevents",
        usecols=["hadm_id", "charttime", "value", "itemid"],
        parse_dates=["charttime"],
    )
    outputs = outputs[
        outputs["hadm_id"].isin(hadms)
        & outputs["itemid"].isin(cfg["mimic_itemids"]["urine_output"])
    ]
    ow = _assign_window(outputs[["hadm_id", "charttime", "value"]], "charttime", t0, edges, labels)
    urine_sum = ow.groupby(["hadm_id", "win"])["value"].sum()

    # Net fluid balance per window -> a_fluid.
    net = (fluid_in.subtract(urine_sum, fill_value=0.0)).rename("net_ml")
    a_fluid = _pivot((net > 0).astype(int), "a_fluid", labels)

    # Urine rate (mL/kg/h) per window.
    urine_wide = _pivot(urine_sum, "urine_ml", labels)
    uo_rate = pd.DataFrame(index=urine_wide.index)
    w = weight.reindex(urine_wide.index)
    for k in labels:
        uo_rate[f"uo_rate_{k}"] = urine_wide[f"urine_ml_{k}"] / (w * win_hours[k])

    # a_map per window.
    a_map = pd.DataFrame(index=map_min.index)
    thr = cfg["exposures"]["map_threshold_primary"]
    for k in labels:
        a_map[f"a_map_{k}"] = (map_min[f"map_min_{k}"] < thr).astype("Int64")

    # Mechanical ventilation overlap per window.
    proc = io.read_table(
        cfg.mimic_icu(),
        "procedureevents",
        usecols=["hadm_id", "starttime", "endtime", "itemid"],
        parse_dates=["starttime", "endtime"],
    )
    vent = proc[
        proc["itemid"].isin(cfg["mimic_itemids"]["vent_procedure"]) & proc["hadm_id"].isin(hadms)
    ].merge(t0, on="hadm_id", how="inner")
    vent_rows: list[tuple[int, int]] = []
    for _, r in vent.iterrows():
        for k, (lo, hi) in zip(labels, cfg["windows_hours"]):
            ws = r["aki_t0"] + pd.Timedelta(hours=lo)
            we = r["aki_t0"] + pd.Timedelta(hours=hi)
            if (r["starttime"] < we) and (r["endtime"] > ws):
                vent_rows.append((r["hadm_id"], k))
    vent_df = pd.DataFrame(vent_rows, columns=["hadm_id", "win"]).drop_duplicates()
    vent_any = _pivot(
        vent_df.assign(v=1).set_index(["hadm_id", "win"])["v"], "vent", labels
    ).fillna(0).astype(int)

    # Nephrotoxin overlap per window -> a_ntx.
    ntx = matched_nephrotoxin_prescriptions(cfg, hadms).merge(t0, on="hadm_id", how="inner")
    ntx_rows: list[tuple[int, int]] = []
    for _, r in ntx.iterrows():
        for k, (lo, hi) in zip(labels, cfg["windows_hours"]):
            ws = r["aki_t0"] + pd.Timedelta(hours=lo)
            we = r["aki_t0"] + pd.Timedelta(hours=hi)
            if (r["starttime"] < we) and (r["stoptime"] > ws):
                ntx_rows.append((r["hadm_id"], k))
    ntx_df = pd.DataFrame(ntx_rows, columns=["hadm_id", "win"]).drop_duplicates()
    a_ntx = _pivot(
        ntx_df.assign(v=1).set_index(["hadm_id", "win"])["v"], "a_ntx", labels
    ).fillna(0).astype(int)

    # Baseline covariates and outcome.
    base_cols = [
        "hadm_id",
        "anchor_age",
        "female",
        "cm_ckd",
        "cm_diabetes",
        "cm_heart_failure",
        "baseline_creatinine",
        "baseline_egfr",
        "aki_max_stage",
        "aki_present_at_admission",
    ]
    panel = cohort[base_cols].set_index("hadm_id")
    y = outcomes.set_index("hadm_id")[["make_h", "make_h_classifiable"]]

    panel = panel.join([map_last, creat_last, lact_last, uo_rate, vaso_any,
                        vent_any, plt_last, bili_last, pao2_last, fio2_last,
                        spo2_last, gcs_total, a_fluid, a_map, a_ntx, y])

    # Structural zeros: absence of a vasopressor, ventilation, or nephrotoxin
    # event in a window means the patient was not exposed (0), not missing.
    structural_zero = (
        [f"vaso_{k}" for k in labels]
        + [f"vent_{k}" for k in labels]
        + [f"a_ntx_{k}" for k in labels]
    )
    for col in structural_zero:
        panel[col] = panel[col].fillna(0).astype(int)

    return panel.reset_index()
