"""Longitudinal panel construction for eICU-CRD (same wide format as MIMIC).

Produces the identical column layout used by the LTMLE stage so that the same R
script runs unchanged. eICU-specific sources:
- MAP, SpO2: vitalPeriodic + vitalAperiodic (cached vitals).
- creatinine, lactate, platelets, bilirubin, PaO2: lab.
- urine, fluid balance: intakeOutput.
- vasopressors: infusionDrug; ventilation: treatment.
- nephrotoxins: medication.
- GCS and FiO2: apacheApsVar (one value per stay, held constant across windows,
  because eICU does not chart these longitudinally in the downloaded tables).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .eicu_signals import get_eicu_creatinine, get_eicu_lab, get_eicu_vitals
from .panel_mimic import _assign_window, _bins_labels, _locf_at_window_ends, _pivot

logger = logging.getLogger(__name__)

_REF = "reference_date"
_WEIGHT_BOUNDS = (20.0, 400.0)
_MAP_BOUNDS = (10.0, 250.0)


def _ref(cfg: Config) -> pd.Timestamp:
    return pd.Timestamp(cfg["eicu"][_REF])


def _offset_time(cfg: Config, minutes: pd.Series) -> pd.Series:
    return _ref(cfg) + pd.to_timedelta(minutes.astype(float), unit="m")


def _overlap_windows(intervals: pd.DataFrame, t0: pd.DataFrame, cfg: Config,
                     labels: list[int], name: str) -> pd.DataFrame:
    """Per-window 0/1 indicator for intervals (start, stop) overlapping each window.

    Vectorized across all intervals: one boolean pass per window (5 passes total)
    rather than a per-row loop.
    """
    iv = intervals.merge(t0, on="hadm_id", how="inner")
    frames: list[pd.DataFrame] = []
    for k, (lo, hi) in zip(labels, cfg["windows_hours"]):
        ws = iv["aki_t0"] + pd.Timedelta(hours=lo)
        we = iv["aki_t0"] + pd.Timedelta(hours=hi)
        mask = (iv["start"] < we) & (iv["stop"] > ws)
        hit = iv.loc[mask, "hadm_id"].unique()
        if len(hit):
            frames.append(pd.DataFrame({"hadm_id": hit, "win": k}))
    if not frames:
        return pd.DataFrame(columns=[f"{name}_{k}" for k in labels])
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return _pivot(df.assign(v=1).set_index(["hadm_id", "win"])["v"], name, labels).fillna(0).astype(int)


def build_panel_eicu(cohort: pd.DataFrame, outcomes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    edges, labels, _ = _bins_labels(cfg)
    his = {k: float(hi) for k, (lo, hi) in zip(labels, cfg["windows_hours"])}
    hadms = set(cohort["hadm_id"])
    t0 = cohort[["hadm_id", "aki_t0"]].copy()

    # Body weight from the patient table.
    patient = io.read_table(cfg.eicu_dir(), "patient",
                            usecols=["patientunitstayid", "admissionweight"])
    patient = patient.rename(columns={"patientunitstayid": "hadm_id", "admissionweight": "weight_kg"})
    lo, hi = _WEIGHT_BOUNDS
    patient = patient[(patient["weight_kg"] >= lo) & (patient["weight_kg"] <= hi)]
    weight = patient.set_index("hadm_id")["weight_kg"]

    # Vitals: MAP (within-window min -> a_map; LOCF -> map_last) and SpO2 (LOCF).
    vit = get_eicu_vitals(cfg)
    vit = vit[vit["hadm_id"].isin(hadms)]
    mp = vit[vit["map"].notna()][["hadm_id", "charttime", "map"]].rename(columns={"map": "valuenum"})
    mp = mp[(mp["valuenum"] >= _MAP_BOUNDS[0]) & (mp["valuenum"] <= _MAP_BOUNDS[1])]
    mpw = _assign_window(mp, "charttime", t0, edges, labels)
    map_min = _pivot(mpw.groupby(["hadm_id", "win"])["valuenum"].min(), "map_min", labels)
    map_last = _locf_at_window_ends(mp, "charttime", t0, labels, his, "map_last")
    spo2 = vit[vit["sao2"].notna()][["hadm_id", "charttime", "sao2"]].rename(columns={"sao2": "valuenum"})
    spo2_last = _locf_at_window_ends(spo2, "charttime", t0, labels, his, "spo2_last")

    # Labs (LOCF).
    creat = get_eicu_creatinine(cfg)
    creat = creat[creat["hadm_id"].isin(hadms)][["hadm_id", "charttime", "valuenum"]]
    creat_last = _locf_at_window_ends(creat, "charttime", t0, labels, his, "creat_last")

    def _lab(labkey: str, name: str) -> pd.DataFrame:
        lab = get_eicu_lab(cfg, cfg["eicu"]["labnames"][labkey], labkey)
        lab = lab[lab["hadm_id"].isin(hadms)][["hadm_id", "charttime", "valuenum"]]
        return _locf_at_window_ends(lab, "charttime", t0, labels, his, name)

    lact_last = _lab("lactate", "lact_last")
    plt_last = _lab("platelet", "platelet_last")
    bili_last = _lab("bilirubin", "bilirubin_last")
    pao2_last = _lab("pao2", "pao2_last")

    # Intake/output: net fluid balance per window (a_fluid) and urine rate.
    iod = io.read_table(cfg.eicu_dir(), "intakeOutput",
                        usecols=["patientunitstayid", "intakeoutputoffset", "intaketotal", "outputtotal"])
    iod = iod[iod["patientunitstayid"].isin(hadms)].rename(columns={"patientunitstayid": "hadm_id"})
    iod["charttime"] = _offset_time(cfg, iod["intakeoutputoffset"])
    iod["net_ml"] = pd.to_numeric(iod["intaketotal"], errors="coerce") - pd.to_numeric(iod["outputtotal"], errors="coerce")
    iod["out_ml"] = pd.to_numeric(iod["outputtotal"], errors="coerce")
    iw = _assign_window(iod[["hadm_id", "charttime", "net_ml", "out_ml"]], "charttime", t0, edges, labels)
    net = iw.groupby(["hadm_id", "win"])["net_ml"].sum()
    a_fluid = _pivot((net > 0).astype(int), "a_fluid", labels)
    urine_sum = _pivot(iw.groupby(["hadm_id", "win"])["out_ml"].sum(), "urine_ml", labels)
    uo_rate = pd.DataFrame(index=urine_sum.index)
    w = weight.reindex(urine_sum.index)
    for k in labels:
        uo_rate[f"uo_rate_{k}"] = urine_sum[f"urine_ml_{k}"] / (w * his[k])

    # a_map within-window indicator.
    a_map = pd.DataFrame(index=map_min.index)
    thr = cfg["exposures"]["map_threshold_primary"]
    for k in labels:
        a_map[f"a_map_{k}"] = (map_min[f"map_min_{k}"] < thr).astype("Int64")

    # Vasopressors (infusionDrug) and nephrotoxins (medication) and ventilation (treatment).
    infu = io.read_table(cfg.eicu_dir(), "infusionDrug",
                         usecols=["patientunitstayid", "infusionoffset", "drugname"])
    infu = infu[infu["patientunitstayid"].isin(hadms)].rename(columns={"patientunitstayid": "hadm_id"})
    vp = "|".join(cfg["eicu"]["vasopressor_patterns"])
    vaso = infu[infu["drugname"].astype(str).str.lower().str.contains(vp, na=False, regex=True)].copy()
    vaso["charttime"] = _offset_time(cfg, vaso["infusionoffset"])
    vw = _assign_window(vaso[["hadm_id", "charttime"]], "charttime", t0, edges, labels)
    vaso_any = _pivot((vw.groupby(["hadm_id", "win"]).size() > 0).astype(int), "vaso", labels).fillna(0).astype(int)

    med = io.read_table(cfg.eicu_dir(), "medication",
                        usecols=["patientunitstayid", "drugstartoffset", "drugstopoffset", "drugname"])
    med = med[med["patientunitstayid"].isin(hadms)].rename(columns={"patientunitstayid": "hadm_id"})
    nx = "|".join(cfg["eicu"]["nephrotoxin_patterns"])
    ntx = med[med["drugname"].astype(str).str.lower().str.contains(nx, na=False, regex=True)].copy()
    ntx["start"] = _offset_time(cfg, ntx["drugstartoffset"])
    ntx["stop"] = _offset_time(cfg, ntx["drugstopoffset"])
    a_ntx = _overlap_windows(ntx[["hadm_id", "start", "stop"]], t0, cfg, labels, "a_ntx")

    treat = io.read_table(cfg.eicu_dir(), "treatment",
                          usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    treat = treat[treat["patientunitstayid"].isin(hadms)].rename(columns={"patientunitstayid": "hadm_id"})
    vk = "|".join(cfg["eicu"]["vent_keywords"])
    vent = treat[treat["treatmentstring"].astype(str).str.lower().str.contains(vk, na=False, regex=True)].copy()
    vent["charttime"] = _offset_time(cfg, vent["treatmentoffset"])
    vtw = _assign_window(vent[["hadm_id", "charttime"]], "charttime", t0, edges, labels)
    vent_any = _pivot((vtw.groupby(["hadm_id", "win"]).size() > 0).astype(int), "vent", labels).fillna(0).astype(int)

    # GCS total and FiO2 from apacheApsVar (one value per stay, held constant).
    apache = io.read_table(cfg.eicu_dir(), "apacheApsVar",
                           usecols=["patientunitstayid", "eyes", "motor", "verbal", "fio2"])
    apache = apache.rename(columns={"patientunitstayid": "hadm_id"})
    for c in ["eyes", "motor", "verbal", "fio2"]:
        apache[c] = pd.to_numeric(apache[c], errors="coerce")
        apache.loc[apache[c] < 0, c] = np.nan
    apache["gcs"] = apache["eyes"] + apache["motor"] + apache["verbal"]
    apache = apache.set_index("hadm_id")
    gcs_total = pd.DataFrame(index=apache.index)
    fio2_last = pd.DataFrame(index=apache.index)
    for k in labels:
        gcs_total[f"gcs_total_{k}"] = apache["gcs"]
        fio2_last[f"fio2_last_{k}"] = apache["fio2"]

    # Assemble.
    base_cols = [
        "hadm_id", "anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
        "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission",
    ]
    panel = cohort[base_cols].set_index("hadm_id")
    y = outcomes.set_index("hadm_id")[["make_h", "make_h_classifiable"]]
    panel = panel.join([map_last, creat_last, lact_last, uo_rate, vaso_any, vent_any,
                        plt_last, bili_last, pao2_last, fio2_last, spo2_last, gcs_total,
                        a_fluid, a_map, a_ntx, y])

    for col in ([f"vaso_{k}" for k in labels] + [f"vent_{k}" for k in labels]
                + [f"a_ntx_{k}" for k in labels]):
        panel[col] = panel[col].fillna(0).astype(int)
    return panel.reset_index()
