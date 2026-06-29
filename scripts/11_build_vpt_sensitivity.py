#!/usr/bin/env python
"""Sensitivity inputs for the well-defined vancomycin + piperacillin-tazobactam
(VPT) target trial.

Two robustness analyses requested at review:

  1. Grace period. The main analysis defines exposure over the first 72 hours
     after AKI onset. Here we rebuild the same well-defined comparison under a
     shorter 48-hour grace period, to show the antibiotic contrast is not an
     artefact of the window length.

  2. Dose-response. Among VPT-exposed patients we compute the cumulative hours of
     piperacillin-tazobactam overlap within the window (converted to a 1/2/3-day
     dose tier), so the renal effect can be tested for a monotonic dose-response.

Outputs, per cohort:
    outputs/<cohort>_vpt_48h.parquet   same layout as <cohort>_vpt.parquet
    outputs/<cohort>_vpt_dose.parquet  VPT-exposed only, with pt_dose_days / tier

Usage:
    python scripts/11_build_vpt_sensitivity.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from makeaki import io  # noqa: E402
from makeaki.config import load_config  # noqa: E402

VANCO = ["vancomycin", "vanco"]
PIPTAZO = ["piperacillin", "zosyn", "tazobactam"]
COMPARATOR = ["cefepime", "maxipime", "meropenem", "merrem", "imipenem",
              "primaxin", "ertapenem", "doripenem"]

XBASE = ["anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
         "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission"]
XEARLY = ["map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
          "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
          "spo2_last_1", "gcs_total_1"]


def _flag(intervals: pd.DataFrame, t0: pd.DataFrame, patterns: list[str], hours: int) -> set[int]:
    pat = "|".join(patterns)
    iv = intervals[intervals["drug"].astype(str).str.lower().str.contains(pat, na=False, regex=True)]
    iv = iv.merge(t0, on="hadm_id", how="inner")
    we = iv["aki_t0"] + pd.Timedelta(hours=hours)
    overlap = (iv["start"] < we) & (iv["stop"] > iv["aki_t0"])
    return set(iv.loc[overlap, "hadm_id"].unique())


def _overlap_hours(intervals: pd.DataFrame, t0: pd.DataFrame, patterns: list[str], hours: int) -> pd.Series:
    """Cumulative hours of drug overlap within [t0, t0+hours], per hadm_id."""
    pat = "|".join(patterns)
    iv = intervals[intervals["drug"].astype(str).str.lower().str.contains(pat, na=False, regex=True)].copy()
    iv = iv.merge(t0, on="hadm_id", how="inner")
    we = iv["aki_t0"] + pd.Timedelta(hours=hours)
    lo = iv[["start", "aki_t0"]].max(axis=1)
    hi = iv[["stop"]].join(we.rename("we")).min(axis=1)
    dur = (hi - lo).dt.total_seconds() / 3600.0
    iv["ov"] = dur.clip(lower=0)
    return iv.groupby("hadm_id")["ov"].sum()


def _mimic_drugs(cfg) -> pd.DataFrame:
    pres = io.read_table(cfg.mimic_hosp(), "prescriptions",
                         usecols=["hadm_id", "starttime", "stoptime", "drug"],
                         parse_dates=["starttime", "stoptime"])
    return pres.rename(columns={"starttime": "start", "stoptime": "stop"})[["hadm_id", "start", "stop", "drug"]]


def _eicu_drugs(cfg) -> pd.DataFrame:
    ref = pd.Timestamp(cfg["eicu"]["reference_date"])
    med = io.read_table(cfg.eicu_dir(), "medication",
                        usecols=["patientunitstayid", "drugstartoffset", "drugstopoffset", "drugname"])
    med = med.rename(columns={"patientunitstayid": "hadm_id", "drugname": "drug"})
    med["start"] = ref + pd.to_timedelta(pd.to_numeric(med["drugstartoffset"], errors="coerce"), unit="m")
    med["stop"] = ref + pd.to_timedelta(pd.to_numeric(med["drugstopoffset"], errors="coerce"), unit="m")
    return med[["hadm_id", "start", "stop", "drug"]]


def _load(cfg, cohort: str):
    out = cfg.output_dir
    panel = pd.read_parquet(out / f"{cohort}_aki_panel.parquet")
    panel = panel[panel["make_h_classifiable"] == True]  # noqa: E712
    oc = pd.read_parquet(out / f"{cohort}_aki_cohort_outcomes.parquet")[["hadm_id", "prd_h"]]
    coh = pd.read_parquet(out / f"{cohort}_aki_cohort.parquet")[["hadm_id", "aki_t0"]]
    drugs = _mimic_drugs(cfg) if cohort == "mimic" else _eicu_drugs(cfg)
    drugs = drugs.dropna(subset=["start"])
    drugs["hadm_id"] = drugs["hadm_id"].astype("int64")
    drugs = drugs[drugs["hadm_id"].isin(set(panel["hadm_id"]))]
    return panel.merge(oc, on="hadm_id", how="left"), coh, drugs


def build_grace(cfg, cohort: str, hours: int) -> pd.DataFrame:
    df, coh, drugs = _load(cfg, cohort)
    vanco = _flag(drugs, coh, VANCO, hours)
    pip = _flag(drugs, coh, PIPTAZO, hours)
    comp = _flag(drugs, coh, COMPARATOR, hours)
    df["on_vanco"] = df["hadm_id"].isin(vanco)
    df["on_pip"] = df["hadm_id"].isin(pip)
    df["on_comp"] = df["hadm_id"].isin(comp)
    vpt = df["on_vanco"] & df["on_pip"] & ~df["on_comp"]
    cmp = df["on_vanco"] & df["on_comp"] & ~df["on_pip"]
    sub = df[vpt | cmp].copy()
    sub["W"] = vpt[vpt | cmp].astype(int).values
    keep = ["hadm_id"] + XBASE + XEARLY + ["W", "make_h", "prd_h"]
    sub = sub[keep]
    sub.to_parquet(cfg.output_dir / f"{cohort}_vpt_48h.parquet", index=False)
    return sub


def build_dose(cfg, cohort: str, hours: int) -> pd.DataFrame:
    df, coh, drugs = _load(cfg, cohort)
    vanco = _flag(drugs, coh, VANCO, hours)
    pip = _flag(drugs, coh, PIPTAZO, hours)
    comp = _flag(drugs, coh, COMPARATOR, hours)
    df["on_vanco"] = df["hadm_id"].isin(vanco)
    df["on_pip"] = df["hadm_id"].isin(pip)
    df["on_comp"] = df["hadm_id"].isin(comp)
    vpt = df["on_vanco"] & df["on_pip"] & ~df["on_comp"]
    sub = df[vpt].copy()
    ov = _overlap_hours(drugs, coh, PIPTAZO, hours)
    sub["pt_overlap_hours"] = sub["hadm_id"].map(ov).fillna(0.0)
    # 1/2/3-day dose tier within the window
    sub["pt_dose_days"] = np.ceil(sub["pt_overlap_hours"] / 24.0).clip(lower=1, upper=hours // 24).astype(int)
    keep = ["hadm_id"] + XBASE + XEARLY + ["pt_overlap_hours", "pt_dose_days", "make_h", "prd_h"]
    sub = sub[keep]
    sub.to_parquet(cfg.output_dir / f"{cohort}_vpt_dose.parquet", index=False)
    return sub


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    for cohort in ("mimic", "eicu"):
        g = build_grace(cfg, cohort, hours=48)
        n, n_vpt = len(g), int(g["W"].sum())
        print(f"\n{cohort.upper()} 48h grace-period VPT population: {n:,d} "
              f"(VPT {n_vpt:,d} / comparator {n - n_vpt:,d})")
        d = build_dose(cfg, cohort, hours=cfg["exposures"]["exposure_window_hours"])
        tiers = d["pt_dose_days"].value_counts().sort_index()
        print(f"  dose tiers (pip-tazo days within window): "
              + ", ".join(f"{int(k)}d={int(v)}" for k, v in tiers.items()))


if __name__ == "__main__":
    main()
