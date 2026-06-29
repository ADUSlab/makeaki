#!/usr/bin/env python
"""Well-defined target-trial exposure: vancomycin + piperacillin-tazobactam (VPT)
versus vancomycin + an alternative anti-pseudomonal beta-lactam (cefepime or a
carbapenem), within the first 72 hours after AKI onset.

This replaces the heterogeneous "avoid nephrotoxins" policy with a specific,
implementable comparison aligned with the clinical literature on VPT-associated
AKI. The analysis population is restricted to patients receiving vancomycin plus
exactly one of the two beta-lactam options, so the contrast is well defined.

Outputs, for each cohort, outputs/<cohort>_vpt.parquet with baseline and early
confounders, the binary exposure (W = 1 for VPT, 0 for comparator), and the
outcomes make_h and prd_h.

Usage:
    python scripts/09_build_vpt.py --config config/config.yaml
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
    """hadm_ids whose drug interval overlaps [t0, t0+hours] and matches any pattern."""
    pat = "|".join(patterns)
    iv = intervals[intervals["drug"].astype(str).str.lower().str.contains(pat, na=False, regex=True)]
    iv = iv.merge(t0, on="hadm_id", how="inner")
    we = iv["aki_t0"] + pd.Timedelta(hours=hours)
    overlap = (iv["start"] < we) & (iv["stop"] > iv["aki_t0"])
    return set(iv.loc[overlap, "hadm_id"].unique())


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


def build(cfg, cohort: str) -> pd.DataFrame:
    out = cfg.output_dir
    panel = pd.read_parquet(out / f"{cohort}_aki_panel.parquet")
    panel = panel[panel["make_h_classifiable"] == True]  # noqa: E712
    oc = pd.read_parquet(out / f"{cohort}_aki_cohort_outcomes.parquet")[["hadm_id", "prd_h"]]
    coh = pd.read_parquet(out / f"{cohort}_aki_cohort.parquet")[["hadm_id", "aki_t0"]]
    hours = cfg["exposures"]["exposure_window_hours"]

    drugs = _mimic_drugs(cfg) if cohort == "mimic" else _eicu_drugs(cfg)
    drugs = drugs.dropna(subset=["start"])
    drugs["hadm_id"] = drugs["hadm_id"].astype("int64")
    drugs = drugs[drugs["hadm_id"].isin(set(panel["hadm_id"]))]

    vanco = _flag(drugs, coh, VANCO, hours)
    pip = _flag(drugs, coh, PIPTAZO, hours)
    comp = _flag(drugs, coh, COMPARATOR, hours)

    df = panel.merge(oc, on="hadm_id", how="left")
    df["on_vanco"] = df["hadm_id"].isin(vanco)
    df["on_pip"] = df["hadm_id"].isin(pip)
    df["on_comp"] = df["hadm_id"].isin(comp)

    # Comparable population: vancomycin plus exactly one beta-lactam option.
    vpt = df["on_vanco"] & df["on_pip"] & ~df["on_comp"]
    cmp = df["on_vanco"] & df["on_comp"] & ~df["on_pip"]
    sub = df[vpt | cmp].copy()
    sub["W"] = vpt[vpt | cmp].astype(int).values

    keep = ["hadm_id"] + XBASE + XEARLY + ["W", "make_h", "prd_h"]
    sub = sub[keep]
    sub.to_parquet(out / f"{cohort}_vpt.parquet", index=False)
    return sub


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    for cohort in ("mimic", "eicu"):
        sub = build(cfg, cohort)
        n, n_vpt = len(sub), int(sub["W"].sum())
        print(f"\n{cohort.upper()} VPT analysis population: {n:,d}  "
              f"(VPT {n_vpt:,d} / comparator {n - n_vpt:,d})")
        print(f"  observed MAKE-H: VPT {sub.loc[sub.W==1,'make_h'].mean():.3f}  "
              f"comparator {sub.loc[sub.W==0,'make_h'].mean():.3f}")


if __name__ == "__main__":
    main()
