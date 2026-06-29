#!/usr/bin/env python
"""Negative-control exposures for falsification.

Common, non-nephrotoxic drugs (acetaminophen, pantoprazole) have no plausible
causal effect on MAKE-H. Estimating their apparent effect tests for residual
confounding: a non-null association indicates that confounding (for example,
drugs given to less acutely ill patients) can manufacture a spurious effect of
the kind seen for the nephrotoxin policy.

Outputs outputs/<cohort>_negcontrol.parquet with baseline and early confounders,
two negative-control exposure flags, and the outcomes make_h and prd_h (the latter
so the falsification can also be applied to the surviving renal signal).

Usage:
    python scripts/10_build_negcontrol.py --config config/config.yaml
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

NEG_CONTROLS = {
    "acetaminophen": ["acetaminophen", "paracetamol", "tylenol", "ofirmev"],
    "pantoprazole": ["pantoprazole", "protonix"],
}

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
    panel = panel.merge(oc, on="hadm_id", how="left")
    coh = pd.read_parquet(out / f"{cohort}_aki_cohort.parquet")[["hadm_id", "aki_t0"]]
    hours = cfg["exposures"]["exposure_window_hours"]

    drugs = _mimic_drugs(cfg) if cohort == "mimic" else _eicu_drugs(cfg)
    drugs = drugs.dropna(subset=["start"])
    drugs["hadm_id"] = drugs["hadm_id"].astype("int64")
    drugs = drugs[drugs["hadm_id"].isin(set(panel["hadm_id"]))]

    df = panel.copy()
    for name, pats in NEG_CONTROLS.items():
        flagged = _flag(drugs, coh, pats, hours)
        df[f"nc_{name}"] = df["hadm_id"].isin(flagged).astype(int)

    keep = ["hadm_id"] + XBASE + XEARLY + [f"nc_{n}" for n in NEG_CONTROLS] + ["make_h", "prd_h"]
    df = df[keep]
    df.to_parquet(out / f"{cohort}_negcontrol.parquet", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    for cohort in ("mimic", "eicu"):
        df = build(cfg, cohort)
        for name in NEG_CONTROLS:
            col = f"nc_{name}"
            print(f"{cohort.upper()} {name}: exposed {int(df[col].sum()):,d} / {len(df):,d} "
                  f"({100 * df[col].mean():.0f}%)")


if __name__ == "__main__":
    main()
