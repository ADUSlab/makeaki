"""MIMIC-IV cohort construction: adult ICU AKI patients with a screening log.

This stage produces the estimation cohort (all eligible AKI patients, regardless
of whether they later develop MAKE) and the per-step screening funnel. Outcome
labeling and exposure construction are implemented in later stages.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import pandas as pd

from . import io
from .config import Config
from .egfr import ckd_epi_2021
from .kdigo import (
    clean_creatinine,
    compute_baseline_creatinine,
    detect_aki_creatinine,
)

logger = logging.getLogger(__name__)

_CREATININE_ITEMID_COLS = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]


def _norm_icd(series: pd.Series) -> pd.Series:
    """Normalise ICD codes to upper case without dots or whitespace."""
    return (
        series.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    )


def _flag_admissions_by_icd(
    diagnoses: pd.DataFrame, codes_exact: list[str], prefixes: list[str]
) -> set[int]:
    """Return hadm_ids whose diagnoses match any exact code or code prefix."""
    code = _norm_icd(diagnoses["icd_code"])
    exact = set(c.upper().replace(".", "") for c in codes_exact)
    mask = code.isin(exact)
    for pref in prefixes:
        mask = mask | code.str.startswith(pref.upper().replace(".", ""))
    return set(diagnoses.loc[mask, "hadm_id"].unique())


def build_mimic_cohort(cfg: Config) -> tuple[pd.DataFrame, "OrderedDict[str, int]"]:
    """Build the MIMIC-IV adult ICU AKI cohort and return it with a screening log."""
    hosp, icu = cfg.mimic_hosp(), cfg.mimic_icu()

    patients = io.read_table(
        hosp, "patients", usecols=["subject_id", "gender", "anchor_age", "dod"]
    )
    admissions = io.read_table(
        hosp,
        "admissions",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icustays = io.read_table(
        icu,
        "icustays",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )

    screening: "OrderedDict[str, int]" = OrderedDict()

    # Adults with at least one ICU stay (admission-level unit of analysis).
    adm = admissions.merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )
    adm = adm[adm["anchor_age"] >= cfg["inclusion"]["min_age_years"]]
    icu_hadm = icustays.groupby("hadm_id")["intime"].min().rename("icu_intime")
    adm = adm.merge(icu_hadm, on="hadm_id", how="inner")
    screening["adult_icu_admissions"] = adm["hadm_id"].nunique()

    # Creatinine: subject-level (for outpatient baseline) and in-hospital (for AKI).
    creat_all = io.read_filtered_by_itemid(
        hosp,
        "labevents",
        itemids=cfg["mimic_itemids"]["creatinine_lab"],
        usecols=_CREATININE_ITEMID_COLS,
        parse_dates=["charttime"],
    )
    creat_all = clean_creatinine(creat_all)
    creat_subject = creat_all[["subject_id", "charttime", "valuenum"]]

    creat_inhosp = creat_all[creat_all["hadm_id"].notna()]
    creat_inhosp = creat_inhosp[creat_inhosp["hadm_id"].isin(adm["hadm_id"])]
    creat_inhosp = creat_inhosp[["hadm_id", "charttime", "valuenum"]].copy()
    creat_inhosp["hadm_id"] = creat_inhosp["hadm_id"].astype("int64")
    screening["with_inhospital_creatinine"] = creat_inhosp["hadm_id"].nunique()

    # Exclusions by diagnosis.
    diagnoses = io.read_table(
        hosp, "diagnoses_icd", usecols=["hadm_id", "icd_code", "icd_version"]
    )
    excl = cfg["exclusion_icd"]
    exclude_hadm = (
        _flag_admissions_by_icd(diagnoses, excl["esrd"], [])
        | _flag_admissions_by_icd(diagnoses, excl["chronic_dialysis"], [])
        | _flag_admissions_by_icd(diagnoses, excl["kidney_transplant"], [])
    )
    before = adm["hadm_id"].nunique()
    adm = adm[~adm["hadm_id"].isin(exclude_hadm)]
    screening["excluded_esrd_dialysis_transplant"] = before - adm["hadm_id"].nunique()

    # Baseline creatinine.
    baseline = compute_baseline_creatinine(
        creat_subject=creat_subject,
        admissions=adm[["subject_id", "hadm_id", "admittime", "dischtime"]],
        outpatient_window_days=tuple(cfg["baseline_creatinine"]["outpatient_window_days"]),
        aggregator=cfg["baseline_creatinine"]["outpatient_aggregator"],
    )
    baseline = baseline[baseline["baseline_creatinine"].notna()]
    adm = adm[adm["hadm_id"].isin(baseline["hadm_id"])]
    screening["with_computable_baseline"] = adm["hadm_id"].nunique()

    # AKI detection (creatinine criterion).
    creat_inhosp = creat_inhosp[creat_inhosp["hadm_id"].isin(adm["hadm_id"])]
    aki = detect_aki_creatinine(
        creat_inhosp=creat_inhosp,
        baseline=baseline,
        scr_abs_increase=cfg["aki"]["scr_abs_increase_mgdl"],
        ratio_stage1=cfg["aki"]["scr_ratio_stage1"],
        ratio_stage2=cfg["aki"]["scr_ratio_stage2"],
        ratio_stage3=cfg["aki"]["scr_ratio_stage3"],
        abs_stage3=cfg["aki"]["scr_abs_stage3_mgdl"],
        delta_window_hours=cfg["aki"]["delta_window_hours"],
    )
    aki_hadm = set(aki.loc[aki["aki"], "hadm_id"])
    screening["kdigo_aki_creatinine"] = len(aki_hadm)

    # Assemble the AKI estimation cohort.
    cohort = adm[adm["hadm_id"].isin(aki_hadm)].merge(
        baseline, on="hadm_id", how="left"
    ).merge(aki, on="hadm_id", how="left")

    cohort["female"] = cohort["gender"].eq("F")
    cohort["baseline_egfr"] = ckd_epi_2021(
        cohort["baseline_creatinine"], cohort["anchor_age"], cohort["female"]
    )
    cohort["aki_present_at_admission"] = (
        cohort["aki_t0"] <= cohort["icu_intime"]
    )

    # Comorbidity flags (baseline confounders).
    comob = cfg["comorbidity_icd_prefix"]
    for name, prefixes in comob.items():
        flagged = _flag_admissions_by_icd(diagnoses, [], prefixes)
        cohort[f"cm_{name}"] = cohort["hadm_id"].isin(flagged)

    screening["aki_cohort_final"] = cohort["hadm_id"].nunique()

    keep = [
        "subject_id",
        "hadm_id",
        "anchor_age",
        "gender",
        "female",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "admission_type",
        "icu_intime",
        "baseline_creatinine",
        "baseline_source",
        "baseline_egfr",
        "aki_t0",
        "aki_max_stage",
        "aki_present_at_admission",
    ] + [f"cm_{name}" for name in comob]
    cohort = cohort[keep].reset_index(drop=True)
    return cohort, screening
