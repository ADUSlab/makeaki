"""eICU-CRD cohort construction: adult ICU AKI patients with a screening log.

eICU records time as integer offsets in minutes from unit admission. We map each
offset to a synthetic timestamp (reference_date + offset) so that the shared
analytic helpers (KDIGO staging, windowing, carry-forward) can be reused
unchanged. The unit of analysis is the ICU stay (patientunitstayid), mapped to
the column name ``hadm_id`` used throughout the pipeline.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import io
from .config import Config
from .egfr import ckd_epi_2021
from .kdigo import clean_creatinine, compute_baseline_creatinine, detect_aki_creatinine

logger = logging.getLogger(__name__)

_COMORBIDITY_PATTERNS = {
    "ckd": "chronic kidney|chronic renal|ckd",
    "diabetes": "diabetes",
    "heart_failure": "heart failure|cardiac failure|chf",
}


def _parse_age(value: object) -> float:
    s = str(value).strip()
    if s in ("", "nan", "NaN"):
        return np.nan
    if ">" in s:  # eICU encodes ages over 89 as "> 89"
        return 90.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _ref(cfg: Config) -> pd.Timestamp:
    return pd.Timestamp(cfg["eicu"]["reference_date"])


def _offset_to_time(cfg: Config, minutes: pd.Series) -> pd.Series:
    return _ref(cfg) + pd.to_timedelta(minutes.astype(float), unit="m")


def build_eicu_cohort(cfg: Config) -> tuple[pd.DataFrame, "OrderedDict[str, int]"]:
    """Build the eICU-CRD adult ICU AKI cohort and return it with a screening log."""
    d = cfg.eicu_dir()
    ref = _ref(cfg)

    patient = io.read_table(
        d,
        "patient",
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "gender",
            "age",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
            "hospitaldischargestatus",
            "admissionweight",
        ],
    )
    patient["age_years"] = patient["age"].map(_parse_age)
    patient["female"] = patient["gender"].astype(str).str.upper().str.startswith("F")

    screening: "OrderedDict[str, int]" = OrderedDict()
    screening["icu_stays"] = patient["patientunitstayid"].nunique()

    adults = patient[patient["age_years"] >= cfg["inclusion"]["min_age_years"]].copy()
    adults = adults.rename(
        columns={"patientunitstayid": "hadm_id", "patienthealthsystemstayid": "subject_id"}
    )
    screening["adult_icu_admissions"] = adults["hadm_id"].nunique()

    # Creatinine (synthetic timestamps).
    lab = io.read_table(
        d, "lab", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"]
    )
    names = [n.lower() for n in cfg["eicu"]["labnames"]["creatinine"]]
    creat = lab[lab["labname"].astype(str).str.lower().isin(names)].copy()
    creat = creat.rename(columns={"patientunitstayid": "hadm_id", "labresult": "valuenum"})
    creat["valuenum"] = pd.to_numeric(creat["valuenum"], errors="coerce")
    creat["charttime"] = _offset_to_time(cfg, creat["labresultoffset"])
    creat = clean_creatinine(creat)
    creat = creat[creat["hadm_id"].isin(adults["hadm_id"])]
    creat["subject_id"] = creat["hadm_id"]
    screening["with_creatinine"] = creat["hadm_id"].nunique()

    # Pseudo-admissions for the baseline-creatinine hierarchy. The baseline helper
    # matches creatinine to admissions on subject_id; in eICU each ICU stay is its
    # own unit, so subject_id is set equal to hadm_id for this computation.
    admissions = adults[["hadm_id", "unitdischargeoffset"]].copy()
    admissions["subject_id"] = admissions["hadm_id"]
    admissions["admittime"] = ref
    admissions["dischtime"] = _offset_to_time(cfg, admissions["unitdischargeoffset"])
    creat["subject_id"] = creat["hadm_id"]

    baseline = compute_baseline_creatinine(
        creat_subject=creat[["subject_id", "charttime", "valuenum"]],
        admissions=admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        outpatient_window_days=tuple(cfg["baseline_creatinine"]["outpatient_window_days"]),
        aggregator=cfg["baseline_creatinine"]["outpatient_aggregator"],
    )
    baseline = baseline[baseline["baseline_creatinine"].notna()]
    adults = adults[adults["hadm_id"].isin(baseline["hadm_id"])]
    screening["with_computable_baseline"] = adults["hadm_id"].nunique()

    # Exclusions from past history and diagnosis strings.
    excl = "|".join(cfg["eicu"]["exclusion_patterns"])
    exclude: set[int] = set()
    for table, col in (("pastHistory", "pasthistorypath"), ("diagnosis", "diagnosisstring")):
        try:
            tab = io.read_table(d, table, usecols=["patientunitstayid", col])
        except (FileNotFoundError, ValueError):
            continue
        hit = tab[tab[col].astype(str).str.lower().str.contains(excl, na=False, regex=True)]
        exclude |= set(hit["patientunitstayid"])
    before = adults["hadm_id"].nunique()
    adults = adults[~adults["hadm_id"].isin(exclude)]
    screening["excluded_esrd_dialysis_transplant"] = before - adults["hadm_id"].nunique()

    # AKI detection (creatinine criterion) on the synthetic timeline.
    creat_inhosp = creat[creat["hadm_id"].isin(adults["hadm_id"])][
        ["hadm_id", "charttime", "valuenum"]
    ]
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

    cohort = (
        adults[adults["hadm_id"].isin(aki_hadm)]
        .merge(baseline, on="hadm_id", how="left")
        .merge(aki, on="hadm_id", how="left")
    )

    cohort["admittime"] = ref
    cohort["dischtime"] = _offset_to_time(cfg, cohort["unitdischargeoffset"])
    expired = cohort["hospitaldischargestatus"].astype(str).str.lower().eq("expired")
    cohort["hospital_expire_flag"] = expired.astype(int)
    cohort["deathtime"] = pd.NaT
    cohort.loc[expired, "deathtime"] = _offset_to_time(cfg, cohort.loc[expired, "hospitaldischargeoffset"])
    cohort["admission_type"] = pd.NA
    cohort["icu_intime"] = ref
    cohort["anchor_age"] = cohort["age_years"]
    cohort["aki_present_at_admission"] = cohort["aki_t0"] <= cohort["icu_intime"]
    cohort["baseline_egfr"] = ckd_epi_2021(
        cohort["baseline_creatinine"], cohort["anchor_age"], cohort["female"]
    )

    # Comorbidity flags from the diagnosis strings.
    try:
        diag = io.read_table(d, "diagnosis", usecols=["patientunitstayid", "diagnosisstring"])
        ds = diag["diagnosisstring"].astype(str).str.lower()
        for name, pattern in _COMORBIDITY_PATTERNS.items():
            flagged = set(diag.loc[ds.str.contains(pattern, na=False, regex=True), "patientunitstayid"])
            cohort[f"cm_{name}"] = cohort["hadm_id"].isin(flagged)
    except (FileNotFoundError, ValueError):
        for name in _COMORBIDITY_PATTERNS:
            cohort[f"cm_{name}"] = False

    screening["aki_cohort_final"] = cohort["hadm_id"].nunique()

    keep = [
        "subject_id", "hadm_id", "anchor_age", "gender", "female",
        "admittime", "dischtime", "deathtime", "hospital_expire_flag",
        "admission_type", "icu_intime", "baseline_creatinine", "baseline_source",
        "baseline_egfr", "aki_t0", "aki_max_stage", "aki_present_at_admission",
        "cm_ckd", "cm_diabetes", "cm_heart_failure",
    ]
    return cohort[keep].reset_index(drop=True), screening
