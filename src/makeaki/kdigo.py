"""KDIGO 2012 acute kidney injury staging (creatinine criterion).

This module implements the prespecified baseline-creatinine hierarchy and the
serum-creatinine arm of KDIGO 2012 staging. The urine-output criterion is added
in a later stage and is intentionally separated here so that the creatinine
logic can be tested in isolation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PLAUSIBLE_SCR = (0.1, 50.0)  # mg/dL bounds for biologically plausible values


def clean_creatinine(creat: pd.DataFrame, value_col: str = "valuenum") -> pd.DataFrame:
    """Drop missing and biologically implausible creatinine measurements."""
    out = creat.dropna(subset=[value_col]).copy()
    lo, hi = _PLAUSIBLE_SCR
    out = out[(out[value_col] >= lo) & (out[value_col] <= hi)]
    return out


def compute_baseline_creatinine(
    creat_subject: pd.DataFrame,
    admissions: pd.DataFrame,
    outpatient_window_days: tuple[int, int],
    aggregator: str = "median",
) -> pd.DataFrame:
    """Compute baseline creatinine per admission using the prespecified hierarchy.

    Hierarchy:
    1. Outpatient creatinine measured between ``window`` days before admission
       (aggregated, default median).
    2. Lowest in-hospital creatinine during the admission (surrogate for the
       lowest pre-AKI value, avoiding circularity).
    3. First creatinine measured during the admission (admission value).

    Parameters
    ----------
    creat_subject:
        Columns ``subject_id``, ``charttime``, ``valuenum`` for all creatinine.
    admissions:
        Columns ``subject_id``, ``hadm_id``, ``admittime``, ``dischtime``.

    Returns
    -------
    DataFrame indexed by ``hadm_id`` with ``baseline_creatinine`` and
    ``baseline_source``.
    """
    lo_days, hi_days = outpatient_window_days
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()

    merged = creat_subject.merge(adm, on="subject_id", how="inner")
    dt_days = (merged["admittime"] - merged["charttime"]).dt.total_seconds() / 86400.0

    # 1. Outpatient window before admission
    outpatient_mask = (dt_days >= lo_days) & (dt_days <= hi_days)
    outpatient = merged.loc[outpatient_mask].groupby("hadm_id")["valuenum"]
    outpatient_base = (
        outpatient.median() if aggregator == "median" else outpatient.min()
    )
    outpatient_base = outpatient_base.rename("baseline_creatinine")

    # 2. Lowest in-hospital value during the admission
    in_hosp_mask = (merged["charttime"] >= merged["admittime"]) & (
        merged["charttime"] <= merged["dischtime"]
    )
    in_hosp = merged.loc[in_hosp_mask]
    in_hosp_min = in_hosp.groupby("hadm_id")["valuenum"].min().rename("baseline_creatinine")

    # 3. First value during the admission
    first_val = (
        in_hosp.sort_values("charttime")
        .groupby("hadm_id")["valuenum"]
        .first()
        .rename("baseline_creatinine")
    )

    result = pd.DataFrame(index=adm["hadm_id"].unique())
    result.index.name = "hadm_id"
    result["baseline_creatinine"] = np.nan
    result["baseline_source"] = pd.NA

    for source, series in (
        ("outpatient", outpatient_base),
        ("inpatient_min", in_hosp_min),
        ("admission_value", first_val),
    ):
        missing = result["baseline_creatinine"].isna()
        fill = series.reindex(result.index)
        take = missing & fill.notna()
        result.loc[take, "baseline_creatinine"] = fill[take]
        result.loc[take, "baseline_source"] = source

    return result.reset_index()


def detect_aki_creatinine(
    creat_inhosp: pd.DataFrame,
    baseline: pd.DataFrame,
    scr_abs_increase: float,
    ratio_stage1: float,
    ratio_stage2: float,
    ratio_stage3: float,
    abs_stage3: float,
    delta_window_hours: int,
) -> pd.DataFrame:
    """Detect AKI and assign the maximum creatinine-based KDIGO stage per admission.

    Parameters
    ----------
    creat_inhosp:
        In-hospital creatinine with columns ``hadm_id``, ``charttime``,
        ``valuenum``.
    baseline:
        Output of :func:`compute_baseline_creatinine`.

    Returns
    -------
    DataFrame with ``hadm_id``, ``aki`` (bool), ``aki_t0`` (timestamp of first
    stage 1 criterion), and ``aki_max_stage`` (1-3, 0 if no AKI).
    """
    base_map = baseline.set_index("hadm_id")["baseline_creatinine"]
    window = f"{int(delta_window_hours)}h"

    records: list[dict[str, object]] = []
    for hadm_id, grp in creat_inhosp.sort_values("charttime").groupby("hadm_id"):
        b = base_map.get(hadm_id, np.nan)
        if not np.isfinite(b) or b <= 0:
            records.append(
                {"hadm_id": hadm_id, "aki": False, "aki_t0": pd.NaT, "aki_max_stage": 0}
            )
            continue

        series = grp.set_index("charttime")["valuenum"].astype(float)
        rolling_min = series.rolling(window).min()
        delta = series - rolling_min
        ratio = series / b

        stage = np.zeros(len(series), dtype=int)
        stage = np.where(
            (delta.to_numpy() >= scr_abs_increase) | (ratio.to_numpy() >= ratio_stage1),
            1,
            stage,
        )
        stage = np.where(ratio.to_numpy() >= ratio_stage2, 2, stage)
        stage = np.where(
            (ratio.to_numpy() >= ratio_stage3) | (series.to_numpy() >= abs_stage3),
            3,
            stage,
        )

        if stage.max() >= 1:
            t0 = series.index[np.argmax(stage >= 1)]
            records.append(
                {
                    "hadm_id": hadm_id,
                    "aki": True,
                    "aki_t0": t0,
                    "aki_max_stage": int(stage.max()),
                }
            )
        else:
            records.append(
                {"hadm_id": hadm_id, "aki": False, "aki_t0": pd.NaT, "aki_max_stage": 0}
            )

    return pd.DataFrame.from_records(records)
