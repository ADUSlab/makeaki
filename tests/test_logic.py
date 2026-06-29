"""Unit tests for the creatinine, baseline, and eGFR logic.

Run with: pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from makeaki.egfr import ckd_epi_2021  # noqa: E402
from makeaki.kdigo import (  # noqa: E402
    clean_creatinine,
    compute_baseline_creatinine,
    detect_aki_creatinine,
)

BASE = pd.Timestamp("2150-01-10 08:00")


def _synthetic_creatinine() -> pd.DataFrame:
    creat = pd.DataFrame(
        {
            "subject_id": [1] * 5,
            "hadm_id": [np.nan, 10, 10, 10, 10],
            "charttime": [
                BASE - pd.Timedelta(days=30),
                BASE,
                BASE + pd.Timedelta(hours=24),
                BASE + pd.Timedelta(hours=48),
                BASE + pd.Timedelta(hours=72),
            ],
            "valuenum": [1.0, 1.0, 1.6, 2.2, 3.5],
        }
    )
    return clean_creatinine(creat)


def test_egfr_plausible_range() -> None:
    egfr = ckd_epi_2021(pd.Series([1.0]), pd.Series([60]), pd.Series([False])).iloc[0]
    assert 80 < egfr < 95


def test_baseline_prefers_outpatient() -> None:
    creat = _synthetic_creatinine()
    adm = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [10],
            "admittime": [BASE],
            "dischtime": [BASE + pd.Timedelta(days=10)],
        }
    )
    bl = compute_baseline_creatinine(
        creat[["subject_id", "charttime", "valuenum"]], adm, (7, 365), "median"
    )
    assert abs(bl.loc[0, "baseline_creatinine"] - 1.0) < 1e-9
    assert bl.loc[0, "baseline_source"] == "outpatient"


def test_aki_staging_reaches_stage_three() -> None:
    creat = _synthetic_creatinine()
    adm = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [10],
            "admittime": [BASE],
            "dischtime": [BASE + pd.Timedelta(days=10)],
        }
    )
    bl = compute_baseline_creatinine(
        creat[["subject_id", "charttime", "valuenum"]], adm, (7, 365), "median"
    )
    inh = creat[creat["hadm_id"].notna()][["hadm_id", "charttime", "valuenum"]].copy()
    inh["hadm_id"] = inh["hadm_id"].astype("int64")
    aki = detect_aki_creatinine(inh, bl, 0.3, 1.5, 2.0, 3.0, 4.0, 48)
    row = aki.iloc[0]
    assert bool(row["aki"]) is True
    assert int(row["aki_max_stage"]) == 3
    assert row["aki_t0"] == BASE + pd.Timedelta(hours=24)
