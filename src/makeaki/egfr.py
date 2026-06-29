"""CKD-EPI 2021 (race-free) estimated glomerular filtration rate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ckd_epi_2021(scr_mgdl: pd.Series, age_years: pd.Series, female: pd.Series) -> pd.Series:
    """Estimate eGFR (mL/min/1.73 m2) with the 2021 CKD-EPI creatinine equation.

    Parameters
    ----------
    scr_mgdl:
        Serum creatinine in mg/dL.
    age_years:
        Age in years.
    female:
        Boolean series, True for female patients.
    """
    scr = scr_mgdl.astype(float)
    age = age_years.astype(float)
    is_female = female.astype(bool)

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    sex_factor = np.where(is_female, 1.012, 1.0)

    ratio = scr / kappa
    egfr = (
        142.0
        * np.minimum(ratio, 1.0) ** alpha
        * np.maximum(ratio, 1.0) ** (-1.200)
        * 0.9938 ** age
        * sex_factor
    )
    return pd.Series(egfr, index=scr_mgdl.index, name="egfr")
