# Estimation stage (R / lmtp)

The confirmatory PIF estimation uses longitudinal targeted maximum likelihood
estimation through the `lmtp` package, which implements doubly-robust estimators
for longitudinal modified treatment policies with Super Learner and
cross-fitting. Data preparation is done in Python; only the estimation runs in R.

## Setup (once)

Install R (4.2 or newer), then:

```bash
Rscript estimation/install_r_deps.R
```

This installs `arrow`, `lmtp`, `SuperLearner`, `glmnet`, and `ranger`.

## Run

From the repository root, after building the panel
(`scripts/05_build_panel_mimic.py`):

```bash
Rscript estimation/lmtp_pif.R
```

Output: `outputs/mimic_pif_results.csv` with, for each policy, the observed risk,
counterfactual risk `psi(g)`, absolute risk reduction with confidence interval,
risk ratio, and the potential impact fraction `PIF` with confidence interval.

## What it estimates

Policies (deterministic dynamic, prespecified):

- `g_F`: avoid net-positive fluid balance in a window, except when the patient is
  vasopressor-dependent in that window (refractory-shock proxy).
- `g_MAP`: no sustained MAP below threshold.
- `g_NTX`: avoid nephrotoxins.
- `g_joint`: all three jointly.

For single policies, the two non-intervened exposures are included as
time-varying confounders. Time-varying confounders are carried forward to each
window boundary; missing values are median-imputed with missingness indicators.

## Notes

- The `PIF` confidence interval is obtained by scaling the influence-curve
  interval of the absolute risk reduction by the observed risk; this treats the
  observed risk as fixed and is a small-variance approximation.
- The Super Learner library and number of folds are set in the script header and
  are part of the prespecified analysis plan.
