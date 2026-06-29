# Reducible burden of MAKE after AKI: a counterfactual causal-inference study

Code for a retrospective, two-cohort causal-inference study that estimates how
much of the burden of major adverse kidney events (MAKE) after acute kidney
injury (AKI) in critically ill patients is potentially reducible by modifying
three bedside exposures: fluid balance, hypotension, and nephrotoxin exposure.
The framework is developed in MIMIC-IV and externally validated in eICU-CRD.

Data extraction and harmonization are in Python; the causal estimation is in R.
The primary estimand is the potential impact fraction of MAKE-H under
prespecified dynamic policies, estimated by longitudinal targeted maximum
likelihood estimation (LTMLE) with Super Learner and cross-fitting.
Treatment-effect heterogeneity and the effect among exposed patients are
estimated with doubly-robust causal forests.

This repository contains no patient data. MIMIC-IV and eICU-CRD are obtained
from PhysioNet by credentialed users under their Data Use Agreements.

## Repository layout

```
config/config.yaml              Locked study parameters (item ids, thresholds, drug lists)
requirements.txt                Python dependencies
src/makeaki/                    Python library
  config.py                     Typed configuration loader
  io.py                         Robust CSV / CSV.gz readers (chunked for large tables)
  egfr.py                       CKD-EPI 2021 eGFR
  kdigo.py                      KDIGO 2012 AKI staging and baseline-creatinine hierarchy
  signals.py                    Cached MIMIC labs and chartevents extraction
  cohort_mimic.py               MIMIC-IV cohort + screening log
  outcomes_mimic.py             MIMIC-IV MAKE-H labeling (death, RRT, PRD)
  exposures_mimic.py            MIMIC-IV 72h exposures (fluid, MAP, nephrotoxin)
  panel_mimic.py                MIMIC-IV longitudinal panel (windows, confounders, treatments)
  eicu_signals.py               Cached eICU labs and vitals (synthetic timestamps)
  eicu_cohort.py                eICU cohort + screening log
  eicu_outcomes.py              eICU MAKE-H labeling
  eicu_panel.py                 eICU longitudinal panel (same column layout as MIMIC)
scripts/
  01_build_cohort_mimic.py      MIMIC: cohort and screening log
  02_label_outcomes_mimic.py    MIMIC: MAKE-H and components
  03_extract_chartevents.py     MIMIC: cache chartevents (MAP, weight, RRT markers, SOFA)
  04_build_exposures_mimic.py   MIMIC: 72h exposures
  05_build_panel_mimic.py       MIMIC: longitudinal panel for estimation
  06_build_cohort_eicu.py       eICU: cohort and screening log
  07_label_outcomes_eicu.py     eICU: MAKE-H and components
  08_build_panel_eicu.py        eICU: longitudinal panel
  09_build_vpt.py               Well-defined target trial exposure (VPT vs comparator)
  10_build_negcontrol.py        Negative-control exposures for falsification
  11_build_vpt_sensitivity.py   VPT sensitivity inputs (48h grace period, pip-tazo dose)
  fig_vpt_sensitivity.py        Render the VPT sensitivity figure (grace period, dose-response)
  fig_study_flow.py             Render the study-flow figure (Figure 1) from the screening logs
  fig_negcontrol.py             Render the negative-control figure (MAKE-H and PRD)
  fig_main_forests.py           Render the AKI-gradient, components, and VPT forest figures
estimation/
  install_r_deps.R              Install R packages
  lmtp_pif.R                    Confirmatory PIF by LTMLE (cohort-aware)
  heterogeneity.R               Causal-forest CATE, among-exposed effect, subgroups
  secondary.R                   Effect on individual MAKE-H components
  vpt_analysis.R                Well-defined VPT vs comparator effect (MAKE-H, PRD)
  vpt_sensitivity.R             VPT robustness: 48h grace period and pip-tazo dose-response
  negcontrol_analysis.R         Negative-control falsification
  transport.R                   Transportability of the effect across cohorts
  run_full.R, run_full_eicu.R   Convenience runners (full library)
  run_med_eicu.R                Convenience runner (medium library, faster)
  run_windows_eicu.R            Self-contained Windows runner
  README_R.md                   R setup and usage
tests/test_logic.py             Unit tests for the core epidemiologic logic
outputs/                        Generated artifacts and figures (git-ignored)
```

## Data access

MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed PhysioNet users
under their Data Use Agreements. Place the downloaded tables under a single
folder and point the pipeline to it with the `DATA_ROOT` environment variable
(or the `data_root` key in `config/config.yaml`). Tables may be `.csv` or
`.csv.gz`; the readers handle both. Expected layout:

```
$DATA_ROOT/mimiciv/hosp/   patients, admissions, diagnoses_icd, labevents, prescriptions, ...
$DATA_ROOT/mimiciv/icu/    icustays, chartevents, inputevents, outputevents, procedureevents, ...
$DATA_ROOT/eicu/           patient, lab, vitalPeriodic, intakeOutput, medication, treatment, ...
```

## Reproduce

Python environment and the data-preparation pipeline:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_ROOT=/path/to/data

# MIMIC-IV (development)
python scripts/01_build_cohort_mimic.py --config config/config.yaml
python scripts/02_label_outcomes_mimic.py --config config/config.yaml
python scripts/03_extract_chartevents.py --config config/config.yaml
python scripts/04_build_exposures_mimic.py --config config/config.yaml
python scripts/05_build_panel_mimic.py --config config/config.yaml

# eICU-CRD (external validation)
python scripts/06_build_cohort_eicu.py --config config/config.yaml
python scripts/07_label_outcomes_eicu.py --config config/config.yaml
python scripts/08_build_panel_eicu.py --config config/config.yaml

# Robustness and falsification inputs
python scripts/09_build_vpt.py --config config/config.yaml
python scripts/10_build_negcontrol.py --config config/config.yaml
python scripts/11_build_vpt_sensitivity.py --config config/config.yaml
```

Estimation in R (see estimation/README_R.md for details):

```bash
Rscript estimation/install_r_deps.R

# Confirmatory potential impact fractions
LMTP_COHORT=mimic Rscript estimation/lmtp_pif.R
LMTP_COHORT=eicu  Rscript estimation/lmtp_pif.R

# Heterogeneity and component analyses
LMTP_COHORT=mimic Rscript estimation/heterogeneity.R
LMTP_COHORT=eicu  Rscript estimation/heterogeneity.R
Rscript estimation/secondary.R

# Robustness, falsification, and transportability
Rscript estimation/vpt_analysis.R
Rscript estimation/vpt_sensitivity.R
Rscript estimation/negcontrol_analysis.R
Rscript estimation/transport.R
```

Each script writes results to `outputs/`. The confirmatory LTMLE runs are
compute-intensive; memory use is controlled by `LMTP_WORKERS` (1 is sequential
and lowest memory).

## Conventions

- All analysis parameters are prespecified in `config/config.yaml` and locked
  prior to causal estimation; random seeds are fixed.
- The eICU panel is built to the identical column layout as the MIMIC panel so
  that the same estimation scripts run on both cohorts unchanged.

## License

MIT. See LICENSE.
