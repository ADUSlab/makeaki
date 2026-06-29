#!/usr/bin/env Rscript
# ============================================================================
# Confirmatory PIF estimation by longitudinal TMLE (lmtp).
#
# For each prespecified dynamic policy (g_F, g_MAP, g_NTX, g_joint) this script
# estimates the counterfactual risk psi(g) = E[Y^g] of MAKE-H, the absolute risk
# reduction versus the natural course, and the potential impact fraction
# PIF(g) = (psi_obs - psi(g)) / psi_obs, with influence-curve confidence
# intervals. Nuisances are fit with a Super Learner library and cross-fitting.
#
# Input : outputs/mimic_aki_panel.parquet   (from scripts/05_build_panel_mimic.py)
# Output: outputs/mimic_pif_results.csv
#
# Usage : Rscript estimation/lmtp_pif.R
# ============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(lmtp)
})

set.seed(20260617)

# Parallelize cross-fitting across cores. Each worker holds its own copy of the
# data and models in memory, so too many workers can exhaust RAM on large
# cohorts. Default to a conservative 2; set LMTP_WORKERS=1 for sequential
# (lowest memory) or a higher number if the machine has plenty of RAM.
workers <- as.integer(Sys.getenv("LMTP_WORKERS", "2"))
if (is.na(workers) || workers < 1L) workers <- 1L
if (workers > 1L && requireNamespace("future", quietly = TRUE)) {
  options(future.globals.maxSize = 2 * 1024^3)
  future::plan(future::multisession, workers = workers)
  message("Parallel backend: ", workers, " workers (LMTP_WORKERS to change; 1 = sequential)")
} else {
  if (requireNamespace("future", quietly = TRUE)) future::plan(future::sequential)
  message("Sequential backend (lowest memory)")
}

K <- 5

# Cohort selector: LMTP_COHORT=mimic (default) or eicu.
COHORT <- Sys.getenv("LMTP_COHORT", "mimic")
PANEL_PATH <- sprintf("outputs/%s_aki_panel.parquet", COHORT)

# Quick mode (LMTP_QUICK=1) runs a fast end-to-end check with a minimal learner
# library and 2 folds. Leave unset for the full prespecified configuration.
if (Sys.getenv("LMTP_QUICK") == "1") {
  FOLDS <- 2
  LEARNERS <- c("SL.glm")
  suffix <- "_quick"
} else if (Sys.getenv("LMTP_MED") == "1") {
  FOLDS <- 5
  LEARNERS <- c("SL.mean", "SL.glm", "SL.glmnet")
  suffix <- "_med"
} else {
  FOLDS <- 5
  LEARNERS <- c("SL.mean", "SL.glm", "SL.glmnet", "SL.ranger", "SL.xgboost")
  suffix <- ""
}
OUT_PATH <- sprintf("outputs/%s_pif_results%s.csv", COHORT, suffix)
message("Cohort: ", COHORT, " | panel: ", PANEL_PATH)

# ---------------------------------------------------------------------------
# Load and prepare the panel.
# ---------------------------------------------------------------------------
panel <- as.data.frame(read_parquet(PANEL_PATH))
panel <- panel[panel$make_h_classifiable == TRUE, ]

logical_cols <- names(panel)[sapply(panel, is.logical)]
panel[logical_cols] <- lapply(panel[logical_cols], as.integer)

baseline <- c(
  "anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
  "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission"
)

# Missingness indicators and median imputation for time-varying confounders.
continuous_confounders <- c(
  "map_last", "lact_last", "uo_rate",
  "platelet_last", "bilirubin_last", "pao2_last", "fio2_last", "spo2_last", "gcs_total"
)
impute_confounders <- unlist(lapply(continuous_confounders, function(x) paste0(x, "_", 1:K)))
for (col in impute_confounders) {
  miss <- is.na(panel[[col]])
  panel[[paste0(col, "_miss")]] <- as.integer(miss)
  panel[[col]][miss] <- stats::median(panel[[col]], na.rm = TRUE)
}

# Structural-zero confounders: absence of a vasopressor or ventilation event in a
# window means the patient was not on that support (value 0, not missing).
for (stub in c("vaso", "vent")) {
  for (k in 1:K) {
    col <- paste0(stub, "_", k)
    panel[[col]][is.na(panel[[col]])] <- 0L
  }
}

# Treatments: a missing within-window treatment means the exposure was not
# recorded in that window; encode as not-exposed (0) with an indicator.
for (stub in c("a_fluid", "a_map", "a_ntx")) {
  for (k in 1:K) {
    col <- paste0(stub, "_", k)
    miss <- is.na(panel[[col]])
    panel[[paste0(col, "_miss")]] <- as.integer(miss)
    panel[[col]][miss] <- 0L
  }
}

fluid <- paste0("a_fluid_", 1:K)
map_t <- paste0("a_map_", 1:K)
ntx <- paste0("a_ntx_", 1:K)

# Time-varying confounders measured at the start of each window.
Lvars <- function(k) {
  base_vars <- c(
    paste0("map_last_", k), paste0("creat_last_", k), paste0("lact_last_", k),
    paste0("uo_rate_", k), paste0("vaso_", k), paste0("vent_", k),
    paste0("platelet_last_", k), paste0("bilirubin_last_", k),
    paste0("pao2_last_", k), paste0("fio2_last_", k), paste0("spo2_last_", k),
    paste0("gcs_total_", k)
  )
  miss_vars <- paste0(continuous_confounders, "_", k, "_miss")
  c(base_vars, miss_vars)
}

# ---------------------------------------------------------------------------
# Deterministic dynamic policies, expressed as pre-computed shifted datasets.
#   g_F : a_fluid -> 0 only when not vasopressor-dependent (refractory-shock proxy)
#   g_MAP, g_NTX : a_map / a_ntx -> 0
# ---------------------------------------------------------------------------
shift_fluid <- function(d) {
  for (k in 1:K) {
    col <- paste0("a_fluid_", k)
    not_shock <- d[[paste0("vaso_", k)]] == 0
    d[[col]] <- ifelse(not_shock, 0L, d[[col]])
  }
  d
}
shift_map <- function(d) {
  for (k in 1:K) d[[paste0("a_map_", k)]] <- 0L
  d
}
shift_ntx <- function(d) {
  for (k in 1:K) d[[paste0("a_ntx_", k)]] <- 0L
  d
}
shift_joint <- function(d) shift_ntx(shift_map(shift_fluid(d)))

# ---------------------------------------------------------------------------
# Run one policy. Non-intervened concurrent exposures enter as confounders.
# ---------------------------------------------------------------------------
run_policy <- function(name, trt_list, build_shifted, extra_tv) {
  time_vary <- lapply(1:K, function(k) c(Lvars(k), extra_tv(k)))
  message("Estimating policy ", name, " ...")
  lmtp_tmle(
    data = panel,
    trt = trt_list,
    outcome = "make_h",
    baseline = baseline,
    time_vary = time_vary,
    shifted = build_shifted(panel),
    outcome_type = "binomial",
    learners_outcome = LEARNERS,
    learners_trt = LEARNERS,
    folds = FOLDS
  )
}

# Natural-course (observed) estimate: no shift.
message("Estimating natural course (observed) ...")
fit_obs <- lmtp_tmle(
  data = panel,
  trt = lapply(1:K, function(k) c(fluid[k], map_t[k], ntx[k])),
  outcome = "make_h",
  baseline = baseline,
  time_vary = lapply(1:K, Lvars),
  shift = NULL,
  outcome_type = "binomial",
  learners_outcome = LEARNERS,
  learners_trt = LEARNERS,
  folds = FOLDS
)

policies <- list(
  g_F = list(
    trt = as.list(fluid),
    shifted = shift_fluid,
    extra_tv = function(k) c(paste0("a_map_", k), paste0("a_ntx_", k))
  ),
  g_MAP = list(
    trt = as.list(map_t),
    shifted = shift_map,
    extra_tv = function(k) c(paste0("a_fluid_", k), paste0("a_ntx_", k))
  ),
  g_NTX = list(
    trt = as.list(ntx),
    shifted = shift_ntx,
    extra_tv = function(k) c(paste0("a_fluid_", k), paste0("a_map_", k))
  ),
  g_joint = list(
    trt = lapply(1:K, function(k) c(fluid[k], map_t[k], ntx[k])),
    shifted = shift_joint,
    extra_tv = function(k) character(0)
  )
)

# One-time diagnostic of the estimate object structure.
message("Structure of fit$estimate:")
utils::str(fit_obs$estimate)

# Robust extraction from the lmtp 'ife' estimate object (a list-like with an
# estimate and standard error under one of several possible field names).
field_of <- function(obj, opts) {
  if (is.numeric(obj)) return(suppressWarnings(as.numeric(obj))[1])
  if (is.list(obj)) {
    for (o in opts) if (!is.null(obj[[o]])) return(suppressWarnings(as.numeric(obj[[o]]))[1])
  }
  for (o in opts) {
    v <- tryCatch(methods::slot(obj, o), error = function(e) NULL)
    if (!is.null(v)) return(suppressWarnings(as.numeric(v))[1])
  }
  NA_real_
}
est_of <- function(fit) field_of(fit$estimate, c("x", "estimate", "theta", "value"))
eif_of <- function(fit) {
  e <- fit$estimate
  v <- tryCatch(methods::slot(e, "eif"), error = function(err) NULL)
  if (is.null(v) && is.list(e)) v <- e[["eif"]]
  as.numeric(v)
}
maxweight_of <- function(fit) {
  d <- fit$density_ratios
  if (is.null(d)) return(NA_real_)
  suppressWarnings(max(as.numeric(as.matrix(d)), na.rm = TRUE))
}

psi_obs <- est_of(fit_obs)
eif_obs <- eif_of(fit_obs)
n <- length(eif_obs)
rows <- list()
for (name in names(policies)) {
  p <- policies[[name]]
  fit <- run_policy(name, p$trt, p$shifted, p$extra_tv)
  psi_g <- est_of(fit)

  # Proper influence-curve standard error of the difference psi_obs - psi(g)
  # from the difference of the two efficient influence functions.
  arr <- psi_obs - psi_g
  eif_diff <- eif_obs - eif_of(fit)
  arr_se <- stats::sd(eif_diff) / sqrt(n)

  rows[[name]] <- data.frame(
    policy = name,
    psi_obs = psi_obs,
    psi_g = psi_g,
    arr = arr,
    arr_ci_low = arr - 1.96 * arr_se,
    arr_ci_high = arr + 1.96 * arr_se,
    rr = psi_g / psi_obs,
    pif = arr / psi_obs,
    pif_ci_low = (arr - 1.96 * arr_se) / psi_obs,
    pif_ci_high = (arr + 1.96 * arr_se) / psi_obs,
    max_weight = maxweight_of(fit)
  )
}

results <- do.call(rbind, rows)
write.csv(results, OUT_PATH, row.names = FALSE)

cat("\nConfirmatory PIF results (MAKE-H):\n")
print(format(results, digits = 3))
cat("\nWritten to ", OUT_PATH, "\n", sep = "")
