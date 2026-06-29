#!/usr/bin/env Rscript
# ============================================================================
# Sensitivity analyses for the well-defined VPT target trial.
#
#   (a) Grace period. Re-estimate the VPT-versus-comparator effect on MAKE-H and
#       persistent renal dysfunction under a 48-hour grace period (main = 72h).
#   (b) Dose-response. Among VPT-exposed patients, estimate the association of
#       cumulative piperacillin-tazobactam dose (1/2/3-day tier) with persistent
#       renal dysfunction, adjusting for the same confounders, as a test for a
#       monotonic dose-response. A doubly-robust forest on the binary contrast
#       (>=2 days vs 1 day) gives the headline; a covariate-adjusted linear fit
#       across tiers gives the trend.
#
#   Rscript estimation/vpt_sensitivity.R
# ============================================================================

suppressPackageStartupMessages({ library(arrow); library(grf) })
set.seed(20260617)

Xvars <- c("anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
           "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission",
           "map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
           "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
           "spo2_last_1", "gcs_total_1")

clean <- function(d) {
  lg <- sapply(d, is.logical); d[lg] <- lapply(d[lg], as.integer)
  for (v in Xvars) { x <- suppressWarnings(as.numeric(d[[v]])); x[is.na(x)] <- stats::median(x, na.rm = TRUE); d[[v]] <- x }
  d
}

# (a) 48h grace-period re-estimation -----------------------------------------
grace <- function(cohort) {
  d <- clean(as.data.frame(read_parquet(sprintf("outputs/%s_vpt_48h.parquet", cohort))))
  X <- as.matrix(d[, Xvars]); W <- as.integer(d$W)
  rows <- list()
  for (oc in c("make_h", "prd_h")) {
    y <- suppressWarnings(as.numeric(d[[oc]])); keep <- !is.na(y)
    if (length(unique(W[keep])) < 2) next
    cf <- causal_forest(X[keep, ], y[keep], W[keep], num.trees = 2000, seed = 1)
    a <- average_treatment_effect(cf, target.sample = "all")
    rows[[oc]] <- data.frame(analysis = "grace_48h", cohort = cohort, outcome = oc,
      n = sum(keep), n_vpt = sum(W[keep]),
      effect = a[[1]], se = a[[2]], ci_low = a[[1]] - 1.96 * a[[2]], ci_high = a[[1]] + 1.96 * a[[2]])
  }
  do.call(rbind, rows)
}

# (b) dose-response among VPT-exposed ----------------------------------------
dose <- function(cohort) {
  d <- clean(as.data.frame(read_parquet(sprintf("outputs/%s_vpt_dose.parquet", cohort))))
  d$pt_dose_days <- as.integer(d$pt_dose_days)
  X <- as.matrix(d[, Xvars])
  rows <- list()
  # binary contrast: >=2 days vs 1 day, doubly-robust forest, outcome PRD
  Wb <- as.integer(d$pt_dose_days >= 2)
  y <- suppressWarnings(as.numeric(d$prd_h)); keep <- !is.na(y)
  if (length(unique(Wb[keep])) == 2) {
    cf <- causal_forest(X[keep, ], y[keep], Wb[keep], num.trees = 2000, seed = 1)
    a <- average_treatment_effect(cf, target.sample = "all")
    rows[["bin"]] <- data.frame(analysis = "dose_ge2_vs_1", cohort = cohort, outcome = "prd_h",
      n = sum(keep), n_vpt = sum(Wb[keep]),
      effect = a[[1]], se = a[[2]], ci_low = a[[1]] - 1.96 * a[[2]], ci_high = a[[1]] + 1.96 * a[[2]])
  }
  # linear trend across tiers, covariate-adjusted GLM (slope per extra day)
  fit <- try(glm(prd_h ~ pt_dose_days + ., data = data.frame(prd_h = y, pt_dose_days = d$pt_dose_days, X)[keep, ],
                 family = binomial()), silent = TRUE)
  if (!inherits(fit, "try-error")) {
    co <- summary(fit)$coefficients
    b <- co["pt_dose_days", "Estimate"]; sb <- co["pt_dose_days", "Std. Error"]
    rows[["trend"]] <- data.frame(analysis = "dose_trend_logOR_per_day", cohort = cohort, outcome = "prd_h",
      n = sum(keep), n_vpt = NA_integer_,
      effect = b, se = sb, ci_low = b - 1.96 * sb, ci_high = b + 1.96 * sb)
  }
  # incidence by tier, for transparency
  inc <- tapply(y[keep], d$pt_dose_days[keep], mean)
  cat(sprintf("  %s PRD incidence by dose tier: %s\n", cohort,
              paste(sprintf("%dd=%.3f", as.integer(names(inc)), inc), collapse = "  ")))
  do.call(rbind, rows)
}

res <- do.call(rbind, c(lapply(c("mimic", "eicu"), grace),
                        lapply(c("mimic", "eicu"), dose)))
write.csv(res, "outputs/vpt_sensitivity_results.csv", row.names = FALSE)
cat("\n=== VPT sensitivity (grace period 48h; dose-response) ===\n")
print(format(res, digits = 3))
cat("\ngrace_48h effect: positive = VPT increases risk vs comparator.\n",
    "dose_ge2_vs_1 effect: positive = more pip-tazo days increase PRD.\n",
    "dose_trend_logOR_per_day: log-odds of PRD per additional pip-tazo day.\n", sep = "")
cat("Written: outputs/vpt_sensitivity_results.csv\n")
