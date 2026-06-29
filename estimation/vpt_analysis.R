#!/usr/bin/env Rscript
# ============================================================================
# Well-defined target trial: vancomycin + piperacillin-tazobactam (VPT) versus
# vancomycin + alternative anti-pseudomonal beta-lactam, on MAKE-H and on the
# persistent-renal-dysfunction component. Doubly-robust causal forest, average
# treatment effect (all and treated). Both cohorts in one run.
#
#   Rscript estimation/vpt_analysis.R
# ============================================================================

suppressPackageStartupMessages({ library(arrow); library(grf) })
set.seed(20260617)

Xvars <- c("anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
           "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission",
           "map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
           "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
           "spo2_last_1", "gcs_total_1")

run <- function(cohort) {
  d <- as.data.frame(read_parquet(sprintf("outputs/%s_vpt.parquet", cohort)))
  lg <- sapply(d, is.logical); d[lg] <- lapply(d[lg], as.integer)
  for (v in Xvars) { x <- suppressWarnings(as.numeric(d[[v]])); x[is.na(x)] <- stats::median(x, na.rm = TRUE); d[[v]] <- x }
  X <- as.matrix(d[, Xvars]); W <- as.integer(d$W)
  rows <- list()
  for (oc in c("make_h", "prd_h")) {
    y <- suppressWarnings(as.numeric(d[[oc]])); keep <- !is.na(y)
    cf <- causal_forest(X[keep, ], y[keep], W[keep], num.trees = 2000, seed = 1)
    ate <- average_treatment_effect(cf, target.sample = "all")
    rows[[oc]] <- data.frame(cohort = cohort, outcome = oc, n = sum(keep), n_vpt = sum(W[keep]),
      ate_vpt_minus_comparator = ate[[1]], se = ate[[2]],
      ci_low = ate[[1]] - 1.96 * ate[[2]], ci_high = ate[[1]] + 1.96 * ate[[2]])
  }
  do.call(rbind, rows)
}

res <- do.call(rbind, lapply(c("mimic", "eicu"), run))
write.csv(res, "outputs/vpt_results.csv", row.names = FALSE)
cat("\n=== VPT vs alternative beta-lactam (positive = VPT increases risk) ===\n")
print(format(res, digits = 3))
cat("\nWritten: outputs/vpt_results.csv\n")
