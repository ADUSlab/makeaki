#!/usr/bin/env Rscript
# ============================================================================
# Falsification with negative-control exposures. Acetaminophen and pantoprazole
# have no plausible causal effect on the kidney; a confidence interval excluding
# zero indicates residual confounding. We apply the test to BOTH the composite
# (MAKE-H) and the persistent-renal-dysfunction component (PRD), so the surviving
# renal signal is held to the same falsification standard as the composite.
# Doubly-robust causal forest, both cohorts.
#
#   Rscript estimation/negcontrol_analysis.R
# ============================================================================

suppressPackageStartupMessages({ library(arrow); library(grf) })
set.seed(20260617)

Xvars <- c("anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
           "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission",
           "map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
           "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
           "spo2_last_1", "gcs_total_1")
controls <- c("nc_acetaminophen", "nc_pantoprazole")
outcomes <- c("make_h", "prd_h")

run <- function(cohort) {
  d <- as.data.frame(read_parquet(sprintf("outputs/%s_negcontrol.parquet", cohort)))
  lg <- sapply(d, is.logical); d[lg] <- lapply(d[lg], as.integer)
  for (v in Xvars) { x <- suppressWarnings(as.numeric(d[[v]])); x[is.na(x)] <- stats::median(x, na.rm = TRUE); d[[v]] <- x }
  Xall <- as.matrix(d[, Xvars])
  rows <- list()
  for (oc in outcomes) {
    y <- suppressWarnings(as.numeric(d[[oc]])); keepY <- !is.na(y)
    for (w in controls) {
      W <- as.integer(d[[w]]); keep <- keepY & !is.na(W)
      if (length(unique(W[keep])) < 2) next
      cf <- causal_forest(Xall[keep, ], y[keep], W[keep], num.trees = 2000, seed = 1)
      ate <- average_treatment_effect(cf, target.sample = "all")
      rows[[paste(oc, w)]] <- data.frame(cohort = cohort, outcome = oc,
        negative_control = sub("nc_", "", w), n = sum(keep), n_exposed = sum(W[keep]),
        apparent_effect = ate[[1]], se = ate[[2]],
        ci_low = ate[[1]] - 1.96 * ate[[2]], ci_high = ate[[1]] + 1.96 * ate[[2]])
    }
  }
  do.call(rbind, rows)
}

res <- do.call(rbind, lapply(c("mimic", "eicu"), run))
write.csv(res, "outputs/negcontrol_results.csv", row.names = FALSE)
cat("\n=== Negative-control exposures on MAKE-H and PRD (should be ~0; non-zero signals confounding) ===\n")
print(format(res, digits = 3))
cat("\nWritten: outputs/negcontrol_results.csv\n")
