#!/usr/bin/env Rscript
# ============================================================================
# Secondary analyses: nephrotoxin-avoidance effect on the individual MAKE-H
# components (in-hospital death, RRT dependence, persistent renal dysfunction)
# to show whether the benefit is renal rather than mortality-driven.
#
# Doubly-robust causal forest, among-exposed (ATT) effect. Fast (minutes).
#
# Run from the code/ folder (does both cohorts in one launch):
#   Rscript estimation/secondary.R
# ============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(grf)
})
set.seed(20260617)

Xbase <- c("anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
           "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission")
Xearly <- c("map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
            "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
            "spo2_last_1", "gcs_total_1")
K <- 5
outcomes_cols <- c("make_h", "death_h", "rrt_h", "prd_h")

run_cohort <- function(cohort) {
  panel <- as.data.frame(read_parquet(sprintf("outputs/%s_aki_panel.parquet", cohort)))
  panel <- panel[panel$make_h_classifiable == TRUE, ]
  oc <- as.data.frame(read_parquet(sprintf("outputs/%s_aki_cohort_outcomes.parquet", cohort)))
  oc <- oc[, c("hadm_id", "death_h", "rrt_h", "prd_h")]
  panel <- merge(panel, oc, by = "hadm_id", all.x = TRUE)

  lg <- sapply(panel, is.logical); panel[lg] <- lapply(panel[lg], as.integer)
  for (v in c(Xbase, Xearly)) {
    x <- suppressWarnings(as.numeric(panel[[v]])); x[is.na(x)] <- stats::median(x, na.rm = TRUE)
    panel[[v]] <- x
  }
  X <- as.matrix(panel[, c(Xbase, Xearly)])
  W <- as.integer(rowSums(sapply(1:K, function(k) {
    v <- suppressWarnings(as.numeric(panel[[paste0("a_ntx_", k)]])); ifelse(is.na(v), 0, v) > 0
  })) > 0)

  rows <- list()
  for (oc_name in outcomes_cols) {
    y <- suppressWarnings(as.numeric(panel[[oc_name]]))
    keep <- !is.na(y)
    cf <- causal_forest(X[keep, ], y[keep], W[keep], num.trees = 2000, seed = 1)
    att <- average_treatment_effect(cf, target.sample = "treated")
    rows[[oc_name]] <- data.frame(
      cohort = cohort, outcome = oc_name, n = sum(keep),
      benefit_exposed = att[[1]], se = att[[2]],
      ci_low = att[[1]] - 1.96 * att[[2]], ci_high = att[[1]] + 1.96 * att[[2]]
    )
  }
  do.call(rbind, rows)
}

res <- do.call(rbind, lapply(c("mimic", "eicu"), run_cohort))
write.csv(res, "outputs/make_components_secondary.csv", row.names = FALSE)
cat("\n=== Nephrotoxin-avoidance benefit on MAKE-H components (among exposed) ===\n")
print(format(res, digits = 3))
cat("\nWritten: outputs/make_components_secondary.csv\n")
