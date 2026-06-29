#!/usr/bin/env Rscript
# ============================================================================
# Transportability: does case-mix explain the smaller nephrotoxin effect in the
# validation cohort? We fit the causal forest in MIMIC-IV, transport the
# individual effects to the eICU-CRD covariate distribution, and compare with the
# effect estimated directly in eICU-CRD.
#
#   Rscript estimation/transport.R
# ============================================================================

suppressPackageStartupMessages({ library(arrow); library(grf) })
set.seed(20260617)

Xvars <- c("anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
           "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission",
           "map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
           "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
           "spo2_last_1", "gcs_total_1")
K <- 5

load_cohort <- function(cohort) {
  d <- as.data.frame(read_parquet(sprintf("outputs/%s_aki_panel.parquet", cohort)))
  d <- d[d$make_h_classifiable == TRUE, ]
  lg <- sapply(d, is.logical); d[lg] <- lapply(d[lg], as.integer)
  for (v in Xvars) { x <- suppressWarnings(as.numeric(d[[v]])); x[is.na(x)] <- stats::median(x, na.rm = TRUE); d[[v]] <- x }
  W <- as.integer(rowSums(sapply(1:K, function(k) {
    v <- suppressWarnings(as.numeric(d[[paste0("a_ntx_", k)]])); ifelse(is.na(v), 0, v) > 0
  })) > 0)
  list(X = as.matrix(d[, Xvars]), Y = as.numeric(d$make_h), W = W)
}

mimic <- load_cohort("mimic")
eicu <- load_cohort("eicu")

cf_m <- causal_forest(mimic$X, mimic$Y, mimic$W, num.trees = 2000, seed = 1)
cf_e <- causal_forest(eicu$X, eicu$Y, eicu$W, num.trees = 2000, seed = 1)

ate_m <- average_treatment_effect(cf_m, target.sample = "all")[[1]]
ate_e <- average_treatment_effect(cf_e, target.sample = "all")[[1]]
# Transport: MIMIC individual effects evaluated on the eICU covariate distribution.
tau_on_e <- predict(cf_m, eicu$X)$predictions
transported <- mean(tau_on_e)

res <- data.frame(
  quantity = c("MIMIC effect (own population)",
               "MIMIC effect transported to eICU case-mix",
               "eICU effect (own population)"),
  ate_make_h = c(ate_m, transported, ate_e)
)
write.csv(res, "outputs/transport_results.csv", row.names = FALSE)
cat("\n=== Transportability of the nephrotoxin effect on MAKE-H (ATE of exposure) ===\n")
print(format(res, digits = 3))
cat("\nIf the transported MIMIC effect approaches the eICU effect, case-mix explains\n",
    "the attenuation; if it stays close to the MIMIC effect, it does not.\n", sep = "")
