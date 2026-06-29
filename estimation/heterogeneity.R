#!/usr/bin/env Rscript
# ============================================================================
# Heterogeneity (CATE) via causal forests for the reducible-burden study.
#
# Produces, for each policy:
#   - the average policy benefit in the whole cohort,
#   - the among-exposed (ATT) benefit, which compares effect SIZE fairly across
#     cohorts (independent of how prevalent the exposure is),
#   - individual patient-level benefits,
# and, for the nephrotoxin policy, the benefit by candidate subgroup (to find
# trial-ready subgroups).
#
# causal_forest uses all cores and is memory-light (no out-of-memory risk).
#
# Run from the code/ folder:
#   LMTP_COHORT=mimic Rscript estimation/heterogeneity.R
#   LMTP_COHORT=eicu  Rscript estimation/heterogeneity.R
# ============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(grf)
})
set.seed(20260617)

COHORT <- Sys.getenv("LMTP_COHORT", "mimic")
panel <- as.data.frame(read_parquet(sprintf("outputs/%s_aki_panel.parquet", COHORT)))
panel <- panel[panel$make_h_classifiable == TRUE, ]
lg <- sapply(panel, is.logical)
panel[lg] <- lapply(panel[lg], as.integer)
K <- 5

Xbase <- c(
  "anchor_age", "female", "cm_ckd", "cm_diabetes", "cm_heart_failure",
  "baseline_creatinine", "baseline_egfr", "aki_max_stage", "aki_present_at_admission"
)
Xearly <- c(
  "map_last_1", "creat_last_1", "lact_last_1", "uo_rate_1", "vaso_1", "vent_1",
  "platelet_last_1", "bilirubin_last_1", "pao2_last_1", "fio2_last_1",
  "spo2_last_1", "gcs_total_1"
)
for (v in c(Xbase, Xearly)) {
  x <- suppressWarnings(as.numeric(panel[[v]]))
  x[is.na(x)] <- stats::median(x, na.rm = TRUE)
  panel[[v]] <- x
}
X <- as.matrix(panel[, c(Xbase, Xearly)])
Y <- panel$make_h

# Collapse each longitudinal policy exposure to a 0-72h binary: exposed in any
# window. The policy benefit is the effect of that exposure on MAKE-H (removing
# the exposure is the intervention), so benefit = +tau.
ever <- function(stub) {
  m <- sapply(1:K, function(k) {
    v <- suppressWarnings(as.numeric(panel[[paste0(stub, "_", k)]]))
    ifelse(is.na(v), 0, v) > 0
  })
  as.integer(rowSums(m) > 0)
}

policies <- list(g_F = "a_fluid", g_MAP = "a_map", g_NTX = "a_ntx")
rows <- list()
benefits <- data.frame(hadm_id = panel$hadm_id)
forests <- list()

for (nm in names(policies)) {
  W <- ever(policies[[nm]])
  cf <- causal_forest(X, Y, W, num.trees = 2000, seed = 1)
  forests[[nm]] <- cf
  tau <- predict(cf)$predictions
  benefits[[paste0("benefit_", nm)]] <- tau
  ate <- average_treatment_effect(cf, target.sample = "all")
  att <- average_treatment_effect(cf, target.sample = "treated")
  rows[[nm]] <- data.frame(
    policy = nm,
    n_exposed = sum(W),
    pct_exposed = round(100 * mean(W), 1),
    benefit_all = ate[[1]], benefit_all_se = ate[[2]],
    benefit_exposed = att[[1]], benefit_exposed_se = att[[2]],
    benefit_exposed_ci_low = att[[1]] - 1.96 * att[[2]],
    benefit_exposed_ci_high = att[[1]] + 1.96 * att[[2]]
  )
}
summ <- do.call(rbind, rows)
write.csv(summ, sprintf("outputs/%s_hetero_summary.csv", COHORT), row.names = FALSE)
write_parquet(benefits, sprintf("outputs/%s_individual_benefits.parquet", COHORT))

# Subgroup benefit for the nephrotoxin policy (the confirmatory-significant one).
cf <- forests[["g_NTX"]]
subs <- list(
  "AKI stage 1" = panel$aki_max_stage == 1,
  "AKI stage 2" = panel$aki_max_stage == 2,
  "AKI stage 3" = panel$aki_max_stage == 3,
  "CKD" = panel$cm_ckd == 1,
  "No CKD" = panel$cm_ckd == 0,
  "Diabetes" = panel$cm_diabetes == 1,
  "Heart failure" = panel$cm_heart_failure == 1,
  "Age >= 65" = panel$anchor_age >= 65,
  "Age < 65" = panel$anchor_age < 65,
  "Female" = panel$female == 1,
  "Male" = panel$female == 0,
  "Vasopressor (w1)" = panel$vaso_1 == 1,
  "Ventilated (w1)" = panel$vent_1 == 1
)
sr <- list()
for (nm in names(subs)) {
  idx <- which(subs[[nm]])
  if (length(idx) < 50) next
  e <- average_treatment_effect(cf, subset = idx)
  sr[[nm]] <- data.frame(
    subgroup = nm, n = length(idx),
    benefit = e[[1]], se = e[[2]],
    ci_low = e[[1]] - 1.96 * e[[2]], ci_high = e[[1]] + 1.96 * e[[2]]
  )
}
sg <- do.call(rbind, sr)
sg <- sg[order(-sg$benefit), ]
write.csv(sg, sprintf("outputs/%s_hetero_subgroups.csv", COHORT), row.names = FALSE)

cat("\n=== Heterogeneity summary (", COHORT, ") ===\n", sep = "")
print(format(summ, digits = 3))
cat("\n=== g_NTX benefit by subgroup (sorted, high to low) ===\n")
print(format(sg, digits = 3))
cat("\nWritten: outputs/", COHORT, "_hetero_summary.csv, _hetero_subgroups.csv, ",
    "_individual_benefits.parquet\n", sep = "")
