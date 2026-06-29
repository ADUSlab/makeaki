# ============================================================================
# Fast (medium library) eICU validation, run from R or RStudio.
#
#   - RStudio: open this file and click "Source".
#   - R console: source("/Volumes/mac1/paper7/code/estimation/run_med_eicu.R")
#
# Uses the eICU panel with the medium learner library (glm + glmnet, no ranger/
# xgboost) for a quick directional read. Results: outputs/eicu_pif_results_med.csv
# ============================================================================

repo <- "/Volumes/mac1/paper7/code"
if (dir.exists(repo)) setwd(repo) else stop("Edit 'repo' to your code folder.")

Sys.unsetenv("LMTP_QUICK")
Sys.setenv(LMTP_COHORT = "eicu")
Sys.setenv(LMTP_MED = "1")
Sys.setenv(LMTP_WORKERS = "1")  # sequential: lowest memory, avoids out-of-memory crashes

cat("Working directory:", getwd(), "\n")
cat("Running eICU validation (medium library) ...\n")
t_start <- Sys.time()
source("estimation/lmtp_pif.R", echo = FALSE)
cat("\nTotal run time: ",
    round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1),
    " minutes\n", sep = "")
