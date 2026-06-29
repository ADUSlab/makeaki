# ============================================================================
# Run the full confirmatory PIF estimation from within R or RStudio.
#
# How to use:
#   - In RStudio: open this file and click "Source" (top-right of the editor).
#   - In an R console: source("/Volumes/mac1/paper7/code/estimation/run_full.R")
#
# This sets the working directory to the repository, forces the full learner
# library (no quick or medium mode), and runs estimation/lmtp_pif.R.
# Results are written to outputs/mimic_pif_results.csv.
# ============================================================================

repo <- "/Volumes/mac1/paper7/code"
if (dir.exists(repo)) {
  setwd(repo)
} else {
  stop("Repository path not found; edit 'repo' in run_full.R to your code folder.")
}

# Force the full prespecified configuration.
Sys.unsetenv("LMTP_QUICK")
Sys.unsetenv("LMTP_MED")

cat("Working directory:", getwd(), "\n")
cat("Starting full LTMLE estimation (this can take a while with ranger) ...\n")

t_start <- Sys.time()
source("estimation/lmtp_pif.R", echo = FALSE)
cat("\nTotal run time: ",
    round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1),
    " minutes\n", sep = "")
