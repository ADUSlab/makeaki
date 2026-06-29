# ============================================================================
# Run the full confirmatory PIF estimation on the eICU-CRD cohort.
#
#   - In RStudio: open this file and click "Source".
#   - In a console: source("/Volumes/mac1/paper7/code/estimation/run_full_eicu.R")
#   - From a shell: caffeinate -i Rscript estimation/run_full_eicu.R
#
# Forces the eICU panel and the full learner library. Results go to
# outputs/eicu_pif_results.csv.
# ============================================================================

repo <- "/Volumes/mac1/paper7/code"
if (dir.exists(repo)) setwd(repo) else stop("Edit 'repo' to your code folder.")

Sys.unsetenv("LMTP_QUICK")
Sys.unsetenv("LMTP_MED")
Sys.setenv(LMTP_COHORT = "eicu")
Sys.setenv(LMTP_WORKERS = "1")  # sequential: avoids out-of-memory crashes (slower)

cat("Working directory:", getwd(), "\n")
cat("Running eICU validation (full library) ...\n")
t_start <- Sys.time()
source("estimation/lmtp_pif.R", echo = FALSE)
cat("\nTotal run time: ",
    round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1),
    " minutes\n", sep = "")
