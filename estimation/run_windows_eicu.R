# ============================================================================
# Full eICU validation runner for Windows (R / RStudio).
#
# Setup for Omer:
#   1. Install R (https://cran.r-project.org) and optionally RStudio.
#   2. Put these three files in one folder, e.g. C:/makeaki :
#         run_windows_eicu.R      (this file)
#         lmtp_pif.R
#         eicu_aki_panel.parquet
#      (optional: mimic_aki_panel.parquet)
#   3. Edit the 'base' line below to that folder.
#   4. In RStudio open this file and click Source, or run:
#         "C:\Program Files\R\R-4.x.x\bin\Rscript.exe" run_windows_eicu.R
#
# It installs the needed packages (first run only), arranges the files, and runs
# the estimation. Result: outputs/eicu_pif_results.csv in the same folder.
# ============================================================================

base <- "C:/makeaki"   # <-- EDIT THIS to your folder (use forward slashes)

stopifnot(dir.exists(base))
setwd(base)

# Install packages on first run (Windows uses fast binary packages).
pkgs <- c("arrow", "lmtp", "SuperLearner", "glmnet", "ranger", "xgboost", "future")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

# Arrange files into the layout the estimation script expects.
dir.create("outputs", showWarnings = FALSE)
dir.create("estimation", showWarnings = FALSE)
file.copy("eicu_aki_panel.parquet", "outputs/eicu_aki_panel.parquet", overwrite = TRUE)
if (file.exists("mimic_aki_panel.parquet")) {
  file.copy("mimic_aki_panel.parquet", "outputs/mimic_aki_panel.parquet", overwrite = TRUE)
}
file.copy("lmtp_pif.R", "estimation/lmtp_pif.R", overwrite = TRUE)

# Configuration.
Sys.unsetenv("LMTP_QUICK")
Sys.unsetenv("LMTP_MED")
Sys.setenv(LMTP_COHORT = "eicu")
# Parallel workers. Each worker copies the data and models in memory, so:
#   16 GB RAM -> 3 or 4 ; 8 GB RAM -> 1 or 2. Lower this if you see a memory error.
Sys.setenv(LMTP_WORKERS = "4")

cat("Working directory:", getwd(), "\n")
cat("Running full eICU validation ...\n")
t_start <- Sys.time()
source("estimation/lmtp_pif.R", echo = FALSE)
cat("\nTotal run time: ",
    round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1),
    " minutes\n", sep = "")
cat("Result written to outputs/eicu_pif_results.csv\n")
