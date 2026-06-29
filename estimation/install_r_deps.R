#!/usr/bin/env Rscript
# Install the R dependencies for the estimation stage.
# Run once:  Rscript estimation/install_r_deps.R

repos <- "https://cloud.r-project.org"
pkgs <- c("arrow", "lmtp", "SuperLearner", "glmnet", "ranger", "xgboost", "future", "grf")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Installing ", p, " ...")
    install.packages(p, repos = repos)
  } else {
    message(p, " already installed.")
  }
}
message("Done. R dependencies are ready.")
