# One-off installer for every R package used in this repository.
# Run once:  Rscript install_r_packages.R
pkgs <- c(
  "dplyr", "psych", "fitdistrplus", "MASS", "ggplot2", "sampling", "DescTools", "boot",
  "car", "rstatix", "Hmisc", "corrplot", "broom", "Matrix", "FactoMineR", "factoextra",
  "optimx", "nloptr", "MonteCarlo", "forecast", "tseries", "lattice", "plotly", "htmlwidgets",
  "caret", "recipes", "agricolae", "vcd", "coin", "rsample", "infer", "vegan", "cluster",
  "mice", "Amelia", "VIM", "rmarkdown", "knitr", "gt"
)
missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
still_missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(still_missing)) cat("Could not install:", paste(still_missing, collapse = ", "), "\n") else cat("All", length(pkgs), "packages are installed.\n")
