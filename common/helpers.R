# Shared helpers for every R script.
# RUN ALL R SCRIPTS FROM THE REPOSITORY ROOT, e.g.
#   Rscript topics/01_central_tendency_dispersion/r/analysis.R

CRAN <- "https://cloud.r-project.org"

# TRUE if the package is (or can be made) available; installs it if missing.
opt <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    try(install.packages(pkg, repos = CRAN, quiet = TRUE), silent = TRUE)
  }
  requireNamespace(pkg, quietly = TRUE)
}

# Load packages, installing any that are missing.
need <- function(...) {
  for (p in c(...)) {
    if (!opt(p)) stop("Package '", p, "' could not be installed")
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }
  invisible(TRUE)
}

load_data <- function(name) read.csv(file.path("data", name), stringsAsFactors = FALSE)

# Start a topic: create outputs/, mirror console output into outputs/results_r.txt, fix the seed.
start_topic <- function(folder) {
  OUT <<- file.path("topics", folder, "outputs")
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  sink(file.path(OUT, "results_r.txt"), split = TRUE)
  set.seed(4131)
}
end_topic <- function() sink()

h <- function(title) cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")

# Save a plot to outputs/r_<name>.png. `expr` is evaluated AFTER the device opens.
# For ggplot objects pass print(p).
save_plot <- function(name, expr, width = 1000, height = 650) {
  png(file.path(OUT, paste0("r_", name, ".png")), width = width, height = height, res = 120)
  on.exit(dev.off())
  force(expr)
  invisible(NULL)
}

skewness <- function(v) { m <- mean(v); mean((v - m)^3) / mean((v - m)^2)^1.5 }
get_mode <- function(v) { u <- unique(v); u[which.max(tabulate(match(v, u)))] }
