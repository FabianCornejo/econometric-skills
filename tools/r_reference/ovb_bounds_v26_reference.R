# Reference numbers for specification.ovb_bounds (v2.6), from R's sensemakr (Cinelli and Hazlett 2020).
#
# Runs once, on a machine with R; the numbers it writes are fixed in tests/test_v26.py with this file as their
# origin. Two parts:
#   1. Darfur, the package's own example (data shipped with sensemakr). The data are NOT written out: sensemakr is
#      GPL-3 and this repository is MIT, so only summary statistics and the bounds go to the output (the Python
#      test feeds those statistics through the numeric core of ovb_bounds).
#   2. tests/data/ovb_v26.csv (built by make_ovb_v26_data.py): the whole function, from a fitted model.
#
# Run from the repository root:
#     Rscript tools/r_reference/ovb_bounds_v26_reference.R
# Packages (installed from CRAN into R_LIBS_USER if missing): sensemakr, jsonlite.
# Output: tools/r_reference/output/ovb_bounds_v26_reference.txt (everything printed) and .json (the numbers).

pkgs <- c("sensemakr", "jsonlite")
user_lib <- Sys.getenv("R_LIBS_USER")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, lib = user_lib, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(sensemakr)
  library(jsonlite)
})

here <- "tools/r_reference"
dir.create(file.path(here, "output"), showWarnings = FALSE)
txt <- file.path(here, "output", "ovb_bounds_v26_reference.txt")
sink(txt, split = TRUE)
cat("R:", R.version.string, "\n")
cat("sensemakr:", as.character(packageVersion("sensemakr")), "\n")
csv <- "tests/data/ovb_v26.csv"
cat("ovb_v26.csv MD5:", unname(tools::md5sum(csv)), "\n\n")

bounds_list <- function(b) {
  b <- as.data.frame(b)
  lapply(seq_len(nrow(b)), function(i) as.list(b[i, , drop = FALSE]))
}
run <- function(label, model, treatment, bench, kd, ky = kd) {
  cat("---", label, "\n")
  w <- NULL
  b <- withCallingHandlers(
    ovb_bounds(model, treatment = treatment, benchmark_covariates = bench, kd = kd, ky = ky),
    warning = function(cnd) { w <<- c(w, conditionMessage(cnd)); invokeRestart("muffleWarning") })
  print(as.data.frame(b))
  if (!is.null(w)) cat("WARNING:", w, sep = "\n  ")
  list(case = label, kd = kd, ky = ky, warnings = if (is.null(w)) list() else as.list(w), bounds = bounds_list(b))
}
err <- function(label, model, treatment, bench, kd) {
  m <- tryCatch({ ovb_bounds(model, treatment = treatment, benchmark_covariates = bench, kd = kd); "no error" },
                error = function(e) conditionMessage(e))
  cat("---", label, "| error:", m, "\n")
  list(case = label, kd = kd, error = m)
}
stats_of <- function(model, terms) {
  co <- coef(summary(model))
  lapply(setNames(terms, terms), function(v) list(estimate = co[v, "Estimate"], se = co[v, "Std. Error"],
                                                  t = co[v, "t value"]))
}

res <- list()

# 1. Darfur ------------------------------------------------------------------------------------------------------
data("darfur", package = "sensemakr")
f_y <- peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village
f_d <- directlyharmed ~ age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village
my <- lm(f_y, data = darfur)
md <- lm(f_d, data = darfur)
occ <- c("farmer_dar", "herder_dar")
res$darfur <- list(
  n = nobs(my), df_outcome = df.residual(my), df_treatment = df.residual(md),
  treatment = stats_of(my, "directlyharmed"),
  outcome_t = stats_of(my, c("female", "pastvoted", occ)),
  treatment_t = stats_of(md, c("female", "pastvoted", occ)),
  r2yxj = as.list(partial_r2(my, covariates = c("female", "pastvoted"))),
  r2dxj = as.list(partial_r2(md, covariates = c("female", "pastvoted"))),
  group_r2y_occupation = group_partial_r2(my, covariates = occ),
  group_r2d_occupation = group_partial_r2(md, covariates = occ),
  runs = list(
    run("darfur female kd 1:3", my, "directlyharmed", "female", 1:3),
    run("darfur pastvoted kd 1:3", my, "directlyharmed", "pastvoted", 1:3),
    run("darfur occupation group kd 1:2", my, "directlyharmed", list(occupation = occ), 1:2),
    run("darfur female kd != ky", my, "directlyharmed", "female", c(1, 2), c(2, 1)),
    run("darfur female cap kd 8", my, "directlyharmed", "female", 8),
    run("darfur female cap kd 9", my, "directlyharmed", "female", 9),
    err("darfur female impossible kd 120", my, "directlyharmed", "female", 120)))

# README table, for the record (the published numbers this run must reproduce)
cat("\nREADME check (1x/2x/3x female): R2dz.x 0.0092/0.0183/0.0275, R2yz.dx 0.1246/0.2493/0.3741,",
    "adjusted estimate 0.0752/0.0529/0.0304, adjusted se 0.0219/0.0204/0.0187\n\n")

# 2. Simulated data, from the fitted model -----------------------------------------------------------------------
sim <- read.csv(csv)
sim$g <- factor(sim$g, levels = c("a", "b", "c", "d"))
ms <- lm(y ~ d + x_strong + x_weak + g, data = sim)
gd <- c("gb", "gc", "gd")
res$sim <- list(
  n = nobs(ms), df_outcome = df.residual(ms), treatment = stats_of(ms, "d"),
  runs = list(
    run("sim x_strong kd 1:3", ms, "d", "x_strong", 1:3),
    run("sim x_weak kd 1:3", ms, "d", "x_weak", 1:3),
    run("sim g group kd 1:2", ms, "d", list(g = gd), 1:2),
    run("sim x_weak kd != ky", ms, "d", "x_weak", c(1, 3), c(2, 0.5))))

sink()
write(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE),
      file.path(here, "output", "ovb_bounds_v26_reference.json"))
cat("\nWrote", txt, "and the .json next to it.\n")
