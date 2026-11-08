# Reference numbers for specification.oster(exact=True) (v2.5.2), from R's robomit (Schaub; based on Stata's psacalc).
#
# Runs once, on a machine with R; the numbers it writes are fixed in tests/test_v252.py with this file as their
# origin. It reads tests/data/oster_v252.csv (built by make_oster_v252_data.py) and R's built-in mtcars, which it
# also writes out so the Python test uses the same data.
#
# Run from the repository root:
#     Rscript tools/r_reference/oster_v252_reference.R
# Packages (installed from CRAN into R_LIBS_USER if missing): robomit, jsonlite.
# Output: tools/r_reference/output/oster_v252_reference.txt (everything printed), .json (the numbers) and
#         tools/r_reference/output/mtcars.csv.

pkgs <- c("robomit", "jsonlite")
user_lib <- Sys.getenv("R_LIBS_USER")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, lib = user_lib, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(robomit)
  library(jsonlite)
})

here <- "tools/r_reference"
dir.create(file.path(here, "output"), showWarnings = FALSE)
txt <- file.path(here, "output", "oster_v252_reference.txt")
sink(txt, split = TRUE)
cat("R:", R.version.string, "\n")
cat("robomit:", as.character(packageVersion("robomit")), "\n\n")

val <- function(tb, name) as.numeric(tb$Value[tb$Name == name])
opt <- function(tb, name) if (name %in% tb$Name) val(tb, name) else NA

run_beta <- function(label, y, x, con, delta, R2max, data) {
  tb <- suppressWarnings(o_beta(y = y, x = x, con = con, delta = delta, R2max = R2max, type = "lm", data = data))
  cat("---", label, "| o_beta delta =", delta, "R2max =", R2max, "\n"); print(as.data.frame(tb))
  list(case = label, fun = "o_beta", delta = delta, R2max = R2max, beta_star = val(tb, "beta*"),
       alt1 = opt(tb, "Alternative Solution 1"), alt2 = opt(tb, "Alternative Solution 2"),
       b0 = val(tb, "Uncontrolled Coefficient"), b1 = val(tb, "Controlled Coefficient"),
       r0 = val(tb, "Uncontrolled R-square"), r1 = val(tb, "Controlled R-square"))
}
run_delta <- function(label, y, x, con, beta, R2max, data) {
  tb <- suppressWarnings(o_delta(y = y, x = x, con = con, beta = beta, R2max = R2max, type = "lm", data = data))
  cat("---", label, "| o_delta beta =", beta, "R2max =", R2max, "\n"); print(as.data.frame(tb))
  list(case = label, fun = "o_delta", beta = beta, R2max = R2max, delta_star = val(tb, "delta*"))
}

res <- list()
# 1. The package's own example: mtcars, mpg on wt with controls hp and qsec.
mt <- mtcars
write.csv(data.frame(mpg = mt$mpg, wt = mt$wt, hp = mt$hp, qsec = mt$qsec),
          file.path(here, "output", "mtcars.csv"), row.names = FALSE)
for (d in c(1, 0.5, 2)) res[[length(res) + 1]] <- run_beta("mtcars", "mpg", "wt", "hp + qsec", d, 0.9, mt)
res[[length(res) + 1]] <- run_delta("mtcars", "mpg", "wt", "hp + qsec", 0, 0.9, mt)

# 2. Simulated proportional selection (true effect 1, delta 2), control w1 only.
sim <- read.csv("tests/data/oster_v252.csv")
for (d in c(0.5, 1, 2)) res[[length(res) + 1]] <- run_beta("sim", "y", "x", "w1", d, 0.885, sim)
res[[length(res) + 1]] <- run_beta("sim", "y", "x", "w1", 2, 1.0, sim)    # one real root (Cardano)
res[[length(res) + 1]] <- run_beta("sim", "y", "x", "w1", 3, 0.86, sim)   # one real root (Cardano)
for (rm in c(0.885, 1.0)) res[[length(res) + 1]] <- run_delta("sim", "y", "x", "w1", 0, rm, sim)
res[[length(res) + 1]] <- run_delta("sim", "y", "x", "w1", 1, 0.885, sim)

sink()
write(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null", pretty = TRUE),
      file.path(here, "output", "oster_v252_reference.json"))
cat("\nWrote", txt, "and the .json and mtcars.csv next to it.\n")
