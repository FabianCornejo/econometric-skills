# Reference numbers for the v2.5.0 DiD functions (causal.did_imputation, causal.twfe_weights).
#
# Runs once, on a machine with R; the numbers it writes are fixed in tests/test_v250.py with this file as their
# origin. It reads the committed panel tests/data/did_v250_panel.csv (built by make_did_v250_panel.py).
#
# Run from the repository root:
#     Rscript tools/r_reference/did_v250_reference.R
# Packages (installed from CRAN if missing): didimputation (Borusyak, Jaravel & Spiess; R port by Kyle Butts),
# TwoWayFEWeights (de Chaisemartin & D'Haultfoeuille), jsonlite.
# Output: tools/r_reference/output/did_v250_reference.txt (everything printed) and .json (the numbers).

pkgs <- c("didimputation", "TwoWayFEWeights", "jsonlite")
# Install into the user's personal library: R's own library (C:/Program Files/R/...) is not writable without admin
# rights, and Rscript, unlike the R console, does not offer to create the personal one.
user_lib <- Sys.getenv("R_LIBS_USER")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, lib = user_lib, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(didimputation)
  library(TwoWayFEWeights)
  library(jsonlite)
})

here <- "tools/r_reference"
dir.create(file.path(here, "output"), showWarnings = FALSE)
txt <- file.path(here, "output", "did_v250_reference.txt")
sink(txt, split = TRUE)

panel_file <- "tests/data/did_v250_panel.csv"
df <- read.csv(panel_file)
cat("panel:", panel_file, " rows:", nrow(df), "\n")
panel_md5 <- unname(tools::md5sum(panel_file))
cat("md5 of the panel:", panel_md5, "(expected '7498036256b9e987f01997bc34a57ea0'; a different value means the file changed, e.g. CRLF line endings)\n")
cat("\nR:", R.version.string, "\n")
for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")

out <- list(panel = panel_file, panel_md5 = panel_md5, R = R.version.string,
            versions = lapply(setNames(pkgs, pkgs), function(p) as.character(packageVersion(p))))
grab <- function(x) tryCatch(as.data.frame(x), error = function(e) paste("error:", conditionMessage(e)))
# print every row; a failure to print must not stop the run (v2.5.0: print(x, n = 50) failed on a data.table)
show_df <- function(x) tryCatch(print(as.data.frame(x), digits = 10), error = function(e) print(x))

# ---- 1. Imputation estimator (balanced panel), never-treated coded as g = 0
cat("\n==== didimputation: static ATT, SEs clustered by unit ====\n")
s1 <- tryCatch(did_imputation(data = df, yname = "y", gname = "g", tname = "t", idname = "id"),
               error = function(e) e)
show_df(s1); out$bjs_static <- grab(s1)

cat("\n==== didimputation: event study (all horizons) with pre-trend coefficients ====\n")
s2 <- tryCatch(did_imputation(data = df, yname = "y", gname = "g", tname = "t", idname = "id",
                              horizon = TRUE, pretrends = TRUE), error = function(e) e)
show_df(s2); out$bjs_event <- grab(s2)

cat("\n==== didimputation: static ATT, SEs clustered by state ====\n")
s3 <- tryCatch(did_imputation(data = df, yname = "y", gname = "g", tname = "t", idname = "id",
                              cluster_var = "state"), error = function(e) e)
show_df(s3); out$bjs_static_state <- grab(s3)

cat("\n==== didimputation: unbalanced version (keep_unb == 1) ====\n")
du <- df[df$keep_unb == 1, ]
s4 <- tryCatch(did_imputation(data = du, yname = "y", gname = "g", tname = "t", idname = "id"),
               error = function(e) e)
show_df(s4); out$bjs_static_unbalanced <- grab(s4)

# ---- 2. TWFE weights (de Chaisemartin & D'Haultfoeuille 2020), binary and non-binary treatment
tw <- function(D) {
  r <- tryCatch(twowayfeweights(df, Y = "y", G = "id", T = "t", D = D, type = "feTR", summary_measures = TRUE),
                error = function(e) e)
  tryCatch(print(r), error = function(e) cat("print failed:", conditionMessage(e), "\n"))
  cat("\n-- str of the result --\n"); tryCatch(str(r, max.level = 2), error = function(e) NULL)
  r
}
cat("\n==== TwoWayFEWeights: binary treatment d (feTR) ====\n")
w1 <- tw("d"); out$twfe_weights_binary <- tryCatch(lapply(unclass(w1), function(z) tryCatch(as.data.frame(z), error = function(e) z)), error = function(e) paste("error:", conditionMessage(e)))
cat("\n==== TwoWayFEWeights: non-binary treatment dose (feTR) ====\n")
w2 <- tw("dose"); out$twfe_weights_dose <- tryCatch(lapply(unclass(w2), function(z) tryCatch(as.data.frame(z), error = function(e) z)), error = function(e) paste("error:", conditionMessage(e)))

# TWFE coefficients computed directly, as an anchor for the weights
out$twfe_beta_binary <- unname(coef(lm(y ~ d + factor(id) + factor(t), data = df))["d"])
out$twfe_beta_dose <- unname(coef(lm(y ~ dose + factor(id) + factor(t), data = df))["dose"])
cat("\nTWFE beta (lm with unit and period dummies): d =", out$twfe_beta_binary, " dose =", out$twfe_beta_dose, "\n")

sink()
json <- file.path(here, "output", "did_v250_reference.json")
write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE, force = TRUE), json)
cat("wrote", txt, "and", json, "\n")
