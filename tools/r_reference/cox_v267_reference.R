# Reference for survival.cox's robust (Lin-Wei) SEs, with tied event times (v2.6.7).
# R survival::coxph handles ties by Efron or Breslow in the score residuals of its robust variance.
# Data: tests/data/cox_v267_ties.csv (built by make_cox_v267_data.py). Run once, from the repository root:
#   Rscript tools/r_reference/cox_v267_reference.R
# Writes tools/r_reference/output/cox_v267_reference.json (and .txt). Base R only (survival ships with R).
library(survival)
d <- read.csv("tests/data/cox_v267_ties.csv")
cases <- list(
  efron_robust          = list(f = Surv(t, e) ~ x + w,              ties = "efron",   cl = "id"),
  breslow_robust        = list(f = Surv(t, e) ~ x + w,              ties = "breslow", cl = "id"),
  efron_cluster         = list(f = Surv(t, e) ~ x + w,              ties = "efron",   cl = "g"),
  efron_strata_robust   = list(f = Surv(t, e) ~ x + w + strata(s),  ties = "efron",   cl = "id"),
  efron_strata_cluster  = list(f = Surv(t, e) ~ x + w + strata(s),  ties = "efron",   cl = "g"),
  breslow_strata_cluster= list(f = Surv(t, e) ~ x + w + strata(s),  ties = "breslow", cl = "g"),
  continuous_efron      = list(f = Surv(tc, e) ~ x + w,             ties = "efron",   cl = "id")
)
num <- function(v) paste0("[", paste(sprintf("%.15g", v), collapse = ", "), "]")
out <- c()
txt <- c(R.version.string, paste("survival", as.character(packageVersion("survival"))))
for (nm in names(cases)) {
  cs <- cases[[nm]]
  fit <- coxph(cs$f, data = d, ties = cs$ties, cluster = d[[cs$cl]], control = coxph.control(eps = 1e-10, iter.max = 100))
  out <- c(out, sprintf('  "%s": {"ties": "%s", "cluster": "%s", "coef": %s, "robust_se": %s, "naive_se": %s}',
                        nm, cs$ties, cs$cl, num(coef(fit)), num(sqrt(diag(fit$var))), num(sqrt(diag(fit$naive.var)))))
  txt <- c(txt, paste("==", nm), capture.output(print(summary(fit)$coefficients)))
}
json <- c("{", sprintf('  "R": "%s", "survival": "%s",', R.version.string, as.character(packageVersion("survival"))),
          paste(out, collapse = ",\n"), "}")
dir.create("tools/r_reference/output", showWarnings = FALSE)
writeLines(json, "tools/r_reference/output/cox_v267_reference.json")
writeLines(txt, "tools/r_reference/output/cox_v267_reference.txt")
cat(json, sep = "\n")
