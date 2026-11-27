"""Documented changes to econometrics/SKILL.md, release by release, as (now, then) pairs. A test that pins the hash of an
earlier SKILL.md undoes exactly these pairs (each must occur once) and compares, so it keeps proving that nothing
else changed (lesson of v2.5.1: never re-pin a hash). Not part of the package.
"""

ROW_GMM = ("| Moment conditions (an Euler equation, a structural equation with instruments): GMM (a linear IV "
           "equation: `diagnostics.iv_report`, row \"Panel data, endogeneity, instruments\"; a dynamic panel: "
           "`systems.dynamic_panel`), Hansen's J, a rank test of whether the moments can identify the parameters; a "
           "likelihood ratio between nested fits with restrictions of any form | `estimation-theory.md` §7–8 | "
           "`estimation.gmm`, `lr_test` |\n")
ROW_SEQ = ("| Data arriving one at a time (an A/B test, quality control, a trial): Wald's sequential test of a simple "
           "against a simple hypothesis, which stops early; the error of peeking at a fixed-sample test and the "
           "per-look level that keeps it; sample size from divergences | `statistics-foundations.md` §14 | "
           "`sequential.sprt_design`, `sprt`, `peeking_size`, `n_for_error`, `divergence` |\n")
CARD_KEYS = (" Its keys are exact: `question`, `estimand`, `design`, `assumption`, `y`, `treatment`, `X`, `W`, `Z`, "
             "`data`, `threats`, `robustness_plan`, `alpha`, `family_of_tests` (the family of section 4), `mde` "
             "(forecasts: `horizon`, `loss`, `benchmarks`, `window`).")
EST_GMM = ("; generic GMM from user-written moments (one-step, two-step, iterated; robust or HAC weight; Hansen's J with "
           "its right-tail p-value; a rank test of identification), likelihood ratio between nested `mle` fits (with the "
           "50:50 boundary mixture for one restriction)")
ROW_SEQ_SCRIPT = ("| `sequential.py` | Wald's SPRT (design with simulated errors and sample sizes, and a run on data in "
                  "arrival order), the type I error of peeking at a fixed-sample test and the Pocock per-look level, KL, "
                  "Jensen-Shannon, Chernoff and Bhattacharyya divergences, KL rates between Markov chains or VAR(1) "
                  "models, sample size of the likelihood-ratio test for target errors, small error probabilities by "
                  "importance sampling |\n")

# v2.8.1 (CHANGELOG v2.8.1): description, two rows in §2, the card keys in §3, estimation.py and sequential.py in §6.
V281 = (
    ("power and sample size, A/B peeking and SPRT, any regression (OLS, IV, panel, logit/probit), GMM, effect of X on Y,",
     "power and sample size, any regression (OLS, IV, panel, logit/probit), effect of X on Y,"),
    ("`wild_cluster_bootstrap` |\n" + ROW_GMM + ROW_SEQ, "`wild_cluster_bootstrap` |\n"),
    ("and a reason." + CARD_KEYS, "and a reason."),
    ("wild cluster bootstrap-t with CI" + EST_GMM + " |\n" + ROW_SEQ_SCRIPT, "wild cluster bootstrap-t with CI |\n"),
)


def undo(text: str, pairs) -> tuple[str, list]:
    """Undo each (now, then) pair once. Returns the text and the pairs that did not occur exactly once."""
    bad = [now for now, _ in pairs if text.count(now) != 1]
    for now, then in pairs:
        text = text.replace(now, then, 1)
    return text, bad
