#!/usr/bin/env Rscript
# run_firth_logistf.R -- generic Firth-penalized logistic regression runner,
# called via Rscript from 53_phase_x_vre_sensitivity_landmark_firth.py.
#
# Why R/logistf and not a Python package: statsmodels has no native Firth
# implementation; `firthlogist` (Python) is not installed in the project's
# 'gpt' conda env (checked 2026-09-02: ModuleNotFoundError). System R already
# has `logistf` installed (checked: library(logistf) loads cleanly) -- used
# here rather than adding a new Python dependency mid-analysis.
#
# Usage: Rscript run_firth_logistf.R <input_csv>
#   input_csv: first column must be named 'y' (binary 0/1 outcome), every
#   remaining column is a predictor, included in the model in file-column
#   order (name the columns exactly as you want them reported).
#
# Prints one line per model term to stdout, machine-parsed by the caller:
#   COEF,<term>,<coef>,<OR>,<CI_low>,<CI_high>,<p>
#   CONV,<last_LL_change>,<max_abs_score>,<last_beta_change>   (fit$conv)
#   N,<n>
# On any fitting failure, prints "FIRTH_FIT_FAILED" and exits non-zero --
# caller must treat this as non-convergence, never crash or fabricate a
# number (same convention as `adjusted_or_logistic()` in script 51).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("FIRTH_FIT_FAILED\n")
  quit(status = 1)
}
in_csv <- args[1]

ok <- requireNamespace("logistf", quietly = TRUE)
if (!ok) {
  cat("FIRTH_FIT_FAILED\n")
  quit(status = 1)
}
suppressMessages(library(logistf))

df <- tryCatch(read.csv(in_csv), error = function(e) NULL)
if (is.null(df) || !("y" %in% colnames(df))) {
  cat("FIRTH_FIT_FAILED\n")
  quit(status = 1)
}
predictors <- setdiff(colnames(df), "y")
if (length(predictors) == 0) {
  cat("FIRTH_FIT_FAILED\n")
  quit(status = 1)
}
formula_str <- paste("y ~", paste(predictors, collapse = " + "))

fit <- tryCatch(
  logistf(as.formula(formula_str), data = df, pl = TRUE),
  error = function(e) NULL,
  warning = function(w) {
    # let warnings through (logistf can warn on marginal cases) but still
    # capture the fit object via withCallingHandlers below if needed
    invokeRestart("muffleWarning")
  }
)
if (is.null(fit)) {
  fit <- tryCatch(
    suppressWarnings(logistf(as.formula(formula_str), data = df, pl = TRUE)),
    error = function(e) NULL
  )
}
if (is.null(fit)) {
  cat("FIRTH_FIT_FAILED\n")
  quit(status = 1)
}

coefs <- fit$coefficients
ors <- exp(coefs)
ci_lo <- exp(fit$ci.lower)
ci_hi <- exp(fit$ci.upper)
pvals <- fit$prob

for (i in seq_along(coefs)) {
  cat(sprintf("COEF,%s,%.8f,%.8f,%.8f,%.8f,%.8g\n",
              names(coefs)[i], coefs[i], ors[i], ci_lo[i], ci_hi[i], pvals[i]))
}
conv <- fit$conv
if (is.null(conv) || length(conv) < 3) conv <- c(NA, NA, NA)
cat(sprintf("CONV,%.8g,%.8g,%.8g\n", conv[1], conv[2], conv[3]))
cat(sprintf("N,%d\n", fit$n))
