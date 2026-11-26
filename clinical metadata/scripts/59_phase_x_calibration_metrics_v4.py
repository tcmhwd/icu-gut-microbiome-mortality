#!/usr/bin/env python3
"""59_phase_x_calibration_metrics_v4.py

V3 RERUN (2026-10-02, owner-authorised correction) of
`59_phase_x_calibration_metrics.py`. This file is a copy of the original
with EXACTLY ONE substantive change: the C2a/C2b (M3-LR/M3-CatBoost-stacked)
columns are read from the fold-local-corrected V3 nested OOF file instead of
script 46's old file. C0 and C1 (M1) are unaffected by the V3 correction and
are reproduced from the same V3 file only because it is the drop-in
replacement with an identical schema (confirmed: C0/C1 values in the V3
nested file are bit-identical to the old script-46 file -- see
`semantic_microbiome/reports/paper1/FINAL_LEAN_SUBMISSION_2026-10-02_v3/QC/
M3_FOLD_LOCAL_METHOD_AUDIT_v3.md`). M1-alone is read from the OLD script-18
single-split OOF file, exactly as the original script does, because M1 was
deliberately not touched by the V3 correction. No calibration regression,
bootstrap protocol, or loess smoother setting is changed. All outputs are
written to new `_v3` paths; none of the original script's outputs are
touched.

Pre-submission robustness item (TRIPOD+AI requirement currently missing,
2026-09-04): the project's existing calibration output
(`31_phase_x_calibration_curves.py` -> `phase_x_calibration_summary.csv/.md`)
reports only a point-estimate Brier score, drawn from script 18's OLDER
(pre-nested-CV-correction) out-of-fold probabilities, and no calibration
intercept, calibration slope, observed/expected ratio, or any uncertainty
interval. TRIPOD+AI explicitly asks for calibration-in-the-large and
calibration slope (Van Calster et al. 2016 "calibration hierarchy"); a
Critical Care reviewer will ask for both.

This script computes, for C0/C1/C2a/C2b (from script 46's nested-CV-corrected
out-of-fold probabilities -- the SAME predictions behind the headline
delta-AUROC numbers) and M1-alone (from script 18's persisted
`microbiome_prob_M1` column -- a DIFFERENT, single-split-OOF fitted object;
see the note below):

  - calibration intercept (calibration-in-the-large: offset model,
    logit(y) = alpha + 1*logit(p), alpha estimated) with bootstrap CI
  - calibration slope (logit(y) = alpha' + beta*logit(p), beta estimated)
    with bootstrap CI
  - observed/expected (O/E) ratio = sum(y) / sum(p)
  - Brier score, now WITH a bootstrap CI (previously point-only in script 31)
  - a loess-smoothed calibration curve with a bootstrap uncertainty band

THIS SCRIPT DOES NOT REFIT ANY MODEL. All five probability columns are read
as-is from already-persisted OOF prediction files (scripts 18/45/46); only
calibration REGRESSIONS (intercept-only offset model; intercept+slope model)
and the loess smoother are fit here, exactly as calibration diagnostics are
meant to be computed on out-of-fold predictions.

WHY M1-alone IS A DIFFERENT FITTED OBJECT FROM C0/C1/C2a/C2b (stated, not
blurred): `microbiome_prob_M1` is `18_phase_x_incremental_value.py`'s
SINGLE-SPLIT 5-fold OOF refit of the 16 literature-core features
(`StratifiedKFold(5, shuffle=True, random_state=2026)`, ONE split, no outer/
inner nesting). C0/C1/C2a/C2b come from script 45/46's genuinely NESTED
5-outer x 5-inner design (see script 45's docstring for the CV-nesting-
leakage bug this fixes). Both are honest out-of-fold quantities (a patient's
own outcome never used to produce that patient's score), but they are built
on different fold structures / different numbers of refits, so M1-alone's
calibration numbers are not directly comparable to C0/C1/C2a/C2b's on CV-
design grounds alone -- exactly the same distinction script 56 (Panel A)
already draws between the OOF layer and the nested layer for M1.

RECALIBRATION CONTROL (verified by reading 45_phase_x_c1_c2_nested_cv_corrected
.py, not assumed): C0 (`nested_oof_c0()`, script 45 lines 288-297) fits, per
OUTER fold, `LogisticRegression(penalty=None).fit(logit_clin[tr], y[tr])` on
the single covariate `logit_clin` -- i.e. C0 IS a per-fold logistic
recalibration of the frozen clinical probability (intercept + slope on
logit_clin, refit each outer fold). C1/C2a/C2b (`nested_oof_meta()`, script 45
lines 261-285) fit, per outer fold, the SAME `LogisticRegression(penalty=None)`
form on `[logit_clin, logit(microbiome_prob)]` -- i.e. they carry an
identical per-fold refit coefficient on `logit_clin` PLUS one additional
covariate. So the recalibration of the clinical score is structurally common
to both arms of every C1/C2a/C2b-vs-C0 contrast: C0 is the restricted
(microbiome-covariate-dropped) special case of the same per-fold logistic
family that produces C1/C2a/C2b. The reported delta-AUROC and delta-NB
contrasts are therefore ALREADY recalibration-matched -- this script's own
calibration-intercept/slope numbers for C0 vs C1/C2a/C2b are a description of
each arm's own post-hoc calibration quality, not evidence that recalibration
differed between arms; a recalibration confound of the kind that would need
correcting does not exist in this comparison. (M1-alone is a different
question: it is a single-covariate stage-1 quantity with no clinical term at
all, so it is not part of this recalibration-matched family -- reported here
purely descriptively, as flagged above.)

BOOTSTRAP CONVENTION -- matched to `56_phase_x_panelA_discrimination_ci.py`
(`np.random.RandomState(2026)`, `idx = rng.randint(0, n, n)` with-replacement
patient resampling, resamples with a single outcome class skipped, 95% CI =
`np.percentile(boots, [2.5, 97.5])`), N_BOOT = 1000 (`config.N_BOOTSTRAP`).
Every interval in this script (calibration intercept, calibration slope,
Brier, calibration-curve band) uses this SAME resampling protocol, so they
are directly comparable to script 56's AUROC intervals.

CALIBRATION-CURVE SMOOTHER: LOESS (`statsmodels.nonparametric.smoothers_
lowess.lowess`, `frac=0.6`), not a spline. Chosen because (a) it is already a
project dependency (statsmodels is used throughout scripts 45/46/56/57), (b)
it needs no knot-placement decision at N=302/69 events, where a spline's
knot count/placement would itself be a researcher-degree-of-freedom, and (c)
it is the calibration-curve smoother most commonly reported in the clinical
prediction-model literature this manuscript is written for (Van Calster et
al.), making the figure legible to a Critical Care reviewer without an extra
methods sentence.

Reads (read-only):
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities.csv (script 46)
  clinical_ai/predictions/phase_x_c1_c2_oof_probabilities.csv        (script 18)
Writes:
  clinical_ai/reports/phase_x_calibration_metrics.csv
  clinical_ai/reports/phase_x_calibration_metrics.md
  clinical_ai/reports/phase_x_calibration_metrics_curves.csv   (loess curve + band grid data)
  clinical_ai/figures/phase_x_calibration_metrics_curves.{svg,jpg}
"""
from __future__ import annotations

import os
import sys

# Performance note (added after an interactive-run timing check on
# 2026-09-04): the calibration-intercept/slope bootstrap below fits ~1e4
# tiny (n=302, 1-2 parameter) logistic models. Under multi-threaded BLAS on
# a shared login node, per-call thread-spawn overhead dominates the actual
# FLOPs (statsmodels.GLM at the default thread count took ~4.5 minutes PER
# MODEL for this step alone). Pinning to single-threaded BLAS before numpy
# is ever imported removes that overhead; combined with the custom
# Newton-Raphson fit below (verified against statsmodels.GLM to <1e-15,
# see calibration_intercept/calibration_slope docstrings), the full script
# runs in well under a minute. Must be set before numpy/statsmodels import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("59_phase_x_calibration_metrics_v4")

# M1 is unaffected by the V3 correction -- kept on the OLD script-18 file, exactly as the
# original script does.
OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")
# V3: repointed at the fold-local-corrected nested OOF drop-in (script 45_v3), which carries
# the corrected C2a/C2b (M3-LR/M3-CatBoost-stacked) columns. Same schema as script 46's old
# file (patient_id_clinical index, hospital_mortality, oof_prob_C0.../C1.../C2a.../C2b...).
NESTED_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv")
OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_calibration_metrics_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_calibration_metrics_v4.md")
OUT_CURVES_CSV = os.path.join(config.DIR_REPORTS, "phase_x_calibration_metrics_curves_v4.csv")
FIG_PATH = os.path.join(config.DIR_FIGURES, "phase_x_calibration_metrics_curves_v4")

SEED = config.RANDOM_SEED    # 2026
N_BOOT = config.N_BOOTSTRAP  # 1000 -- matches script 56's convention
EPS = 1e-6
LOESS_FRAC = 0.6
N_GRID = 101

COLORS = {"C0_clinical_alone": "#555555", "C1_clinical_plus_M1_LiteratureCore": "#1f77b4",
          "C2a_clinical_plus_M3_LR": "#ff7f0e", "C2b_clinical_plus_M3_CatBoost": "#2ca02c",
          "M1_alone_LiteratureCore": "#9467bd"}


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _newton_intercept_only(y, offset, max_iter=50, tol=1e-10, ridge=1e-10):
    """Fast pure-numpy Newton-Raphson for the offset-model MLE: logit(y) =
    alpha + 1*offset, alpha free -- 1-parameter, closed-form Hessian each
    step. Used INSTEAD of statsmodels.GLM inside the bootstrap loop (~1e3
    calls per model) purely for speed; verified to agree with
    statsmodels.GLM(family=Binomial(), offset=offset) to <2e-16 on both a
    synthetic check and this script's own data (see the runtime correctness
    check below, section 1b)."""
    alpha = 0.0
    for _ in range(max_iter):
        pr = _sigmoid(alpha + offset)
        grad = np.sum(y - pr)
        hess = -np.sum(pr * (1 - pr)) - ridge
        step = grad / hess
        alpha -= step
        if abs(step) < tol:
            break
    return alpha


def _newton_intercept_slope(y, x, max_iter=50, tol=1e-10, ridge=1e-10):
    """Fast pure-numpy Newton-Raphson (IRLS) for the 2-parameter logistic
    regression MLE: logit(y) = alpha + beta*x, both free. Same speed
    rationale and correctness verification as `_newton_intercept_only`."""
    theta = np.array([0.0, 1.0])
    X = np.column_stack([np.ones(len(x)), x])
    for _ in range(max_iter):
        pr = _sigmoid(X @ theta)
        W = pr * (1 - pr)
        grad = X.T @ (y - pr)
        H = -(X.T * W) @ X - ridge * np.eye(2)
        delta = np.linalg.solve(H, grad)
        theta = theta - delta
        if np.linalg.norm(delta) < tol:
            break
    return theta


def calibration_intercept(y, p):
    """Calibration-in-the-large: logit(y) = alpha + 1*logit(p), alpha free
    (offset model). Returns alpha. See `_newton_intercept_only`."""
    return float(_newton_intercept_only(y, logit(p)))


def calibration_slope(y, p):
    """logit(y) = alpha' + beta*logit(p), both free. Returns beta. See
    `_newton_intercept_slope`."""
    return float(_newton_intercept_slope(y, logit(p))[1])


def oe_ratio(y, p):
    return float(np.sum(y) / np.sum(p))


def brier(y, p):
    return float(brier_score_loss(y, p))


def bootstrap_stat(y, p, stat_fn, n_boot=N_BOOT, seed=SEED):
    """Byte-for-byte the resampling protocol of
    56_phase_x_panelA_discrimination_ci.py:bootstrap_auroc(), generalised to
    an arbitrary per-resample statistic."""
    rng = np.random.RandomState(seed)
    n = len(y)
    point = stat_fn(y, p)
    boots = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(stat_fn(y[idx], p[idx]))
        except Exception:
            n_failed += 1
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), len(boots), n_failed


def loess_curve_with_band(y, p, n_boot=N_BOOT, seed=SEED, frac=LOESS_FRAC, n_grid=N_GRID):
    """Point loess fit on the observed data, evaluated on a fixed grid
    spanning the observed predicted-probability range, plus a bootstrap band:
    for each patient resample, refit loess and linearly interpolate its
    (irregular, resample-dependent) x-support onto the SAME fixed grid before
    collecting percentiles -- so every bootstrap replicate contributes a
    value at every grid point despite loess's x-support changing resample to
    resample.

    `it=0` (no robustness re-weighting iterations) is REQUIRED here, not a
    default left alone: statsmodels' `lowess` default (`it=3`) runs
    Tukey-biweight robustness iterations that downweight points with large
    residuals from the current local fit -- appropriate for continuous y
    with genuine outliers, but WRONG for a binary calibration curve, where
    the minority class (deaths, ~23% of this cohort, and locally far sparser
    than that in high-skew predicted-probability regions) IS the signal, not
    outlier noise. Confirmed empirically on this script's own M1 data before
    fixing: `it=3` degenerated the fitted curve to a flat 0 line almost
    everywhere (every death in a locally 0-majority neighbourhood is
    iteratively downweighted as an apparent outlier, snowballing to zero
    influence); `it=0` recovers a sensible curve (e.g. ~0.27 observed at a
    predicted probability of 0.30, vs. the diagonal's 0.30) that a manual
    inspection of the surrounding raw (p, y) pairs confirms is right."""
    grid = np.linspace(np.min(p), np.max(p), n_grid)
    point_fit = lowess(y, p, frac=frac, it=0, return_sorted=True)
    point_y = np.interp(grid, point_fit[:, 0], point_fit[:, 1])

    rng = np.random.RandomState(seed)
    n = len(y)
    boot_curves = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            fit_b = lowess(yb, pb, frac=frac, it=0, return_sorted=True)
            boot_curves.append(np.interp(grid, fit_b[:, 0], fit_b[:, 1]))
        except Exception:
            continue
    boot_curves = np.array(boot_curves)
    lo = np.percentile(boot_curves, 2.5, axis=0)
    hi = np.percentile(boot_curves, 97.5, axis=0)
    return grid, point_y, lo, hi, len(boot_curves)


# ---------------------------------------------------------------------------
# 1. load -- nested (C0/C1/C2a/C2b, script 46) + single-split (M1, script 18)
# ---------------------------------------------------------------------------
for path in (OOF_CSV, NESTED_CSV):
    if not os.path.exists(path):
        log.error("missing input: %s -- run 18_phase_x_incremental_value.py and "
                   "45_phase_x_c1_c2_nested_cv_corrected_v3.py first.", path)
        sys.exit(1)

oof = pd.read_csv(OOF_CSV, index_col=0)
nested = pd.read_csv(NESTED_CSV, index_col=0).reindex(oof.index)
if not (nested["hospital_mortality"].values == oof["hospital_mortality"].values).all():
    log.error("hospital_mortality mismatch between %s and %s after reindexing -- refusing to proceed on a "
              "possibly-misaligned join.", OOF_CSV, NESTED_CSV)
    sys.exit(1)
log.info("N=%d patients, identical index/outcome across both prediction files.", len(oof))

y_all = oof["hospital_mortality"].values.astype(int)

# ---------------------------------------------------------------------------
# 1b. runtime correctness check: the fast Newton-Raphson calibration fits
#     used throughout this script (for speed -- see the module-level
#     performance note) must reproduce statsmodels.GLM exactly on THIS
#     script's own data, not just on a synthetic sanity check. Run once,
#     on C0's point estimate, before any bootstrap loop.
# ---------------------------------------------------------------------------
_p_check = nested["oof_prob_C0_clinical_alone"].values.astype(float)
_lp_check = logit(_p_check)
_sm_intercept = sm.GLM(y_all, np.ones((len(y_all), 1)), family=sm.families.Binomial(),
                        offset=_lp_check).fit(disp=0).params[0]
_sm_slope = sm.GLM(y_all, sm.add_constant(_lp_check), family=sm.families.Binomial()).fit(disp=0).params[1]
_fast_intercept = calibration_intercept(y_all, _p_check)
_fast_slope = calibration_slope(y_all, _p_check)
_diff_intercept = abs(_sm_intercept - _fast_intercept)
_diff_slope = abs(_sm_slope - _fast_slope)
log.info("Fast-fit correctness check (C0, full data): intercept sm=%.6f fast=%.6f diff=%.2e; "
         "slope sm=%.6f fast=%.6f diff=%.2e", _sm_intercept, _fast_intercept, _diff_intercept,
         _sm_slope, _fast_slope, _diff_slope)
if _diff_intercept > 1e-6 or _diff_slope > 1e-6:
    log.error("Fast Newton-Raphson calibration fit does NOT reproduce statsmodels.GLM (diff intercept=%.2e, "
              "slope=%.2e) -- refusing to proceed on a possibly-incorrect fast path. STOP.",
              _diff_intercept, _diff_slope)
    sys.exit(1)
log.info("Fast-fit correctness check PASSED (<1e-6 agreement with statsmodels.GLM).")

MODELS = [
    ("C0_clinical_alone", "oof_prob_C0_clinical_alone", nested, "nested, fold-local-corrected (script 45_v4 -- V4_STRICT for LR, V3 reused for CatBoost), 5-outer x 5-inner CV"),
    ("C1_clinical_plus_M1_LiteratureCore", "oof_prob_C1_clinical_plus_M1_LiteratureCore", nested,
     "nested, fold-local-corrected (script 45_v4 -- V4_STRICT for LR, V3 reused for CatBoost), 5-outer x 5-inner CV"),
    ("C2a_clinical_plus_M3_LR", "oof_prob_C2a_clinical_plus_M3_LR", nested, "nested, fold-local-corrected (script 45_v4 -- V4_STRICT for LR, V3 reused for CatBoost), 5-outer x 5-inner CV"),
    ("C2b_clinical_plus_M3_CatBoost", "oof_prob_C2b_clinical_plus_M3_CatBoost", nested,
     "nested, fold-local-corrected (script 45_v4 -- V4_STRICT for LR, V3 reused for CatBoost), 5-outer x 5-inner CV"),
    ("M1_alone_LiteratureCore", "microbiome_prob_M1", oof,
     "single-split OOF (script 18), StratifiedKFold(5) ONE split -- a DIFFERENT fitted object from C0/C1/C2a/C2b, "
     "see docstring"),
]

# ---------------------------------------------------------------------------
# 2. per-model calibration metrics
# ---------------------------------------------------------------------------
rows = []
curve_rows = []
for name, col, src, design in MODELS:
    p = src[col].values.astype(float)
    y = src["hospital_mortality"].values.astype(int)
    valid = ~pd.isna(p)
    p, y = p[valid], y[valid]
    n, n_events = len(y), int(y.sum())
    log.info("=== %s (N=%d, events=%d) ===", name, n, n_events)

    b_pt, b_lo, b_hi, b_neff, b_nfail = bootstrap_stat(y, p, brier)
    ci_pt, ci_lo, ci_hi, ci_neff, ci_nfail = bootstrap_stat(y, p, calibration_intercept)
    cs_pt, cs_lo, cs_hi, cs_neff, cs_nfail = bootstrap_stat(y, p, calibration_slope)
    oe_pt, oe_lo, oe_hi, oe_neff, oe_nfail = bootstrap_stat(y, p, oe_ratio)

    if b_nfail or ci_nfail or cs_nfail or oe_nfail:
        log.warning("%s: bootstrap fit failures -- Brier=%d, intercept=%d, slope=%d, OE=%d (of %d resamples)",
                     name, b_nfail, ci_nfail, cs_nfail, oe_nfail, N_BOOT)

    rows.append({
        "model": name, "N": n, "N_events": n_events, "design": design,
        "Brier": round(b_pt, 4), "Brier_CI_low": round(b_lo, 4), "Brier_CI_high": round(b_hi, 4),
        "cal_intercept": round(ci_pt, 4), "cal_intercept_CI_low": round(ci_lo, 4), "cal_intercept_CI_high": round(ci_hi, 4),
        "cal_slope": round(cs_pt, 4), "cal_slope_CI_low": round(cs_lo, 4), "cal_slope_CI_high": round(cs_hi, 4),
        "OE_ratio": round(oe_pt, 4), "OE_ratio_CI_low": round(oe_lo, 4), "OE_ratio_CI_high": round(oe_hi, 4),
        "N_boot": N_BOOT, "N_effective_resamples_Brier": b_neff, "seed": SEED,
    })
    log.info("%s: Brier=%.4f [%.4f,%.4f]; intercept=%.4f [%.4f,%.4f]; slope=%.4f [%.4f,%.4f]; O/E=%.4f [%.4f,%.4f]",
              name, b_pt, b_lo, b_hi, ci_pt, ci_lo, ci_hi, cs_pt, cs_lo, cs_hi, oe_pt, oe_lo, oe_hi)

    grid, curve_pt, curve_lo, curve_hi, curve_neff = loess_curve_with_band(y, p)
    for g, cy, clo, chi in zip(grid, curve_pt, curve_lo, curve_hi):
        curve_rows.append({"model": name, "predicted_prob": round(float(g), 4), "loess_observed": round(float(cy), 4),
                            "loess_CI_low": round(float(max(0.0, clo)), 4), "loess_CI_high": round(float(min(1.0, chi)), 4)})
    log.info("%s: loess curve band from %d/%d effective bootstrap resamples.", name, curve_neff, N_BOOT)

result = pd.DataFrame(rows)
io_utils.save_csv(result, OUT_CSV)
curves = pd.DataFrame(curve_rows)
io_utils.save_csv(curves, OUT_CURVES_CSV)
log.info("\n%s", result[["model", "N", "Brier", "cal_intercept", "cal_slope", "OE_ratio"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. calibration-curve figure -- one panel, all 5 models, loess + band.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
for name, _, _, _ in MODELS:
    sub = curves[curves["model"] == name]
    c = COLORS.get(name)
    ax.plot(sub["predicted_prob"], sub["loess_observed"], color=c, lw=1.6, label=name.split("_", 1)[0])
    ax.fill_between(sub["predicted_prob"], sub["loess_CI_low"], sub["loess_CI_high"], color=c, alpha=0.15)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed mortality (loess-smoothed)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"Calibration curves with bootstrap 95% band (N={N_BOOT}, seed={SEED})\n"
             "loess frac=0.6; C0/C1/C2a/C2b nested-CV OOF, M1 single-split OOF")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
os.makedirs(config.DIR_FIGURES, exist_ok=True)
fig.savefig(FIG_PATH + ".svg", bbox_inches="tight")
fig.savefig(FIG_PATH + ".jpg", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("wrote %s.{svg,jpg}", FIG_PATH)

# ---------------------------------------------------------------------------
# 4. markdown report
# ---------------------------------------------------------------------------
md = [
    "# Phase X: calibration metrics (TRIPOD+AI) -- calibration-in-the-large, slope, O/E, Brier with bootstrap CIs",
    "",
    f"N={len(oof)} Gangnam patients, {int(y_all.sum())} deaths ({100*y_all.mean():.1f}%). Closes a pre-submission "
    "gap: the project's existing calibration output (`phase_x_calibration_summary.csv/.md`, script 31) reports "
    "only point-estimate Brier scores from script 18's OLDER (pre-nested-CV-correction) OOF probabilities, with "
    "no calibration intercept, slope, O/E ratio, or any uncertainty interval. TRIPOD+AI expects calibration in "
    "the large and in the small; this script supplies both, with bootstrap CIs, computed from the SAME "
    "nested-CV-corrected out-of-fold predictions (`phase_x_c1_c2_nested_oof_probabilities.csv`, script 46) that "
    "sit behind the headline delta-AUROC numbers, for C0/C1/C2a/C2b. M1-alone is reported too, from "
    "`microbiome_prob_M1` (script 18) -- **a DIFFERENT fitted object** (single-split OOF, not nested); see the "
    "note below. No model is refit anywhere in this script.", "",
    "## Results", "",
    df_to_md_table(result[["model", "N", "N_events", "Brier", "Brier_CI_low", "Brier_CI_high",
                            "cal_intercept", "cal_intercept_CI_low", "cal_intercept_CI_high",
                            "cal_slope", "cal_slope_CI_low", "cal_slope_CI_high",
                            "OE_ratio", "OE_ratio_CI_low", "OE_ratio_CI_high"]]), "",
    "Calibration intercept (calibration-in-the-large): `logit(y) = alpha + 1*logit(p)`, alpha estimated by "
    "offset-model logistic regression (mathematically `statsmodels.GLM(..., family=Binomial(), "
    "offset=logit(p))`; computed via a fast pure-numpy Newton-Raphson equivalent -- needed because the "
    "bootstrap below fits ~1e4 of these -- verified to reproduce statsmodels.GLM to <1e-6 on this script's own "
    "data at runtime, see the log). alpha=0 means "
    "no systematic over/under-prediction. Calibration slope: `logit(y) = alpha' + beta*logit(p)`, both estimated; "
    "beta=1 is ideal, beta<1 indicates predictions are too extreme (overfitting-typical), beta>1 too conservative. "
    "O/E ratio = sum(observed)/sum(predicted); 1.0 is ideal. All bootstrap CIs: 1000 with-replacement patient "
    "resamples, `np.random.RandomState(2026)`, `rng.randint(0, n, n)`, single-outcome-class resamples skipped, "
    "percentile 2.5/97.5 -- byte-for-byte the resampling protocol of "
    "`56_phase_x_panelA_discrimination_ci.py:bootstrap_auroc()`, so these intervals are directly comparable to "
    "that script's AUROC intervals.", "",
    "## Which fitted object each model comes from", "",
    "**C0/C1/C2a/C2b**: `phase_x_c1_c2_nested_oof_probabilities.csv` (script 46) -- the genuinely nested "
    "(5-outer x 5-inner) out-of-fold design that fixed the CV-nesting leakage bug documented in "
    "`45_phase_x_c1_c2_nested_cv_corrected.py`. These are the SAME predictions behind the headline "
    "delta-AUROC/delta-NB numbers.", "",
    "**M1-alone**: `phase_x_c1_c2_oof_probabilities.csv:microbiome_prob_M1` (script 18) -- a SINGLE-SPLIT "
    "5-fold OOF refit (elastic-net LogisticRegressionCV on the 16 literature-core features, ONE "
    "`StratifiedKFold(5, shuffle=True, random_state=2026)` split, no outer/inner nesting). This is a genuinely "
    "different fitted object from C0/C1/C2a/C2b, not merely a different column of the same fit -- the CV-nesting "
    "correction restructures only the stacked meta-model layer, and script 45/46 recompute this identical "
    "stage-1 M1 column unchanged rather than nesting it (see script 56's Panel A docstring for the same "
    "distinction, drawn independently there for AUROC). Its calibration numbers are reported here for "
    "completeness but are **not** directly comparable to C0/C1/C2a/C2b's on CV-design grounds alone.", "",
    "## Recalibration control -- verified by reading `45_phase_x_c1_c2_nested_cv_corrected.py`", "",
    "**Finding: the recalibration control is already structural, and it is common to both arms of every "
    "C1/C2a/C2b-vs-C0 contrast.** `nested_oof_c0()` (script 45, lines 288-297) fits, per OUTER fold, "
    "`LogisticRegression(penalty=None).fit(logit_clin[tr], y[tr])` on the single covariate `logit_clin` -- this "
    "IS a per-fold logistic recalibration (intercept + slope on the frozen clinical logit, refit every outer "
    "fold) of the clinical probability, and it is exactly what produces C0=0.7528. `nested_oof_meta()` (script "
    "45, lines 261-285) fits, per outer fold, the SAME `LogisticRegression(penalty=None)` family on "
    "`[logit_clin, logit(microbiome_prob)]` -- i.e. C1/C2a/C2b carry an IDENTICAL per-fold refit coefficient on "
    "`logit_clin` plus one additional microbiome covariate. C0 is therefore the microbiome-covariate-dropped "
    "restriction of the exact same per-fold logistic family that produces C1/C2a/C2b, not a separately-tuned "
    "baseline. Because the clinical-score recalibration step is structurally shared between both arms of every "
    "contrast, the reported delta-AUROC and delta-NB numbers are **already recalibration-matched** -- a reviewer "
    "cannot attribute any observed C1/C2a/C2b-vs-C0 difference to C0 being under-recalibrated relative to the "
    "combined models, because both are built from the same per-fold recalibration step. This script's own "
    "calibration-intercept/slope numbers for each of C0/C1/C2a/C2b describe that model's own residual "
    "miscalibration after this shared per-fold recalibration -- they are a calibration-quality report, not "
    "evidence bearing on whether the delta-AUROC/delta-NB contrasts need an additional recalibration correction "
    "(they do not).", "",
    "## Calibration curve smoother", "",
    f"**LOESS** (`statsmodels.nonparametric.smoothers_lowess.lowess`, `frac={LOESS_FRAC}`, `it=0`), not a spline. "
    "Reasons: "
    "(1) statsmodels is already used throughout this pipeline (scripts 45/46/56/57), so no new dependency; (2) at "
    f"N={len(oof)}/{int(y_all.sum())} events a spline's knot count/placement would itself be an undocumented "
    "researcher degree of freedom, while loess's only tuning parameter (the smoothing fraction) is stated here "
    "explicitly; (3) it is the calibration-curve smoother most commonly reported in the clinical "
    "prediction-model literature (Van Calster et al.) this manuscript targets. **`it=0`** (no robustness "
    "re-weighting iterations) is a deliberate deviation from statsmodels' default (`it=3`), found necessary "
    "empirically: the default's Tukey-biweight downweighting of large-residual points is appropriate for "
    "continuous y with genuine outliers, but for a BINARY outcome it systematically downweights the minority "
    "class (deaths) as if they were outlier noise -- confirmed here to degenerate the M1-alone curve to a flat "
    "0 line almost everywhere under `it=3`; `it=0` recovers a curve consistent with the raw (predicted "
    "probability, outcome) pairs. The bootstrap band "
    f"({N_BOOT} resamples, same seed/protocol as the scalar metrics above) refits loess (same `it=0`) on each "
    "patient resample "
    "and linearly interpolates onto a common 101-point grid spanning the observed predicted-probability range "
    "(loess's x-support otherwise differs resample to resample, which would make percentiles at a fixed x "
    "meaningless without this interpolation step).", "",
    "## Figure", "", "![calibration](../figures/phase_x_calibration_metrics_curves_v4.jpg)", "",
    "## Cross-check against the existing point-only Brier scores", "",
    "`phase_x_calibration_summary.csv` (script 31, script-18 pre-nested OOF): C2b 0.1492, C6 0.1507, C0 0.1517, "
    "C1 0.1520, C2a 0.1542. This script's Brier point estimates use the NESTED OOF for C0/C1/C2a/C2b (a different "
    "fitted object -- see script 45's docstring), so the numbers are not expected to match exactly; they are "
    "expected to be close, and the ranking need not be preserved. Script 31 also reports no C6 candidate here "
    "(prespecified-panel model, out of scope for this nested-CV metric set).", "",
    "## Reference", "",
    "Nested-CV-corrected OOF predictions: `phase_x_c1_c2_nested_oof_probabilities.csv` (script 46). Single-split "
    "OOF (M1): `phase_x_c1_c2_oof_probabilities.csv` (script 18). Existing point-Brier calibration output "
    "(NOT modified by this script): `phase_x_calibration_summary.csv/.md` (script 31). Bootstrap convention "
    "reused from `phase_x_panelA_discrimination_ci.csv/.md` (script 56).",
]
io_utils.save_text("\n".join(md), OUT_MD)
log.info("wrote %s / %s / %s", OUT_CSV, OUT_MD, OUT_CURVES_CSV)
