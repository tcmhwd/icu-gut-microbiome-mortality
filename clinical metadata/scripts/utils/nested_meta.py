#!/usr/bin/env python3
"""nested_meta.py — shared nested-CV base/meta layer for the Phase X
adaptive-combination-family rebuild (script 57), per
`clinical_ai/reports/adaptive_nested_rebuild_DESIGN.md` section 3.

Reused, UNMODIFIED, from `45_phase_x_c1_c2_nested_cv_corrected.py` (lines
135-210 / 198-285): `logit()`, `make_lr_pipe()`, `fit_predict_lr()`, and the
outer/inner `StratifiedKFold` nesting structure itself. `45`/`46` are NOT
edited — this is a new sibling module, per this project's standing
new-sibling-script-not-in-place-edit precedent (script 45's own docstring;
`lit_M4_leakage_free_refit.py`).

Two functions, per the design's explicit split (script 45's `nested_oof_meta`
cannot be reused as-is: it discards the intermediate inner-OOF / outer-fit
microbiome-probability vectors that the adaptive/interaction models need):

1. `nested_base_probs()` — the cacheable base layer. One outer K-fold pass
   of the M1 base model (elastic-net LR). Runs ONCE per (X_micro, seed,
   n_folds) and is reused by every moderator/subgroup/script's-worth of
   downstream meta-model rows. Returns, per outer fold, the inner-OOF
   (train-fold) and outer-fit (test-fold) microbiome probabilities as a
   single FULL-LENGTH (n,) array indexed by GLOBAL patient position --
   never by position-within-tr_outer. Script 45's own `inner_prob` is
   indexed within `tr_outer` (after `.reset_index(drop=True)`); converting
   to a full-length, globally-positioned array exactly once, at the point
   of assignment (`p[tr_outer] = inner_prob_local`), is what the design
   calls out as eliminating the entire off-by-one bug class that appears
   once this base layer also has to be subset by moderator missingness and
   by fold-dependent subgroup membership downstream.

2. `nested_meta_eval()` — the thin per-row meta layer. Given the cached
   folds, a moderator (or a dict of several, for script 28 Part B's
   multivariate combination), and an optional `row_mask_fn` (fold-dependent
   subgroup membership, e.g. the clinical-flagged-HIGH/LOW strata), fits
   THREE nested arms per outer fold on the SAME analysed rows: C0
   (clinical-alone, for the "AUROC_C0_nested_same_rows" anchor -- design
   section 2's ladder-visibility requirement), plain (clinical + microbiome,
   no interaction), and adaptive (+ moderator interaction term(s)). The
   moderator is standardised INSIDE each outer training fold (design section
   3's "(c) Standardise the moderator inside the fold anyway" -- proven
   mathematically inconsequential for this unpenalized estimator, but free
   and removes the objection; a degenerate fold sigma is guarded and skips
   that fold for ALL THREE arms together, so plain/adaptive/C0 stay paired
   on identical analysed rows for the bootstrap).

   Row-filtering order (design section 3, formula "tr = tr_outer ∩
   finite(moderator) ∩ subgroup(tr)"): moderator-completeness is applied
   FIRST (tr_outer -> tr_valid), THEN `row_mask_fn(tr_valid, te_valid, y)`
   computes subgroup membership -- this ordering is what makes the lifted
   script-28(A)-lines-135-141 `row_mask_fn` (Youden threshold + subgroup
   mask, re-derived per outer TRAINING fold) reproduce that script's exact
   algorithm, which computes the fold threshold on the moderator-valid
   subset of tr_outer, not the raw tr_outer.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

EPS = 1e-6
SEED = config.RANDOM_SEED  # 2026, shared convention -- matches script 45/46 exactly


def logit(p):
    """Identical to 45_phase_x_c1_c2_nested_cv_corrected.py's logit()."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_lr_pipe():
    """Identical to script 45's make_lr_pipe() (lines 198-204) -- the base
    model spec is NOT changed by this rebuild, only the CV nesting that
    calls it. n_jobs=config.get_n_jobs() is what makes --cpus-per-task
    meaningful once this module is used (it was not, for the original
    unpenalized-LR-only scripts 25/27/28)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                    random_state=SEED, n_jobs=config.get_n_jobs()))])


def fit_predict_lr(X_tr, y_tr, X_te):
    """Identical to script 45's fit_predict_lr() (lines 207-209)."""
    pipe = make_lr_pipe().fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# Function 1 -- the cacheable base layer (design section 3, "Function 1").
# ---------------------------------------------------------------------------
def nested_base_probs(X_micro, y, fit_predict_fn, n_folds=5, seed=SEED):
    """One outer `n_folds`-fold pass, identical algorithm to script 45's
    `nested_oof_meta` base layer (lines 265-279), with the meta fit removed
    and the intermediate probabilities returned (not dropped).

    Parameters
    ----------
    X_micro : pd.DataFrame, positionally aligned with `y` (row i of X_micro
        is the same patient as y[i], everywhere else in the calling script).
    y : array-like (n,) binary outcome.
    fit_predict_fn : callable(X_tr, y_tr, X_te) -> predicted probabilities.
        Pass `fit_predict_lr` for M1 (the only base model the adaptive
        family ever uses -- it never uses M3).
    n_folds : outer/inner fold count (5, matching script 45/46).
    seed : StratifiedKFold random_state for BOTH outer and (per-outer-fold)
        inner splits -- matches script 45/46's SEED (2026) exactly, so the
        outer fold partition is bit-identical to scripts 45/46's.

    Returns
    -------
    folds : list of dict, one per outer fold, in outer_skf.split() order:
        {"tr_outer": np.ndarray, "te_outer": np.ndarray,
         "p": np.ndarray shape (n,), NaN outside tr_outer|te_outer;
              p[tr_outer] = inner-OOF microbiome probability (script 45 step a)
              p[te_outer] = base model fit on the FULL tr_outer, applied to
                            te_outer (script 45 step b) -- the only place
                            te_outer ever interacts with any model.}
    n_base_fits : int, total base-model fits performed. For n_folds=5:
        5 outer folds x (5 inner fits + 1 outer-full fit) = 30 -- "30 base
        fits, run once" is the design's mandatory verification anchor.
    """
    n = len(y)
    y = np.asarray(y)
    outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    n_base_fits = 0
    for tr_outer, te_outer in outer_skf.split(X_micro, y):
        y_tr_outer = y[tr_outer]
        X_tr_outer = X_micro.iloc[tr_outer].reset_index(drop=True)

        # (a) inner OOF microbiome_prob for tr_outer ONLY -- te_outer never
        # touched. Computed in tr_outer-LOCAL position order (matching
        # script 45 exactly), then converted to full-length GLOBAL position
        # exactly once, below -- this is the single point of conversion the
        # design calls out to eliminate the off-by-one bug class.
        inner_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        inner_prob_local = np.full(len(tr_outer), np.nan)
        for tr_inner, te_inner in inner_skf.split(X_tr_outer, y_tr_outer):
            inner_prob_local[te_inner] = fit_predict_fn(X_tr_outer.iloc[tr_inner], y_tr_outer[tr_inner],
                                                          X_tr_outer.iloc[te_inner])
            n_base_fits += 1

        # (b) base model fit on the FULL tr_outer, applied to te_outer.
        te_outer_prob = fit_predict_fn(X_micro.iloc[tr_outer], y[tr_outer], X_micro.iloc[te_outer])
        n_base_fits += 1

        p = np.full(n, np.nan)
        p[tr_outer] = inner_prob_local  # local -> global position, ONCE, here
        p[te_outer] = te_outer_prob
        folds.append({"tr_outer": tr_outer, "te_outer": te_outer, "p": p})
    return folds, n_base_fits


# ---------------------------------------------------------------------------
# Function 2 -- the thin per-row meta layer (design section 3, "Function 2").
# ---------------------------------------------------------------------------
def nested_meta_eval(folds, y, logit_clin, moderator=None, row_mask_fn=None, seed=SEED):
    """Fit and evaluate C0 (clinical-alone), plain (clinical+microbiome),
    and (if `moderator` given) adaptive (+ moderator interaction) nested
    meta-models on the SAME per-fold analysed rows, using the cached base
    probabilities in `folds`.

    Parameters
    ----------
    folds : list of dict, from `nested_base_probs()`.
    y : array-like (n,) binary outcome, SAME order/index as `folds`' arrays.
    logit_clin : array-like (n,) logit(clinical_AI_probability_PRIMARY).
    moderator : None, a 1-D array-like (n,) (single moderator), or a dict
        {name: 1-D array-like (n,)} (script 28 Part B's SOFA+Cr+PLT+AST
        multivariate combination -- one interaction term per entry, summed
        into the adaptive design matrix). None means "no adaptive arm" --
        used to recompute the bare nested C0/plain vectors for the
        verification anchor (must reproduce script 46's persisted
        `oof_prob_C1_clinical_plus_M1_LiteratureCore` / C0 exactly).
    row_mask_fn : optional callable(tr_valid, te_valid, y) ->
        (tr_keep, te_keep) GLOBAL index arrays, called AFTER
        moderator-completeness filtering (tr_outer/te_outer -> tr_valid/
        te_valid), so a threshold-based row_mask_fn computes its threshold
        on the moderator-valid subset -- this reproduces script 28(A) lines
        135-141 EXACTLY when lifted as-is (see `make_subgroup_row_mask_fn`
        below). None means no additional row restriction (whole-cohort
        rows).
    seed : unused directly (folds already carry their own seed-derived
        splits); kept for signature symmetry / future use.

    Returns
    -------
    oof_c0, oof_plain, oof_adapt : np.ndarray (n,), NaN outside the analysed
        rows. oof_adapt is None if moderator is None.
    analysed_mask : np.ndarray (n,) bool -- across all outer folds, was this
        patient's test-fold prediction included (i.e. moderator-complete AND,
        if row_mask_fn given, in-subgroup for that fold).
    diagnostics : list of per-fold dict (fold index, n_tr, n_te, skip reason
        or None) -- for `n_folds_evaluated` and audit.
    """
    n = len(y)
    y = np.asarray(y)
    logit_clin = np.asarray(logit_clin, dtype=float)

    if moderator is None:
        mods = {}
    elif isinstance(moderator, dict):
        mods = {k: np.asarray(v, dtype=float) for k, v in moderator.items()}
    else:
        mods = {"_single": np.asarray(moderator, dtype=float)}
    has_moderator = len(mods) > 0

    finite_mod = np.ones(n, dtype=bool)
    for arr in mods.values():
        finite_mod &= np.isfinite(arr)

    oof_c0 = np.full(n, np.nan)
    oof_plain = np.full(n, np.nan)
    oof_adapt = np.full(n, np.nan) if has_moderator else None
    analysed_mask = np.zeros(n, dtype=bool)
    diagnostics = []

    for fold_i, f in enumerate(folds):
        tr_outer, te_outer, p = f["tr_outer"], f["te_outer"], f["p"]

        tr_valid = tr_outer[finite_mod[tr_outer]]
        te_valid = te_outer[finite_mod[te_outer]]

        if row_mask_fn is not None:
            tr, te = row_mask_fn(tr_valid, te_valid, y)
            tr, te = np.asarray(tr), np.asarray(te)
        else:
            tr, te = tr_valid, te_valid

        diag = {"fold": fold_i, "n_tr": int(len(tr)), "n_te": int(len(te)), "skipped": None}

        if len(te) == 0 or len(tr) < 2 or len(np.unique(y[tr])) < 2 or y[tr].sum() < 3:
            diag["skipped"] = "degenerate_fold_after_masking"
            diagnostics.append(diag)
            continue

        z_cols = {}
        if has_moderator:
            degenerate = False
            for name, arr in mods.items():
                mu, sd = arr[tr].mean(), arr[tr].std()
                if not np.isfinite(sd) or sd == 0:
                    degenerate = True
                    break
                z_cols[name] = (arr - mu) / sd
            if degenerate:
                diag["skipped"] = "degenerate_moderator_sd"
                diagnostics.append(diag)
                continue

        lm_tr, lm_te = logit(p[tr]), logit(p[te])

        X_c0_tr, X_c0_te = logit_clin[tr].reshape(-1, 1), logit_clin[te].reshape(-1, 1)
        X_plain_tr = np.column_stack([logit_clin[tr], lm_tr])
        X_plain_te = np.column_stack([logit_clin[te], lm_te])

        c0_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_c0_tr, y[tr])
        plain_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_plain_tr, y[tr])
        oof_c0[te] = c0_model.predict_proba(X_c0_te)[:, 1]
        oof_plain[te] = plain_model.predict_proba(X_plain_te)[:, 1]

        if has_moderator:
            inter_tr = [z_cols[name][tr] * lm_tr for name in mods]
            inter_te = [z_cols[name][te] * lm_te for name in mods]
            X_adapt_tr = np.column_stack([logit_clin[tr], lm_tr] + inter_tr)
            X_adapt_te = np.column_stack([logit_clin[te], lm_te] + inter_te)
            adapt_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_adapt_tr, y[tr])
            oof_adapt[te] = adapt_model.predict_proba(X_adapt_te)[:, 1]

        analysed_mask[te] = True
        diagnostics.append(diag)

    return oof_c0, oof_plain, oof_adapt, analysed_mask, diagnostics


def make_subgroup_row_mask_fn(c0, direction):
    """Factory for a `row_mask_fn` implementing script 28(A) lines 135-141
    lifted verbatim: the clinical Youden threshold is re-derived from
    ROC(y[tr_valid], c0[tr_valid]) -- i.e. on the moderator-valid subset of
    the outer TRAINING fold ONLY -- and applied to both tr_valid and
    te_valid to define subgroup membership. `c0` = the frozen
    clinical_AI_probability_PRIMARY vector (never refit here -- see script
    45's provenance check). `direction`: "HIGH" (c0 >= fold_thr) or "LOW"
    (c0 < fold_thr).
    """
    from sklearn.metrics import roc_curve

    if direction not in ("HIGH", "LOW"):
        raise ValueError(f"direction must be 'HIGH' or 'LOW', got {direction!r}")

    def row_mask_fn(tr_valid, te_valid, y):
        fpr, tpr, thr = roc_curve(y[tr_valid], c0[tr_valid])
        fold_thr = thr[np.argmax(tpr - fpr)]
        if direction == "HIGH":
            tr_keep = tr_valid[c0[tr_valid] >= fold_thr]
            te_keep = te_valid[c0[te_valid] >= fold_thr]
        else:
            tr_keep = tr_valid[c0[tr_valid] < fold_thr]
            te_keep = te_valid[c0[te_valid] < fold_thr]
        return tr_keep, te_keep

    return row_mask_fn


def paired_bootstrap_delta_auroc(y_row, plain_row, adapt_row, n_boot=2000, seed=SEED):
    """Paired bootstrap delta AUROC(adaptive) - AUROC(plain), matching
    scripts 21/25/27/28/46's algorithm EXACTLY: with-replacement patient
    resampling, the SAME resample index applied to both arms each
    iteration, resamples with <2 outcome classes skipped, 95% CI from
    bootstrap percentiles, two-sided p = 2*min(tail proportions crossing
    zero). `N_BOOT=2000` is deliberately retained (not config.N_BOOTSTRAP's
    1000) to match scripts 21/25/27/28/46 -- see script docstring note.

    A FRESH `np.random.RandomState(seed)` is created inside this call every
    time it is invoked -- i.e. "fresh per row" when called once per output
    row -- fixing script 28's shared-single-`rng`-across-parts bug (its
    Parts B and D silently depended on how many draws Part A had already
    consumed).
    """
    rng = np.random.RandomState(seed)
    n = len(y_row)
    y_row = np.asarray(y_row)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_row[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y_row[idx], adapt_row[idx]) - roc_auc_score(y_row[idx], plain_row[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0)), 1.0)
    point = roc_auc_score(y_row, adapt_row) - roc_auc_score(y_row, plain_row)
    return round(point, 4), round(lo, 4), round(hi, 4), round(p, 4), len(deltas)
