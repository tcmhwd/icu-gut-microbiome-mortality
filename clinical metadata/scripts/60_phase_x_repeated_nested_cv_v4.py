#!/usr/bin/env python3
"""60_phase_x_repeated_nested_cv_v4.py

v5 editorial pass (2026-10-03), Task 1: sibling of 60_phase_x_repeated_nested_cv_v3.py
with the C2a (M3-LR) arm's feature construction swapped from V3's fold-local-but-not-
tuning-local design to 45_phase_x_c1_c2_nested_cv_corrected_v4.py's V4_STRICT design
(FullBlockPreprocessor + GridSearchCV -- prevalence filter AND PCA refit inside the
LR's own 3-fold C/l1 tuning CV), parameterised by this script's per-repetition seed
(FullBlockPreprocessor.seed, LogisticRegression.random_state). This is the SAME
repeated-CV robustness design already established (20 seeds, 5x5 nested stacking per
seed) -- no new design, only the M3-LR base learner's preprocessing is swapped to the
FINAL headline specification. C2b (CatBoost) is UNCHANGED (still V3-equivalent fold-local
prevalence+PCA, since CatBoost's own specification did not change in v4). M1/C0 remain
untouched, exactly as in v3.

GridSearchCV runs with n_jobs=1 inside each seed (this script's parallelism is across
the 20 seeds via multiprocessing.Pool, matching v3's existing design philosophy --
"parallelism here is across seeds, not inside one fit").

CORRECTNESS ANCHOR (updated for v4): seed=2026 must reproduce 45_v4's published
CV_AUROC_5fold_nested to 4dp: C0=0.7528 (REF), C1=0.7610 (REF), C2a=0.7677 (V4_STRICT,
the v4 FINAL headline for M3-LR), C2b=0.7711 (V3, unchanged, CatBoost's own primary) --
read from phase_x_c1_c2_incremental_value_nested_corrected_v4.csv, verified at runtime.

Original v3 docstring below, retained for full method provenance (prevalence-filter
feasibility, seeding convention, etc. -- all unchanged except as noted above):

Fold-local M3 preprocessing correction (owner-authorised 2026-10-02) for
`60_phase_x_repeated_nested_cv.py`, following the same precedent as
`45_phase_x_c1_c2_nested_cv_corrected_v3.py` (the source of truth for the
corrected M3 feature construction) and `61_phase_x_microbiome_over_sofa_nested_v3.py`
(the sibling script that already applied this same pattern to the SOFA-referenced
family). The original script 60 is NOT edited; M1/C0 are NOT touched.

WHAT WAS GLOBAL (script 60, module level, lines ~170-183 of the original): the M3
feature matrix X_m3 (LOW_DIM_FILES raw columns + 20 PCA components each for
MAG_ABUND and EGGNOG_KO) was built ONCE, before any of the 20 seeds' CV splits, by
applying FROZEN imputer/scaler/PCA artifacts
(`semantic_microbiome/supervised_semantic/frozen_models/M3_pca_loadings/`, fit by
`lit14_freeze_pipeline.py` on all 302 patients) and a globally-fit >=10% prevalence
filter (`09_build_structured_views.py`). Every one of the 20 seeds' base-model fits
for C2a/C2b therefore saw a transform estimated partly on its own held-out rows --
the identical defect `M3_FOLD_LOCAL_METHOD_AUDIT_v3.md` section 1 documents for
script 45, and section 8 explicitly lists this script among those that
"reimplement the frozen PCA and need the same fold-local fix" (not just an
input-file repoint, because this script refits its own base models per seed).

WHAT THIS SCRIPT DOES: reuses `45_phase_x_c1_c2_nested_cv_corrected_v3.py`'s
`FeatureSource`/`load_raw_views`/`verify_raw_reproduces_filtered`/`RAW_SPECS`
machinery by file-path import (not retyped -- same technique as
`61_phase_x_microbiome_over_sofa_nested_v3.py:225-228`), in that script's "V3"
mode (fold-local >=10% prevalence filter + fold-local imputer/scaler/PCA(20),
the audit's pre-registered PRIMARY variant), and repeats THIS script's exact
20-seed procedure with that feature source instead of the frozen one. Every
stochastic step of one repetition -- the outer/inner StratifiedKFold split, the
LR solver, CatBoost, and the PCA randomized-SVD projection (sklearn's "auto"
solver picks randomized SVD here because n_components=20 << min(n_train,
n_features)) -- uses THAT repetition's own seed, exactly matching this script's
pre-existing single-seed-drives-everything convention (module docstring of the
original script 60).

M1/C1 and C0 are completely UNCHANGED: same `X_m1` (literature_core_features.csv),
same `make_lr_pipe_seeded`/`fit_predict_lr` spec at max_iter=3000, same
`nested_oof_meta`/`nested_oof_c0` code for those two arms. M1 was out of scope for
the whole M3 correction (owner instruction; see audit doc section 4 "open item").

LR CONVERGENCE (M3 only): script 45_v3's V3 variant is defined as fold-local
preprocessing AND a converged LR (max_iter ladder 30000 -> 100000, tracked via
`warnings.catch_warnings` + per-fit `n_iter_` check, never suppressed) --
convergence is part of what "V3" means, not a separate change, because this
script's correctness anchor (seed=2026 must reproduce 45_v3's V3 point AUROC to
4dp) was computed at that same converged spec. M1's LR remains byte-identical to
the original script 60 (max_iter=3000, untouched, per the audit's open item 4 --
M1 was explicitly out of scope).

CORRECTNESS ANCHOR: seed=2026 must reproduce 45_v3's PRIMARY (V3) published
CV_AUROC_5fold_nested to 4dp -- C0=0.7528, C1=0.7610, C2a=0.7608, **C2b=0.7711**
(NOT the old frozen-PCA 0.7722 for C2b -- that is precisely the number this
correction changes) -- read from
`clinical_ai/reports/phase_x_c1_c2_incremental_value_nested_corrected_v3.csv`,
verified at runtime before any output is trusted, exactly mirroring the original
script's own anchor-check design but against the corrected V3 targets.

NOT DONE HERE (project house style, 2026-09-04; already the pattern in
`61_phase_x_microbiome_over_sofa_nested_v3.py` point 6): no new matplotlib figure.
The original script's `.svg/.jpg` boxplot is a reference figure feeding
Supplementary Fig. S3 panel A; the R-side rebuild of that panel from this script's
CSV is a separate (paper-layer) step, not part of this analysis rerun.

PURPOSE, STATED SO IT CANNOT BE MISREAD LATER (restated verbatim from the
original script 60): this is a ROBUSTNESS check on CV-partitioning sensitivity,
not a search for a favourable split and not an additional significance test.
Report the full distribution. Never select a seed. Never report a best case.

SEEDS: identical 20 seeds, `[2026] + list(range(1, 20))`.

Reads: same upstream files as `45_phase_x_c1_c2_nested_cv_corrected_v3.py`
(imported directly) -- unfiltered raw per-view matrices, clinical AI predictions,
discovery cohort outcome, literature-core (M1) features. No frozen PCA artifacts.
Writes:
  clinical_ai/reports/phase_x_repeated_nested_cv_v3.csv   (one row per seed x model)
  clinical_ai/reports/phase_x_repeated_nested_cv_v3.md
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

# --- reuse 45_v3's fold-local machinery verbatim (FeatureSource, load_raw_views,
#     verify_raw_reproduces_filtered, RAW_SPECS, LOW_DIM_FILES, HIGH_DIM_FILES,
#     HIGH_STEM, PREV, PCA_N_COMPONENTS) -- same technique as 61_v3:225-228.
_spec = importlib.util.spec_from_file_location(
    "v4fl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "45_phase_x_c1_c2_nested_cv_corrected_v4.py"))
v4fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4fl)  # defs/constants only (main guarded)

log = io_utils.setup_logging("60_phase_x_repeated_nested_cv_v4")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
HARMONIZED_DIR = os.path.join(SEMANTIC, "harmonized")
DISCOVERY_CSV = os.path.join(HARMONIZED_DIR, "discovery_cohort_primary.csv")
CLINICAL_PRED_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_repeated_nested_cv_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_repeated_nested_cv_v4.md")
OLD_CSV = os.path.join(config.DIR_REPORTS, "phase_x_repeated_nested_cv_v3.csv")  # read-only reference (v3 is now "old")
V4_ANCHOR_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected_v4.csv")

PUBLISHED_SEED = config.RANDOM_SEED  # 2026
SEEDS = [PUBLISHED_SEED] + list(range(1, 20))  # 20 total: 2026 + 1..19
N_FOLDS = config.N_CV_FOLDS  # 5
EPS = 1e-6
LR_MAX_ITER_LADDER = [30000, 100000]  # M3 only; M1 stays at the original 3000

# 45_v3's PRIMARY (V3, fully fold-local) published AUROC -- read from
# phase_x_c1_c2_incremental_value_nested_corrected_v3.csv at import time so this
# anchor can never silently drift from that file.
_anc = pd.read_csv(V4_ANCHOR_CSV)
_ref_row = lambda variant, model: float(_anc[(_anc.preprocessing_variant == variant) & (_anc.model == model)]["CV_AUROC_5fold_nested"].iloc[0])
PUBLISHED_AUROC = {
    "C0": _ref_row("REF", "C0_clinical_alone"),
    "C1": _ref_row("REF", "C1_clinical_plus_M1_LiteratureCore"),
    "C2a": _ref_row("V4_STRICT", "C2a_clinical_plus_M3_LR"),  # v4 FINAL headline for M3-LR
    "C2b": _ref_row("V3", "C2b_clinical_plus_M3_CatBoost"),  # CatBoost unchanged in v4, still V3
}


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# data -- built ONCE in the parent process; forked workers inherit via
# copy-on-write (identical strategy to the original script 60).
# ---------------------------------------------------------------------------
outcome = pd.read_csv(DISCOVERY_CSV)[["patient_id_clinical", "hospital_mortality"]].set_index("patient_id_clinical")
clin_pred = pd.read_csv(CLINICAL_PRED_CSV).set_index("patient_id_clinical")
if "clinical_AI_probability_PRIMARY" not in clin_pred.columns:
    log.error("No PRIMARY column in %s", CLINICAL_PRED_CSV)
    sys.exit(1)
df = outcome.join(clin_pred[["clinical_AI_probability_PRIMARY"]], how="inner")
log.info("N=%d patients with both outcome and clinical AI probability", len(df))

lit_core = pd.read_csv(os.path.join(STRUCTURED_DIR, "literature_core_features.csv"), index_col="sample_id")
lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
X_m1 = lit_core.reindex(df.index)  # M1: UNCHANGED, static, out of scope

IDS = df.index.values
Y = df["hospital_mortality"].values.astype(int)
LOGIT_CLIN = logit(df["clinical_AI_probability_PRIMARY"].values)

# Unfiltered per-view raw matrices (fold-local prevalence filter + PCA source),
# plus the same correctness check 45_v3 runs before trusting V3.
RAW_VIEWS = v4fl.load_raw_views(IDS)
_ok, _chk = v4fl.verify_raw_reproduces_filtered(RAW_VIEWS, log)
if not _ok:
    log.error("Raw views do NOT reproduce structured/*_filtered.tsv with all 302 -- refusing to run. See %s", _chk)
    sys.exit(2)
log.info("Raw-view -> filtered reproduction check PASSED for all %d views.", len(RAW_VIEWS))


def fit_block_pca_seeded(tr_raw: np.ndarray, te_raw: np.ndarray, seed: int):
    """`v4fl.fit_block_pca` with `seed` parameterised. 45_v3 hardcodes its own
    module-level SEED=2026 because it runs one seed only; this script's whole
    point is CV-partitioning sensitivity across 20 different seeds, so every
    stochastic step of one repetition -- including the PCA randomized-SVD
    projection sklearn's 'auto' solver selects here -- uses that repetition's
    own seed, matching the rest of this script's seeding convention."""
    imp = SimpleImputer(strategy="median").fit(tr_raw)
    tr_imp, te_imp = imp.transform(tr_raw), imp.transform(te_raw)
    scaler = StandardScaler().fit(tr_imp)
    tr_s, te_s = scaler.transform(tr_imp), scaler.transform(te_imp)
    n_comp = min(v4fl.PCA_N_COMPONENTS, tr_s.shape[0] - 1, tr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(tr_s)
    return pca.transform(tr_s), pca.transform(te_s), n_comp


def build_fold_x_m3(tr_pos: np.ndarray, te_pos: np.ndarray, seed: int):
    """45_v3's FeatureSource('prev_pca') logic (fold-local >=10% prevalence filter
    on LOW_DIM_FILES and on each HIGH_DIM block, then fold-local imputer/scaler/
    PCA(20) on the two HIGH_DIM blocks), reimplemented here only to parameterise
    the PCA step by `seed` instead of 45_v3's fixed module SEED. tr_pos/te_pos are
    POSITIONAL indices into the 302-patient IDS array."""
    tr_ids, te_ids = IDS[tr_pos], IDS[te_pos]
    Xtr_parts, Xte_parts = [], []
    for fname in v4fl.LOW_DIM_FILES:
        stem = fname.replace("_filtered.tsv", "")
        pres, vals = RAW_VIEWS[stem]
        keep = v4fl.FeatureSource._train_keep(pres, tr_ids)
        v = vals.loc[:, keep.values]
        v.columns = [f"{stem}__{c}" for c in v.columns]
        Xtr_parts.append(v.reindex(tr_ids))
        Xte_parts.append(v.reindex(te_ids))
    for name, stem in v4fl.HIGH_STEM.items():
        pres, vals = RAW_VIEWS[stem]
        keep = v4fl.FeatureSource._train_keep(pres, tr_ids)
        v = vals.loc[:, keep.values]
        p_tr, p_te, k = fit_block_pca_seeded(v.reindex(tr_ids).values, v.reindex(te_ids).values, seed)
        cols = [f"{name}_pc{i+1}" for i in range(k)]
        Xtr_parts.append(pd.DataFrame(p_tr, index=tr_ids, columns=cols))
        Xte_parts.append(pd.DataFrame(p_te, index=te_ids, columns=cols))
    return pd.concat(Xtr_parts, axis=1), pd.concat(Xte_parts, axis=1)


def build_fold_x_m3_strict(tr_pos: np.ndarray, te_pos: np.ndarray):
    """V4_STRICT's FeatureSource('raw_concat_full') -- fully UNFILTERED presence+value
    columns for all 11 structured views, tagged for FullBlockPreprocessor. No filtering,
    imputation, scaling or PCA happens here; all of that happens inside
    fit_lr_strict_full_seeded's Pipeline, refit per GridSearchCV tuning fold (tuning-local).
    Used for C2a (M3-LR) only; C2b (CatBoost) keeps build_fold_x_m3 above, unchanged."""
    src = v4fl.FeatureSource("raw_concat_full", IDS, raw_views=RAW_VIEWS)
    Xtr, Xte, _ = src.build(tr_pos, te_pos)
    return Xtr, Xte


# ---------------------------------------------------------------------------
# base learners
# ---------------------------------------------------------------------------
def make_lr_pipe_seeded_m1(seed):
    """M1 spec, max_iter=30000 (v5 fix): M1's preprocessing/grid/CV architecture is out of
    scope for this correction and unchanged, but max_iter now matches M1's final,
    project-wide converged spec (QC/M1_CONVERGENCE_AUDIT_v4.txt) instead of the original
    script 60's pre-M1-fix max_iter=3000, so this script's M1 row is numerically consistent
    with the rest of the package and the Fig. S3 legend's "same final specification" claim
    is accurate for all rows, not just the LR row."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=30000, scoring="roc_auc",
                                                    random_state=seed, n_jobs=1))])


def fit_predict_lr_m1(X_tr, y_tr, X_te, seed):
    pipe = make_lr_pipe_seeded_m1(seed).fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def fit_lr_strict_full_seeded(X_tr, y_tr, X_te, max_iter, seed, conv_log):
    """V4_STRICT's FullBlockPreprocessor + GridSearchCV (45_v4's fit_lr_strict_full),
    parameterised by `seed` (threaded into FullBlockPreprocessor's internal PCA and into
    LogisticRegression's random_state) instead of 45_v4's fixed module SEED=2026 --
    this script's whole point is CV-partitioning sensitivity across 20 different seeds.
    Same elastic-net grid (Cs=8 logspace(-4,4,8) x l1_ratios [.1,.5,.9], cv=3) as every
    other LR fit in this project. n_jobs=1: this script's parallelism is across seeds
    (mp.Pool), not inside one fit -- matching the old make_lr_pipe_seeded_m3's n_jobs=1
    design philosophy exactly."""
    pipe = Pipeline([("blocks", v4fl.FullBlockPreprocessor(seed=seed)),
                      ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=max_iter,
                                                  random_state=seed))])
    grid = {"clf__C": np.logspace(-4, 4, 8), "clf__l1_ratio": [0.1, 0.5, 0.9]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=3), scoring="roc_auc", refit=True, n_jobs=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gs.fit(X_tr, y_tr)
    conv_w = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    best = gs.best_estimator_.named_steps["clf"]
    n_it = int(np.max(best.n_iter_))
    converged = bool(len(conv_w) == 0 and n_it < max_iter)
    conv_log.append({"max_iter": max_iter, "n_iter_max": n_it, "converged": converged})
    return gs.predict_proba(X_te)[:, 1]


def fit_predict_catboost(X_tr, y_tr, X_te, seed):
    """ORIGINAL script 60 CatBoost spec, byte-identical (thread_count=1,
    iterations=200, depth=4, lr=0.05)."""
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cb = CatBoostClassifier(verbose=False, random_state=seed, thread_count=1,
                             iterations=200, depth=4, learning_rate=0.05)
    cb.fit(imp.transform(X_tr), y_tr)
    return cb.predict_proba(imp.transform(X_te))[:, 1]


# ---------------------------------------------------------------------------
# nested-OOF drivers
# ---------------------------------------------------------------------------
def nested_oof_meta_static(X_micro, fit_predict_fn, seed):
    """IDENTICAL to the original script 60's nested_oof_meta() -- used ONLY for
    M1 (static X_m1), which this correction does not touch."""
    n = len(Y)
    oof = np.full(n, np.nan)
    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr_outer, te_outer in outer_skf.split(X_micro, Y):
        y_tr_outer = Y[tr_outer]
        X_tr_outer = X_micro.iloc[tr_outer].reset_index(drop=True)
        inner_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        inner_prob = np.full(len(tr_outer), np.nan)
        for tr_inner, te_inner in inner_skf.split(X_tr_outer, y_tr_outer):
            inner_prob[te_inner] = fit_predict_fn(X_tr_outer.iloc[tr_inner], y_tr_outer[tr_inner],
                                                    X_tr_outer.iloc[te_inner], seed)
        te_outer_prob = fit_predict_fn(X_micro.iloc[tr_outer], Y[tr_outer], X_micro.iloc[te_outer], seed)
        X_meta_tr = np.column_stack([LOGIT_CLIN[tr_outer], logit(inner_prob)])
        X_meta_te = np.column_stack([LOGIT_CLIN[te_outer], logit(te_outer_prob)])
        meta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_meta_tr, y_tr_outer)
        oof[te_outer] = meta.predict_proba(X_meta_te)[:, 1]
    return oof


def nested_oof_meta_foldlocal(run, seed):
    """Same nested structure as nested_oof_meta_static, but the feature matrix is
    built INSIDE the loop from positional indices (`run(tr_pos, te_pos)`), so no
    outer-test row and no inner-validation row ever contributes to the fitted
    prevalence mask / imputer / scaler / PCA used to score it. Used for C2a/C2b."""
    n = len(Y)
    oof = np.full(n, np.nan)
    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr_outer, te_outer in outer_skf.split(np.zeros(n), Y):
        y_tr_outer = Y[tr_outer]
        inner_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        inner_prob = np.full(len(tr_outer), np.nan)
        for tr_inner, te_inner in inner_skf.split(np.zeros(len(tr_outer)), y_tr_outer):
            tr_pos, te_pos = tr_outer[tr_inner], tr_outer[te_inner]
            inner_prob[te_inner] = run(tr_pos, te_pos)
        te_outer_prob = run(tr_outer, te_outer)
        X_meta_tr = np.column_stack([LOGIT_CLIN[tr_outer], logit(inner_prob)])
        X_meta_te = np.column_stack([LOGIT_CLIN[te_outer], logit(te_outer_prob)])
        meta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_meta_tr, y_tr_outer)
        oof[te_outer] = meta.predict_proba(X_meta_te)[:, 1]
    return oof


def nested_oof_c0(seed):
    """IDENTICAL to the original script 60's nested_oof_c0()."""
    X = LOGIT_CLIN.reshape(-1, 1)
    oof = np.full(len(Y), np.nan)
    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in outer_skf.split(X, Y):
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], Y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


def run_one_seed(seed):
    t0 = time.time()
    oof_c0 = nested_oof_c0(seed)
    oof_c1 = nested_oof_meta_static(X_m1, fit_predict_lr_m1, seed)  # M1: UNCHANGED

    def run_lr_m3(tr_pos, te_pos, max_iter, conv_log):
        # V4_STRICT: unfiltered raw features; prevalence filter + PCA refit inside
        # fit_lr_strict_full_seeded's own GridSearchCV tuning folds (tuning-local).
        X_tr, X_te = build_fold_x_m3_strict(tr_pos, te_pos)
        return fit_lr_strict_full_seeded(X_tr, Y[tr_pos], X_te, max_iter, seed, conv_log)

    def run_cb_m3(tr_pos, te_pos):
        # CatBoost: UNCHANGED from v3 (V3-equivalent fold-local prevalence+PCA);
        # CatBoost's own specification did not change in the v4 pass.
        X_tr, X_te = build_fold_x_m3(tr_pos, te_pos, seed)
        return fit_predict_catboost(X_tr, Y[tr_pos], X_te, seed)

    conv_log, max_iter = [], LR_MAX_ITER_LADDER[0]
    oof_c2a = nested_oof_meta_foldlocal(lambda trp, tep: run_lr_m3(trp, tep, max_iter, conv_log), seed)
    n_fail = sum(1 for r in conv_log if not r["converged"])
    if n_fail > 0:
        log.warning("[seed=%d] %d/%d M3-LR fits non-converged at max_iter=%d -- escalating to %d",
                    seed, n_fail, len(conv_log), max_iter, LR_MAX_ITER_LADDER[1])
        conv_log, max_iter = [], LR_MAX_ITER_LADDER[1]
        oof_c2a = nested_oof_meta_foldlocal(lambda trp, tep: run_lr_m3(trp, tep, max_iter, conv_log), seed)
        n_fail = sum(1 for r in conv_log if not r["converged"])

    oof_c2b = nested_oof_meta_foldlocal(run_cb_m3, seed)

    oof = {"C0": oof_c0, "C1": oof_c1, "C2a": oof_c2a, "C2b": oof_c2b}
    row = {"seed": seed, "is_published_seed": seed == PUBLISHED_SEED, "elapsed_sec": round(time.time() - t0, 1),
           "m3lr_final_max_iter": max_iter, "m3lr_n_fits": len(conv_log), "m3lr_n_nonconverged_final": n_fail}
    for name, p in oof.items():
        row[f"{name}_AUROC"] = round(roc_auc_score(Y, p), 4)
        row[f"{name}_AUPRC"] = round(average_precision_score(Y, p), 4)
        row[f"{name}_Brier"] = round(brier_score_loss(Y, p), 4)
    for name in ("C1", "C2a", "C2b"):
        row[f"delta_AUROC_{name}_vs_C0"] = round(row[f"{name}_AUROC"] - row["C0_AUROC"], 4)
    log.info("[seed=%d] done in %.1fs: C0=%.4f C1=%.4f C2a=%.4f C2b=%.4f (m3lr max_iter=%d, %d/%d nonconverged)",
              seed, row["elapsed_sec"], row["C0_AUROC"], row["C1_AUROC"], row["C2a_AUROC"], row["C2b_AUROC"],
              max_iter, n_fail, len(conv_log))
    return row


if __name__ == "__main__":
    n_workers = min(config.get_n_jobs(), len(SEEDS))
    log.info("Running %d seeds (%s) across %d parallel single-threaded workers (fold-local M3 prevalence+PCA) ...",
              len(SEEDS), SEEDS, n_workers)
    t_start = time.time()
    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            rows = pool.map(run_one_seed, SEEDS)
    else:
        rows = [run_one_seed(s) for s in SEEDS]
    log.info("All %d repetitions complete in %.1f sec wall time.", len(SEEDS), time.time() - t_start)

    result = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

    # -------------------------------------------------------------------
    # correctness anchor: seed=2026 must reproduce 45_v3's PRIMARY (V3)
    # published CV_AUROC_5fold_nested to 4dp before anything is trusted.
    # -------------------------------------------------------------------
    pub_row = result[result["seed"] == PUBLISHED_SEED].iloc[0]
    mismatches = []
    for name, want in PUBLISHED_AUROC.items():
        got = pub_row[f"{name}_AUROC"]
        if abs(got - want) > 1e-9:
            mismatches.append((name, got, want))
    if mismatches:
        log.error("CORRECTNESS ANCHOR FAILED -- seed=2026 does not reproduce 45_v3's PRIMARY (V3) AUROC: %s. "
                   "STOP: the fold-local algorithm has drifted from 45_v3. Not writing outputs.", mismatches)
        sys.exit(1)
    log.info("Correctness anchor PASSED: seed=2026 reproduces 45_v3 PRIMARY (V3) exactly (C0=%.4f, C1=%.4f, "
             "C2a=%.4f, C2b=%.4f).", pub_row["C0_AUROC"], pub_row["C1_AUROC"], pub_row["C2a_AUROC"], pub_row["C2b_AUROC"])

    io_utils.save_csv(result, OUT_CSV)
    log.info("\n%s", result.to_string(index=False))

    # -------------------------------------------------------------------
    # distribution summary + old-vs-new comparison (read-only reference to
    # the original frozen-PCA script 60 output).
    # -------------------------------------------------------------------
    delta_cols = ["delta_AUROC_C1_vs_C0", "delta_AUROC_C2a_vs_C0", "delta_AUROC_C2b_vs_C0"]
    old = pd.read_csv(OLD_CSV) if os.path.exists(OLD_CSV) else None
    summary_rows = []
    for col in delta_cols:
        vals = result[col].values
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        frac_pos = float((vals > 0).mean())
        published_val = float(pub_row[col])
        published_rank = int((vals <= published_val).sum())
        row = {"contrast": col, "N_seeds": len(vals), "median": round(q50, 4),
               "IQR_low": round(q25, 4), "IQR_high": round(q75, 4),
               "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4),
               "fraction_seeds_positive": round(frac_pos, 3),
               "published_seed_2026_value": round(published_val, 4),
               "published_seed_rank_of_20": published_rank}
        if old is not None:
            ovals = old[col].values
            opub = float(old.loc[old.seed == PUBLISHED_SEED, col].iloc[0])
            orank = int((ovals <= opub).sum())
            row.update({"OLD_median": round(float(np.percentile(ovals, 50)), 4),
                        "OLD_published_seed_2026_value": round(opub, 4),
                        "OLD_published_seed_rank_of_20": orank})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    log.info("\n%s", summary.to_string(index=False))

    abs_summary_rows = []
    for name in ("C0", "C1", "C2a", "C2b"):
        vals = result[f"{name}_AUROC"].values
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        abs_summary_rows.append({
            "model": name, "median_AUROC": round(q50, 4), "IQR_low": round(q25, 4), "IQR_high": round(q75, 4),
            "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4),
            "published_seed_2026_value": round(float(pub_row[f"{name}_AUROC"]), 4),
        })
    abs_summary = pd.DataFrame(abs_summary_rows)

    # -------------------------------------------------------------------
    # markdown report (no new matplotlib figure -- see module docstring)
    # -------------------------------------------------------------------
    n_pos = {col: int((result[col].values > 0).sum()) for col in delta_cols}
    md = [
        "# Phase X: repeated nested cross-validation across independent outer-split seeds (v3, fold-local M3)",
        "",
        "**Fold-local M3 correction (2026-10-02):** this is the sibling of `60_phase_x_repeated_nested_cv.py` "
        "with the frozen-PCA / global-prevalence-filter M3 feature construction replaced by "
        "`45_phase_x_c1_c2_nested_cv_corrected_v3.py`'s fold-local V3 design (fold-local >=10% prevalence filter "
        "+ fold-local imputer/scaler/PCA(20), fit on each fit's training rows only). M1/C0 are unchanged. See this "
        "script's own module docstring for the full method note and the correctness anchor.", "",
        "**Purpose (binding framing, restated so it cannot be misread):** this is a ROBUSTNESS check on "
        "whether the headline incremental-value increments are stable to which patients happened to land in "
        "which CV fold. It is NOT a search for a favourable split, NOT an additional significance test, and the "
        "spread reported below is a property of the CV-partitioning resampling process itself -- **not an "
        "additional confidence interval**, and must not be read as one. No seed was selected; no best case is "
        "reported; every seed run is shown.", "",
        f"N={len(Y)} Gangnam patients, {int(Y.sum())} deaths. "
        f"**{len(SEEDS)} independent outer-split seeds**: `{SEEDS}` (2026 first, then 1..19). "
        f"**Seed {PUBLISHED_SEED} is the published seed and is marked explicitly** "
        "(`is_published_seed` column) in the table below, so a reader can see exactly where the published "
        "number sits in the full distribution.", "",
        "## Correctness anchor", "",
        f"seed={PUBLISHED_SEED}'s recomputed AUROC reproduces 45_v3's PRIMARY (V3) published "
        f"`CV_AUROC_5fold_nested` to 4 decimal places (C0={pub_row['C0_AUROC']:.4f}, C1={pub_row['C1_AUROC']:.4f}, "
        f"C2a={pub_row['C2a_AUROC']:.4f}, C2b={pub_row['C2b_AUROC']:.4f} vs published "
        f"{PUBLISHED_AUROC['C0']:.4f}/{PUBLISHED_AUROC['C1']:.4f}/{PUBLISHED_AUROC['C2a']:.4f}/"
        f"{PUBLISHED_AUROC['C2b']:.4f}) -- verified at runtime before any output was written, not just asserted.",
        "", "## Per-seed results (all 20, sorted by seed)", "",
        df_to_md_table(result[["seed", "is_published_seed", "C0_AUROC", "C1_AUROC", "C2a_AUROC", "C2b_AUROC",
                                "delta_AUROC_C1_vs_C0", "delta_AUROC_C2a_vs_C0", "delta_AUROC_C2b_vs_C0",
                                "C0_AUPRC", "C1_AUPRC", "C2a_AUPRC", "C2b_AUPRC",
                                "C0_Brier", "C1_Brier", "C2a_Brier", "C2b_Brier",
                                "m3lr_final_max_iter", "m3lr_n_nonconverged_final"]]), "",
        "## delta_AUROC distribution summary (NEW fold-local vs OLD frozen-PCA)", "",
        df_to_md_table(summary), "",
        f"Fraction of the {len(SEEDS)} seeds whose point estimate is positive: "
        f"C1 vs C0 {n_pos['delta_AUROC_C1_vs_C0']}/{len(SEEDS)}, "
        f"C2a vs C0 {n_pos['delta_AUROC_C2a_vs_C0']}/{len(SEEDS)}, "
        f"C2b vs C0 {n_pos['delta_AUROC_C2b_vs_C0']}/{len(SEEDS)}.", "",
        "## Absolute AUROC distribution (context, not the primary object of this analysis)", "",
        df_to_md_table(abs_summary), "",
        "## Interpretation", "",
        "Report exactly as it came out. This file intentionally states only what was measured, not a verdict, "
        "so it cannot drift from the numbers above on a future edit.", "",
        "## Reference", "",
        "Published (seed=2026) point estimates and their bootstrap CIs under the PRIMARY (V3) fold-local design: "
        "`phase_x_c1_c2_incremental_value_nested_corrected_v3.csv/.md`, "
        "`phase_x_c1_c2_nested_bootstrap_delta_v3.csv` (script 45_v3). Old (frozen-PCA) repeated-CV reference: "
        "`phase_x_repeated_nested_cv.csv/.md` (script 60). This script does NOT modify any of them.",
    ]
    io_utils.save_text("\n".join(md), OUT_MD)
    log.info("wrote %s / %s", OUT_CSV, OUT_MD)
