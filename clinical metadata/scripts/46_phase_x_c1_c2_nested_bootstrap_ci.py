#!/usr/bin/env python3
"""46_phase_x_c1_c2_nested_bootstrap_ci.py

Manuscript-blocking gap (2026-09-02): `manuscript_numbers.json`'s
`delta_AUROC_C1_vs_C0` / `delta_AUROC_C2b_vs_C0` entries explicitly state
"no corrected CI/bootstrap_p has been computed for the nested design" --
the manuscript currently pairs the nested-CV-corrected POINT estimates
(script 45, C1 +0.0082 / C2a +0.0080 / C2b +0.0194) with CI/p-values taken
from the older, confirmed-leaky single-split design (script 21:
[-0.0159,+0.0284] p=0.515 for C1; [-0.0086,+0.0480] p=0.189 for C2b).

This script closes that gap. It does NOT re-derive the nested-CV fix itself
(script 45 already did that, and is not modified here, following this
project's own precedent of new-sibling-script-not-in-place-edit for a
correction, restated in 45's own docstring). It:

  1. Recomputes the SAME nested (outer 5-fold honest-eval / inner 5-fold
     meta-training) out-of-fold probabilities as script 45, using IDENTICAL
     code (feature construction, base-model specs, meta-model spec,
     SEED=2026, StratifiedKFold order) -- this is a correctness check as
     much as a re-derivation: the AUROCs are asserted to match script 45's
     published CV_AUROC_5fold_nested_corrected values before anything else
     runs.
  2. Applies EXACTLY the paired-bootstrap algorithm already used in
     `21_phase_x_paired_bootstrap_and_interaction.py` (2000 resamples,
     seed=2026, `rng.randint(0, n, n)` with-replacement patient resampling,
     SAME resample index used for both models each iteration, skip a
     resample if it has <2 outcome classes, delta_AUROC/AUPRC/Brier with
     percentile 95% CI and two-sided bootstrap p = 2*min(tail proportions))
     to the nested OOF probabilities, for C1/C2a/C2b vs C0.
  3. SOFA-referenced contrasts (task item 2's "if script 45 produced those
     OOF probabilities"): script 45 does NOT compute any SOFA-referenced
     candidate (S0/S1/S2a/S2b/S3* live only in
     `41_phase_x_microbiome_over_sofa.py`, which consumes script 18's
     ORIGINAL leaky-CV OOF probabilities, not script 45's nested-corrected
     ones) -- so there is nothing to bootstrap here for SOFA under the
     nested design. Explicitly noted as out of scope, not silently skipped.

Reads:
  Same upstream files as 45_phase_x_c1_c2_nested_cv_corrected.py (clinical
  AI predictions, discovery cohort outcome, structured microbiome feature
  files, frozen M3 PCA loadings).
Writes:
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities.csv
  clinical_ai/reports/phase_x_c1_c2_nested_bootstrap_delta.csv
  clinical_ai/reports/phase_x_c1_c2_nested_bootstrap_delta.md
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("46_phase_x_c1_c2_nested_bootstrap_ci")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
FROZEN_DIR = os.path.join(SEMANTIC, "supervised_semantic", "frozen_models")
DISCOVERY_CSV = os.path.join(SEMANTIC, "harmonized", "discovery_cohort_primary.csv")
CLINICAL_PRED_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")
SELECTED_JSON = os.path.join(config.DIR_REPORTS, "selected_reduced_model.json")

NESTED_CORRECTED_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected.csv")
OUT_OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities.csv")
OUT_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_nested_bootstrap_delta.csv")
OUT_BOOT_MD = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_nested_bootstrap_delta.md")

SEED = config.RANDOM_SEED  # 2026
N_FOLDS = config.N_CV_FOLDS  # 5
N_BOOT = 2000  # matches 21_phase_x_paired_bootstrap_and_interaction.py exactly
EPS = 1e-6

LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}


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
# 1. load outcome + clinical AI probability -- identical to script 45.
# ---------------------------------------------------------------------------
outcome = pd.read_csv(DISCOVERY_CSV)[["patient_id_clinical", "hospital_mortality"]].set_index("patient_id_clinical")
clin_pred = pd.read_csv(CLINICAL_PRED_CSV).set_index("patient_id_clinical")
if "clinical_AI_probability_PRIMARY" not in clin_pred.columns:
    log.error("No PRIMARY column in %s -- run 15_select_best_reduced_model.py then re-run "
              "10_apply_reduced_model_to_microbiome_cohort.py first.", CLINICAL_PRED_CSV)
    sys.exit(1)

with open(SELECTED_JSON) as fh:
    selected_meta = json.load(fh)

df = outcome.join(clin_pred[["clinical_AI_probability_PRIMARY"]], how="inner")
log.info("N=%d patients with both outcome and clinical AI probability", len(df))

# ---------------------------------------------------------------------------
# 2. microbiome feature matrices -- identical to script 45.
# ---------------------------------------------------------------------------
lit_core = pd.read_csv(os.path.join(STRUCTURED_DIR, "literature_core_features.csv"), index_col="sample_id")
lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
X_m1 = lit_core.reindex(df.index)

low_parts = []
for fname in LOW_DIM_FILES:
    d = pd.read_csv(os.path.join(STRUCTURED_DIR, fname), sep="\t", index_col="patient_id")
    d.columns = [f"{fname.replace('_filtered.tsv', '')}__{c}" for c in d.columns]
    low_parts.append(d)
X_low = pd.concat(low_parts, axis=1).reindex(df.index)

pca_parts = [X_low]
pca_dir = os.path.join(FROZEN_DIR, "M3_pca_loadings")
for name, fname in HIGH_DIM_FILES.items():
    raw = pd.read_csv(os.path.join(STRUCTURED_DIR, fname), sep="\t", index_col="patient_id").reindex(df.index)
    imp = joblib.load(os.path.join(pca_dir, f"{name}_imputer.joblib"))
    raw_imp = imp.transform(raw.values)
    scaler_mean = np.load(os.path.join(pca_dir, f"{name}_scaler_mean.npy"))
    scaler_scale = np.load(os.path.join(pca_dir, f"{name}_scaler_scale.npy"))
    raw_scaled = (raw_imp - scaler_mean) / scaler_scale
    pca_mean = np.load(os.path.join(pca_dir, f"{name}_mean.npy"))
    loadings = np.load(os.path.join(pca_dir, f"{name}_loadings.npy"))
    pcs = (raw_scaled - pca_mean) @ loadings.T
    pca_parts.append(pd.DataFrame(pcs, index=df.index, columns=[f"{name}_pc{i+1}" for i in range(loadings.shape[0])]))
X_m3 = pd.concat(pca_parts, axis=1)

y = df["hospital_mortality"].values
n = len(y)
logit_clin = logit(df["clinical_AI_probability_PRIMARY"].values)


def make_lr_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                    random_state=SEED, n_jobs=config.get_n_jobs()))])


def fit_predict_lr(X_tr, y_tr, X_te):
    pipe = make_lr_pipe().fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def fit_predict_catboost(X_tr, y_tr, X_te):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cb = CatBoostClassifier(verbose=False, random_state=SEED, thread_count=config.get_n_jobs(),
                             iterations=200, depth=4, learning_rate=0.05)
    cb.fit(imp.transform(X_tr), y_tr)
    return cb.predict_proba(imp.transform(X_te))[:, 1]


# ---------------------------------------------------------------------------
# 3. nested OOF -- IDENTICAL logic/seeds to script 45's nested_oof_meta /
#    nested_oof_c0, so the recomputed OOF arrays here are the exact same
#    ones that produced 45's published CV_AUROC_5fold_nested_corrected
#    values (asserted below, not just assumed).
# ---------------------------------------------------------------------------
outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def nested_oof_meta(X_micro, fit_predict_fn):
    oof = np.full(n, np.nan)
    for tr_outer, te_outer in outer_skf.split(X_micro, y):
        y_tr_outer = y[tr_outer]
        X_tr_outer = X_micro.iloc[tr_outer].reset_index(drop=True)

        inner_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        inner_prob = np.full(len(tr_outer), np.nan)
        for tr_inner, te_inner in inner_skf.split(X_tr_outer, y_tr_outer):
            inner_prob[te_inner] = fit_predict_fn(X_tr_outer.iloc[tr_inner], y_tr_outer[tr_inner], X_tr_outer.iloc[te_inner])

        te_outer_prob = fit_predict_fn(X_micro.iloc[tr_outer], y[tr_outer], X_micro.iloc[te_outer])

        X_meta_tr = np.column_stack([logit_clin[tr_outer], logit(inner_prob)])
        X_meta_te = np.column_stack([logit_clin[te_outer], logit(te_outer_prob)])
        meta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_meta_tr, y_tr_outer)
        oof[te_outer] = meta.predict_proba(X_meta_te)[:, 1]
    return oof


def nested_oof_c0():
    X = logit_clin.reshape(-1, 1)
    oof = np.full(n, np.nan)
    for tr, te in outer_skf.split(X, y):
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


CANDIDATES = {
    "C0_clinical_alone": None,
    "C1_clinical_plus_M1_LiteratureCore": (X_m1, fit_predict_lr),
    "C2a_clinical_plus_M3_LR": (X_m3, fit_predict_lr),
    "C2b_clinical_plus_M3_CatBoost": (X_m3, fit_predict_catboost),
}

oof_arrays = {}
for name, spec in CANDIDATES.items():
    log.info("Recomputing nested OOF for %s ...", name)
    oof_arrays[name] = nested_oof_c0() if spec is None else nested_oof_meta(*spec)
    log.info("%s: recomputed CV_AUROC_5fold_nested = %.4f", name, roc_auc_score(y, oof_arrays[name]))

oof_df = pd.DataFrame({"hospital_mortality": y}, index=df.index)
for name, arr in oof_arrays.items():
    oof_df[f"oof_prob_{name}"] = arr
oof_df["clinical_AI_probability_PRIMARY"] = df["clinical_AI_probability_PRIMARY"].values
io_utils.save_csv(oof_df, OUT_OOF_CSV, index=True)
log.info("wrote %s", OUT_OOF_CSV)

# ---------------------------------------------------------------------------
# 3b. correctness check: recomputed AUROCs must match script 45's published
#     CV_AUROC_5fold_nested_corrected values (same code, same seed).
# ---------------------------------------------------------------------------
if os.path.exists(NESTED_CORRECTED_CSV):
    ref = pd.read_csv(NESTED_CORRECTED_CSV).set_index("model")["CV_AUROC_5fold_nested_corrected"]
    for name, arr in oof_arrays.items():
        recomputed = round(roc_auc_score(y, arr), 4)
        published = ref.get(name)
        status = "MATCH" if published is not None and abs(recomputed - published) < 1e-9 else "MISMATCH"
        log.info("Correctness check %s: recomputed=%.4f published(script 45)=%s -> %s",
                  name, recomputed, published, status)
        if status == "MISMATCH":
            log.error("Recomputed nested OOF does not reproduce script 45's published AUROC for %s "
                       "-- refusing to proceed on a possibly-divergent recomputation.", name)
            sys.exit(1)
else:
    log.warning("%s not found -- skipping correctness check against script 45's published values.",
                NESTED_CORRECTED_CSV)

# ---------------------------------------------------------------------------
# 4. paired bootstrap -- EXACTLY 21_phase_x_paired_bootstrap_and_interaction
#    .py's algorithm, applied to the nested OOF probabilities instead of the
#    original single-split ones.
# ---------------------------------------------------------------------------
CONTRAST_CANDIDATES = {
    "C1_clinical_plus_M1_LiteratureCore": "oof_prob_C1_clinical_plus_M1_LiteratureCore",
    "C2a_clinical_plus_M3_LR": "oof_prob_C2a_clinical_plus_M3_LR",
    "C2b_clinical_plus_M3_CatBoost": "oof_prob_C2b_clinical_plus_M3_CatBoost",
}
c0 = oof_df["oof_prob_C0_clinical_alone"].values

rng = np.random.RandomState(SEED)
boot_rows = []
for name, col in CONTRAST_CANDIDATES.items():
    ck = oof_df[col].values
    deltas_auroc, deltas_auprc, deltas_brier = [], [], []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        c0b, ckb = c0[idx], ck[idx]
        deltas_auroc.append(roc_auc_score(yb, ckb) - roc_auc_score(yb, c0b))
        deltas_auprc.append(average_precision_score(yb, ckb) - average_precision_score(yb, c0b))
        deltas_brier.append(brier_score_loss(yb, c0b) - brier_score_loss(yb, ckb))  # positive = ck has LOWER (better) Brier

    def summarize(deltas, point):
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_two_sided = 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0))
        return round(point, 4), round(lo, 4), round(hi, 4), round(min(p_two_sided, 1.0), 4)

    d_auroc = roc_auc_score(y, ck) - roc_auc_score(y, c0)
    d_auprc = average_precision_score(y, ck) - average_precision_score(y, c0)
    d_brier = brier_score_loss(y, c0) - brier_score_loss(y, ck)
    auroc_pt, auroc_lo, auroc_hi, auroc_p = summarize(deltas_auroc, d_auroc)
    auprc_pt, auprc_lo, auprc_hi, auprc_p = summarize(deltas_auprc, d_auprc)
    brier_pt, brier_lo, brier_hi, brier_p = summarize(deltas_brier, d_brier)

    boot_rows.append({
        "model": name, "N_boot": N_BOOT, "N_effective_resamples": len(deltas_auroc),
        "delta_AUROC": auroc_pt, "delta_AUROC_CI_low": auroc_lo, "delta_AUROC_CI_high": auroc_hi, "delta_AUROC_boot_p": auroc_p,
        "delta_AUPRC": auprc_pt, "delta_AUPRC_CI_low": auprc_lo, "delta_AUPRC_CI_high": auprc_hi, "delta_AUPRC_boot_p": auprc_p,
        "delta_Brier_improvement": brier_pt, "delta_Brier_CI_low": brier_lo, "delta_Brier_CI_high": brier_hi, "delta_Brier_boot_p": brier_p,
    })

boot_df = pd.DataFrame(boot_rows)

# side-by-side with the original leaky-design bootstrap, for the explicit
# old-vs-new comparison the task asks for.
ORIG_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_paired_bootstrap_delta.csv")
if os.path.exists(ORIG_BOOT_CSV):
    orig = pd.read_csv(ORIG_BOOT_CSV)[["model", "delta_AUROC", "delta_AUROC_CI_low", "delta_AUROC_CI_high", "delta_AUROC_boot_p"]].rename(
        columns={"delta_AUROC": "delta_AUROC_original_leaky", "delta_AUROC_CI_low": "CI_low_original_leaky",
                 "delta_AUROC_CI_high": "CI_high_original_leaky", "delta_AUROC_boot_p": "boot_p_original_leaky"})
    boot_df = boot_df.merge(orig, on="model", how="left")

io_utils.save_csv(boot_df, OUT_BOOT_CSV)
log.info("\n%s", boot_df.to_string(index=False))

md = [
    "# Phase X: nested-CV-corrected paired bootstrap delta AUROC/AUPRC/Brier (C1/C2a/C2b vs C0)", "",
    f"N={n} Gangnam patients, {int(y.sum())} deaths ({100*y.mean():.1f}%). Fills the gap flagged "
    "explicitly in `manuscript_numbers.json` (\"no corrected CI/bootstrap_p has been computed for "
    "the nested design\"): the nested-CV-corrected point estimates from "
    "`45_phase_x_c1_c2_nested_cv_corrected.py` were previously reported alongside CI/p-values taken "
    "from the older, confirmed CV-nesting-leakage-affected single-split design "
    "(`phase_x_paired_bootstrap_delta.md`, revision_log.md issue 19).", "",
    "**Method**: nested OOF probabilities recomputed with IDENTICAL code/seeds to script 45 "
    "(correctness-checked below to reproduce script 45's published AUROCs exactly), then the SAME "
    f"paired-bootstrap algorithm as `21_phase_x_paired_bootstrap_and_interaction.py` ({N_BOOT} "
    "with-replacement patient resamples, seed=2026, the SAME resample index applied to both C0 and "
    "the candidate each iteration, resamples with <2 outcome classes skipped, 95% CI from bootstrap "
    "percentiles, two-sided bootstrap p = 2 x min(tail proportions crossing zero)).", "",
    "**SOFA-referenced contrasts**: NOT computed here. Script 45 (the nested-CV correction) produced "
    "OOF probabilities only for C0/C1/C2a/C2b; the SOFA-referenced S0-S3 candidates live in "
    "`41_phase_x_microbiome_over_sofa.py`, which consumes script 18's ORIGINAL leaky-CV OOF "
    "probabilities, not script 45's nested ones -- there is no nested-CV SOFA OOF to bootstrap. "
    "Explicitly out of scope, not silently skipped.", "",
    "## Nested-corrected paired bootstrap (this script)", "",
    df_to_md_table(boot_df), "",
    "## Reference", "",
    "Nested-CV-corrected point estimates: `phase_x_c1_c2_incremental_value_nested_corrected.csv/.md` "
    "(script 45). Original (leaky-CV, pre-nested) paired bootstrap: `phase_x_paired_bootstrap_delta.csv/.md` "
    "(script 21, revision_log.md issue 19).",
]
with open(OUT_BOOT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s / %s", OUT_BOOT_CSV, OUT_BOOT_MD)
