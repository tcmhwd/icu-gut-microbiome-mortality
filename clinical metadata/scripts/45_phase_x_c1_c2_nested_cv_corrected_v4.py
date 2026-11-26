#!/usr/bin/env python3
"""45_phase_x_c1_c2_nested_cv_corrected_v4.py

Sibling of 45_phase_x_c1_c2_nested_cv_corrected_v3.py (itself untouched),
owner-authorised 2026-10-02 "final technical closure pass". Two changes on
top of v3:

(1) M1 convergence fix. v3 deliberately pinned M1 ("REF" variant) at the old
    max_iter=3000 as a reproduction anchor (7 of 35 LR fits non-converged).
    That instruction is now superseded: M1 goes through the SAME convergence
    ladder (30000 -> 100000) already used for M3, with the 16-marker
    definition, feature construction, penalty, C grid, l1_ratio grid, CV
    architecture, outcome and cohort all held byte-identical. Only the
    numerical optimisation changes.

(2) V4_STRICT variant: the "V3+tuning-local" combination v3's own audit
    (QC/M3_FOLD_LOCAL_METHOD_AUDIT_v3.md section 6) flagged as "not run here...
    if the owner wants it, it should be pre-registered". V2S (v3) made PCA
    tuning-local but left the prevalence filter global. V4_STRICT makes BOTH
    the prevalence filter AND the PCA tuning-local -- i.e. refit inside
    LogisticRegressionCV's own hyperparameter-selection CV, not just inside
    the outer/stacking-inner folds -- via a GridSearchCV(Pipeline[...]) with
    the identical elastic-net grid. See FullBlockPreprocessor below. This
    was NOT pre-registered before V2S's partial result (C2a crossing nominal
    significance) was already seen; that tension is disclosed rather than
    hidden in QC/DATA_DRIVEN_LR_STRICT_METHOD_AUDIT_v4.md, per instruction.

WHAT WAS GLOBAL (script 45, lines 179-192; identically 18:152-176, 46:138-151,
60/61 same block): the M3 feature matrix X_m3 was built ONCE, before any fold
split, by applying FROZEN imputer -> scaler -> PCA(20) artifacts
(semantic_microbiome/supervised_semantic/frozen_models/M3_pca_loadings/, fit by
lit14_freeze_pipeline.py:102-118 on all 302 patients) to the two high-dimensional
blocks MAG_ABUND / EGGNOG_KO. Every base-model fit (single-level OOF, nested
inner, nested outer-full) therefore saw a transform estimated partly on its own
held-out rows. A second, upstream global step: the >=10% prevalence filter of
semantic_microbiome/scripts/09_build_structured_views.py (common.prevalence_filter)
decided which columns exist in structured/*_filtered.tsv, using all 302 patients.

WHAT THIS SCRIPT DOES: every base-model fit receives POSITIONAL INDICES
(train_idx, test_idx) into the 302-patient frame, and the feature matrix is
built inside the fit from the training rows only (FeatureSource.build). The
nested structure of script 45 is reproduced exactly (same StratifiedKFold
objects / seeds, so identical partitions); only what is fitted where changes:

  stage                         transform fit on          applied to
  single-level OOF (standalone) tr (4/5 of 302)           te
  nested (a) inner OOF          tr_outer[tr_inner]        tr_outer[te_inner]
  nested (b) outer-full         tr_outer                  te_outer
  nested (c) meta LR            (no microbiome transform; inputs are (a)/(b) probs)

So no outer-test row and no inner-validation row ever contributes to any fitted
imputer / scaler / PCA / prevalence mask used to score it.

VARIANTS (run one per SLURM array task with --variant; then --aggregate):
  REF       C0 + C1 (M1) only -- 16-marker definition/penalty/grids/CV architecture
            byte-identical to 45; ONLY max_iter now goes through the ladder (v4 fix).
  V0        frozen PCA, LR max_iter=3000      -- must reproduce 45/46/18 exactly (anchor)
  V1        frozen PCA, LR converged          -- isolates the convergence change
  V2        fold-local PCA (global prevalence filter), LR converged  -- requested PCA fix
  V3        fold-local prevalence filter + fold-local PCA, LR converged -- fully fold-local
  V2S       sensitivity to V2: PCA also refit inside LR's own 3-fold C/l1 tuning CV
            (GridSearchCV over a Pipeline with the same grid as LogisticRegressionCV).
            Same model class, same grid; labelled sensitivity, not primary (v3).
  V4_STRICT v4 FINAL headline for the data-driven LR model: V3's fold-local prevalence
            filter made ALSO tuning-local (FullBlockPreprocessor), combined with V2S's
            tuning-local PCA, both inside one Pipeline refit by GridSearchCV's cv=3.
            C2b (CatBoost) is untouched -- it has no internal tuning CV, so it was
            already maximally fold-local in V3; V4_STRICT reuses V3's C2b verbatim.

CONVERGENCE: LR max_iter ladder (30000 -> 100000). A fit counts as converged
only if (i) no ConvergenceWarning was emitted during .fit() (captured with
warnings.catch_warnings(record=True) + simplefilter("always"); NOT suppressed)
AND (ii) no LogisticRegressionCV path has n_iter_ >= max_iter. (The LRCV refit's
iteration count is discarded by sklearn 1.4.2, so (i) is the only check for the
refit -- it runs in the calling thread, where warnings are reliably captured.)
If any M3-LR fit of a variant fails at 30000, the WHOLE variant is rerun at
100000 so one max_iter applies to every fit of a variant. If it still fails,
the variant is marked NONCONVERGED (blocker), outputs kept but flagged.

CatBoost spec (iterations=200, depth=4, lr=0.05, fixed) and meta-model spec are
unchanged from 45. No hyperparameter search added.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
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

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
sys.path.insert(0, os.path.join(SEMANTIC, "scripts"))
import common as sm_common  # noqa: E402  (load_cohort_matrix, prevalence_filter, PREVALENCE_THRESHOLD)

STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
FROZEN_DIR = os.path.join(SEMANTIC, "supervised_semantic", "frozen_models")
HARMONIZED_DIR = os.path.join(SEMANTIC, "harmonized")
DISCOVERY_CSV = os.path.join(HARMONIZED_DIR, "discovery_cohort_primary.csv")
MAPPING_CSV = os.path.join(HARMONIZED_DIR, "sample_id_mapping.csv")
CLINICAL_PRED_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")

# old (frozen-PCA) reference outputs -- READ ONLY
OLD_OOF18_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")
OLD_NESTED46_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities.csv")
OLD_45_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected.csv")
OLD_46_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_nested_bootstrap_delta.csv")

WORK_DIR = os.path.join(config.DIR_PREDICTIONS, "v4_fold_local")  # separate from v3's v3_fold_local/
OUT_OOF_LONG = os.path.join(config.DIR_PREDICTIONS, "phase_x_m3_fold_local_oof_long_v4.csv")
OUT_REPORT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected_v4.csv")
OUT_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_nested_bootstrap_delta_v4.csv")
OUT_STANDALONE_CSV = os.path.join(config.DIR_REPORTS, "phase_x_m3_standalone_auroc_v4.csv")
OUT_CONV_CSV = os.path.join(config.DIR_REPORTS, "phase_x_m3_fold_local_convergence_v4.csv")
OUT_ANCHOR_CSV = os.path.join(config.DIR_REPORTS, "phase_x_m3_fold_local_reproduction_anchor_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected_v4.md")

SEED = config.RANDOM_SEED  # 2026
N_FOLDS = config.N_CV_FOLDS  # 5
EPS = 1e-6
PCA_N_COMPONENTS = 20  # lit14_freeze_pipeline.py:93
PREV = sm_common.PREVALENCE_THRESHOLD  # 0.10
N_BOOT_DELTA = 2000  # script 46
N_BOOT_AUROC = config.N_BOOTSTRAP  # 1000, script 56
ORIG_LR_MAX_ITER = 3000
MAX_ITER_LADDER = [30000, 100000]

LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}

# Raw (pre-prevalence-filter) sources, exactly as 09_build_structured_views.py
# reads them. key = filtered-file stem (column prefix used by 45).
MO = os.path.join(MICROBIOME_BASE, "merge", "output")
MS = os.path.join(MO, "mag_summary")
RAW_SPECS = {
    "taxonomy_species": ("MetaPhlAn", os.path.join(MO, "metaphlan3", "metaphlan3_species_merged.tsv"), "clade_name", "plain"),
    "taxonomy_genus": ("MetaPhlAn", os.path.join(MO, "metaphlan3", "metaphlan3_genus_merged.tsv"), "clade_name", "plain"),
    "humann_pathway": ("HUMAnN3", os.path.join(MO, "humann3", "humann_pathabundance_wide.unstratified.tsv"), "# Pathway", "humann"),
    "contig_rgi_drugclass": ("contig_RGI", os.path.join(MS, "rgi_contig", "rgi_contig_drug_class_per_sample.tsv"), "feature", "plain"),
    "contig_rgi_rpkm_drugclass": ("RPKM_RGI_contig", os.path.join(MS, "rpkm_rgi", "rpkm_rgi_drug_class_per_sample.tsv"), "feature", "plain"),
    "mag_rgi_drugclass": ("MAG_RGI", os.path.join(MS, "rgi", "rgi_drug_class_per_sample.tsv"), "feature", "plain"),
    "vfdb_category": ("VFDB", os.path.join(MS, "vfdb", "vfdb_category_per_sample.tsv"), "feature", "plain"),
    "antismash_bgc_types": ("antiSMASH", os.path.join(MS, "antismash", "antismash_bgc_types_per_sample.tsv"), "feature", "antismash"),
    "eggnog_cog": ("eggNOG", os.path.join(MS, "eggnog", "eggnog_cog_per_sample.tsv"), "feature", "plain"),
    "mag_abundance": ("MAG_abundance_CoverM", os.path.join(MS, "coverm", "coverm_abundance_matrix.tsv"), "MAG", "coverm"),
    "eggnog_kegg_ko": ("eggNOG", os.path.join(MS, "eggnog", "eggnog_kegg_ko_per_sample.tsv"), "feature", "plain"),
}
HIGH_STEM = {"MAG_ABUND": "mag_abundance", "EGGNOG_KO": "eggnog_kegg_ko"}

MODEL_LABELS = {
    "M1": "M1 literature-core score (elastic-net LR, frozen 16-marker feature set)",
    "M3_LR": "M3 data-driven elastic-net logistic regression",
    "M3_CatBoost": "M3 data-driven CatBoost",
    "C0_clinical_alone": "Reduced clinical AI alone (C0)",
    "C1_clinical_plus_M1_LiteratureCore": "Clinical AI + M1 (C1)",
    "C2a_clinical_plus_M3_LR": "Clinical AI + M3 elastic-net LR (C2a)",
    "C2b_clinical_plus_M3_CatBoost": "Clinical AI + M3 CatBoost (C2b)",
}
VARIANT_DESC = {
    "REF": "C0/C1 (M1): 16-marker/penalty/grid/CV spec unchanged, LR max_iter now on the convergence ladder (v4 fix)",
    "V0": "frozen global PCA, LR max_iter=3000 (reproduction of published 45/46/18)",
    "V1": "frozen global PCA, LR converged (convergence effect only)",
    "V2": "fold-local imputer/scaler/PCA; global prevalence filter; LR converged",
    "V3": "fold-local prevalence filter + fold-local imputer/scaler/PCA; LR converged",
    "V2S": "V2 + PCA refit inside LR's 3-fold C/l1 tuning CV (GridSearchCV, same grid) -- sensitivity",
    "V4_STRICT": "V3's prevalence filter AND PCA both refit inside LR's 3-fold C/l1 tuning CV -- v4 FINAL headline for data-driven LR",
}
LOW_STEMS = [f.replace("_filtered.tsv", "") for f in LOW_DIM_FILES]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row.values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data (identical to 45 section 1/2)
# ---------------------------------------------------------------------------
def load_base():
    outcome = pd.read_csv(DISCOVERY_CSV)[["patient_id_clinical", "hospital_mortality"]].set_index("patient_id_clinical")
    clin_pred = pd.read_csv(CLINICAL_PRED_CSV).set_index("patient_id_clinical")
    df = outcome.join(clin_pred[["clinical_AI_probability_PRIMARY"]], how="inner")
    lit_core = pd.read_csv(os.path.join(STRUCTURED_DIR, "literature_core_features.csv"), index_col="sample_id")
    lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
    X_m1 = lit_core.reindex(df.index)
    low_parts = []
    for fname in LOW_DIM_FILES:
        d = pd.read_csv(os.path.join(STRUCTURED_DIR, fname), sep="\t", index_col="patient_id")
        d.columns = [f"{fname.replace('_filtered.tsv', '')}__{c}" for c in d.columns]
        low_parts.append(d)
    X_low = pd.concat(low_parts, axis=1).reindex(df.index)
    high_raw = {name: pd.read_csv(os.path.join(STRUCTURED_DIR, fname), sep="\t", index_col="patient_id").reindex(df.index)
                for name, fname in HIGH_DIM_FILES.items()}
    return df, X_m1, X_low, high_raw


def frozen_x_m3(df, X_low):
    """Byte-for-byte 45:179-192 (frozen artifacts)."""
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
    return pd.concat(pca_parts, axis=1)


def load_raw_views(cohort_ids):
    """Unfiltered patients x features per view, reproducing 09_build_structured_views.py's
    loading/normalisation exactly. Returns {stem: (presence_matrix, value_matrix)} where the
    prevalence filter is computed on presence_matrix (>0) and values are taken from value_matrix
    (they differ only for HUMAnN3: prevalence on raw RPK, values = per-sample relative proportion)."""
    mapping = pd.read_csv(MAPPING_CSV)
    cohort = pd.read_csv(DISCOVERY_CSV)
    out = {}
    for stem, (view, path, id_col, kind) in RAW_SPECS.items():
        mat = sm_common.load_cohort_matrix(view, Path(path), id_col, mapping, cohort)
        if kind == "humann":
            special = [c for c in mat.columns if c in ("UNMAPPED", "UNINTEGRATED")]
            mat_bio = mat.drop(columns=special)
            total = mat_bio.sum(axis=1)
            rel = mat_bio.div(total.replace(0, np.nan), axis=0).fillna(0.0)
            out[stem] = (mat_bio, rel)
        elif kind == "coverm":
            if "unmapped" in mat.columns:
                mat = mat.drop(columns=["unmapped"])
            out[stem] = (mat, mat)
        elif kind == "antismash":
            m_total = mapping[(mapping.view == "antiSMASH") & (mapping.match_method != "unmatched")]
            m_total = m_total[m_total.patient_id_clinical.isin(set(cohort["patient_id_clinical"]))]
            full_ids = sorted(set(m_total["patient_id_clinical"]))
            mat = mat.reindex(full_ids, fill_value=0.0)
            out[stem] = (mat, mat)
        else:
            out[stem] = (mat, mat)
    return out


def verify_raw_reproduces_filtered(raw_views, log):
    """Correctness anchor for V3: applying the prevalence filter to ALL rows of each raw
    view must reproduce structured/*_filtered.tsv's columns and values exactly."""
    rows = []
    ok_all = True
    for stem, (pres, vals) in raw_views.items():
        ref = pd.read_csv(os.path.join(STRUCTURED_DIR, f"{stem}_filtered.tsv"), sep="\t", index_col="patient_id")
        keep = (pres > 0).sum(axis=0) >= (PREV * len(pres))
        mine = vals.loc[:, keep.values]
        same_cols = list(mine.columns.astype(str)) == list(ref.columns.astype(str))
        same_rows = set(mine.index) == set(ref.index)
        maxdiff = np.nan
        if same_cols and same_rows:
            maxdiff = float(np.nanmax(np.abs(mine.reindex(ref.index).values.astype(float) - ref.values.astype(float))))
        ok = same_cols and same_rows and maxdiff < 1e-9
        ok_all &= ok
        rows.append({"view": stem, "n_patients_raw": len(pres), "n_features_raw": pres.shape[1],
                     "n_features_filtered_all302": int(keep.sum()), "n_features_reference_file": ref.shape[1],
                     "columns_identical": same_cols, "rows_identical": same_rows, "max_abs_value_diff": maxdiff, "PASS": ok})
        log.info("raw->filtered reproduction %s: raw=%d feats, filtered=%d vs ref=%d, cols_same=%s rows_same=%s maxdiff=%s -> %s",
                 stem, pres.shape[1], int(keep.sum()), ref.shape[1], same_cols, same_rows, maxdiff, "PASS" if ok else "FAIL")
    return ok_all, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fold-local feature construction
# ---------------------------------------------------------------------------
def fit_block_pca(tr_raw: np.ndarray, te_raw: np.ndarray):
    """lit09_data_driven_nested_cv.py:build_fold_matrix() per-block logic, TRAIN-only fit."""
    imp = SimpleImputer(strategy="median").fit(tr_raw)
    tr_imp, te_imp = imp.transform(tr_raw), imp.transform(te_raw)
    scaler = StandardScaler().fit(tr_imp)
    tr_s, te_s = scaler.transform(tr_imp), scaler.transform(te_imp)
    n_comp = min(PCA_N_COMPONENTS, tr_s.shape[0] - 1, tr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(tr_s)
    return pca.transform(tr_s), pca.transform(te_s), n_comp


class FeatureSource:
    """build(train_idx, test_idx) -> (X_tr, X_te, info). Indices are POSITIONS into the
    302-patient frame. Every fitted transform uses only train_idx rows."""

    def __init__(self, mode, ids, X_static=None, X_low=None, high_raw=None, raw_views=None):
        self.mode, self.ids = mode, ids
        self.X_static, self.X_low, self.high_raw, self.raw_views = X_static, X_low, high_raw, raw_views

    def build(self, tr, te):
        if self.mode == "static":  # M1, or frozen-PCA X_m3
            return self.X_static.iloc[tr], self.X_static.iloc[te], {}
        tr_ids, te_ids = self.ids[tr], self.ids[te]
        info = {}
        if self.mode == "pca":  # global prevalence filter, fold-local PCA
            Xtr_parts, Xte_parts = [self.X_low.iloc[tr]], [self.X_low.iloc[te]]
            for name, raw in self.high_raw.items():
                p_tr, p_te, k = fit_block_pca(raw.iloc[tr].values, raw.iloc[te].values)
                cols = [f"{name}_pc{i+1}" for i in range(k)]
                Xtr_parts.append(pd.DataFrame(p_tr, index=tr_ids, columns=cols))
                Xte_parts.append(pd.DataFrame(p_te, index=te_ids, columns=cols))
            return pd.concat(Xtr_parts, axis=1), pd.concat(Xte_parts, axis=1), info
        if self.mode == "prev_pca":  # fold-local prevalence filter + fold-local PCA
            Xtr_parts, Xte_parts = [], []
            for fname in LOW_DIM_FILES:
                stem = fname.replace("_filtered.tsv", "")
                pres, vals = self.raw_views[stem]
                keep = self._train_keep(pres, tr_ids)
                info[f"nfeat_{stem}"] = int(keep.sum())
                v = vals.loc[:, keep.values]
                v.columns = [f"{stem}__{c}" for c in v.columns]
                Xtr_parts.append(v.reindex(tr_ids))
                Xte_parts.append(v.reindex(te_ids))
            for name, stem in HIGH_STEM.items():
                pres, vals = self.raw_views[stem]
                keep = self._train_keep(pres, tr_ids)
                info[f"nfeat_{stem}"] = int(keep.sum())
                v = vals.loc[:, keep.values]
                p_tr, p_te, k = fit_block_pca(v.reindex(tr_ids).values, v.reindex(te_ids).values)
                cols = [f"{name}_pc{i+1}" for i in range(k)]
                Xtr_parts.append(pd.DataFrame(p_tr, index=tr_ids, columns=cols))
                Xte_parts.append(pd.DataFrame(p_te, index=te_ids, columns=cols))
            return pd.concat(Xtr_parts, axis=1), pd.concat(Xte_parts, axis=1), info
        if self.mode == "raw_concat":  # V2S: low-dim + raw high-dim; PCA happens inside the Pipeline
            parts_tr, parts_te = [self.X_low.iloc[tr]], [self.X_low.iloc[te]]
            for name, raw in self.high_raw.items():
                r = raw.copy()
                r.columns = [f"{name}__raw__{c}" for c in r.columns]
                parts_tr.append(r.iloc[tr])
                parts_te.append(r.iloc[te])
            return pd.concat(parts_tr, axis=1), pd.concat(parts_te, axis=1), info
        if self.mode == "raw_concat_full":
            # V4_STRICT: fully unfiltered presence+value columns for ALL 11 structured
            # views (9 low-dim + 2 high-dim), tagged so FullBlockPreprocessor can find
            # them. Filtering AND PCA happen inside the Pipeline (tuning-local).
            parts_tr, parts_te = [], []
            for stem in LOW_STEMS + list(HIGH_STEM.values()):
                pres, vals = self.raw_views[stem]
                pres_t, val_t = pres.copy(), vals.copy()
                pres_t.columns = [f"__PRES__{stem}__{c}" for c in pres.columns]
                val_t.columns = [f"__VAL__{stem}__{c}" for c in vals.columns]
                parts_tr.append(pres_t.reindex(tr_ids))
                parts_tr.append(val_t.reindex(tr_ids))
                parts_te.append(pres_t.reindex(te_ids))
                parts_te.append(val_t.reindex(te_ids))
            return pd.concat(parts_tr, axis=1), pd.concat(parts_te, axis=1), info
        raise ValueError(self.mode)

    @staticmethod
    def _train_keep(pres, tr_ids):
        """09's rule (>0 in >= 10% of the view's patients), denominator = TRAINING patients
        that have this view (09 used all cohort patients that have the view)."""
        sub = pres.loc[pres.index.intersection(tr_ids)]
        return (sub > 0).sum(axis=0) >= (PREV * len(sub))


class HighDimBlockPCA(BaseEstimator, TransformerMixin):
    """V2S only: per-block imputer->scaler->PCA as a Pipeline step, so that
    GridSearchCV's own tuning folds refit it."""

    def __init__(self, n_components=PCA_N_COMPONENTS, seed=SEED):
        self.n_components, self.seed = n_components, seed

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.blocks_ = {name: [c for c in X.columns if str(c).startswith(f"{name}__raw__")] for name in HIGH_DIM_FILES}
        used = set(c for cols in self.blocks_.values() for c in cols)
        self.low_cols_ = [c for c in X.columns if c not in used]
        self.fitted_ = {}
        for name, cols in self.blocks_.items():
            raw = X[cols].values
            imp = SimpleImputer(strategy="median").fit(raw)
            ri = imp.transform(raw)
            sc = StandardScaler().fit(ri)
            k = min(self.n_components, ri.shape[0] - 1, ri.shape[1])
            pca = PCA(n_components=k, random_state=self.seed).fit(sc.transform(ri))
            self.fitted_[name] = (imp, sc, pca)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        parts = [X[self.low_cols_].values]
        for name, cols in self.blocks_.items():
            imp, sc, pca = self.fitted_[name]
            parts.append(pca.transform(sc.transform(imp.transform(X[cols].values))))
        return np.hstack(parts)


class FullBlockPreprocessor(BaseEstimator, TransformerMixin):
    """V4_STRICT only: prevalence filter for ALL 11 structured views (9 low-dim
    column-selection-only, 2 high-dim filter+imputer+scaler+PCA), as ONE Pipeline
    step, so GridSearchCV's own 3-fold tuning CV refits the ENTIRE chain --
    filter included, not just PCA (cf. HighDimBlockPCA used by V2S, which keeps
    the global prevalence filter). fit() sees only the rows GridSearchCV hands
    it for that tuning-training fold; prevalence is >0-count >= PREV fraction of
    those rows, matching FeatureSource._train_keep's rule exactly."""

    def __init__(self, n_components=PCA_N_COMPONENTS, seed=SEED, prevalence=PREV):
        self.n_components, self.seed, self.prevalence = n_components, seed, prevalence

    @staticmethod
    def _cols(X, stem, tag):
        return [c for c in X.columns if c.startswith(f"__{tag}__{stem}__")]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_cols_ = {}
        self.high_fit_ = {}
        for stem in LOW_STEMS:
            pres_cols, val_cols = self._cols(X, stem, "PRES"), self._cols(X, stem, "VAL")
            pres = X[pres_cols].values
            # Denominator must be rows that HAVE this view (non-NaN after the upstream
            # reindex to the full fold), matching FeatureSource._train_keep exactly --
            # a row lacking the view entirely is NaN across the whole presence block and
            # must be excluded from both numerator and denominator, not counted as absent.
            has_view = ~np.isnan(pres).any(axis=1)
            pres_v = pres[has_view]
            keep = (pres_v > 0).sum(axis=0) >= (self.prevalence * pres_v.shape[0])
            self.low_cols_[stem] = list(np.array(val_cols)[keep])
        for name, stem in HIGH_STEM.items():
            pres_cols, val_cols = self._cols(X, stem, "PRES"), self._cols(X, stem, "VAL")
            pres = X[pres_cols].values
            has_view = ~np.isnan(pres).any(axis=1)
            pres_v = pres[has_view]
            keep = (pres_v > 0).sum(axis=0) >= (self.prevalence * pres_v.shape[0])
            kept_cols = list(np.array(val_cols)[keep])
            raw = X[kept_cols].values
            imp = SimpleImputer(strategy="median").fit(raw)
            ri = imp.transform(raw)
            sc = StandardScaler().fit(ri)
            k = min(self.n_components, ri.shape[0] - 1, ri.shape[1])
            pca = PCA(n_components=k, random_state=self.seed).fit(sc.transform(ri))
            self.high_fit_[name] = (kept_cols, imp, sc, pca)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        parts = [X[cols].values for cols in self.low_cols_.values()]
        for name, (cols, imp, sc, pca) in self.high_fit_.items():
            parts.append(pca.transform(sc.transform(imp.transform(X[cols].values))))
        return np.hstack(parts)


# ---------------------------------------------------------------------------
# base learners with convergence tracking
# ---------------------------------------------------------------------------
CONV_LOG: list[dict] = []
FEAT_LOG: list[dict] = []


def make_lr_pipe(max_iter):
    """45's make_lr_pipe() with max_iter as the ONLY change."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=max_iter, scoring="roc_auc",
                                                    random_state=SEED, n_jobs=config.get_n_jobs()))])


def fit_lr(X_tr, y_tr, X_te, max_iter, ctx):
    pipe = make_lr_pipe(max_iter)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X_tr, y_tr)
    conv_w = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    clf = pipe.named_steps["clf"]
    n_iter = np.asarray(clf.n_iter_)
    n_at_max = int((n_iter >= max_iter).sum())
    CONV_LOG.append({**ctx, "learner": "LogisticRegressionCV(saga, elasticnet)", "max_iter": max_iter,
                     "n_train": len(y_tr), "n_features_in": X_tr.shape[1],
                     "n_cv_paths": int(n_iter.size), "n_iter_max_over_cv_paths": int(n_iter.max()),
                     "n_iter_median_over_cv_paths": float(np.median(n_iter)),
                     "n_cv_paths_reaching_max_iter": n_at_max, "n_convergence_warnings_incl_refit": len(conv_w),
                     "selected_C": float(clf.C_[0]), "selected_l1_ratio": float(clf.l1_ratio_[0]),
                     "converged": bool(n_at_max == 0 and len(conv_w) == 0)})
    return pipe.predict_proba(X_te)[:, 1]


def fit_lr_strict(X_tr, y_tr, X_te, max_iter, ctx):
    """V2S: same elastic-net grid as LRCV(Cs=8 -> logspace(-4,4,8), l1 in {.1,.5,.9}, cv=3
    unshuffled StratifiedKFold, roc_auc), but PCA is a Pipeline step so it is refit in each
    tuning fold. Threading backend so warnings from tuning fits are captured in-process."""
    pipe = Pipeline([("blocks", HighDimBlockPCA()), ("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=max_iter,
                                                random_state=SEED))])
    grid = {"clf__C": np.logspace(-4, 4, 8), "clf__l1_ratio": [0.1, 0.5, 0.9]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=3), scoring="roc_auc", refit=True,
                      n_jobs=config.get_n_jobs())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with joblib.parallel_backend("threading", n_jobs=config.get_n_jobs()):
            gs.fit(X_tr, y_tr)
    conv_w = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    best = gs.best_estimator_.named_steps["clf"]
    n_it = int(np.max(best.n_iter_))
    CONV_LOG.append({**ctx, "learner": "GridSearchCV(Pipeline[BlockPCA,LR saga elasticnet])", "max_iter": max_iter,
                     "n_train": len(y_tr), "n_features_in": X_tr.shape[1],
                     "n_cv_paths": 72, "n_iter_max_over_cv_paths": np.nan, "n_iter_median_over_cv_paths": np.nan,
                     "n_cv_paths_reaching_max_iter": np.nan, "n_iter_refit": n_it,
                     "n_convergence_warnings_incl_refit": len(conv_w),
                     "selected_C": float(gs.best_params_["clf__C"]), "selected_l1_ratio": float(gs.best_params_["clf__l1_ratio"]),
                     "converged": bool(len(conv_w) == 0 and n_it < max_iter)})
    return gs.predict_proba(X_te)[:, 1]


def fit_lr_strict_full(X_tr, y_tr, X_te, max_iter, ctx):
    """V4_STRICT: identical elastic-net grid/CV as fit_lr_strict, but the
    Pipeline's first step is FullBlockPreprocessor instead of HighDimBlockPCA,
    so the prevalence filter (not just PCA) is also refit inside GridSearchCV's
    cv=3 tuning folds -- the fully tuning-local combination."""
    pipe = Pipeline([("blocks", FullBlockPreprocessor()), ("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=max_iter,
                                                random_state=SEED))])
    grid = {"clf__C": np.logspace(-4, 4, 8), "clf__l1_ratio": [0.1, 0.5, 0.9]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=3), scoring="roc_auc", refit=True,
                      n_jobs=config.get_n_jobs())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with joblib.parallel_backend("threading", n_jobs=config.get_n_jobs()):
            gs.fit(X_tr, y_tr)
    conv_w = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    best = gs.best_estimator_.named_steps["clf"]
    n_it = int(np.max(best.n_iter_))
    blocks = gs.best_estimator_.named_steps["blocks"]
    CONV_LOG.append({**ctx, "learner": "GridSearchCV(Pipeline[FullBlockPreprocessor,LR saga elasticnet])", "max_iter": max_iter,
                     "n_train": len(y_tr), "n_features_in": X_tr.shape[1],
                     "n_cv_paths": 72, "n_iter_max_over_cv_paths": np.nan, "n_iter_median_over_cv_paths": np.nan,
                     "n_cv_paths_reaching_max_iter": np.nan, "n_iter_refit": n_it,
                     "n_convergence_warnings_incl_refit": len(conv_w),
                     "selected_C": float(gs.best_params_["clf__C"]), "selected_l1_ratio": float(gs.best_params_["clf__l1_ratio"]),
                     "converged": bool(len(conv_w) == 0 and n_it < max_iter)})
    FEAT_LOG.append({**ctx, "n_train": len(y_tr),
                     **{f"nfeat_{k}": len(v) for k, v in blocks.low_cols_.items()},
                     **{f"nfeat_{stem}": len(blocks.high_fit_[name][0]) for name, stem in HIGH_STEM.items()}})
    return gs.predict_proba(X_te)[:, 1]


def fit_catboost(X_tr, y_tr, X_te, max_iter, ctx):
    """45's fit_predict_catboost(), unchanged."""
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cb = CatBoostClassifier(verbose=False, random_state=SEED, thread_count=config.get_n_jobs(),
                             iterations=200, depth=4, learning_rate=0.05)
    cb.fit(imp.transform(X_tr), y_tr)
    return cb.predict_proba(imp.transform(X_te))[:, 1]


# ---------------------------------------------------------------------------
# CV drivers (partitions identical to 45: same StratifiedKFold args, same y)
# ---------------------------------------------------------------------------
def make_runner(source, learner, y, max_iter, variant, model_code):
    def run(tr, te, stage, outer_fold, inner_fold):
        X_tr, X_te, info = source.build(tr, te)
        ctx = {"variant": variant, "model_code": model_code, "stage": stage,
               "outer_fold": outer_fold, "inner_fold": inner_fold}
        if info:
            FEAT_LOG.append({**ctx, "n_train": len(tr), **info})
        return learner(X_tr, y[tr], X_te, max_iter, ctx)
    return run


def single_level_oof(run, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    p = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        p[te] = run(tr, te, "single_level_oof", k, -1)
        fold[te] = k
    return p, fold


def nested_oof_meta(run, y, logit_clin):
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    meta_log = []
    for k, (tr_o, te_o) in enumerate(outer.split(np.zeros(len(y)), y)):
        y_tr_o = y[tr_o]
        inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        inner_prob = np.full(len(tr_o), np.nan)
        for j, (tr_i, te_i) in enumerate(inner.split(np.zeros(len(tr_o)), y_tr_o)):
            # (a) transform + base model fit on tr_outer[tr_inner] ONLY
            inner_prob[te_i] = run(tr_o[tr_i], tr_o[te_i], "nested_inner", k, j)
        # (b) transform + base model fit on the full tr_outer, applied to te_outer
        te_prob = run(tr_o, te_o, "nested_outer_full", k, -1)
        # (c) meta model, unchanged from 45
        X_meta_tr = np.column_stack([logit_clin[tr_o], logit(inner_prob)])
        X_meta_te = np.column_stack([logit_clin[te_o], logit(te_prob)])
        meta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_meta_tr, y_tr_o)
        meta_log.append({"outer_fold": k, "meta_n_iter": int(meta.n_iter_[0]), "meta_converged": bool(meta.n_iter_[0] < 1000)})
        oof[te_o] = meta.predict_proba(X_meta_te)[:, 1]
        fold[te_o] = k
    return oof, fold, meta_log


def nested_oof_c0(y, logit_clin):
    X = logit_clin.reshape(-1, 1)
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
        fold[te] = k
    return oof, fold


# ---------------------------------------------------------------------------
# per-variant run
# ---------------------------------------------------------------------------
def run_variant(variant, log):
    os.makedirs(WORK_DIR, exist_ok=True)
    t0 = time.time()
    df, X_m1, X_low, high_raw = load_base()
    ids = df.index.values
    y = df["hospital_mortality"].values.astype(int)
    logit_clin = logit(df["clinical_AI_probability_PRIMARY"].values)
    log.info("[%s] N=%d, events=%d, X_low=%s, high=%s, n_jobs=%d", variant, len(y), y.sum(), X_low.shape,
             {k: v.shape for k, v in high_raw.items()}, config.get_n_jobs())
    status = {"variant": variant, "description": VARIANT_DESC[variant]}

    if variant == "REF":
        src = FeatureSource("static", ids, X_static=X_m1)
        jobs = [("M1", src, fit_lr)]
        ladder = MAX_ITER_LADDER  # v4 fix: M1 now escalates like M3 instead of staying pinned at 3000
    elif variant in ("V0", "V1"):
        src = FeatureSource("static", ids, X_static=frozen_x_m3(df, X_low))
        jobs = [("M3_LR", src, fit_lr), ("M3_CatBoost", src, fit_catboost)]
        ladder = [ORIG_LR_MAX_ITER] if variant == "V0" else MAX_ITER_LADDER
    elif variant == "V2":
        src = FeatureSource("pca", ids, X_low=X_low, high_raw=high_raw)
        jobs = [("M3_LR", src, fit_lr), ("M3_CatBoost", src, fit_catboost)]
        ladder = MAX_ITER_LADDER
    elif variant == "V3":
        raw_views = load_raw_views(ids)
        ok, chk = verify_raw_reproduces_filtered(raw_views, log)
        io_utils.save_csv(chk, os.path.join(WORK_DIR, "prevalence_filter_reproduction_check.csv"))
        status["prevalence_reproduction_PASS"] = bool(ok)
        if not ok:
            log.error("Raw views do NOT reproduce structured/*_filtered.tsv with all 302 -- refusing to run V3.")
            status["state"] = "BLOCKED_prevalence_reproduction_failed"
            json.dump(status, open(os.path.join(WORK_DIR, f"status_{variant}.json"), "w"), indent=1)
            sys.exit(2)
        src = FeatureSource("prev_pca", ids, raw_views=raw_views)
        jobs = [("M3_LR", src, fit_lr), ("M3_CatBoost", src, fit_catboost)]
        ladder = MAX_ITER_LADDER
    elif variant == "V2S":
        src = FeatureSource("raw_concat", ids, X_low=X_low, high_raw=high_raw)
        jobs = [("M3_LR", src, fit_lr_strict)]
        ladder = MAX_ITER_LADDER
    elif variant == "V4_STRICT":
        raw_views = load_raw_views(ids)
        ok, chk = verify_raw_reproduces_filtered(raw_views, log)
        io_utils.save_csv(chk, os.path.join(WORK_DIR, "prevalence_filter_reproduction_check_V4STRICT.csv"))
        status["prevalence_reproduction_PASS"] = bool(ok)
        if not ok:
            log.error("Raw views do NOT reproduce structured/*_filtered.tsv with all 302 -- refusing to run V4_STRICT.")
            status["state"] = "BLOCKED_prevalence_reproduction_failed"
            json.dump(status, open(os.path.join(WORK_DIR, f"status_{variant}.json"), "w"), indent=1)
            sys.exit(2)
        src = FeatureSource("raw_concat_full", ids, raw_views=raw_views)
        jobs = [("M3_LR", src, fit_lr_strict_full)]
        ladder = MAX_ITER_LADDER
    else:
        raise ValueError(variant)

    # CatBoost has no max_iter: run it once, outside the ladder.
    long_rows = []
    attempts = []
    final_max_iter = None
    lr_jobs = [j for j in jobs if j[2] is not fit_catboost]
    cb_jobs = [j for j in jobs if j[2] is fit_catboost]

    def run_jobs(job_list, max_iter, tag):
        rows = []
        for code, source, learner in job_list:
            run = make_runner(source, learner, y, max_iter, variant, code)
            p_single, f_single = single_level_oof(run, y)
            log.info("[%s|%s|max_iter=%s] standalone single-level OOF AUROC = %.4f", variant, code, max_iter, roc_auc_score(y, p_single))
            stacked = "C1_clinical_plus_M1_LiteratureCore" if code == "M1" else (
                "C2a_clinical_plus_M3_LR" if code == "M3_LR" else "C2b_clinical_plus_M3_CatBoost")
            p_nest, f_nest, meta_log = nested_oof_meta(run, y, logit_clin)
            log.info("[%s|%s|max_iter=%s] %s nested AUROC = %.4f", variant, code, max_iter, stacked, roc_auc_score(y, p_nest))
            for i in range(len(y)):
                rows.append({"patient_id_clinical": ids[i], "hospital_mortality": int(y[i]), "outer_fold": int(f_single[i]),
                             "analysis": "standalone_single_level_5fold_OOF", "internal_model_code": code,
                             "reader_facing_model": MODEL_LABELS[code], "oof_probability": p_single[i]})
                rows.append({"patient_id_clinical": ids[i], "hospital_mortality": int(y[i]), "outer_fold": int(f_nest[i]),
                             "analysis": "stacked_nested_5x5_OOF", "internal_model_code": stacked,
                             "reader_facing_model": MODEL_LABELS[stacked], "oof_probability": p_nest[i]})
            for m in meta_log:
                CONV_LOG.append({"variant": variant, "model_code": stacked, "stage": "meta_LR", "outer_fold": m["outer_fold"],
                                 "inner_fold": -1, "learner": "LogisticRegression(lbfgs, no penalty)", "max_iter": 1000,
                                 "n_iter_max_over_cv_paths": m["meta_n_iter"], "converged": m["meta_converged"], "attempt": tag})
        return rows

    for max_iter in ladder:
        n_before = len(CONV_LOG)
        rows = run_jobs(lr_jobs, max_iter, f"max_iter={max_iter}")
        new = [r for r in CONV_LOG[n_before:] if r.get("stage") != "meta_LR"]
        for r in CONV_LOG[n_before:]:
            r.setdefault("attempt", f"max_iter={max_iter}")
        n_fail = sum(1 for r in new if not r["converged"])
        attempts.append({"max_iter": max_iter, "n_lr_fits": len(new), "n_nonconverged_fits": n_fail})
        log.info("[%s] max_iter=%d: %d LR fits, %d non-converged", variant, max_iter, len(new), n_fail)
        final_max_iter = max_iter
        long_rows_lr = rows
        if n_fail == 0 or variant == "V0":
            break
    long_rows.extend(long_rows_lr)
    if cb_jobs:
        long_rows.extend(run_jobs(cb_jobs, None, "catboost"))

    out = pd.DataFrame(long_rows)
    out.insert(0, "preprocessing_variant", variant)
    out["lr_max_iter"] = final_max_iter
    out["seed"] = SEED
    io_utils.save_csv(out, os.path.join(WORK_DIR, f"oof_long_{variant}.csv"))
    conv = pd.DataFrame(CONV_LOG)
    io_utils.save_csv(conv, os.path.join(WORK_DIR, f"convergence_{variant}.csv"))
    if FEAT_LOG:
        io_utils.save_csv(pd.DataFrame(FEAT_LOG), os.path.join(WORK_DIR, f"fold_feature_counts_{variant}.csv"))
    final_fail = attempts[-1]["n_nonconverged_fits"]
    status.update({"attempts": attempts, "final_lr_max_iter": final_max_iter,
                   "final_n_nonconverged_lr_fits": final_fail,
                   "state": ("OK" if final_fail == 0 else
                             ("NONCONVERGED_REFERENCE_SPEC_UNCHANGED" if variant == "V0" else "NONCONVERGED_BLOCKER")),
                   "runtime_sec": round(time.time() - t0, 1)})
    json.dump(status, open(os.path.join(WORK_DIR, f"status_{variant}.json"), "w"), indent=1)
    log.info("[%s] done: %s", variant, status)


# ---------------------------------------------------------------------------
# aggregation: anchors, bootstrap, comparison, report
# ---------------------------------------------------------------------------
def bootstrap_auroc(y, score, n_boot=N_BOOT_AUROC, seed=SEED):
    """Script 56's protocol (byte-for-byte)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y, score), lo, hi


def paired_delta_boot(y, c0, cands: dict):
    """Script 46's algorithm, one RandomState(2026) shared sequentially over C1, C2a, C2b
    (same order as 46, so every variant uses the identical resample indices)."""
    rng = np.random.RandomState(SEED)
    n = len(y)
    res = {}
    for name, ck in cands.items():
        da, dp, db = [], [], []
        for _ in range(N_BOOT_DELTA):
            idx = rng.randint(0, n, n)
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            da.append(roc_auc_score(yb, ck[idx]) - roc_auc_score(yb, c0[idx]))
            dp.append(average_precision_score(yb, ck[idx]) - average_precision_score(yb, c0[idx]))
            db.append(brier_score_loss(yb, c0[idx]) - brier_score_loss(yb, ck[idx]))

        def summ(d, pt):
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))
            return round(pt, 4), round(lo, 4), round(hi, 4), round(min(p, 1.0), 4)
        res[name] = {
            "AUROC": summ(da, roc_auc_score(y, ck) - roc_auc_score(y, c0)),
            "AUPRC": summ(dp, average_precision_score(y, ck) - average_precision_score(y, c0)),
            "Brier": summ(db, brier_score_loss(y, c0) - brier_score_loss(y, ck)),
            "n_eff": len(da)}
    return res


def adjusted_or(y, logit_clin, p_micro):
    """45's adjusted-OR step (full-data Logit on [logit_clin, logit(single-level OOF prob)])."""
    Xc = sm.add_constant(np.column_stack([logit_clin, logit(p_micro)]))
    f = sm.Logit(y, Xc).fit(disp=0)
    c, se = f.params[2], f.bse[2]
    return round(np.exp(c), 3), round(np.exp(c - 1.96 * se), 3), round(np.exp(c + 1.96 * se), 3), round(f.pvalues[2], 4), bool(f.mle_retvals.get("converged", True))


def aggregate(log):
    variants = ["REF", "V0", "V1", "V2", "V3", "V2S", "V4_STRICT"]
    statuses, longs, convs = {}, [], []
    for v in variants:
        sp = os.path.join(WORK_DIR, f"status_{v}.json")
        if not os.path.exists(sp):
            log.error("missing %s -- variant %s did not finish", sp, v)
            sys.exit(1)
        statuses[v] = json.load(open(sp))
        longs.append(pd.read_csv(os.path.join(WORK_DIR, f"oof_long_{v}.csv")))
        convs.append(pd.read_csv(os.path.join(WORK_DIR, f"convergence_{v}.csv")))
    long = pd.concat(longs, ignore_index=True)
    conv = pd.concat(convs, ignore_index=True)

    df, _, _, _ = load_base()
    ids = df.index.values
    y = df["hospital_mortality"].values.astype(int)
    logit_clin = logit(df["clinical_AI_probability_PRIMARY"].values)
    c0, c0_fold = nested_oof_c0(y, logit_clin)

    # C0 rows appended (not variant-specific)
    c0_rows = pd.DataFrame({"preprocessing_variant": "REF", "patient_id_clinical": ids, "hospital_mortality": y,
                            "outer_fold": c0_fold, "analysis": "stacked_nested_5x5_OOF",
                            "internal_model_code": "C0_clinical_alone", "reader_facing_model": MODEL_LABELS["C0_clinical_alone"],
                            "oof_probability": c0, "lr_max_iter": np.nan, "seed": SEED})
    long = pd.concat([c0_rows, long], ignore_index=True)
    # v4: CatBoost's primary stays V3 (untouched -- no internal tuning CV, already
    # maximally fold-local); the data-driven LR model's primary moves to V4_STRICT
    # (final headline per instruction) -- V3's own LR/C2a rows are therefore no
    # longer flagged primary, since V4_STRICT supersedes them.
    is_ref = long["preprocessing_variant"] == "REF"
    is_v3_catboost = (long["preprocessing_variant"] == "V3") & long["internal_model_code"].isin(
        ["M3_CatBoost", "C2b_clinical_plus_M3_CatBoost"])
    is_v4strict_lr = (long["preprocessing_variant"] == "V4_STRICT") & long["internal_model_code"].isin(
        ["M3_LR", "C2a_clinical_plus_M3_LR"])
    long["recommended_primary"] = is_ref | is_v3_catboost | is_v4strict_lr
    io_utils.save_csv(long, OUT_OOF_LONG)
    io_utils.save_csv(conv, OUT_CONV_CSV)

    def get(v, code):
        s = long[(long.preprocessing_variant == v) & (long.internal_model_code == code)].set_index("patient_id_clinical")
        return s.reindex(ids)["oof_probability"].values

    # ---- reproduction anchors vs published frozen-PCA outputs --------------------------------
    old18 = pd.read_csv(OLD_OOF18_CSV).set_index("patient_id_clinical").reindex(ids)
    old46 = pd.read_csv(OLD_NESTED46_CSV).set_index("patient_id_clinical").reindex(ids)
    anchors = []
    for label, new, old in [
        ("C0 nested vs 46", c0, old46["oof_prob_C0_clinical_alone"].values),
        ("REF M1 single-level vs 18", get("REF", "M1"), old18["microbiome_prob_M1"].values),
        ("REF C1 nested vs 46", get("REF", "C1_clinical_plus_M1_LiteratureCore"), old46["oof_prob_C1_clinical_plus_M1_LiteratureCore"].values),
        ("V0 M3_LR single-level vs 18", get("V0", "M3_LR"), old18["microbiome_prob_M3_LR"].values),
        ("V0 M3_CatBoost single-level vs 18", get("V0", "M3_CatBoost"), old18["microbiome_prob_M3_CatBoost"].values),
        ("V0 C2a nested vs 46", get("V0", "C2a_clinical_plus_M3_LR"), old46["oof_prob_C2a_clinical_plus_M3_LR"].values),
        ("V0 C2b nested vs 46", get("V0", "C2b_clinical_plus_M3_CatBoost"), old46["oof_prob_C2b_clinical_plus_M3_CatBoost"].values),
    ]:
        d = float(np.max(np.abs(new - old)))
        anchors.append({"check": label, "max_abs_prob_diff": d, "AUROC_new": round(roc_auc_score(y, new), 4),
                        "AUROC_published": round(roc_auc_score(y, old), 4),
                        "AUROC_match_4dp": bool(round(roc_auc_score(y, new), 4) == round(roc_auc_score(y, old), 4)),
                        "PASS_prob_identical_1e-6": bool(d < 1e-6)})
    anchors = pd.DataFrame(anchors)
    io_utils.save_csv(anchors, OUT_ANCHOR_CSV)
    log.info("\n%s", anchors.to_string(index=False))

    # ---- per-variant numbers ------------------------------------------------------------------
    rep_rows, boot_rows, sa_rows = [], [], []
    p_c1 = get("REF", "C1_clinical_plus_M1_LiteratureCore")
    NO_CATBOOST = ("V2S", "V4_STRICT")  # variants that only ran the LR job, no CatBoost job
    for v in ["V0", "V1", "V2", "V3", "V2S", "V4_STRICT"]:
        cands = {"C1_clinical_plus_M1_LiteratureCore": p_c1, "C2a_clinical_plus_M3_LR": get(v, "C2a_clinical_plus_M3_LR")}
        if v not in NO_CATBOOST:
            cands["C2b_clinical_plus_M3_CatBoost"] = get(v, "C2b_clinical_plus_M3_CatBoost")
        elif v == "V2S":  # 46's rng order needs C2b present to keep identical resamples; reuse V2's C2b (V2S does not touch CatBoost)
            cands["C2b_clinical_plus_M3_CatBoost"] = get("V2", "C2b_clinical_plus_M3_CatBoost")
        else:  # V4_STRICT: reuse V3's C2b verbatim (CatBoost untouched, already maximally fold-local)
            cands["C2b_clinical_plus_M3_CatBoost"] = get("V3", "C2b_clinical_plus_M3_CatBoost")
        boot = paired_delta_boot(y, c0, cands)
        for name, b in boot.items():
            if name.startswith("C1") and v != "V0":
                continue
            if v in NO_CATBOOST and name.startswith("C2b"):
                continue
            boot_rows.append({"preprocessing_variant": v, "model": name, "N_boot": N_BOOT_DELTA, "N_effective_resamples": b["n_eff"],
                              "delta_AUROC": b["AUROC"][0], "delta_AUROC_CI_low": b["AUROC"][1], "delta_AUROC_CI_high": b["AUROC"][2], "delta_AUROC_boot_p": b["AUROC"][3],
                              "delta_AUPRC": b["AUPRC"][0], "delta_AUPRC_CI_low": b["AUPRC"][1], "delta_AUPRC_CI_high": b["AUPRC"][2], "delta_AUPRC_boot_p": b["AUPRC"][3],
                              "delta_Brier_improvement": b["Brier"][0], "delta_Brier_CI_low": b["Brier"][1], "delta_Brier_CI_high": b["Brier"][2], "delta_Brier_boot_p": b["Brier"][3]})
        for code, stacked in [("M3_LR", "C2a_clinical_plus_M3_LR"), ("M3_CatBoost", "C2b_clinical_plus_M3_CatBoost")]:
            if v in NO_CATBOOST and code == "M3_CatBoost":
                continue
            p_sa = get(v, code)
            a, lo, hi = bootstrap_auroc(y, p_sa)
            sa_rows.append({"preprocessing_variant": v, "model": code, "AUROC": round(a, 4), "AUROC_CI_low": round(lo, 4),
                            "AUROC_CI_high": round(hi, 4), "N_boot": N_BOOT_AUROC, "seed": SEED})
            o, olo, ohi, op, oconv = adjusted_or(y, logit_clin, p_sa)
            rep_rows.append({"preprocessing_variant": v, "model": stacked, "N": len(y), "N_events": int(y.sum()),
                             "CV_AUROC_5fold_nested": round(roc_auc_score(y, get(v, stacked)), 4),
                             "delta_AUROC_vs_C0": round(roc_auc_score(y, get(v, stacked)) - roc_auc_score(y, c0), 4),
                             "microbiome_adjusted_OR": o, "microbiome_adjusted_OR_CI_low": olo,
                             "microbiome_adjusted_OR_CI_high": ohi, "microbiome_adjusted_p_value": op,
                             "adjusted_OR_logit_converged": oconv, "lr_max_iter": statuses[v].get("final_lr_max_iter"),
                             "variant_state": statuses[v].get("state")})
    for v, code in [("REF", "M1")]:
        a, lo, hi = bootstrap_auroc(y, get(v, code))
        sa_rows.append({"preprocessing_variant": v, "model": code, "AUROC": round(a, 4), "AUROC_CI_low": round(lo, 4),
                        "AUROC_CI_high": round(hi, 4), "N_boot": N_BOOT_AUROC, "seed": SEED})
    rep = pd.DataFrame(rep_rows)
    rep = pd.concat([pd.DataFrame([{"preprocessing_variant": "REF", "model": "C0_clinical_alone", "N": len(y), "N_events": int(y.sum()),
                                     "CV_AUROC_5fold_nested": round(roc_auc_score(y, c0), 4), "delta_AUROC_vs_C0": 0.0},
                                    {"preprocessing_variant": "REF", "model": "C1_clinical_plus_M1_LiteratureCore", "N": len(y),
                                     "N_events": int(y.sum()), "CV_AUROC_5fold_nested": round(roc_auc_score(y, p_c1), 4),
                                     "delta_AUROC_vs_C0": round(roc_auc_score(y, p_c1) - roc_auc_score(y, c0), 4),
                                     "lr_max_iter": statuses["REF"].get("final_lr_max_iter"), "variant_state": statuses["REF"].get("state")}]), rep],
                    ignore_index=True)
    boot = pd.DataFrame(boot_rows)
    sa = pd.DataFrame(sa_rows)
    io_utils.save_csv(rep, OUT_REPORT_CSV)
    io_utils.save_csv(boot, OUT_BOOT_CSV)
    io_utils.save_csv(sa, OUT_STANDALONE_CSV)
    log.info("\n%s\n%s\n%s", rep.to_string(index=False), boot.to_string(index=False), sa.to_string(index=False))

    # ---- drop-in wide files (same column layout as the files downstream scripts read) ----------
    # v4 writes its own _v4_{v} files (distinct from v3's _v3_{v} files) so v3's drop-ins are
    # never overwritten. V4_STRICT has no CatBoost job of its own -> reuse V3's C2b verbatim
    # (CatBoost untouched, already maximally fold-local; same precedent as V2S reusing V2's C2b).
    for v in ["V2", "V3", "V4_STRICT"]:
        c2b_src = "V3" if v == "V4_STRICT" else v
        wide18 = pd.DataFrame({"patient_id_clinical": ids, "hospital_mortality": y,
                               "clinical_AI_probability_PRIMARY": df["clinical_AI_probability_PRIMARY"].values,
                               "microbiome_prob_M1": get("REF", "M1"),
                               "microbiome_prob_M3_LR": get(v, "M3_LR"),
                               "microbiome_prob_M3_CatBoost": get(c2b_src, "M3_CatBoost")})
        io_utils.save_csv(wide18, os.path.join(config.DIR_PREDICTIONS, f"phase_x_m3_single_level_oof_probabilities_v4_{v}.csv"))
        wide46 = pd.DataFrame({"patient_id_clinical": ids, "hospital_mortality": y,
                               "oof_prob_C0_clinical_alone": c0,
                               "oof_prob_C1_clinical_plus_M1_LiteratureCore": p_c1,
                               "oof_prob_C2a_clinical_plus_M3_LR": get(v, "C2a_clinical_plus_M3_LR"),
                               "oof_prob_C2b_clinical_plus_M3_CatBoost": get(c2b_src, "C2b_clinical_plus_M3_CatBoost"),
                               "clinical_AI_probability_PRIMARY": df["clinical_AI_probability_PRIMARY"].values})
        io_utils.save_csv(wide46, os.path.join(config.DIR_PREDICTIONS, f"phase_x_c1_c2_nested_oof_probabilities_v4_{v}.csv"))

    # ---- markdown ------------------------------------------------------------------------------
    st = pd.DataFrame([{"variant": v, "description": VARIANT_DESC[v], "state": s.get("state"),
                        "final_lr_max_iter": s.get("final_lr_max_iter"), "attempts": json.dumps(s.get("attempts")),
                        "runtime_sec": s.get("runtime_sec")} for v, s in statuses.items()])
    md = ["# Phase X C1/C2 nested CV -- M1 convergence fix + strict tuning-local LR (v4)", "",
          f"N={len(y)}, events={int(y.sum())}. Script `45_phase_x_c1_c2_nested_cv_corrected_v4.py`.", "",
          "## Variant status", "", df_to_md_table(st), "",
          "## Reproduction anchors (V0/REF vs published frozen-PCA outputs)", "", df_to_md_table(anchors), "",
          "## Stacked nested AUROC / adjusted OR", "", df_to_md_table(rep), "",
          "## Paired bootstrap delta vs C0 (script 46 algorithm)", "", df_to_md_table(boot), "",
          "## Standalone single-level OOF AUROC (script 56 bootstrap)", "", df_to_md_table(sa), ""]
    open(OUT_MD, "w").write("\n".join(md))
    log.info("wrote %s", OUT_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["REF", "V0", "V1", "V2", "V3", "V2S", "V4_STRICT"])
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    logger = io_utils.setup_logging(f"45_v4_{'aggregate' if a.aggregate else a.variant}")
    if a.aggregate:
        aggregate(logger)
    elif a.variant:
        run_variant(a.variant, logger)
    else:
        ap.error("--variant or --aggregate required")
