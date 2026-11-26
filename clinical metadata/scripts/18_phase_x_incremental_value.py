#!/usr/bin/env python3
"""18_phase_x_incremental_value.py

Phase X, C1/C2 (spec sec. 63-68): does the microbiome score add INCREMENTAL
prognostic value on top of the reduced clinical AI model (script 16's C0),
for the 302 Gangnam ICU patients? This is the first script in this project
that reads BOTH the clinical-AI side AND the microbiome-semantic side --
by design, ONLY after both were independently developed/frozen (clinical AI:
scripts 09-17 here; microbiome: semantic_microbiome/FROZEN_MICROBIOME_PIPELINE.json,
GATE 4, frozen 2026-08-14 BEFORE any public-cohort outcome was read).

Microbiome candidates used, PER THE MICROBIOME PROJECT'S OWN FREEZE
MANIFEST recommendation (FROZEN_MICROBIOME_PIPELINE.json ->
"structured_models"): only M1 (Literature Core, LR) and M3 (Data-driven,
LR + CatBoost) -- both explicitly marked "FROZEN, recommended candidate for
Phase VIII/X" there. Deliberately EXCLUDED, matching that same manifest:
  - M4 Hybrid / Structured+Semantic Hybrid: confirmed LEAKED (internal-CV
    0.84 vs external-validated 0.50-0.59) -- not a valid candidate.
  - S1/S2 semantic embeddings: not frozen as standalone supervised models
    (S1 chance-level 0.504; S2 modest 0.622 but "adds no clear advantage
    over structured features" per the manifest's own assessment) -- adding
    them here would require inventing a NEW, unvalidated combination method
    (their only frozen artifacts are unsupervised clustering PCA/KMeans, not
    a supervised embedding-to-outcome model), so they are left for future
    work rather than combined via an ad hoc, never-validated pipeline.

For each microbiome candidate (M1, M3-LR, M3-CatBoost):
  1. Apply the FROZEN microbiome model (no refitting) to get a microbiome
     probability for each of the 302 patients -- reproduces
     semantic_microbiome/scripts/lit14_freeze_pipeline.py's own feature
     construction exactly (same structured files, same frozen PCA/imputer/
     scaler for M3), so the microbiome side is used completely unchanged.
  2. Combine logit(clinical_AI_probability) + logit(microbiome_probability)
     in a 2-term logistic regression, 5-fold stratified CV (seed=2026,
     matching this project's own N_CV_FOLDS/RANDOM_SEED convention) for an
     honest (not in-sample) AUROC comparison against C0 (clinical alone).
  3. Full-data (non-CV) adjusted logistic regression (statsmodels) for the
     microbiome term's adjusted OR/95% CI, controlling for clinical risk
     (spec sec. 67).

Reads (read-only, no modification to either side):
  clinical_ai/predictions/reduced_model_microbiome_cohort_predictions.csv (script 10)
  clinical_ai/reports/selected_reduced_model.json (script 15)
  semantic_microbiome/harmonized/discovery_cohort_primary.csv (hospital_mortality)
  semantic_microbiome/structured/literature_core_features.csv
  semantic_microbiome/structured/{taxonomy,humann,rgi,vfdb,antismash,eggnog}*_filtered.tsv
  semantic_microbiome/supervised_semantic/frozen_models/*
Writes:
  clinical_ai/reports/phase_x_c1_c2_incremental_value.csv
  clinical_ai/reports/phase_x_c1_c2_incremental_value.md
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("18_phase_x_incremental_value")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
FROZEN_DIR = os.path.join(SEMANTIC, "supervised_semantic", "frozen_models")
DISCOVERY_CSV = os.path.join(SEMANTIC, "harmonized", "discovery_cohort_primary.csv")
CLINICAL_PRED_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")
SELECTED_JSON = os.path.join(config.DIR_REPORTS, "selected_reduced_model.json")
OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value.md")

SEED = config.RANDOM_SEED  # 2026, shared convention across both projects
N_FOLDS = config.N_CV_FOLDS  # 5
EPS = 1e-6

LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}
PCA_N_COMPONENTS = 20


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
# 1. load outcome + clinical AI probability (PRIMARY, reduced model)
# ---------------------------------------------------------------------------
outcome = pd.read_csv(DISCOVERY_CSV)[["patient_id_clinical", "hospital_mortality"]].set_index("patient_id_clinical")
clin_pred = pd.read_csv(CLINICAL_PRED_CSV).set_index("patient_id_clinical")
if "clinical_AI_probability_PRIMARY" not in clin_pred.columns:
    log.error("No PRIMARY column in %s -- run 15_select_best_reduced_model.py then re-run "
              "10_apply_reduced_model_to_microbiome_cohort.py first.", CLINICAL_PRED_CSV)
    sys.exit(1)

with open(SELECTED_JSON) as fh:
    selected_meta = json.load(fh)
log.info("Clinical AI PRIMARY model: %s (dev=%s), %s, Gangnam-relevant AUROC=%.4f",
          selected_meta["selected_model_name"], selected_meta["selected_development_dataset"],
          "PROVISIONAL" if selected_meta["selection_is_provisional"] else "FINAL",
          selected_meta["selected_Gangnam_relevant_AUROC"])

df = outcome.join(clin_pred[["clinical_AI_probability_PRIMARY"]], how="inner")
log.info("N=%d patients with both outcome and clinical AI probability", len(df))

# ---------------------------------------------------------------------------
# 2/3. microbiome scores -- OUT-OF-FOLD, NOT the frozen in-sample models.
#
# IMPORTANT (bug found + fixed while building this script, 2026-08-14): the
# FROZEN M1/M3 joblib/cbm models were refit on ALL 302 Gangnam patients
# (lit14_freeze_pipeline.py, by design, for later application to EXTERNAL
# public cohorts where in-sample fit is irrelevant). Calling
# frozen_model.predict_proba() on these SAME 302 patients is therefore
# in-sample/overfit scoring -- an initial version of this script did exactly
# that and produced an implausible C2 AUROC (~0.96, adjusted OR ~24-50),
# wildly higher than M3's own internal 25-fold nested-CV AUROC (0.62-0.65,
# per FROZEN_MICROBIOME_PIPELINE.json) -- a clear sign of the same class of
# leakage already documented for the (separately excluded) M4 Hybrid arm.
#
# Fix: generate genuine OUT-OF-FOLD microbiome probabilities by REFITTING
# the same model architecture (elastic-net LR for M1; LR + CatBoost for M3)
# within a fresh 5-fold stratified CV (seed=2026) here, exactly as
# 12_tune_ml_reduced.py/16 already do for the clinical-AI side. The frozen,
# UNSUPERVISED M3 PCA (loadings/mean/scaler, outcome-blind, fit once on all
# 302) is still reused unchanged -- consistent with this project's own
# established convention (e.g. the Phase-2 prevalence filter is likewise
# unsupervised and computed once on all 302, per FROZEN_MICROBIOME_PIPELINE.
# json's own "known_limitation" note) -- only the SUPERVISED (outcome-fit)
# step is redone per-fold.
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

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

oof_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_lr_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                    random_state=SEED, n_jobs=config.get_n_jobs()))])


def oof_scores_lr(X):
    p = np.full(len(y), np.nan)
    for tr, te in oof_skf.split(X, y):
        pipe = make_lr_pipe().fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def oof_scores_catboost(X):
    p = np.full(len(y), np.nan)
    for tr, te in oof_skf.split(X, y):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        cb = CatBoostClassifier(verbose=False, random_state=SEED, thread_count=config.get_n_jobs(),
                                 iterations=200, depth=4, learning_rate=0.05)
        cb.fit(imp.transform(X.iloc[tr]), y[tr])
        p[te] = cb.predict_proba(imp.transform(X.iloc[te]))[:, 1]
    return p


y = df["hospital_mortality"].values  # needed by the OOF helper closures above
df["microbiome_prob_M1"] = oof_scores_lr(X_m1)
df["microbiome_prob_M3_LR"] = oof_scores_lr(X_m3)
df["microbiome_prob_M3_CatBoost"] = oof_scores_catboost(X_m3)
log.info("M1 (Literature Core, OOF) mean prob=%.3f; M3-LR (OOF) mean prob=%.3f; M3-CatBoost (OOF) mean prob=%.3f",
          df["microbiome_prob_M1"].mean(), df["microbiome_prob_M3_LR"].mean(), df["microbiome_prob_M3_CatBoost"].mean())

# ---------------------------------------------------------------------------
# 4. C0 (clinical alone) vs C1/C2 (clinical + each microbiome candidate):
#    5-fold CV AUROC (honest, out-of-fold) + full-data adjusted OR
# ---------------------------------------------------------------------------
logit_clin = logit(df["clinical_AI_probability_PRIMARY"].values)

CANDIDATES = {
    "C0_clinical_alone": None,
    "C1_clinical_plus_M1_LiteratureCore": "microbiome_prob_M1",
    "C2a_clinical_plus_M3_LR": "microbiome_prob_M3_LR",
    "C2b_clinical_plus_M3_CatBoost": "microbiome_prob_M3_CatBoost",
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
rows = []
for name, micro_col in CANDIDATES.items():
    if micro_col is None:
        X = logit_clin.reshape(-1, 1)
    else:
        X = np.column_stack([logit_clin, logit(df[micro_col].values)])

    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    cv_auroc = roc_auc_score(y, oof)
    df[f"oof_prob_{name}"] = oof  # persisted below for reuse by 20_phase_x_residual_and_decision_curves.py

    row = {"model": name, "N": len(y), "N_events": int(y.sum()), "CV_AUROC_5fold": round(cv_auroc, 4)}

    if micro_col is not None:
        Xc = sm.add_constant(X)  # [const, logit_clinical, logit_microbiome]
        try:
            sm_fit = sm.Logit(y, Xc).fit(disp=0)
            coef, se = sm_fit.params[2], sm_fit.bse[2]
            or_ = np.exp(coef)
            ci_lo, ci_hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
            row.update({"microbiome_adjusted_OR": round(or_, 3),
                        "microbiome_adjusted_OR_CI_low": round(ci_lo, 3),
                        "microbiome_adjusted_OR_CI_high": round(ci_hi, 3),
                        "microbiome_adjusted_p_value": round(sm_fit.pvalues[2], 4)})
        except Exception as e:
            log.warning("%s: adjusted logistic regression failed (%s)", name, e)
    rows.append(row)

out = pd.DataFrame(rows)
c0_auroc = out.loc[out["model"] == "C0_clinical_alone", "CV_AUROC_5fold"].iloc[0]
out["delta_AUROC_vs_C0"] = (out["CV_AUROC_5fold"] - c0_auroc).round(4)

io_utils.save_csv(out, OUT_CSV)
log.info("wrote %s", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

md = ["# Phase X, C1/C2: incremental value of microbiome score over clinical AI alone", "",
      f"N={len(y)} Gangnam patients, {int(y.sum())} deaths ({100*y.mean():.1f}%). "
      f"Clinical AI = PRIMARY reduced model ({selected_meta['selected_model_name']}, "
      f"dev={selected_meta['selected_development_dataset']}, "
      f"{'PROVISIONAL selection -- re-run once all 18 reduced models finish' if selected_meta['selection_is_provisional'] else 'FINAL selection'}). "
      "Microbiome candidates (M1 Literature Core, M3 Data-driven LR/CatBoost) scored via a fresh, "
      "self-contained 5-fold out-of-fold refit here (NOT the all-302 frozen models -- see the section 2/3 "
      "code comment above for why: applying the frozen models directly was found to be in-sample/overfit). "
      "The unsupervised, outcome-blind M3 PCA (dimensionality reduction only) is still reused frozen. "
      "CV_AUROC_5fold is out-of-fold (honest); adjusted OR/CI is from a full-data logistic regression of "
      "outcome ~ logit(clinical) + logit(microbiome), for the microbiome term, controlling for clinical risk.",
      "", df_to_md_table(out), "",
      "**Note**: M4 Hybrid is deliberately excluded (confirmed leaked in the microbiome project's own validation). "
      "S1/S2 semantic-embedding combinations are evaluated separately in "
      "19_phase_x_c4_c5_semantic_incremental.py (a fully fold-local design, since no frozen supervised S1/S2 "
      "model exists to reuse).",
      ]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)

# Persist per-patient OOF probabilities (clinical alone + each combined
# model) for reuse by 20_phase_x_residual_and_decision_curves.py, so that
# script doesn't need to redo the ~2.5min M1/M3 OOF refit.
oof_cols = ["hospital_mortality", "clinical_AI_probability_PRIMARY",
            "microbiome_prob_M1", "microbiome_prob_M3_LR", "microbiome_prob_M3_CatBoost"] + \
           [c for c in df.columns if c.startswith("oof_prob_")]
OOF_OUT_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")
df[oof_cols].to_csv(OOF_OUT_CSV)
log.info("wrote per-patient OOF probabilities to %s", OOF_OUT_CSV)
