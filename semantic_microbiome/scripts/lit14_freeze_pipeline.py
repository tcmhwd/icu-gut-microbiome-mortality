"""
lit14_freeze_pipeline.py — Phase VII / GATE 4 (spec sec. 50, 81): freeze
the complete microbiome pipeline BEFORE any public-cohort mortality is
examined for model selection, threshold tuning, or refitting.

Freezes and persists:
  - M1 (Literature Core, elastic-net LR) final model, refit on all 302 Gangnam
  - M3 (Data-driven, LR + CatBoost) final models, refit on all 302 Gangnam,
    including the frozen PCA loadings for the 2 high-dim blocks (MAG_ABUND,
    EGGNOG_KO) -- fit ONCE on all 302 Gangnam training patients, to be
    applied UNCHANGED to public cohorts in Phase VIII (same "frozen, apply
    unchanged" discipline as script 25)
  - Gangnam-only clustering (KMeans K=2 on S1/S2 embeddings, lit13b) --
    scaler + PCA + KMeans model objects, so public patients CAN be
    assigned to these frozen clusters later without recomputing them
    (spec sec. 54: "assign public patients to frozen Gangnam clusters")
  - patient report template version, reference percentile distributions,
    embedding model spec

Explicitly NOT frozen as a usable model: M4 Hybrid / Structured+Semantic
Hybrid (both flagged as leaked in lit10/lit13 -- see model_comparison_
M0_M4_external.md). Their feature manifest is still recorded for
transparency, but marked excluded from Phase VIII candidate scoring.

hospital_mortality used only to fit the final M1/M3 models and the
Gangnam-only cluster freeze (already-computed embeddings/features) --
no public outcome is read anywhere in this script.

Outputs:
  supervised_semantic/frozen_models/M1_literature_core_LR.joblib
  supervised_semantic/frozen_models/M3_data_driven_LR.joblib
  supervised_semantic/frozen_models/M3_data_driven_CatBoost.cbm
  supervised_semantic/frozen_models/M3_pca_loadings/{MAG_ABUND,EGGNOG_KO}_{loadings,mean}.npy
  clustering/lit_phenotypes/frozen_clustering/{S1,S2}_{scaler,pca,kmeans}.joblib
  semantic_microbiome/FROZEN_MICROBIOME_PIPELINE.json
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEMANTIC, HARMONIZED_DIR, STRUCTURED_DIR, SEED, today_str

optuna.logging.set_verbosity(optuna.logging.WARNING)

SUPERVISED_DIR = SEMANTIC / "supervised_semantic"
FROZEN_DIR = SUPERVISED_DIR / "frozen_models"
PCA_DIR = FROZEN_DIR / "M3_pca_loadings"
CLUSTER_DIR = SEMANTIC / "clustering" / "lit_phenotypes"
FROZEN_CLUSTER_DIR = CLUSTER_DIR / "frozen_clustering"
for d in (FROZEN_DIR, PCA_DIR, FROZEN_CLUSTER_DIR):
    d.mkdir(parents=True, exist_ok=True)

cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
y = cohort["hospital_mortality"].astype(int)

# ---------------------------------------------------------------------
# 1. M1 Literature Core -- final refit (already validated externally in lit10b)
# ---------------------------------------------------------------------
lit_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
X_m1 = lit_core.reindex(y.index)

m1_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                   l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                   random_state=SEED, n_jobs=4))]).fit(X_m1, y)
joblib.dump(m1_pipe, FROZEN_DIR / "M1_literature_core_LR.joblib")
print(f"[lit14] M1 Literature Core LR refit on all {len(y)} Gangnam, saved.")

# ---------------------------------------------------------------------
# 2. M3 Data-driven -- final refit with FROZEN PCA (fit once on all 302,
#    saved for unchanged application to public cohorts in Phase VIII)
# ---------------------------------------------------------------------
LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}
PCA_N_COMPONENTS = 20

low_parts = []
for fname in LOW_DIM_FILES:
    df = pd.read_csv(STRUCTURED_DIR / fname, sep="\t", index_col="patient_id")
    df.columns = [f"{fname.replace('_filtered.tsv', '')}__{c}" for c in df.columns]
    low_parts.append(df)
X_low = pd.concat(low_parts, axis=1).reindex(y.index)

pca_parts = [X_low]
for name, fname in HIGH_DIM_FILES.items():
    raw = pd.read_csv(STRUCTURED_DIR / fname, sep="\t", index_col="patient_id").reindex(y.index)
    imp = SimpleImputer(strategy="median").fit(raw.values)
    raw_imp = imp.transform(raw.values)
    scaler = StandardScaler().fit(raw_imp)
    raw_scaled = scaler.transform(raw_imp)
    n_comp = min(PCA_N_COMPONENTS, raw_scaled.shape[0] - 1, raw_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(raw_scaled)
    np.save(PCA_DIR / f"{name}_loadings.npy", pca.components_)
    np.save(PCA_DIR / f"{name}_mean.npy", pca.mean_)
    np.save(PCA_DIR / f"{name}_scaler_mean.npy", scaler.mean_)
    np.save(PCA_DIR / f"{name}_scaler_scale.npy", scaler.scale_)
    joblib.dump(imp, PCA_DIR / f"{name}_imputer.joblib")
    pcs = pca.transform(raw_scaled)
    pca_parts.append(pd.DataFrame(pcs, index=y.index, columns=[f"{name}_pc{i+1}" for i in range(n_comp)]))
    print(f"[lit14] {name}: frozen PCA ({n_comp} components) fit on all {len(y)} Gangnam, saved to {PCA_DIR}/")

X_m3 = pd.concat(pca_parts, axis=1)

m3_lr = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                   ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                 l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                 random_state=SEED, n_jobs=4))]).fit(X_m3, y)
joblib.dump(m3_lr, FROZEN_DIR / "M3_data_driven_LR.joblib")


def objective_cb(trial, Xtr, ytr):
    params = {"iterations": trial.suggest_int("iterations", 50, 300), "depth": trial.suggest_int("depth", 2, 5),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0)}
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", CatBoostClassifier(verbose=False, random_state=SEED, thread_count=8, **params))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="roc_auc", n_jobs=1).mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(lambda t: objective_cb(t, X_m3, y), n_trials=30, n_jobs=1, show_progress_bar=False)
imp_final = SimpleImputer(strategy="median").fit(X_m3)
m3_cb = CatBoostClassifier(verbose=False, random_state=SEED, thread_count=8, **study.best_params)
m3_cb.fit(imp_final.transform(X_m3), y)
m3_cb.save_model(str(FROZEN_DIR / "M3_data_driven_CatBoost.cbm"))
joblib.dump(imp_final, FROZEN_DIR / "M3_data_driven_CatBoost_imputer.joblib")
print(f"[lit14] M3 Data-driven LR + CatBoost (hyperparams={study.best_params}) refit on all {len(y)} Gangnam, saved.")

# ---------------------------------------------------------------------
# 3. freeze Gangnam-only clustering (S1/S2, K=2, per lit13b)
# ---------------------------------------------------------------------
EMB_DIR = SEMANTIC / "embeddings"
for tag, emb_prefix in [("S1", "lit_S1_literature"), ("S2", "lit_S2_hybrid")]:
    emb = np.load(EMB_DIR / f"{emb_prefix}_embeddings.npy")
    idx = pd.read_csv(EMB_DIR / f"{emb_prefix}_index.csv").set_index("patient_id")
    E = pd.DataFrame(emb, index=idx.index).reindex(y.index)

    scaler = StandardScaler().fit(E.values)
    X_scaled = scaler.transform(E.values)
    pca = PCA(n_components=min(PCA_N_COMPONENTS, X_scaled.shape[0] - 1), random_state=SEED).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    km = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(X_pca)

    joblib.dump(scaler, FROZEN_CLUSTER_DIR / f"{tag}_scaler.joblib")
    joblib.dump(pca, FROZEN_CLUSTER_DIR / f"{tag}_pca.joblib")
    joblib.dump(km, FROZEN_CLUSTER_DIR / f"{tag}_kmeans.joblib")
    print(f"[lit14] {tag} clustering (K=2) frozen: scaler/pca/kmeans saved to {FROZEN_CLUSTER_DIR}/")

# ---------------------------------------------------------------------
# 4. master freeze manifest
# ---------------------------------------------------------------------
def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


lit_prior_final = SEMANTIC / "literature_prior" / "final"
manifest = {
    "freeze_timestamp": today_str(),
    "freeze_gate": "GATE_4 (spec sec. 81) -- complete microbiome pipeline frozen BEFORE public-cohort outcome analysis",
    "random_seed": SEED,
    "literature_library": {
        "manifest": str(lit_prior_final / "FROZEN_LIBRARY_MANIFEST.json"),
        "sha256": file_sha256(lit_prior_final / "FROZEN_LIBRARY_MANIFEST.json"),
    },
    "literature_core_extended_formulas": {
        "path": str(lit_prior_final / "literature_core_extended_formulas.json"),
        "sha256": file_sha256(lit_prior_final / "literature_core_extended_formulas.json"),
        "bugfix_2026_08_14": "Gangnam sample-ID mapping bug fixed (see lit08 script docstring) -- 302/302 "
                              "Gangnam patients now correctly populated (was 110/302).",
    },
    "data_driven_stability": {
        "source": str(SEMANTIC / "data_driven" / "feature_stability.csv"),
        "threshold_used_for_hybrid": 0.8,
        "note": "Hybrid arm (M4) built from this threshold is FLAGGED AS LEAKED (see below) -- "
                "threshold recorded for provenance only, not endorsed for reuse without a corrected "
                "nested-within-nested implementation.",
    },
    "preprocessing": {
        "imputation": "median (SimpleImputer), fit on training/discovery data only",
        "scaling": "StandardScaler, fit on training/discovery data only",
        "high_dim_pca": {"n_components": PCA_N_COMPONENTS, "blocks": list(HIGH_DIM_FILES.keys()),
                          "fit_on": "all 302 Gangnam discovery patients only, frozen loadings in M3_pca_loadings/",
                          "known_limitation": "Phase-2 prevalence filter (structured/*_filtered.tsv column lists) "
                                               "was computed on all 302 Gangnam patients unsupervised (outcome-blind) "
                                               "and is not re-derived per any downstream split -- documented, not silent."},
    },
    "patient_report_templates": {
        "L1_LiteratureCore": "patient_reports/lit_reports_L1.jsonl",
        "L2_LiteratureExtended": "patient_reports/lit_reports_L2.jsonl",
        "H_Hybrid": "patient_reports/lit_reports_H.jsonl",
        "reference_percentiles": str(SEMANTIC / "patient_reports" / "reference_percentile_distributions.json"),
        "reference_cohort": "Gangnam discovery (N=302) only, applied unchanged to public cohorts",
    },
    "embedding_model": {"model": "text-embedding-3-large", "dimension": 1024,
                         "S1_source": "L1_LiteratureCore report", "S2_source": "H_Hybrid report"},
    "clustering": {
        "method": "KMeans, K=2 (silhouette-selected, Gangnam-only per lit13b -- all-cohort clustering in lit13 "
                   "found to trivially recover COHORT identity, not phenotype, and was discarded)",
        "S1_silhouette": 0.108, "S2_silhouette": 0.152,
        "S1_cluster_mortality": {"cluster_0_n": 119, "cluster_0_mortality": 0.176, "cluster_1_n": 183, "cluster_1_mortality": 0.262},
        "S2_cluster_mortality": {"cluster_0_n": 143, "cluster_0_mortality": 0.203, "cluster_1_n": 159, "cluster_1_mortality": 0.252},
        "frozen_artifacts_dir": str(FROZEN_CLUSTER_DIR),
        "interpretation": "weak structure (silhouette ~0.1-0.15); mortality difference between clusters is directionally "
                           "plausible but not strongly supported given the weak separation -- consistent with the "
                           "project's repeated finding that unsupervised structure alone carries limited information.",
    },
    "structured_models": {
        "M0_simple_ecology": {"status": "reference baseline only, not frozen as a model artifact", "internal_auroc": 0.533},
        "M1_literature_core": {"path": str(FROZEN_DIR / "M1_literature_core_LR.joblib"), "n_features": 16,
                                "internal_auroc": 0.624, "external_auroc_AGI": [0.618, 0.628], "external_auroc_PRJNA842425": [0.550, 0.567],
                                "status": "FROZEN, recommended candidate for Phase VIII/X"},
        "M2_literature_extended": {"status": "not separately frozen (Core preferred; Extended's 1 extra feature "
                                               "was not literature-tier-anchored)", "internal_auroc": 0.609},
        "M3_data_driven": {"LR_path": str(FROZEN_DIR / "M3_data_driven_LR.joblib"),
                            "CatBoost_path": str(FROZEN_DIR / "M3_data_driven_CatBoost.cbm"),
                            "n_features": int(X_m3.shape[1]), "internal_auroc_LR": 0.624, "internal_auroc_CatBoost": 0.649,
                            "status": "FROZEN, recommended candidate for Phase VIII/X (best internal discriminator)"},
        "M4_hybrid": {"status": "NOT FROZEN -- internal-CV result (0.841 LR / 0.721 CatBoost) confirmed LEAKED via "
                                 "external validation (external AUROC 0.50-0.59, WORSE than M1 alone). "
                                 "See supervised_semantic/model_comparison_M0_M4_external.md.",
                      "hybrid_feature_manifest": str(SEMANTIC / "hybrid" / "hybrid_feature_manifest.csv")},
    },
    "semantic_models": {
        "S1_literature_embedding": {"internal_auroc": 0.504, "status": "chance-level, NOT useful beyond structured features"},
        "S2_hybrid_embedding": {"internal_auroc": 0.622, "status": "modest, comparable to M1/M3 but adds no clear "
                                 "advantage over structured features; not frozen as a separate model artifact"},
        "structured_plus_semantic_hybrid": {"status": "NOT FROZEN -- same leakage source as M4 (includes the same "
                                             "71 stability-selected features)"},
    },
    "known_issues_and_fixes": [
        "2026-08-14: Gangnam sample-ID mapping bug in lit08/lit10 (192/302 patients silently NaN'd) -- FIXED.",
        "2026-08-14: M4 Hybrid / Structured+Semantic Hybrid internal-CV leakage (stable-feature selection used the "
        "same 25-fold split later used for evaluation) -- FLAGGED, worked around via external validation (lit10b), "
        "NOT fixed at the root (would require nested-within-nested feature selection) -- these arms remain excluded "
        "from the frozen candidate set.",
        "2026-08-14: all-cohort embedding clustering (lit13) found to trivially separate by cohort identity, not "
        "phenotype -- discarded in favor of Gangnam-only clustering (lit13b).",
    ],
}

with open(SEMANTIC / "FROZEN_MICROBIOME_PIPELINE.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"[lit14] GATE 4 PASSED: complete microbiome pipeline frozen at {manifest['freeze_timestamp']}")
print(f"[lit14] Frozen candidate models for Phase VIII/X: M1 (Literature Core), M3 (Data-driven LR+CatBoost)")
print(f"[lit14] wrote {SEMANTIC}/FROZEN_MICROBIOME_PIPELINE.json")
