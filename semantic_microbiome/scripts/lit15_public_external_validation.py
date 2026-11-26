"""
lit15_public_external_validation.py — Phase VIII / GATE 4->5 transition
(spec sec. 51-55): apply the FROZEN Gangnam pipeline (lit14) unchanged to
public cohorts. No new feature selection, no literature feature
modification, no percentile recalculation, no PCA refitting, no public
reclustering, no hyperparameter retuning, no threshold optimization
(sec. 51) -- every transform below loads an already-fit object from
lit14's frozen_models/frozen_clustering directories and only calls
.transform()/.predict()/.predict_proba().

52: per-cohort validation report (N, deaths, mortality endpoint, sampling,
    layer availability).
53: public model performance (M1 Literature Core, M3 Data-driven) --
    AUROC/AUPRC/Brier/calibration intercept+slope, bootstrap 95% CI.
    (M4 Hybrid / Structured+Semantic Hybrid excluded -- flagged as leaked
    in lit14, never made it into the frozen candidate set.)
54: public phenotype validation -- public patients assigned to the
    FROZEN Gangnam-only clusters (never independently re-clustered),
    then cluster prevalence / cluster-mortality / feature-pattern report.
55: cohort domain shift -- can a classifier trivially tell Gangnam from
    Public using (a) structured Literature Core features, (b) S1
    embeddings? Nested CV AUROC for "predict source cohort" -- if near
    1.0, strong domain shift.

hospital_mortality used only for evaluation (AUROC/calibration/cluster
cross-tab), never for any fitting step in this script.

Outputs:
  public_validation/cohort_summary.tsv
  public_validation/model_performance_M1_M3.tsv/.md   (+ bootstrap CIs)
  public_validation/cluster_assignment_public.csv
  public_validation/cluster_mortality_public.tsv
  public_validation/domain_shift_results.tsv/.md
  public_validation/figures/calibration_curves.svg+jpg
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEMANTIC, ICU_PUBLIC, HARMONIZED_DIR, STRUCTURED_DIR, SEED, save_fig, today_str

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PUBVAL_DIR = SEMANTIC / "public_validation"
FIGURES_DIR = PUBVAL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_DIR = SEMANTIC / "supervised_semantic" / "frozen_models"
PCA_DIR = FROZEN_DIR / "M3_pca_loadings"
FROZEN_CLUSTER_DIR = SEMANTIC / "clustering" / "lit_phenotypes" / "frozen_clustering"
EMB_DIR = SEMANTIC / "embeddings"

MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
sm = pd.read_csv(MERGE_OUTPUT / "sample_metadata.tsv", sep="\t", dtype=str)
sm["mortality"] = pd.to_numeric(sm["mortality"], errors="coerce")
public = sm[sm.dataset != "ICU"].set_index("sample_id")

cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
y_gangnam = cohort["hospital_mortality"].astype(int)

MORTALITY_ENDPOINT = {"AGI_BGISEQ": "in-hospital (Title S/D)", "PRJNA842425": "mortality_label (xlsx)",
                       "PRJNA1100796": "all-death (100%, sanity-check only)", "PRJNA1102991": "all-death (100%, sanity-check only)"}

# ---------------------------------------------------------------------
# 52. per-cohort validation report
# ---------------------------------------------------------------------
lit_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")

rows = []
for cname, grp in public.groupby("subcohort"):
    n_with_core = lit_core.reindex(grp.index).notna().all(axis=1).sum()
    rows.append({"cohort": cname, "N": len(grp), "deaths": int(grp.mortality.sum(skipna=True)),
                 "survivors": int((grp.mortality == 0).sum()), "mortality_endpoint": MORTALITY_ENDPOINT.get(cname, "?"),
                 "literature_core_complete": int(n_with_core), "literature_core_pct": round(100 * n_with_core / len(grp), 1)})
cohort_summary = pd.DataFrame(rows)
cohort_summary.to_csv(PUBVAL_DIR / "cohort_summary.tsv", sep="\t", index=False)
print(f"[lit15] cohort summary:\n{cohort_summary.to_string(index=False)}")

# ---------------------------------------------------------------------
# 53. apply frozen M1 (Literature Core LR) and M3 (Data-driven LR+CatBoost)
# ---------------------------------------------------------------------
m1_pipe = joblib.load(FROZEN_DIR / "M1_literature_core_LR.joblib")
lit_core_cols = [c for c in lit_core.columns]  # already LitCore_-prefixed? check
# lit08 output has raw feature names; M1 was fit on LitCore_-prefixed columns in lit14 -- reproduce that prefixing
lit_core_prefixed = lit_core.copy()
lit_core_prefixed.columns = [f"LitCore_{c}" for c in lit_core.columns]

LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}

# low-dim raw sources (pooled, public sample_ids already correct -- same pattern as script 25)
LOW_DIM_RAW = {
    "taxonomy_species_filtered": (SEMANTIC.parent.parent / "merge" / "output" / "metaphlan3" / "metaphlan3_species_merged.tsv", "clade_name"),
}


def build_public_m3_matrix(ids):
    # low-dim: reuse frozen COLUMN LIST from structured/*_filtered.tsv, values from pooled raw sources (script 25 pattern)
    from common import METAPHLAN_DIR, HUMANN_DIR
    MAG_SUMMARY = ICU_PUBLIC / "merge" / "output" / "mag_summary"
    RAW_SOURCES = {
        "taxonomy_species_filtered.tsv": (METAPHLAN_DIR / "metaphlan3_species_merged.tsv", "clade_name"),
        "taxonomy_genus_filtered.tsv": (METAPHLAN_DIR / "metaphlan3_genus_merged.tsv", "clade_name"),
        "humann_pathway_filtered.tsv": (HUMANN_DIR / "humann_pathabundance_wide.unstratified.tsv", "# Pathway"),
        "contig_rgi_drugclass_filtered.tsv": (MAG_SUMMARY / "rgi_contig" / "rgi_contig_drug_class_per_sample.tsv", "feature"),
        "contig_rgi_rpkm_drugclass_filtered.tsv": (MAG_SUMMARY / "rpkm_rgi" / "rpkm_rgi_drug_class_per_sample.tsv", "feature"),
        "mag_rgi_drugclass_filtered.tsv": (MAG_SUMMARY / "rgi" / "rgi_drug_class_per_sample.tsv", "feature"),
        "vfdb_category_filtered.tsv": (MAG_SUMMARY / "vfdb" / "vfdb_category_per_sample.tsv", "feature"),
        "antismash_bgc_types_filtered.tsv": (MAG_SUMMARY / "antismash" / "antismash_bgc_types_per_sample.tsv", "feature"),
        "eggnog_cog_filtered.tsv": (MAG_SUMMARY / "eggnog" / "eggnog_cog_per_sample.tsv", "feature"),
    }
    low_parts = []
    for fname in LOW_DIM_FILES:
        frozen_cols = pd.read_csv(STRUCTURED_DIR / fname, sep="\t", index_col="patient_id", nrows=0).columns.tolist()
        raw_path, id_col = RAW_SOURCES[fname]
        raw = pd.read_csv(raw_path, sep="\t", index_col=id_col)
        present = [c for c in frozen_cols if c in raw.index]
        sub = raw.loc[present].T.reindex(index=ids, columns=frozen_cols)
        sub.columns = [f"{fname.replace('_filtered.tsv', '')}__{c}" for c in sub.columns]
        low_parts.append(sub)
    X_low = pd.concat(low_parts, axis=1)

    pca_parts = [X_low]
    for name, fname in HIGH_DIM_FILES.items():
        frozen_cols = pd.read_csv(STRUCTURED_DIR / fname, sep="\t", index_col="patient_id", nrows=0).columns.tolist()
        raw_path, id_col = RAW_SOURCES[fname] if fname in RAW_SOURCES else (None, None)
        MAG_MAP = {"mag_abundance_filtered.tsv": (MAG_SUMMARY / "coverm" / "coverm_abundance_matrix.tsv", "MAG"),
                   "eggnog_kegg_ko_filtered.tsv": (MAG_SUMMARY / "eggnog" / "eggnog_kegg_ko_per_sample.tsv", "feature")}
        raw_path, id_col = MAG_MAP[fname]
        raw = pd.read_csv(raw_path, sep="\t", index_col=id_col)
        if id_col == "MAG":
            raw = raw.drop(index="unmapped", errors="ignore")
        present = [c for c in frozen_cols if c in raw.index]
        sub_raw = raw.loc[present].T.reindex(index=ids, columns=frozen_cols)

        imputer = joblib.load(PCA_DIR / f"{name}_imputer.joblib")
        scaler_mean = np.load(PCA_DIR / f"{name}_scaler_mean.npy")
        scaler_scale = np.load(PCA_DIR / f"{name}_scaler_scale.npy")
        loadings = np.load(PCA_DIR / f"{name}_loadings.npy")
        pca_mean = np.load(PCA_DIR / f"{name}_mean.npy")

        imputed = imputer.transform(sub_raw.values)
        scaled = (imputed - scaler_mean) / scaler_scale
        pcs = (scaled - pca_mean) @ loadings.T
        pca_parts.append(pd.DataFrame(pcs, index=ids, columns=[f"{name}_pc{i+1}" for i in range(loadings.shape[0])]))
    return pd.concat(pca_parts, axis=1)


bootstrap_rows = []
perf_rows = []
np.random.seed(SEED)


def bootstrap_auc_ci(y_true, p, n_boot=2000):
    aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        bidx = np.random.choice(idx, len(idx), replace=True)
        if len(set(y_true[bidx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[bidx], p[bidx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def calibration_intercept_slope(y_true, p):
    eps = 1e-6
    logit_p = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression().fit(logit_p, y_true)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return intercept, slope


fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, cname in zip(axes, ["AGI_BGISEQ", "PRJNA842425"]):
    ids = public[public.subcohort == cname].index.tolist()
    y_ext = public.loc[ids, "mortality"].astype(int).values

    X_m1 = lit_core_prefixed.reindex(ids)
    p_m1 = m1_pipe.predict_proba(X_m1)[:, 1]

    m3_lr = joblib.load(FROZEN_DIR / "M3_data_driven_LR.joblib")
    m3_cb = CatBoostClassifier()
    m3_cb.load_model(str(FROZEN_DIR / "M3_data_driven_CatBoost.cbm"))
    m3_cb_imputer = joblib.load(FROZEN_DIR / "M3_data_driven_CatBoost_imputer.joblib")
    X_m3 = build_public_m3_matrix(ids)
    p_m3_lr = m3_lr.predict_proba(X_m3)[:, 1]
    p_m3_cb = m3_cb.predict_proba(m3_cb_imputer.transform(X_m3))[:, 1]

    for model_name, p in [("M1_LiteratureCore", p_m1), ("M3_DataDriven_LR", p_m3_lr), ("M3_DataDriven_CatBoost", p_m3_cb)]:
        auc = roc_auc_score(y_ext, p)
        lo, hi = bootstrap_auc_ci(y_ext, p)
        brier = brier_score_loss(y_ext, p)
        intercept, slope = calibration_intercept_slope(y_ext, p)
        perf_rows.append({"cohort": cname, "model": model_name, "N": len(ids), "AUROC": round(auc, 3),
                          "AUROC_CI_low": round(lo, 3), "AUROC_CI_high": round(hi, 3), "Brier": round(brier, 4),
                          "calibration_intercept": round(intercept, 3), "calibration_slope": round(slope, 3)})

    order = np.argsort(p_m1)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for model_name, p in [("M1", p_m1), ("M3-LR", p_m3_lr), ("M3-CB", p_m3_cb)]:
        bins = np.linspace(0, 1, 6)
        bin_ids = np.digitize(p, bins) - 1
        obs, pred = [], []
        for b in range(5):
            mask = bin_ids == b
            if mask.sum() > 0:
                obs.append(y_ext[mask].mean()); pred.append(p[mask].mean())
        ax.plot(pred, obs, marker="o", label=model_name)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed mortality"); ax.set_title(cname); ax.legend(fontsize=7)

save_fig(fig, "calibration_curves", figures_dir=FIGURES_DIR); plt.close(fig)

# sanity check: all-death cohorts (median predicted prob only, no AUROC possible)
for cname in ["PRJNA1100796", "PRJNA1102991"]:
    ids = public[public.subcohort == cname].index.tolist()
    X_m1 = lit_core_prefixed.reindex(ids)
    p_m1 = m1_pipe.predict_proba(X_m1)[:, 1]
    perf_rows.append({"cohort": cname, "model": "M1_LiteratureCore", "N": len(ids), "AUROC": None,
                      "median_predicted_proba": round(float(np.median(p_m1)), 3), "note": "all-death, sanity check only"})

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(PUBVAL_DIR / "model_performance_M1_M3.tsv", sep="\t", index=False)
print(f"[lit15] public model performance:\n{perf_df.to_string(index=False)}")

# ---------------------------------------------------------------------
# 54. assign public patients to FROZEN Gangnam-only clusters (no reclustering)
# ---------------------------------------------------------------------
cluster_rows = []
for tag, emb_prefix in [("S1", "lit_S1_literature"), ("S2", "lit_S2_hybrid")]:
    emb = np.load(EMB_DIR / f"{emb_prefix}_embeddings.npy")
    idx = pd.read_csv(EMB_DIR / f"{emb_prefix}_index.csv").set_index("patient_id")
    scaler = joblib.load(FROZEN_CLUSTER_DIR / f"{tag}_scaler.joblib")
    pca = joblib.load(FROZEN_CLUSTER_DIR / f"{tag}_pca.joblib")
    km = joblib.load(FROZEN_CLUSTER_DIR / f"{tag}_kmeans.joblib")

    public_ids = [i for i in idx.index if i in public.index]
    E = pd.DataFrame(emb, index=idx.index).reindex(public_ids)
    labels = km.predict(pca.transform(scaler.transform(E.values)))
    for pid, lab in zip(public_ids, labels):
        cluster_rows.append({"sample_id": pid, "embedding": tag, "cluster": lab,
                             "subcohort": public.loc[pid, "subcohort"], "mortality": public.loc[pid, "mortality"]})

cluster_df = pd.DataFrame(cluster_rows)
cluster_df.to_csv(PUBVAL_DIR / "cluster_assignment_public.csv", index=False)

cluster_mortality = cluster_df[cluster_df.subcohort.isin(["AGI_BGISEQ", "PRJNA842425"])].groupby(
    ["embedding", "subcohort", "cluster"]).agg(n=("mortality", "size"), n_deaths=("mortality", "sum")).reset_index()
cluster_mortality["mortality_rate"] = cluster_mortality["n_deaths"] / cluster_mortality["n"]
cluster_mortality.to_csv(PUBVAL_DIR / "cluster_mortality_public.tsv", sep="\t", index=False)
print(f"[lit15] public cluster x mortality:\n{cluster_mortality.to_string(index=False)}")

prevalence = cluster_df.groupby(["embedding", "subcohort", "cluster"]).size().reset_index(name="n")
print(f"[lit15] cluster prevalence by cohort:\n{prevalence.to_string(index=False)}")

# ---------------------------------------------------------------------
# 55. cohort domain shift -- can Gangnam vs Public be trivially told apart?
# ---------------------------------------------------------------------
domain_rows = []
gangnam_ids = y_gangnam.index.tolist()
agi_ids = public[public.subcohort == "AGI_BGISEQ"].index.tolist()

# (a) structured Literature Core features
X_struct = lit_core_prefixed.reindex(gangnam_ids + agi_ids)
y_source = pd.Series([1] * len(gangnam_ids) + [0] * len(agi_ids), index=gangnam_ids + agi_ids)
pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                 ("clf", LogisticRegressionCV(Cs=5, cv=3, max_iter=2000, random_state=SEED, n_jobs=4))])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_struct = cross_val_score(pipe, X_struct, y_source, cv=skf, scoring="roc_auc", n_jobs=1).mean()
domain_rows.append({"representation": "Literature_Core_structured", "domain_shift_AUROC": round(auc_struct, 3)})

# (b) S1 embeddings
emb = np.load(EMB_DIR / "lit_S1_literature_embeddings.npy")
idx = pd.read_csv(EMB_DIR / "lit_S1_literature_index.csv").set_index("patient_id")
E = pd.DataFrame(emb, index=idx.index).reindex(gangnam_ids + agi_ids)
auc_emb = cross_val_score(pipe, E, y_source, cv=skf, scoring="roc_auc", n_jobs=1).mean()
domain_rows.append({"representation": "S1_Literature_embedding", "domain_shift_AUROC": round(auc_emb, 3)})

domain_df = pd.DataFrame(domain_rows)
domain_df.to_csv(PUBVAL_DIR / "domain_shift_results.tsv", sep="\t", index=False)
print(f"[lit15] domain shift (Gangnam vs AGI source-prediction AUROC):\n{domain_df.to_string(index=False)}")

with open(PUBVAL_DIR / "domain_shift_results.md", "w") as f:
    f.write(f"""# Phase VIII: cohort domain shift (spec sec. 55)

Generated {today_str()}. AUROC for a classifier predicting "is this patient
Gangnam or AGI" using only microbiome features (5-fold CV, no leakage --
this is a diagnostic, not a mortality model).

{domain_df.to_string(index=False)}

AUROC near 1.0 = features almost perfectly reveal which cohort/country a
sample came from (strong technical/population domain shift -- expected
given completely different sequencing platforms: Gangnam Illumina vs AGI
BGISEQ). AUROC near 0.5 = features do not reveal cohort identity.
""")

print(f"[lit15] wrote {PUBVAL_DIR}/")
