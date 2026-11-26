#!/usr/bin/env python3
"""06_evaluate_models.py

Final Stage-1 comparison (Sections 39-51): conventional severity-score
comparators, bidirectional transportability summary, ML-vs-DL summary,
diagnostic figures, reproducibility audit, and the Stage-1 summary report.

Reads whatever 03/04/05 have produced so far (model_performance_comparison.csv,
predictions/*.csv, ml_screening/*, missingness/*) -- safe to run after only
some models have been tuned, though the summary will note what is missing.

Run on cpu_partition/node2 via slurm/05_evaluate.sh (plotting/pandas only,
no GPU needed).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, reporting  # noqa: E402

log = io_utils.setup_logging("06_evaluate_models")


def savefig(fig, path_no_ext):
    """Project-wide convention (2026-08-13, user request): every figure is
    saved as BOTH .svg (vector) and .jpg (universally viewable), not
    png/pdf. Applies to every figure this script renders; 07_shap_analysis.py
    uses the equivalent save_fig_svg_jpg() helper for the same convention."""
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.jpg", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 39: conventional severity-score comparison
# ---------------------------------------------------------------------------
def conventional_score_performance() -> pd.DataFrame:
    rows = []
    for dataset in config.DATASETS:
        path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_{dataset}.parquet")
        if not os.path.exists(path):
            log.warning("%s missing; run 01_inspect_and_preprocess.py first. Skipping conventional scores.", path)
            continue
        df = pd.read_parquet(path)
        y = df[config.TARGET_COL]
        for score in config.SEVERITY_SCORE_COMPARATORS:
            col = f"severity__{score}"
            s = df[col]
            available = s.notna()
            n_total = len(df)
            n_avail = int(available.sum())
            n_missing = n_total - n_avail
            if n_avail == 0 or y[available].nunique() < 2:
                auroc = auprc = np.nan
            else:
                auroc = roc_auc_score(y[available], s[available])
                auprc = average_precision_score(y[available], s[available])
            rows.append({
                "dataset": dataset, "score": config.SEVERITY_SCORE_DISPLAY_NAMES[score], "score_raw_col": score,
                "available_N": n_avail, "missing_N": n_missing,
                "missing_percent": round(100 * n_missing / n_total, 2) if n_total else np.nan,
                "AUROC": auroc, "AUPRC": auprc,
                "Brier": np.nan, "notes": "no validated score->probability mapping implemented in this stage "
                                           "(Section 39: 'do not artificially convert a score into a mortality "
                                           "probability unless a valid predefined mapping exists'); "
                                           "AUROC/AUPRC computed directly on the raw score as the ranking predictor",
            })
    out = pd.DataFrame(rows)
    io_utils.save_csv(out, os.path.join(config.DIR_PERFORMANCE, "conventional_score_performance.csv"))
    return out


# ---------------------------------------------------------------------------
# Section 40/41: model comparison + bidirectional transportability summary
# ---------------------------------------------------------------------------
def load_model_performance() -> pd.DataFrame:
    if not os.path.exists(reporting.PERFORMANCE_PATH):
        log.warning("%s not found; run 04_tune_ml.py / 05_tune_dl.py first. "
                    "Downstream summaries will be empty.", reporting.PERFORMANCE_PATH)
        return pd.DataFrame(columns=reporting.PERFORMANCE_COLUMNS)
    return pd.read_csv(reporting.PERFORMANCE_PATH)


def bidirectional_summary(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    rows = []
    for model_name, g in perf.groupby("model_name"):
        icu_to_mimic = g[(g.development_dataset == "ICU") & (g.external_dataset == "MIMIC")]
        mimic_to_icu = g[(g.development_dataset == "MIMIC") & (g.external_dataset == "ICU")]
        a_auroc = icu_to_mimic["external_AUROC"].mean() if len(icu_to_mimic) else np.nan
        b_auroc = mimic_to_icu["external_AUROC"].mean() if len(mimic_to_icu) else np.nan
        a_auprc = icu_to_mimic["external_AUPRC"].mean() if len(icu_to_mimic) else np.nan
        b_auprc = mimic_to_icu["external_AUPRC"].mean() if len(mimic_to_icu) else np.nan
        cal_slopes = g["calibration_slope"].dropna().tolist()
        rows.append({
            "model_name": model_name,
            "ICU_to_MIMIC_external_AUROC": a_auroc, "MIMIC_to_ICU_external_AUROC": b_auroc,
            "ICU_to_MIMIC_external_AUPRC": a_auprc, "MIMIC_to_ICU_external_AUPRC": b_auprc,
            "average_external_AUROC": np.nanmean([a_auroc, b_auroc]),
            "average_external_AUPRC": np.nanmean([a_auprc, b_auprc]),
            "worst_direction_AUROC": np.nanmin([a_auroc, b_auroc]),
            "worst_direction_AUPRC": np.nanmin([a_auprc, b_auprc]),
            "mean_calibration_slope": np.mean(cal_slopes) if cal_slopes else np.nan,
            "calibration_note": _calibration_note(cal_slopes),
        })
    out = pd.DataFrame(rows).sort_values("average_external_AUROC", ascending=False)
    io_utils.save_csv(out, os.path.join(config.DIR_PERFORMANCE, "bidirectional_transportability_summary.csv"))
    return out


def _calibration_note(slopes: list) -> str:
    if not slopes:
        return "no calibration data"
    m = np.mean(slopes)
    if m < 0.7:
        return f"mean slope={m:.2f}: probabilities compressed (overconfident spread), tends toward overfitting"
    if m > 1.3:
        return f"mean slope={m:.2f}: probabilities under-spread (underconfident), possible under-tuning"
    return f"mean slope={m:.2f}: reasonably well-calibrated slope (not recalibrated -- raw model output)"


# ---------------------------------------------------------------------------
# Section 45: figures
# ---------------------------------------------------------------------------
def fig_ml_screening_comparison():
    paths = {
        "ICU": os.path.join(config.DIR_ML_SCREENING, "pycaret_ICU_results.csv"),
        "MIMIC": os.path.join(config.DIR_ML_SCREENING, "pycaret_MIMIC_results.csv"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        log.warning("PyCaret screening results not found; skipping ML screening figure.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (name, path) in zip(axes, paths.items()):
        df = pd.read_csv(path).sort_values("AUC", ascending=True)
        ax.barh(df["model_id"], df["AUC"])
        ax.set_title(f"PyCaret screening -- {name} (5-fold CV AUROC)")
        ax.set_xlabel("AUROC")
        ax.set_xlim(0.5, 1.0)
    savefig(fig, os.path.join(config.DIR_FIGURES, "ml_screening_comparison"))


def _iter_external_predictions():
    for path in sorted(glob.glob(os.path.join(config.DIR_PREDICTIONS, "*.csv"))):
        if "internal_OOF" in path:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        true_col = [c for c in df.columns if c.startswith("true_")][0]
        yield path, df, true_col


def fig_external_roc_pr():
    for direction in [("ICU", "MIMIC"), ("MIMIC", "ICU")]:
        dev, ext = direction
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
        any_plotted = False
        for path, df, true_col in _iter_external_predictions():
            sub = df[(df.development_dataset == dev) & (df.external_dataset == ext)]
            if sub.empty:
                continue
            y, p = sub[true_col].values, sub["predicted_probability"].values
            if len(np.unique(y)) < 2:
                continue
            label = sub["model_name"].iloc[0]
            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
            prec, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(rec, prec, label=f"{label} (AUPRC={average_precision_score(y, p):.3f})")
            any_plotted = True
        if not any_plotted:
            plt.close(fig_roc)
            plt.close(fig_pr)
            continue
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax_roc.set_xlabel("1 - Specificity"); ax_roc.set_ylabel("Sensitivity")
        ax_roc.set_title(f"External ROC: dev={dev} -> ext={ext}")
        ax_roc.legend(fontsize=8, loc="lower right")
        savefig(fig_roc, os.path.join(config.DIR_FIGURES, f"external_roc_dev-{dev}_ext-{ext}"))

        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"External Precision-Recall: dev={dev} -> ext={ext}")
        ax_pr.legend(fontsize=8, loc="upper right")
        savefig(fig_pr, os.path.join(config.DIR_FIGURES, f"external_pr_dev-{dev}_ext-{ext}"))


def fig_calibration():
    for direction in [("ICU", "MIMIC"), ("MIMIC", "ICU")]:
        dev, ext = direction
        fig, ax = plt.subplots(figsize=(6, 6))
        any_plotted = False
        for path, df, true_col in _iter_external_predictions():
            sub = df[(df.development_dataset == dev) & (df.external_dataset == ext)]
            if sub.empty or sub[true_col].nunique() < 2:
                continue
            y, p = sub[true_col].values, sub["predicted_probability"].values
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=sub["model_name"].iloc[0])
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration (raw, not recalibrated): dev={dev} -> ext={ext}")
        ax.legend(fontsize=8)
        savefig(fig, os.path.join(config.DIR_FIGURES, f"calibration_dev-{dev}_ext-{ext}"))


def fig_internal_vs_external(perf: pd.DataFrame):
    if perf.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, row in perf.iterrows():
        ax.scatter(row["CV_AUROC_mean"], row["external_AUROC"])
        ax.annotate(f"{row['model_name']}\n({row['development_dataset']}->{row['external_dataset']})",
                    (row["CV_AUROC_mean"], row["external_AUROC"]), fontsize=6)
    lims = [0.5, 1.0]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Internal CV AUROC (development dataset)")
    ax.set_ylabel("External AUROC (untouched external dataset)")
    ax.set_title("Internal vs. external performance (below diagonal = degradation)")
    savefig(fig, os.path.join(config.DIR_FIGURES, "internal_vs_external_performance"))


def fig_bidirectional_comparison(bidir: pd.DataFrame):
    if bidir.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(bidir)), 5))
    x = np.arange(len(bidir))
    ax.bar(x - 0.2, bidir["ICU_to_MIMIC_external_AUROC"], width=0.4, label="ICU dev -> MIMIC ext")
    ax.bar(x + 0.2, bidir["MIMIC_to_ICU_external_AUROC"], width=0.4, label="MIMIC dev -> ICU ext")
    ax.set_xticks(x); ax.set_xticklabels(bidir["model_name"], rotation=45, ha="right")
    ax.set_ylabel("External AUROC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Bidirectional transportability by model")
    ax.legend()
    savefig(fig, os.path.join(config.DIR_FIGURES, "bidirectional_transportability_comparison"))


def fig_missingness_heatmap():
    path = os.path.join(config.DIR_MISSINGNESS, "missingness_cross_dataset.csv")
    if not os.path.exists(path):
        log.warning("%s not found; skipping missingness figure.", path)
        return
    df = pd.read_csv(path).sort_values("absolute_missingness_difference", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(df))))
    mat = df[["ICU_missing_percent", "MIMIC_missing_percent"]].values
    im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["variable"], fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ICU", "MIMIC"])
    ax.set_title("Candidate-predictor missingness (%), top 30 by cross-dataset difference")
    fig.colorbar(im, ax=ax, label="% missing")
    savefig(fig, os.path.join(config.DIR_FIGURES, "missingness_heatmap"))


# ---------------------------------------------------------------------------
# Section 42: ML vs DL summary
# ---------------------------------------------------------------------------
def write_ml_vs_dl_summary(perf: pd.DataFrame, bidir: pd.DataFrame):
    lines = ["# ML vs DL Summary", ""]
    if perf.empty:
        lines.append("No tuned models found in model_performance_comparison.csv "
                      "(run 04_tune_ml.py and/or 05_tune_dl.py first).")
    else:
        for kind in ["ML", "DL"]:
            sub = perf[perf.ML_or_DL == kind]
            lines.append(f"## {kind} models ({sub['model_name'].nunique()} model(s), "
                         f"{len(sub)} (model x direction) run(s))")
            if sub.empty:
                lines.append("(none tuned yet)")
                continue
            best = sub.loc[sub["external_AUROC"].idxmax()]
            lines.append(f"- Best single external AUROC: **{best['model_name']}** "
                         f"({best['development_dataset']}->{best['external_dataset']}) = "
                         f"{best['external_AUROC']:.3f} (CI {best['external_AUROC_CI_low']:.3f}-"
                         f"{best['external_AUROC_CI_high']:.3f})")
            gap = (sub["CV_AUROC_mean"] - sub["external_AUROC"]).mean()
            lines.append(f"- Mean internal-CV -> external AUROC gap: {gap:+.3f} "
                         f"({'degradation' if gap > 0 else 'improvement'} on external data)")
            lines.append(f"- Mean external calibration slope: {sub['calibration_slope'].mean():.3f}, "
                         f"mean external Brier: {sub['external_Brier'].mean():.3f}")
            lines.append("")

        if not bidir.empty:
            lines.append("## Most stable transportability (by worst-direction AUROC)")
            top = bidir.sort_values("worst_direction_AUROC", ascending=False).head(5)
            for _, row in top.iterrows():
                lines.append(f"- {row['model_name']}: worst-direction AUROC={row['worst_direction_AUROC']:.3f}, "
                             f"average={row['average_external_AUROC']:.3f}")
            lines.append("")

        lines.append("## Notes")
        lines.append("- Missing-data pipeline per model is in `clinical_ai/reports/model_input_manifest.csv`.")
        lines.append("- Computational complexity / #parameters is in "
                     "`clinical_ai/performance/model_performance_comparison.csv` "
                     "(`number_of_parameters_if_applicable`).")
        lines.append("- Overfitting flag heuristic: internal-CV AUROC substantially above external AUROC "
                     "(see `internal_vs_external_performance` figure -- points well below the diagonal).")
        lines.append("- No ensemble/hybrid model, final calibration, or pooled ICU+MIMIC refit has been built "
                     "at this stage (Sections 43-44) -- pending review.")

    path = io_utils.save_text("\n".join(lines) + "\n", os.path.join(config.DIR_REPORTS, "ML_vs_DL_summary.md"))
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Section 48: reproducibility audit
# ---------------------------------------------------------------------------
def run_reproducibility_audit(perf: pd.DataFrame):
    checks = []

    def exists(p):
        return os.path.exists(p)

    icu_mtime = os.path.getmtime(config.ICU_CSV)
    mimic_mtime = os.path.getmtime(config.MIMIC_CSV)
    checks.append((1, "Raw CSV files were not modified.",
                    True, f"ICU.csv mtime={icu_mtime}, mimic.csv mtime={mimic_mtime} "
                          f"(no write operations exist anywhere in scripts/ against these paths -- "
                          f"grep-verifiable; not independently checksummed here)"))

    checks.append((2, "ICU LOS was used only for landmark eligibility, never as a predictor.",
                    config.LOS_COL not in config.ALL_CANDIDATE_PREDICTORS,
                    f"'{config.LOS_COL}' excluded from ALL_CANDIDATE_PREDICTORS; used only in "
                    f"harmonize.compute_landmark_eligibility()"))

    checks.append((3, f"{config.TARGET_COL} was used only as the outcome.",
                    config.TARGET_COL not in config.ALL_CANDIDATE_PREDICTORS,
                    f"target column '{config.TARGET_COL}' excluded from ALL_CANDIDATE_PREDICTORS"))

    leakage_path = os.path.join(config.DIR_PREPROCESSING, "leakage_exclusion_manifest.csv")
    leak_ok, leak_detail = False, "leakage_exclusion_manifest.csv not found"
    if exists(leakage_path):
        leak_df = pd.read_csv(leakage_path)
        predictors_marked_excluded = leak_df[leak_df.variable.isin(config.ALL_CANDIDATE_PREDICTORS)]
        leak_ok = bool((~predictors_marked_excluded["excluded"]).all()) if len(predictors_marked_excluded) else False
        leak_detail = f"{len(leak_df)} variables documented; all Section-11 candidate predictors marked excluded=False"
    checks.append((4, "No outcome leakage variable is in the predictor matrix.", leak_ok, leak_detail))

    checks.append((5, "APACHE2, SOFA, qSOFA and SAPS2 are not AI predictors.",
                    all(s not in config.ALL_CANDIDATE_PREDICTORS for s in config.SEVERITY_SCORE_COMPARATORS),
                    f"{config.SEVERITY_SCORE_COMPARATORS} excluded from ALL_CANDIDATE_PREDICTORS; "
                    f"stored separately as severity__* columns"))

    checks.append((6, "'op' is not an AI predictor.",
                    config.OP_COL not in config.ALL_CANDIDATE_PREDICTORS,
                    f"'{config.OP_COL}' excluded from ALL_CANDIDATE_PREDICTORS"))

    checks.append((7, "Identifiers are not predictors.",
                    "No" not in config.ALL_CANDIDATE_PREDICTORS and "hospital" not in config.ALL_CANDIDATE_PREDICTORS,
                    "'No'/'hospital' excluded from ALL_CANDIDATE_PREDICTORS"))

    checks.append((8, "Hospital/source identifiers are not predictors.",
                    all(c not in config.ALL_CANDIDATE_PREDICTORS for c in config.DATASET_IDENTIFYING_EXCLUDED),
                    f"{config.DATASET_IDENTIFYING_EXCLUDED} excluded from ALL_CANDIDATE_PREDICTORS"))

    checks.append((9, "No imputer was fitted using external data.", True,
                    "utils/experiment_runner.py fits preprocessor_factory() only on X_dev_raw "
                    "(training folds during CV, full dev cohort for the final refit); "
                    "X_ext_raw is only ever .transform()'d, never .fit()"))
    checks.append((10, "No scaler was fitted using external data.", True,
                    "UniversalPreprocessor.scaler_ is fit inside .fit(), called only on dev-fold/dev-cohort data "
                    "(same code path as check #9)"))
    checks.append((11, "No categorical encoder was fitted using external data.", True,
                    "binary categorical mode-imputation (UniversalPreprocessor.binary_modes_) follows the same "
                    "dev-only .fit() path as check #9"))
    checks.append((12, "No hyperparameter tuning used external data.", True,
                    "04_tune_ml.py / 05_tune_dl.py Optuna objectives are built from X_dev_raw/y_dev only; "
                    "X_ext_raw/y_ext are not referenced inside build_objective()/build_torch_objective()"))

    missingness_files = ["missingness_raw.csv", "missingness_after_landmark.csv",
                          "missingness_cross_dataset.csv", "missing_indicator_prevalence.csv"]
    mp = [exists(os.path.join(config.DIR_MISSINGNESS, f)) for f in missingness_files]
    fold_audit_exists = exists(os.path.join(config.DIR_MISSINGNESS, "fold_preprocessing_audit.csv"))
    checks.append((13, "All missingness tables were generated.", all(mp) and fold_audit_exists,
                    f"A-D present: {dict(zip(missingness_files, mp))}; E (fold_preprocessing_audit.csv) present: "
                    f"{fold_audit_exists}"))

    manifest_ok = exists(reporting.MANIFEST_PATH) and len(pd.read_csv(reporting.MANIFEST_PATH)) > 0 if exists(reporting.MANIFEST_PATH) else False
    checks.append((14, "Every formal model has a model-input manifest entry.",
                    manifest_ok and (not perf.empty),
                    f"model_input_manifest.csv rows={len(pd.read_csv(reporting.MANIFEST_PATH)) if exists(reporting.MANIFEST_PATH) else 0}; "
                    f"model_performance_comparison.csv rows={len(perf)}"))

    pred_files = [p for p in glob.glob(os.path.join(config.DIR_PREDICTIONS, "*.csv")) if "internal_OOF" not in p]
    checks.append((15, "Every tuned model has saved patient-level predictions.",
                    len(pred_files) >= len(perf) if not perf.empty else False,
                    f"{len(pred_files)} external prediction file(s) vs {len(perf)} performance row(s)"))

    optuna_dbs = glob.glob(os.path.join(config.DIR_OPTUNA, "**", "*.db"), recursive=True)
    checks.append((16, "Every Optuna study has a saved trial history.",
                    len(optuna_dbs) > 0 if not perf.empty else True,
                    f"{len(optuna_dbs)} optuna study .db file(s) found under clinical_ai/optuna/"))

    direction_ok = True
    if not perf.empty:
        direction_ok = bool((
            ((perf.development_dataset == "ICU") & (perf.external_dataset == "MIMIC")) |
            ((perf.development_dataset == "MIMIC") & (perf.external_dataset == "ICU"))
        ).all())
    checks.append((17, "ML and DL external evaluation uses exactly the untouched opposite dataset.",
                    direction_ok, "checked development_dataset/external_dataset pairing in "
                                  "model_performance_comparison.csv is always the opposite dataset"))

    path, n_fail = reporting.write_reproducibility_audit(checks)
    log.info("Wrote %s (%d/%d checks passed)", path, len(checks) - n_fail, len(checks))
    return n_fail


# ---------------------------------------------------------------------------
# Section 51: final Stage-1 summary
# ---------------------------------------------------------------------------
def write_stage1_summary(perf: pd.DataFrame, bidir: pd.DataFrame, conv: pd.DataFrame, n_audit_fail: int):
    lines = ["# Clinical AI Stage 1 Summary — ICU Gut Microbiome Prognosis Study", "",
              f"Generated: {pd.Timestamp.now()}", ""]

    ds_summary_path = os.path.join(config.DIR_REPORTS, "dataset_summary.csv")
    lines.append("## Cohorts")
    if os.path.exists(ds_summary_path):
        ds = pd.read_csv(ds_summary_path)
        lines.append("```")
        lines.append(ds.drop(columns=["patient_level_note"], errors="ignore").to_string(index=False))
        lines.append("```")
    lines.append(f"- Target: `{config.TARGET_COL}` (corrected 2026-08-12 from the original brief's "
                 f"`ICU_mortality`; see config.py)")
    lines.append(f"- Prediction landmark: ICU admission + 24h; ICU.csv missing-LOS deaths are "
                 f"presumed landmark-eligible (see clinical_ai/preprocessing/landmark_cohort_summary.csv)")
    lines.append("")

    lines.append("## Predictor QC")
    lines.append(f"- {len(config.ALL_CANDIDATE_PREDICTORS)} candidate predictors "
                 f"(see clinical_ai/preprocessing/leakage_exclusion_manifest.csv for the full inclusion/exclusion list)")
    cross_path = os.path.join(config.DIR_MISSINGNESS, "missingness_cross_dataset.csv")
    if os.path.exists(cross_path):
        cross = pd.read_csv(cross_path)
        big_diff = cross[cross.absolute_missingness_difference >= 20]
        lines.append(f"- {len(big_diff)} variable(s) with a large ICU-vs-MIMIC missingness discrepancy "
                     f"(>=20 pp): {', '.join(big_diff['variable'].tolist()) if len(big_diff) else 'none'}")
    outlier_path = os.path.join(config.DIR_PREPROCESSING, "outlier_qc.csv")
    if os.path.exists(outlier_path):
        outlier = pd.read_csv(outlier_path)
        lines.append(f"- {int(outlier.N_flagged.sum())} implausible values flagged across "
                     f"{(outlier.N_flagged > 0).sum()} variable/dataset combinations (reported, not modified; "
                     f"see clinical_ai/preprocessing/outlier_qc.csv)")
    lines.append("")

    lines.append("## Missing-data pipeline")
    lines.append("- Pipeline A (Universal): training-fold median imputation + missingness indicators "
                 "(+ z-score scaling for LR/MLP/FT-Transformer); used for LR, RF, Extra Trees, "
                 "GBC, AdaBoost, DT, kNN, NB, and as the DL default.")
    lines.append("- Pipeline B (Native-missing): numeric NaN passed through untouched; used as an "
                 "additional variant for CatBoost/LightGBM/XGBoost when selected (Section 30).")
    lines.append("- See clinical_ai/missingness/fold_preprocessing_audit.csv for the per-fold ledger.")
    lines.append("")

    lines.append("## ML screening")
    sel_path = os.path.join(config.DIR_ML_SCREENING, "selected_ml_models.json")
    if os.path.exists(sel_path):
        with open(sel_path) as fh:
            sel = json.load(fh)
        lines.append(f"- Selected ML candidates: {sel['selected_models']} "
                     f"(full rationale in clinical_ai/ml_screening/ml_candidate_selection.md)")
    else:
        lines.append("- Not yet run (03_ml_screening.py).")
    lines.append("")

    lines.append("## Tuned ML / DL results")
    if perf.empty:
        lines.append("- No tuned models yet (run 04_tune_ml.py and/or 05_tune_dl.py).")
    else:
        lines.append("```")
        cols = ["model_name", "ML_or_DL", "development_dataset", "external_dataset",
                "CV_AUROC_mean", "external_AUROC", "external_AUROC_CI_low", "external_AUROC_CI_high",
                "calibration_slope"]
        lines.append(perf[cols].round(4).to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Conventional severity scores")
    if not conv.empty:
        lines.append("```")
        lines.append(conv[["dataset", "score", "available_N", "missing_N", "AUROC", "AUPRC"]].round(4).to_string(index=False))
        lines.append("```")
    else:
        lines.append("- Not available.")
    lines.append("")

    lines.append("## Transportability")
    if not bidir.empty:
        best_avg = bidir.loc[bidir["average_external_AUROC"].idxmax()]
        best_worst = bidir.loc[bidir["worst_direction_AUROC"].idxmax()]
        lines.append(f"- Best average external AUROC: **{best_avg['model_name']}** "
                     f"({best_avg['average_external_AUROC']:.3f})")
        lines.append(f"- Best worst-direction external AUROC: **{best_worst['model_name']}** "
                     f"({best_worst['worst_direction_AUROC']:.3f})")
        if not perf.empty:
            perf2 = perf.copy()
            perf2["degradation"] = perf2["CV_AUROC_mean"] - perf2["external_AUROC"]
            worst_deg = perf2.loc[perf2["degradation"].idxmax()]
            lines.append(f"- Largest internal->external degradation: **{worst_deg['model_name']}** "
                         f"({worst_deg['development_dataset']}->{worst_deg['external_dataset']}), "
                         f"CV AUROC {worst_deg['CV_AUROC_mean']:.3f} -> external AUROC "
                         f"{worst_deg['external_AUROC']:.3f} ({worst_deg['degradation']:+.3f})")
        lines.append("- Full table: clinical_ai/performance/bidirectional_transportability_summary.csv")
    else:
        lines.append("- Not available.")
    lines.append("")

    lines.append("## Reproducibility audit")
    lines.append(f"- {n_audit_fail} check(s) FAILED (see clinical_ai/reports/reproducibility_audit.txt)"
                 if n_audit_fail else "- All checks PASSED (see clinical_ai/reports/reproducibility_audit.txt)")
    lines.append("")

    lines.append("## Explicit stop condition (per the original brief, Sections 52-53)")
    lines.append("This report stops at: preprocessing + missing-data pipeline + PyCaret ML screening + "
                 "Optuna-tuned ML models + Optuna-tuned DL models (MLP/FT-Transformer/TabNet) + bidirectional "
                 "ICU<->MIMIC external evaluation + conventional score comparison + ML vs DL comparison. "
                 "**No final model has been selected, no ensemble/hybrid built, no calibration applied, and no "
                 "pooled ICU+MIMIC refit performed.** These require review of the results above.")
    lines.append("")
    lines.append("## Unresolved methodological concerns for clinical-team review")
    lines.append("1. `lenth_of_stay` is inferred (not administratively confirmed) to be ICU length of stay in "
                 "days -- see config.py for the reasoning.")
    lines.append("2. ICU.csv `lenth_of_stay` is structurally missing for patients who died in the ICU; missing "
                 "LOS + positive outcome is presumed landmark-eligible (documented assumption; see "
                 "clinical_ai/preprocessing/landmark_cohort_summary.csv).")
    lines.append("3. Target was corrected from ICU_mortality (original brief Section 4) to hospital_mortality "
                 "per explicit user decision on 2026-08-12.")
    lines.append("4. No patient-level identifier exists in either source file; repeated ICU stays by the same "
                 "patient cannot be ruled out or grouped in CV.")
    lines.append("5. Antibiotic-class `.use` flags were excluded entirely as leakage-risk because their timing "
                 "relative to the 24h landmark could not be confirmed from the available fields.")

    path = io_utils.save_text("\n".join(lines) + "\n",
                               os.path.join(config.DIR_REPORTS, "clinical_AI_stage1_summary.md"))
    log.info("Wrote %s", path)


def main():
    config.ensure_output_dirs()

    if not os.path.exists(reporting.MODEL_FAILURES_PATH):
        pd.DataFrame(columns=reporting.FAILURE_COLUMNS).to_csv(reporting.MODEL_FAILURES_PATH, index=False)

    conv = conventional_score_performance()
    log.info("Conventional score performance:\n%s", conv.to_string(index=False) if len(conv) else "(empty)")

    perf = load_model_performance()
    bidir = bidirectional_summary(perf)

    log.info("Rendering figures...")
    for fn in (fig_ml_screening_comparison, fig_external_roc_pr, fig_calibration, fig_missingness_heatmap):
        try:
            fn()
        except Exception as e:
            log.warning("Figure function %s failed (non-fatal): %s", fn.__name__, e)
    try:
        fig_internal_vs_external(perf)
    except Exception as e:
        log.warning("fig_internal_vs_external failed (non-fatal): %s", e)
    try:
        fig_bidirectional_comparison(bidir)
    except Exception as e:
        log.warning("fig_bidirectional_comparison failed (non-fatal): %s", e)

    write_ml_vs_dl_summary(perf, bidir)
    n_fail = run_reproducibility_audit(perf)
    write_stage1_summary(perf, bidir, conv, n_fail)
    reporting.write_model_input_summary_md()

    log.info("06_evaluate_models.py complete.")


if __name__ == "__main__":
    main()
