"""experiment_runner.py — shared bidirectional-transportability harness used
by both 04_tune_ml.py and 05_tune_dl.py, so ML and DL models are evaluated
identically (Sections 34-40).

For one (model, development_dataset -> external_dataset) combination:
  1. K-fold CV strictly inside the development dataset (preprocessing fit
     per-training-fold only) -> internal CV AUROC/AUPRC + fold audit rows.
  2. Final refit on the FULL development cohort (preprocessing fit once on
     all of dev, still never touching external data) -> external evaluation
     with bootstrap CIs, calibration, patient-level predictions saved.

`train_fn(X_tr, y_tr, X_va, y_va) -> (predict_proba_fn, fitted_object)` is
the only thing that differs between an sklearn/boosting estimator (04) and a
PyTorch/TabNet model (05); X_va/y_va are None during the final refit and may
be ignored by callers that don't need a validation split (plain sklearn) or
used for early stopping (DL / TabNet).

Every model is saved via utils.model_io.save_bundle() as a single loadable
artifact (model + preprocessor + exact raw feature list), per user request
2026-08-12 -- see utils/model_io.py for the load_bundle()/predict_proba() API.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, cv, metrics, missingness, model_io, reporting


def run_bidirectional_experiment(
    *, model_name: str, model_family: str, ml_or_dl: str, tuning_method: str,
    pipeline_name: str, class_imbalance_strategy: str,
    dev_dataset: str, ext_dataset: str,
    X_dev_raw: pd.DataFrame, y_dev: pd.Series, dev_sample_ids: pd.Series,
    X_ext_raw: pd.DataFrame, y_ext: pd.Series, ext_sample_ids: pd.Series,
    preprocessor_factory, train_fn,
    model_kind: str = None,
    n_folds: int = config.N_CV_FOLDS, seed: int = config.RANDOM_SEED,
    n_bootstrap: int = config.N_BOOTSTRAP,
    extra_manifest_notes: str = "",
):
    experiment_id = f"{model_name}__dev-{dev_dataset}__ext-{ext_dataset}"
    folds = cv.make_folds(X_dev_raw, y_dev.values, n_folds, seed)

    oof_pred = np.full(len(y_dev), np.nan)
    fold_audit_rows = []
    fold_aucs, fold_auprcs = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        X_tr_raw, X_va_raw = X_dev_raw.iloc[tr_idx], X_dev_raw.iloc[va_idx]
        y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]

        prep = preprocessor_factory()
        X_tr = prep.fit_transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        predict_fn, _ = train_fn(X_tr, y_tr.values, X_va, y_va.values)
        p_va = predict_fn(X_va)
        oof_pred[va_idx] = p_va

        if y_va.nunique() > 1:
            fold_aucs.append(roc_auc_score(y_va, p_va))
            fold_auprcs.append(average_precision_score(y_va, p_va))
        fold_audit_rows += prep.audit_rows(X_tr_raw, X_va_raw, experiment_id, dev_dataset, model_name, fold_idx)

    missingness.append_fold_audit(fold_audit_rows)

    cv_auroc_mean = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")
    cv_auroc_sd = float(np.nanstd(fold_aucs)) if fold_aucs else float("nan")
    cv_auprc_mean = float(np.nanmean(fold_auprcs)) if fold_auprcs else float("nan")

    # --- final refit on the full development cohort (dev-only) ---
    prep_final = preprocessor_factory()
    X_dev_final = prep_final.fit_transform(X_dev_raw)
    X_ext_final = prep_final.transform(X_ext_raw)
    predict_fn_final, fitted_obj = train_fn(X_dev_final, y_dev.values, None, None)
    ext_prob = predict_fn_final(X_ext_final)

    final_audit_rows = prep_final.audit_rows(X_dev_raw, X_ext_raw, experiment_id, dev_dataset,
                                              model_name, "final_refit_dev_all__eval_external")
    missingness.append_fold_audit(final_audit_rows)

    ext_metrics = metrics.compute_metrics(y_ext.values, ext_prob)
    boot = metrics.bootstrap_ci(y_ext.values, ext_prob, metrics.DEFAULT_BOOTSTRAP_METRICS, n_boot=n_bootstrap, seed=seed)
    auroc_pt, auroc_lo, auroc_hi = boot["AUROC"]
    auprc_pt, auprc_lo, auprc_hi = boot["AUPRC"]

    # --- patient-level predictions (external) ---
    # Column name is f"true_{config.TARGET_COL}" (currently 'true_hospital_mortality')
    # rather than a hardcoded 'true_ICU_mortality', since the target was
    # corrected 2026-08-12 to hospital_mortality -- see config.py.
    true_col = f"true_{config.TARGET_COL}"
    pred_df = pd.DataFrame({
        "patient_id": ext_sample_ids.values,
        true_col: y_ext.values,
        "predicted_probability": ext_prob,
        "development_dataset": dev_dataset,
        "external_dataset": ext_dataset,
        "model_name": model_name,
        "pipeline_name": pipeline_name,
    })
    pred_path = os.path.join(config.DIR_PREDICTIONS, f"{experiment_id}.csv")
    pred_df.to_csv(pred_path, index=False)

    # --- internal (dev, OOF) predictions, saved for completeness/paired analysis ---
    oof_df = pd.DataFrame({
        "patient_id": dev_sample_ids.values,
        true_col: y_dev.values,
        "predicted_probability": oof_pred,
        "development_dataset": dev_dataset,
        "external_dataset": f"{dev_dataset}_internal_OOF",
        "model_name": model_name,
        "pipeline_name": pipeline_name,
    })
    oof_path = os.path.join(config.DIR_PREDICTIONS, f"{experiment_id}__internal_OOF.csv")
    oof_df.to_csv(oof_path, index=False)

    # --- model artifact (Section 46), single loadable bundle (2026-08-12) ---
    if model_kind is not None:
        artifact_dir = os.path.join(config.DIR_TRAINED_MODELS, ml_or_dl, dev_dataset, model_name)
        model_io.save_bundle(artifact_dir, model_kind, fitted_obj, prep_final, metadata={
            "model_name": model_name, "model_family": model_family, "ML_or_DL": ml_or_dl,
            "development_dataset": dev_dataset, "external_test_dataset": ext_dataset,
            "raw_feature_columns": list(X_dev_raw.columns), "random_seed": seed,
            "target": config.TARGET_COL, "prediction_landmark": "ICU admission + 24h",
            "pipeline_name": pipeline_name, "external_AUROC": float(auroc_pt),
        })

    perf_row = {
        "model_name": model_name, "model_family": model_family, "ML_or_DL": ml_or_dl,
        "pipeline_name": pipeline_name, "development_dataset": dev_dataset, "external_dataset": ext_dataset,
        "CV_AUROC_mean": cv_auroc_mean, "CV_AUROC_SD": cv_auroc_sd, "CV_AUPRC_mean": cv_auprc_mean,
        "external_AUROC": auroc_pt, "external_AUROC_CI_low": auroc_lo, "external_AUROC_CI_high": auroc_hi,
        "external_AUPRC": auprc_pt, "external_AUPRC_CI_low": auprc_lo, "external_AUPRC_CI_high": auprc_hi,
        "external_Brier": ext_metrics["Brier"], "external_logloss": ext_metrics["logloss"],
        "calibration_intercept": ext_metrics["calibration_intercept"],
        "calibration_slope": ext_metrics["calibration_slope"],
        "external_sensitivity": ext_metrics["sensitivity"], "external_specificity": ext_metrics["specificity"],
        "external_PPV": ext_metrics["PPV"], "external_NPV": ext_metrics["NPV"], "external_F1": ext_metrics["F1"],
        "external_N": ext_metrics["N"], "external_N_events": ext_metrics["N_events"],
        "number_of_predictors": X_dev_raw.shape[1],
        "number_of_parameters_if_applicable": _n_params(fitted_obj),
    }

    manifest_row = {
        "model_name": model_name, "model_family": model_family, "ML_or_DL": ml_or_dl,
        "development_dataset": dev_dataset, "external_test_dataset": ext_dataset,
        "input_feature_count_original": X_dev_raw.shape[1],
        "input_feature_count_after_encoding": X_dev_final.shape[1],
        "numeric_missing_strategy": _describe_prep(prep_final, "numeric"),
        "categorical_missing_strategy": _describe_prep(prep_final, "categorical"),
        "missing_indicator": _describe_prep(prep_final, "indicator"),
        "scaling": _describe_prep(prep_final, "scaling"),
        "categorical_encoding": "binary 0/1 (Yes/No, M/F, medical/surgical already binary in source)",
        "native_NaN_handling": _describe_prep(prep_final, "native"),
        "class_imbalance_strategy": class_imbalance_strategy,
        "hyperparameter_tuning_method": tuning_method,
        "CV_strategy": f"{n_folds}-fold stratified (plain, no patient grouping -- see utils/cv.py)",
        "target": config.TARGET_COL, "prediction_landmark": "ICU admission + 24h (Section 5)",
        "notes": extra_manifest_notes,
    }
    reporting.append_manifest_rows([manifest_row])

    return {
        "perf_row": perf_row, "manifest_row": manifest_row,
        "oof_pred": oof_pred, "ext_prob": ext_prob, "ext_metrics": ext_metrics,
        "prediction_path": pred_path, "fitted_obj": fitted_obj, "preprocessor": prep_final,
    }


def _n_params(fitted_obj):
    try:
        import torch
        if isinstance(fitted_obj, torch.nn.Module):
            return int(sum(p.numel() for p in fitted_obj.parameters()))
    except Exception:
        pass
    for attr in ("n_estimators", "tree_count_"):
        if hasattr(fitted_obj, attr):
            return getattr(fitted_obj, attr)
    return float("nan")


def _describe_prep(prep, kind):
    from .preprocessing_pipeline import NativeMissingPreprocessor, UniversalPreprocessor
    if isinstance(prep, UniversalPreprocessor):
        return {
            "numeric": "training-fold median imputation",
            "categorical": "training-fold mode imputation (binary Yes/No/M/F/medical-surgical)",
            "indicator": "yes, per numeric and binary predictor",
            "scaling": "training-fold z-score (StandardScaler)" if prep.scale else "none",
            "native": "no",
        }[kind]
    if isinstance(prep, NativeMissingPreprocessor):
        return {
            "numeric": "none -- NaN preserved natively",
            "categorical": "none -- NaN preserved natively",
            "indicator": "no",
            "scaling": "none",
            "native": "yes (numeric NaN passed through to boosting library)",
        }[kind]
    return "unknown"
