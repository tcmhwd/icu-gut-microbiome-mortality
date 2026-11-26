#!/usr/bin/env python3
"""04_tune_ml.py

Formal Optuna hyperparameter tuning of the ML candidates selected by
03_ml_screening.py (default: read clinical_ai/ml_screening/selected_ml_models.json;
override with --models), outside the PyCaret black box (Section 29).

For each selected model x each direction (ICU dev -> MIMIC ext, MIMIC dev ->
ICU ext) x each applicable missing-data pipeline variant (Universal always;
Native-missing additionally for CatBoost/LightGBM/XGBoost, Section 30):
  - 5-fold stratified CV *inside the development dataset only* is the Optuna
    objective (mean validation AUROC); each fold's preprocessing is fit on
    that fold's training split only.
  - Best hyperparameters are refit on the FULL development cohort and
    evaluated on the untouched external dataset (bootstrap 95% CI).
  - Study, trial history, hyperparameters, plots, model artifact,
    patient-level predictions, fold-preprocessing audit rows, model-input
    manifest row, and a model_performance_comparison.csv row are all saved.

Run on cpu_partition/node2 via slurm/03_tune_ml.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, cv, experiment_runner, io_utils, model_registry, preprocessing_pipeline, reporting  # noqa: E402
from utils.preprocessing_pipeline import NativeMissingPreprocessor, UniversalPreprocessor  # noqa: E402

log = io_utils.setup_logging("04_tune_ml")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = preprocessing_pipeline.ALL_CLEAN_COLS


def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run 01_inspect_and_preprocess.py first")
    return pd.read_parquet(path)


def make_preprocessor(model_id: str, variant: str):
    if variant == "native":
        return NativeMissingPreprocessor()
    return UniversalPreprocessor(scale=(model_id == "lr"))


def build_estimator(model_id: str, params: dict, seed: int, n_jobs: int, y_train=None):
    build_fn = model_registry.REGISTRY[model_id]["build"]
    if model_id == "xgboost":
        return build_fn(params, seed, n_jobs, y_train=y_train)
    return build_fn(params, seed, n_jobs)


def class_imbalance_note(model_id: str, params: dict) -> str:
    base = "original outcome distribution preserved (no SMOTE/oversampling, Section 26)"
    key = {"lr": "class_weight", "rf": "class_weight", "et": "class_weight", "dt": "class_weight",
           "catboost": "auto_class_weights", "lightgbm": "is_unbalance", "xgboost": "scale_pos_weight_mode"}.get(model_id)
    if key and key in params:
        return f"{base}; tunable {key}={params[key]}"
    return base


def build_objective(model_id: str, variant: str, X_dev_raw: pd.DataFrame, y_dev: pd.Series,
                     seed: int, n_jobs: int, n_folds: int):
    folds = cv.make_folds(X_dev_raw, y_dev.values, n_folds, seed)

    def objective(trial: optuna.Trial):
        params = model_registry.REGISTRY[model_id]["space"](trial)
        aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            X_tr_raw, X_va_raw = X_dev_raw.iloc[tr_idx], X_dev_raw.iloc[va_idx]
            y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]

            prep = make_preprocessor(model_id, variant)
            X_tr = prep.fit_transform(X_tr_raw)
            X_va = prep.transform(X_va_raw)

            estimator = build_estimator(model_id, params, seed, n_jobs, y_tr)
            estimator.fit(X_tr, y_tr.values)
            p_va = estimator.predict_proba(X_va)[:, 1]
            if y_va.nunique() > 1:
                aucs.append(roc_auc_score(y_va, p_va))

            trial.report(float(np.mean(aucs)) if aucs else 0.0, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs)) if aucs else 0.0

    return objective


def save_optuna_artifacts(study: optuna.Study, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)
    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as fh:
        json.dump({"best_value_AUROC": study.best_value, "best_params": study.best_params}, fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        for fn, name in [(plot_optimization_history, "optimization_history"),
                          (plot_param_importances, "param_importance")]:
            try:
                ax = fn(study)
                fig = ax.get_figure() if hasattr(ax, "get_figure") else ax.figure
                # project-wide figure convention, 2026-08-13: svg (vector) + jpg (viewable)
                fig.savefig(os.path.join(out_dir, f"{name}.svg"), bbox_inches="tight")
                fig.savefig(os.path.join(out_dir, f"{name}.jpg"), bbox_inches="tight", dpi=200)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                log.warning("Could not render %s for %s: %s", name, out_dir, e)
    except Exception as e:
        log.warning("Optuna plotting unavailable (%s); trial history CSV still saved.", e)


def run_one(model_id: str, variant: str, dev: str, ext: str, args, seed: int, n_jobs: int, datasets: dict):
    model_name = model_id if variant == "universal" else f"{model_id}_native"
    log.info("=== %s | dev=%s -> ext=%s | pipeline=%s ===", model_name, dev, ext, variant)

    df_dev, df_ext = datasets[dev], datasets[ext]
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_ML, dev, model_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{dev}_{model_name}", storage=storage, load_if_exists=True,
        sampler=TPESampler(seed=seed), pruner=MedianPruner(n_warmup_steps=1),
    )
    n_remaining = max(args.n_trials - len(study.trials), 0)

    # Idempotent resubmission: if tuning already finished AND this exact
    # (model, direction) combo already has a performance row, don't re-run
    # the final refit/eval/append -- that would silently duplicate rows in
    # model_performance_comparison.csv / model_input_manifest.csv.
    if n_remaining == 0 and reporting.performance_row_exists(model_name, dev, ext):
        log.info("%s dev=%s ext=%s already fully tuned (%d trials) and recorded; skipping.",
                  model_name, dev, ext, len(study.trials))
        return

    objective = build_objective(model_id, variant, X_dev_raw, y_dev, seed, n_jobs, args.n_folds)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False)
    else:
        log.info("Study already has >= %d trials; skipping optimize().", args.n_trials)
    save_optuna_artifacts(study, study_dir)

    best_params = study.best_params
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    def train_fn(X_tr, y_tr, X_va, y_va):
        estimator = build_estimator(model_id, best_params, seed, n_jobs, y_tr)
        estimator.fit(X_tr, y_tr)

        def predict(X):
            return estimator.predict_proba(X)[:, 1]
        return predict, estimator

    result = experiment_runner.run_bidirectional_experiment(
        model_name=model_name, model_family=model_registry.REGISTRY[model_id]["display"],
        ml_or_dl="ML", tuning_method=f"Optuna TPE sampler + MedianPruner, {len(study.trials)} trials, "
                                      f"{args.n_folds}-fold CV AUROC objective (dev-only)",
        pipeline_name="Universal" if variant == "universal" else "NativeMissing",
        class_imbalance_strategy=class_imbalance_note(model_id, best_params),
        dev_dataset=dev, ext_dataset=ext,
        X_dev_raw=X_dev_raw, y_dev=y_dev, dev_sample_ids=dev_ids,
        X_ext_raw=X_ext_raw, y_ext=y_ext, ext_sample_ids=ext_ids,
        preprocessor_factory=lambda: make_preprocessor(model_id, variant),
        train_fn=train_fn, model_kind="sklearn",
        n_folds=args.n_folds, seed=seed, n_bootstrap=args.n_bootstrap,
        extra_manifest_notes=f"Optuna-tuned {model_id}; pipeline variant={variant}; "
                              f"study={storage}",
    )
    reporting.append_performance_rows([result["perf_row"]])
    log.info("%s dev=%s ext=%s external AUROC=%.4f (CI %.4f-%.4f)", model_name, dev, ext,
              result["perf_row"]["external_AUROC"], result["perf_row"]["external_AUROC_CI_low"],
              result["perf_row"]["external_AUROC_CI_high"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                     help="Override selected_ml_models.json, e.g. --models lr catboost et")
    ap.add_argument("--n-trials", type=int, default=config.N_OPTUNA_TRIALS_ML)
    ap.add_argument("--n-folds", type=int, default=config.N_CV_FOLDS)
    ap.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--skip-native-variant", action="store_true",
                     help="Skip the Section 30 Universal-vs-Native comparison for boosting models "
                          "(fit Universal pipeline only, faster).")
    ap.add_argument("--smoke-test-sample-n", type=int, default=None,
                     help="DEV/CI ONLY: stratified-subsample each dataset to ~N rows before tuning. "
                          "Never use for a real run.")
    args = ap.parse_args()

    config.ensure_output_dirs()
    n_jobs = args.n_jobs or config.get_n_jobs()
    log.info("Using n_jobs=%d, n_trials=%d, seed=%d", n_jobs, args.n_trials, args.seed)

    if args.models:
        selected = args.models
    else:
        sel_path = os.path.join(config.DIR_ML_SCREENING, "selected_ml_models.json")
        if not os.path.exists(sel_path):
            raise FileNotFoundError(f"{sel_path} not found -- run 03_ml_screening.py first, or pass --models")
        with open(sel_path) as fh:
            selected = json.load(fh)["selected_models"]
    log.info("Tuning ML candidates: %s", selected)

    datasets = {n: load_dataset(n) for n in config.DATASETS}
    if args.smoke_test_sample_n:
        for n in config.DATASETS:
            df = datasets[n]
            datasets[n] = df.groupby(config.TARGET_COL, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(args.smoke_test_sample_n // 2, 20)), random_state=args.seed)
            )
            log.warning("dataset=%s: SMOKE-TEST subsample active (N=%d) -- not a real tuning run",
                        n, len(datasets[n]))

    for model_id in selected:
        if model_id not in model_registry.REGISTRY:
            log.error("Unknown model id '%s' -- skipping", model_id)
            reporting.append_failure_rows([{
                "stage": "04_tune_ml", "model_name": model_id, "development_dataset": "n/a",
                "error_type": "UnknownModelId", "error_message": f"'{model_id}' not in model_registry.REGISTRY",
                "resolution": "skipped",
            }])
            continue

        variants = ["universal"]
        if model_registry.REGISTRY[model_id]["supports_native_nan"] and not args.skip_native_variant:
            variants.append("native")

        for variant in variants:
            for dev, ext in [("ICU", "MIMIC"), ("MIMIC", "ICU")]:
                try:
                    run_one(model_id, variant, dev, ext, args, args.seed, n_jobs, datasets)
                except Exception as e:
                    log.error("FAILED: model=%s variant=%s dev=%s ext=%s: %s", model_id, variant, dev, ext, e)
                    traceback.print_exc()
                    reporting.append_failure_rows([{
                        "stage": "04_tune_ml", "model_name": f"{model_id}_{variant}",
                        "development_dataset": dev, "error_type": type(e).__name__,
                        "error_message": str(e)[:1000],
                        "resolution": "skipped this (model, direction, pipeline-variant) combination; "
                                      "other combinations continued",
                    }])
                    continue

    reporting.write_model_input_summary_md()
    log.info("04_tune_ml.py complete.")


if __name__ == "__main__":
    main()
