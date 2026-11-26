#!/usr/bin/env python3
"""12_tune_ml_reduced.py

Reduced-feature (29-column, see utils/reduced_features.py) ML training,
extending 09_train_reduced_clinical_model.py from 2 algorithms (lr,
catboost) to the full Stage-1 ML trio (lr, catboost, et) AND from a 2-cohort
(ICU-pooled, MIMIC) design to a 3-SITE design (Gangnam, Sinchon, MIMIC),
per explicit user request 2026-08-14 -- see 08b_preprocess_reduced_by_site.py
for the site-split rationale.

Reuses the exact Stage-1 infrastructure (utils.model_registry,
utils.experiment_runner.run_bidirectional_experiment,
utils.preprocessing_pipeline) for full methodological parity, only:
  - substituting the reduced 29-column feature set (utils.reduced_features)
    in place of preprocessing_pipeline.ALL_CLEAN_COLS, and
  - substituting a 3-site dev/pooled-external design (dev=one site,
    ext=the OTHER TWO SITES POOLED) in place of the 2-dataset ICU<->MIMIC
    bidirectional design.
ml_or_dl="ML_reduced" keeps every output (trained_models/, performance/,
reports/) in its own namespace, never overwriting or mixing with Stage 1's
ML/ tree.

Run on cpu_partition/node2 via slurm/12_tune_ml_reduced.sh.
Usage: python 12_tune_ml_reduced.py [--models lr catboost et] [--sites Gangnam Sinchon MIMIC]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, cv, experiment_runner, io_utils, model_registry, reduced_features, reporting  # noqa: E402
from utils.preprocessing_pipeline import NativeMissingPreprocessor, UniversalPreprocessor  # noqa: E402

log = io_utils.setup_logging("12_tune_ml_reduced")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_OR_DL = "ML_reduced"
SITES = ["Gangnam", "Sinchon", "MIMIC"]
FEATURE_COLS = reduced_features.REDUCED_ALL_COLS
NUMERIC_COLS = reduced_features.REDUCED_NUMERIC_COLS
BINARY_COLS = reduced_features.REDUCED_BINARY_COLS


def load_site(name: str) -> pd.DataFrame:
    path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_reduced_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run 08b_preprocess_reduced_by_site.py first")
    return pd.read_parquet(path)


def make_preprocessor(model_id: str, variant: str):
    if variant == "native":
        return NativeMissingPreprocessor(numeric_cols=list(NUMERIC_COLS), binary_cols=list(BINARY_COLS))
    return UniversalPreprocessor(scale=(model_id == "lr"), numeric_cols=list(NUMERIC_COLS), binary_cols=list(BINARY_COLS))


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

            build_fn = model_registry.REGISTRY[model_id]["build"]
            estimator = build_fn(params, seed, n_jobs)
            estimator.fit(X_tr, y_tr.values)
            p_va = estimator.predict_proba(X_va)[:, 1]
            if y_va.nunique() > 1:
                aucs.append(roc_auc_score(y_va, p_va))
            trial.report(float(pd.Series(aucs).mean()) if aucs else 0.0, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(pd.Series(aucs).mean()) if aucs else 0.0

    return objective


def run_one(model_id: str, variant: str, dev_site: str, ext_sites: list, args, seed: int,
            n_jobs: int, frames: dict):
    model_name = model_id if variant == "universal" else f"{model_id}_native"
    ext_label = "+".join(ext_sites)
    log.info("=== %s | dev=%s -> ext=%s | pipeline=%s ===", model_name, dev_site, ext_label, variant)

    df_dev = frames[dev_site]
    df_ext = pd.concat([frames[s] for s in ext_sites], ignore_index=True)
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_ML, "reduced_" + dev_site, model_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{dev_site}_{model_name}_reduced", storage=storage, load_if_exists=True,
        sampler=TPESampler(seed=seed), pruner=MedianPruner(n_warmup_steps=1),
    )
    n_remaining = max(args.n_trials - len(study.trials), 0)

    if n_remaining == 0 and reporting.performance_row_exists(model_name, dev_site, ext_label):
        log.info("%s dev=%s ext=%s already fully tuned and recorded; skipping.", model_name, dev_site, ext_label)
        return

    objective = build_objective(model_id, variant, X_dev_raw, y_dev, seed, n_jobs, args.n_folds)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False)
    save_optuna_artifacts(study, study_dir)

    best_params = study.best_params
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    def train_fn(X_tr, y_tr, X_va, y_va):
        build_fn = model_registry.REGISTRY[model_id]["build"]
        estimator = build_fn(best_params, seed, n_jobs)
        estimator.fit(X_tr, y_tr)

        def predict(X):
            return estimator.predict_proba(X)[:, 1]
        return predict, estimator

    result = experiment_runner.run_bidirectional_experiment(
        model_name=model_name, model_family=model_registry.REGISTRY[model_id]["display"],
        ml_or_dl=ML_OR_DL, tuning_method=f"Optuna TPE sampler + MedianPruner, {len(study.trials)} trials, "
                                          f"{args.n_folds}-fold CV AUROC objective (dev-only)",
        pipeline_name="Universal" if variant == "universal" else "NativeMissing",
        class_imbalance_strategy="original outcome distribution preserved (no SMOTE/oversampling)",
        dev_dataset=dev_site, ext_dataset=ext_label,
        X_dev_raw=X_dev_raw, y_dev=y_dev, dev_sample_ids=dev_ids,
        X_ext_raw=X_ext_raw, y_ext=y_ext, ext_sample_ids=ext_ids,
        preprocessor_factory=lambda: make_preprocessor(model_id, variant),
        train_fn=train_fn, model_kind="sklearn",
        n_folds=args.n_folds, seed=seed, n_bootstrap=args.n_bootstrap,
        extra_manifest_notes=f"REDUCED 29-feature model (utils/reduced_features.py); "
                              f"3-site design dev={dev_site} ext_pooled={ext_label}; study={storage}",
    )
    reporting.append_performance_rows([result["perf_row"]])
    log.info("%s dev=%s ext=%s external AUROC=%.4f (CI %.4f-%.4f)", model_name, dev_site, ext_label,
              result["perf_row"]["external_AUROC"], result["perf_row"]["external_AUROC_CI_low"],
              result["perf_row"]["external_AUROC_CI_high"])

    # per-site breakdown of the pooled external prediction (cheap, no
    # extra fitting) -- makes it easy to see e.g. "Gangnam-dev model on
    # Sinchon alone" vs "...on MIMIC alone" without training 2 separate models
    pred_df = pd.read_csv(result["prediction_path"])
    pred_df["site"] = pred_df["patient_id"].str.extract(r"^(Gangnam|Sinchon|MIMIC)_")
    breakdown = pred_df.groupby("site").apply(
        lambda g: pd.Series({
            "N": len(g), "N_events": int(g[f"true_{config.TARGET_COL}"].sum()),
            "AUROC": roc_auc_score(g[f"true_{config.TARGET_COL}"], g["predicted_probability"])
            if g[f"true_{config.TARGET_COL}"].nunique() > 1 else float("nan"),
        })
    ).reset_index()
    breakdown_path = os.path.join(config.DIR_PERFORMANCE, f"reduced_ml_per_site_breakdown_{model_name}_dev-{dev_site}.csv")
    breakdown.to_csv(breakdown_path, index=False)
    log.info("per-site external breakdown:\n%s", breakdown.to_string(index=False))


def save_optuna_artifacts(study: optuna.Study, out_dir: str):
    import json
    os.makedirs(out_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)
    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as fh:
        json.dump({"best_value_AUROC": study.best_value, "best_params": study.best_params}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lr", "catboost", "et"])
    ap.add_argument("--sites", nargs="+", default=SITES, choices=SITES,
                     help="Which site(s) to use as DEV in this job (lets sites be split across parallel jobs).")
    ap.add_argument("--n-trials", type=int, default=config.N_OPTUNA_TRIALS_ML)
    ap.add_argument("--n-folds", type=int, default=config.N_CV_FOLDS)
    ap.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--skip-native-variant", action="store_true")
    args = ap.parse_args()

    config.ensure_output_dirs()
    n_jobs = args.n_jobs or config.get_n_jobs()
    log.info("Using n_jobs=%d, n_trials=%d, seed=%d, models=%s, sites=%s",
              n_jobs, args.n_trials, args.seed, args.models, args.sites)

    frames = {s: load_site(s) for s in SITES}
    for s in SITES:
        log.info("site=%s: N=%d, events=%d (%.1f%%), %d reduced features",
                  s, len(frames[s]), int(frames[s][config.TARGET_COL].sum()),
                  100 * frames[s][config.TARGET_COL].mean(), len(FEATURE_COLS))

    for model_id in args.models:
        if model_id not in model_registry.REGISTRY:
            log.error("Unknown model id '%s' -- skipping", model_id)
            continue
        variants = ["universal"]
        if model_registry.REGISTRY[model_id]["supports_native_nan"] and not args.skip_native_variant:
            variants.append("native")

        for variant in variants:
            for dev_site in args.sites:
                ext_sites = [s for s in SITES if s != dev_site]
                try:
                    run_one(model_id, variant, dev_site, ext_sites, args, args.seed, n_jobs, frames)
                except Exception as e:
                    log.error("FAILED: model=%s variant=%s dev=%s: %s", model_id, variant, dev_site, e)
                    traceback.print_exc()
                    reporting.append_failure_rows([{
                        "stage": "12_tune_ml_reduced", "model_name": f"{model_id}_{variant}",
                        "development_dataset": dev_site, "error_type": type(e).__name__,
                        "error_message": str(e)[:1000], "resolution": "skipped this combination; others continued",
                    }])
                    continue

    reporting.write_model_input_summary_md()
    log.info("12_tune_ml_reduced.py complete.")


if __name__ == "__main__":
    main()
