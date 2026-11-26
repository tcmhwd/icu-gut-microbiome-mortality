#!/usr/bin/env python3
"""05_tune_dl.py

Optuna tuning of the 3 tabular DL families (Section 31): MLP, FT-Transformer,
TabNet. Same bidirectional-transportability design as 04_tune_ml.py, sharing
utils.experiment_runner so ML and DL are evaluated identically (Section 42).

All 3 DL families use the Universal missing-data pipeline (median impute +
missingness indicator + training-fold scaling, Section 32) -- there is no
DL-native-NaN variant (unlike CatBoost/LightGBM/XGBoost in 04_tune_ml.py).

Within each Optuna trial: 5-fold CV inside the development dataset; each
fold gets its own train/val split for early stopping (Section 33) as well as
for the CV-AUROC objective, i.e. the held-out fold IS the early-stopping
validation set for that fold (no separate triple-split) -- this is the
standard, resource-efficient CV+early-stopping compromise, documented here
so it's not accidentally read as leaking external data.

Best hyperparameters are then refit on the full development cohort with an
internal 90/10 split reserved ONLY for early-stopping (never for the
external evaluation), and evaluated on the untouched external dataset.

Run on gpu_partition/node1 via slurm/04_tune_dl.sh.
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
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, cv, dl_models, experiment_runner, io_utils, preprocessing_pipeline, reporting  # noqa: E402
from utils.preprocessing_pipeline import UniversalPreprocessor  # noqa: E402

log = io_utils.setup_logging("05_tune_dl")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = preprocessing_pipeline.ALL_CLEAN_COLS

DL_BUILDERS = {
    "MLP": (dl_models.mlp_optuna_space, dl_models.build_mlp),
    "FTTransformer": (dl_models.fttransformer_optuna_space, dl_models.build_fttransformer),
}


def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run 01_inspect_and_preprocess.py first")
    return pd.read_parquet(path)


def make_preprocessor():
    return UniversalPreprocessor(scale=True)


# ---------------------------------------------------------------------------
# MLP / FT-Transformer objective (shared torch training loop)
# ---------------------------------------------------------------------------
def build_torch_objective(family: str, X_dev_raw: pd.DataFrame, y_dev: pd.Series,
                           seed: int, n_folds: int, max_epochs: int, patience: int):
    space_fn, build_fn = DL_BUILDERS[family]
    folds = cv.make_folds(X_dev_raw, y_dev.values, n_folds, seed)

    def objective(trial: optuna.Trial):
        params = space_fn(trial)
        aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            X_tr_raw, X_va_raw = X_dev_raw.iloc[tr_idx], X_dev_raw.iloc[va_idx]
            y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]

            prep = make_preprocessor()
            X_tr = prep.fit_transform(X_tr_raw).values.astype(np.float32)
            X_va = prep.transform(X_va_raw).values.astype(np.float32)

            model = build_fn(X_tr.shape[1], params)
            _, fold_auc = dl_models.train_torch_classifier(
                model, X_tr, y_tr.values, X_va, y_va.values, params,
                max_epochs=max_epochs, patience=patience, seed=seed, trial=trial,
                step_offset=fold_idx * max_epochs,
            )
            if not np.isnan(fold_auc):
                aucs.append(fold_auc)
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


def run_torch_family(family: str, dev: str, ext: str, args, seed: int, datasets: dict):
    model_name = family
    log.info("=== %s | dev=%s -> ext=%s ===", model_name, dev, ext)
    df_dev, df_ext = datasets[dev], datasets[ext]
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_DL, dev, model_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{dev}_{model_name}", storage=storage, load_if_exists=True,
        sampler=TPESampler(seed=seed), pruner=MedianPruner(n_warmup_steps=1),
    )
    n_remaining = max(args.n_trials - len(study.trials), 0)

    # Idempotent resubmission -- see 04_tune_ml.py run_one() for rationale.
    if n_remaining == 0 and reporting.performance_row_exists(model_name, dev, ext):
        log.info("%s dev=%s ext=%s already fully tuned (%d trials) and recorded; skipping.",
                  model_name, dev, ext, len(study.trials))
        return

    objective = build_torch_objective(family, X_dev_raw, y_dev, seed, args.n_folds, args.max_epochs, args.patience)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False,
                        catch=(RuntimeError,))
    save_optuna_artifacts(study, study_dir)

    if args.trials_only:
        log.info("%s dev=%s: --trials-only, exiting after contributing trials to the shared study "
                  "(current best CV AUROC=%.4f). The primary job for this (family, dev) will do the "
                  "final refit once its own n_remaining reaches 0.", model_name, dev, study.best_value)
        return

    best_params = study.best_params
    if family == "MLP":
        # 'hidden_dims' is a derived list (hidden_dim_0, hidden_dim_1, ...),
        # never its own suggested key -- reconstruct before refitting.
        best_params = dl_models.mlp_reconstruct_params(best_params)
    _, build_fn = DL_BUILDERS[family]
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    def train_fn(X_tr, y_tr, X_va, y_va):
        X_tr = np.asarray(X_tr, dtype=np.float32)
        if X_va is None:
            # final refit: carve an internal early-stopping split out of dev
            # training data ONLY (never external data)
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_tr, y_tr, test_size=0.1, random_state=seed, stratify=y_tr,
            )
        else:
            X_es, y_es = np.asarray(X_va, dtype=np.float32), y_va
        model = build_fn(X_tr.shape[1], best_params)
        model, _ = dl_models.train_torch_classifier(
            model, X_tr, y_tr, X_es, y_es, best_params,
            max_epochs=args.max_epochs, patience=args.patience, seed=seed,
        )

        def predict(X):
            return dl_models.predict_proba_torch(model, np.asarray(X, dtype=np.float32))
        return predict, model

    result = experiment_runner.run_bidirectional_experiment(
        model_name=model_name, model_family=family, ml_or_dl="DL",
        tuning_method=f"Optuna TPE sampler + MedianPruner, {len(study.trials)} trials, "
                      f"{args.n_folds}-fold CV AUROC objective (dev-only), early stopping patience={args.patience}",
        pipeline_name="Universal", class_imbalance_strategy=(
            "original outcome distribution preserved (no SMOTE/oversampling, Section 26); "
            f"weighted BCEWithLogitsLoss tunable pos_weight_mode={best_params.get('pos_weight_mode')}"
        ),
        dev_dataset=dev, ext_dataset=ext,
        X_dev_raw=X_dev_raw, y_dev=y_dev, dev_sample_ids=dev_ids,
        X_ext_raw=X_ext_raw, y_ext=y_ext, ext_sample_ids=ext_ids,
        preprocessor_factory=make_preprocessor, train_fn=train_fn, model_kind="torch",
        n_folds=args.n_folds, seed=seed, n_bootstrap=args.n_bootstrap,
        extra_manifest_notes=f"Optuna-tuned {family} (PyTorch); study={storage}; "
                              f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
    )
    reporting.append_performance_rows([result["perf_row"]])
    log.info("%s dev=%s ext=%s external AUROC=%.4f (CI %.4f-%.4f)", model_name, dev, ext,
              result["perf_row"]["external_AUROC"], result["perf_row"]["external_AUROC_CI_low"],
              result["perf_row"]["external_AUROC_CI_high"])


# ---------------------------------------------------------------------------
# TabNet (own training loop via pytorch-tabnet)
# ---------------------------------------------------------------------------
def build_tabnet_objective(X_dev_raw: pd.DataFrame, y_dev: pd.Series, seed: int, n_folds: int,
                            max_epochs: int, patience: int):
    folds = cv.make_folds(X_dev_raw, y_dev.values, n_folds, seed)

    def objective(trial: optuna.Trial):
        params = dl_models.tabnet_optuna_space(trial)
        aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            X_tr_raw, X_va_raw = X_dev_raw.iloc[tr_idx], X_dev_raw.iloc[va_idx]
            y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]

            prep = make_preprocessor()
            X_tr = prep.fit_transform(X_tr_raw).values.astype(np.float32)
            X_va = prep.transform(X_va_raw).values.astype(np.float32)

            clf = dl_models.build_and_train_tabnet(
                X_tr, y_tr.values, X_va, y_va.values, params, seed=seed,
                max_epochs=max_epochs, patience=patience,
            )
            p_va = dl_models.tabnet_predict_proba(clf, X_va)
            if y_va.nunique() > 1:
                aucs.append(roc_auc_score(y_va, p_va))
            trial.report(float(np.mean(aucs)) if aucs else 0.0, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs)) if aucs else 0.0

    return objective


def run_tabnet(dev: str, ext: str, args, seed: int, datasets: dict):
    model_name = "TabNet"
    log.info("=== %s | dev=%s -> ext=%s ===", model_name, dev, ext)
    df_dev, df_ext = datasets[dev], datasets[ext]
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_DL, dev, model_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{dev}_{model_name}", storage=storage, load_if_exists=True,
        sampler=TPESampler(seed=seed), pruner=MedianPruner(n_warmup_steps=1),
    )
    n_remaining = max(args.n_trials - len(study.trials), 0)

    # Idempotent resubmission -- see 04_tune_ml.py run_one() for rationale.
    if n_remaining == 0 and reporting.performance_row_exists(model_name, dev, ext):
        log.info("%s dev=%s ext=%s already fully tuned (%d trials) and recorded; skipping.",
                  model_name, dev, ext, len(study.trials))
        return

    objective = build_tabnet_objective(X_dev_raw, y_dev, seed, args.n_folds, args.max_epochs, args.patience)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False)
    save_optuna_artifacts(study, study_dir)

    if args.trials_only:
        log.info("%s dev=%s: --trials-only, exiting after contributing trials to the shared study "
                  "(current best CV AUROC=%.4f).", model_name, dev, study.best_value)
        return

    best_params = dict(study.best_params)
    # 'n_a' is tied to 'n_d' in tabnet_optuna_space() via direct assignment
    # (not trial.suggest_*), so it never appears in study.best_params -- must
    # be re-derived here before refitting, or build_and_train_tabnet() KeyErrors.
    best_params["n_a"] = best_params["n_d"]
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    def train_fn(X_tr, y_tr, X_va, y_va):
        X_tr = np.asarray(X_tr, dtype=np.float32)
        if X_va is None:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_tr, y_tr, test_size=0.1, random_state=seed, stratify=y_tr,
            )
        else:
            X_es, y_es = np.asarray(X_va, dtype=np.float32), y_va
        clf = dl_models.build_and_train_tabnet(
            X_tr, y_tr, X_es, y_es, best_params, seed=seed,
            max_epochs=args.max_epochs, patience=args.patience,
        )

        def predict(X):
            return dl_models.tabnet_predict_proba(clf, np.asarray(X, dtype=np.float32))
        return predict, clf

    result = experiment_runner.run_bidirectional_experiment(
        model_name=model_name, model_family="TabNet", ml_or_dl="DL",
        tuning_method=f"Optuna TPE sampler + MedianPruner, {len(study.trials)} trials, "
                      f"{args.n_folds}-fold CV AUROC objective (dev-only), early stopping patience={args.patience}",
        pipeline_name="Universal", class_imbalance_strategy=(
            "original outcome distribution preserved (no SMOTE/oversampling, Section 26); "
            "pytorch-tabnet default weighting"
        ),
        dev_dataset=dev, ext_dataset=ext,
        X_dev_raw=X_dev_raw, y_dev=y_dev, dev_sample_ids=dev_ids,
        X_ext_raw=X_ext_raw, y_ext=y_ext, ext_sample_ids=ext_ids,
        preprocessor_factory=make_preprocessor, train_fn=train_fn, model_kind="tabnet",
        n_folds=args.n_folds, seed=seed, n_bootstrap=args.n_bootstrap,
        extra_manifest_notes=f"Optuna-tuned TabNet (pytorch-tabnet); study={storage}; "
                              f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
    )
    reporting.append_performance_rows([result["perf_row"]])
    log.info("%s dev=%s ext=%s external AUROC=%.4f (CI %.4f-%.4f)", model_name, dev, ext,
              result["perf_row"]["external_AUROC"], result["perf_row"]["external_AUROC_CI_low"],
              result["perf_row"]["external_AUROC_CI_high"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(config.DL_MODEL_FAMILIES),
                     choices=list(config.DL_MODEL_FAMILIES))
    ap.add_argument("--n-trials", type=int, default=config.N_OPTUNA_TRIALS_DL)
    ap.add_argument("--n-folds", type=int, default=config.N_CV_FOLDS)
    ap.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--smoke-test-sample-n", type=int, default=None,
                     help="DEV/CI ONLY: stratified-subsample each dataset to ~N rows before tuning.")
    ap.add_argument("--n-jobs", type=int, default=None,
                     help="CPU thread budget for torch (only matters when running on CPU). "
                          "Defaults to SLURM_CPUS_PER_TASK / os.cpu_count() via config.get_n_jobs().")
    ap.add_argument("--dev-datasets", nargs="+", default=list(config.DATASETS), choices=list(config.DATASETS),
                     help="Restrict to one direction's development dataset (e.g. --dev-datasets ICU runs only "
                          "dev=ICU->ext=MIMIC). Lets a family be split into per-direction parallel SLURM jobs "
                          "in addition to per-family jobs. Default: both directions in one job.")
    ap.add_argument("--trials-only", action="store_true",
                     help="Contribute Optuna trials to the (family, dev) study and exit -- skip the final "
                          "refit/external-eval/artifact-save step entirely. Lets several 'helper' worker "
                          "jobs point at the SAME study (same study_name+storage, i.e. same --families/"
                          "--dev-datasets as an already-running 'primary' job) to churn through the trial "
                          "budget in true parallel (Optuna's own distributed-optimization pattern: multiple "
                          "processes calling study.optimize() against shared SQLite storage), without any "
                          "of them racing to do the final refit -- only a job started WITHOUT this flag "
                          "does that, once ITS OWN view of n_remaining reaches 0. Safe to launch alongside "
                          "an already-running primary job for the same (family, dev); nothing is lost if a "
                          "helper is cancelled (only its current in-flight trial, same as always).")
    args = ap.parse_args()

    config.ensure_output_dirs()
    log.info("torch.cuda.is_available() = %s", torch.cuda.is_available())
    if not torch.cuda.is_available():
        # CPU-only mode (GPU temporarily unavailable on this cluster, 2026-08-12 --
        # slurm/04_tune_dl.sh now submits to cpu_partition/node2; see that file's
        # header comment for how to switch back to gpu_partition/node1 later).
        # Cap torch's intra-op thread pool to the SLURM CPU allocation so this
        # process doesn't oversubscribe the shared node (Section 3 policy).
        # NOTE (2026-08-12): torch.set_num_interop_threads() was tried here too,
        # but caused a reproducible SIGSEGV specifically in the TabNet path when
        # run under SLURM's cgroup-constrained CPU affinity on node1/node2 (MLP/
        # FT-Transformer were unaffected; the crash did not occur locally without
        # cgroup CPU pinning either). set_num_threads() alone is the standard,
        # well-supported API and does not reproduce the crash -- left at torch's
        # default interop pool instead of forcing a custom value.
        n_jobs = args.n_jobs or config.get_n_jobs()
        torch.set_num_threads(n_jobs)
        log.info("Running on CPU: torch.set_num_threads(%d)", n_jobs)

    datasets = {n: load_dataset(n) for n in config.DATASETS}
    if args.smoke_test_sample_n:
        for n in config.DATASETS:
            df = datasets[n]
            datasets[n] = df.groupby(config.TARGET_COL, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(args.smoke_test_sample_n // 2, 20)), random_state=args.seed)
            )
            log.warning("dataset=%s: SMOKE-TEST subsample active (N=%d) -- not a real tuning run",
                        n, len(datasets[n]))

    all_directions = [("ICU", "MIMIC"), ("MIMIC", "ICU")]
    directions = [(dev, ext) for dev, ext in all_directions if dev in args.dev_datasets]

    for family in args.families:
        for dev, ext in directions:
            try:
                if family == "TabNet":
                    run_tabnet(dev, ext, args, args.seed, datasets)
                else:
                    run_torch_family(family, dev, ext, args, args.seed, datasets)
            except Exception as e:
                log.error("FAILED: family=%s dev=%s ext=%s: %s", family, dev, ext, e)
                traceback.print_exc()
                reporting.append_failure_rows([{
                    "stage": "05_tune_dl", "model_name": family, "development_dataset": dev,
                    "error_type": type(e).__name__, "error_message": str(e)[:1000],
                    "resolution": "skipped this (family, direction) combination; others continued",
                }])
                continue

    reporting.write_model_input_summary_md()
    log.info("05_tune_dl.py complete.")


if __name__ == "__main__":
    main()
