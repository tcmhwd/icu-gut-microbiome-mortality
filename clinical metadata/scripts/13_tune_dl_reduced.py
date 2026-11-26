#!/usr/bin/env python3
"""13_tune_dl_reduced.py

Reduced-feature (29-column) DL training -- MLP, FT-Transformer, TabNet --
mirroring 05_tune_dl.py's exact methodology (Optuna TPE + MedianPruner,
5-fold CV objective, final refit with internal 90/10 early-stopping split,
utils.dl_models training loops), generalized to the 3-SITE design (Gangnam,
Sinchon, MIMIC; dev=one site, ext=pooled other two) used by
12_tune_ml_reduced.py. See 08b_preprocess_reduced_by_site.py for the
site-split rationale and utils/reduced_features.py for the feature set.

GPU note (user decision 2026-08-14): this cluster's GPU is occupied by other
work right now, so this always runs CPU-only regardless of which node/
partition it lands on (torch.set_num_threads(SLURM_CPUS_PER_TASK), same as
05_tune_dl.py's existing CPU-fallback path) -- node1/node2 are both used
purely for their free CPU capacity here, not GPU.

FT-Transformer parallel-helper pattern (user request 2026-08-14, mirroring
the Stage-1 job layout already seen in log/04_helper_FT_*.out): pass
--trials-only to run one or more "helper" processes that contribute Optuna
trials to the SAME shared SQLite study (same family/site => same
study_name+storage) without doing the final refit; a job started WITHOUT
--trials-only ("primary") does the refit once ITS OWN view of the trial
budget is exhausted. Safe to launch several helpers alongside one primary
for the same (family, site).

ml_or_dl="DL_reduced" keeps every output in its own namespace (never mixes
with Stage 1's DL/ tree or with ML_reduced/).

Run via slurm/13_tune_dl_reduced.sh (several site/family/role variants,
spread across node1 and node2 -- see that file).
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
from utils import config, cv, dl_models, io_utils, reduced_features, reporting  # noqa: E402
from utils.preprocessing_pipeline import UniversalPreprocessor  # noqa: E402

log = io_utils.setup_logging("13_tune_dl_reduced")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_OR_DL = "DL_reduced"
SITES = ["Gangnam", "Sinchon", "MIMIC"]
FEATURE_COLS = reduced_features.REDUCED_ALL_COLS
NUMERIC_COLS = reduced_features.REDUCED_NUMERIC_COLS
BINARY_COLS = reduced_features.REDUCED_BINARY_COLS

DL_BUILDERS = {
    "MLP": (dl_models.mlp_optuna_space, dl_models.build_mlp),
    "FTTransformer": (dl_models.fttransformer_optuna_space, dl_models.build_fttransformer),
}


def load_site(name: str) -> pd.DataFrame:
    path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_reduced_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run 08b_preprocess_reduced_by_site.py first")
    return pd.read_parquet(path)


def make_preprocessor():
    return UniversalPreprocessor(scale=True, numeric_cols=list(NUMERIC_COLS), binary_cols=list(BINARY_COLS))


def save_optuna_artifacts(study: optuna.Study, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)
    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as fh:
        json.dump({"best_value_AUROC": study.best_value, "best_params": study.best_params}, fh, indent=2)


def per_site_breakdown(pred_path: str, model_name: str, dev_site: str):
    pred_df = pd.read_csv(pred_path)
    pred_df["site"] = pred_df["patient_id"].str.extract(r"^(Gangnam|Sinchon|MIMIC)_")
    true_col = f"true_{config.TARGET_COL}"
    rows = []
    for site, g in pred_df.groupby("site"):
        auc = roc_auc_score(g[true_col], g["predicted_probability"]) if g[true_col].nunique() > 1 else float("nan")
        rows.append({"site": site, "N": len(g), "N_events": int(g[true_col].sum()), "AUROC": auc})
    breakdown = pd.DataFrame(rows)
    out_path = os.path.join(config.DIR_PERFORMANCE, f"reduced_dl_per_site_breakdown_{model_name}_dev-{dev_site}.csv")
    breakdown.to_csv(out_path, index=False)
    log.info("per-site external breakdown:\n%s", breakdown.to_string(index=False))


# ---------------------------------------------------------------------------
# MLP / FT-Transformer (shared torch training loop, dl_models.train_torch_classifier)
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


def run_torch_family(family: str, dev_site: str, ext_sites: list, args, seed: int, frames: dict):
    model_name = family
    ext_label = "+".join(ext_sites)
    log.info("=== %s | dev=%s -> ext=%s (REDUCED) ===", model_name, dev_site, ext_label)

    df_dev = frames[dev_site]
    df_ext = pd.concat([frames[s] for s in ext_sites], ignore_index=True)
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_DL, "reduced_" + dev_site, model_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", study_name=f"{dev_site}_{model_name}_reduced", storage=storage, load_if_exists=True,
        sampler=TPESampler(seed=seed), pruner=MedianPruner(n_warmup_steps=1),
    )
    n_remaining = max(args.n_trials - len(study.trials), 0)

    if n_remaining == 0 and reporting.performance_row_exists(model_name, dev_site, ext_label):
        log.info("%s dev=%s ext=%s already fully tuned (%d trials) and recorded; skipping.",
                  model_name, dev_site, ext_label, len(study.trials))
        return

    objective = build_torch_objective(family, X_dev_raw, y_dev, seed, args.n_folds, args.max_epochs, args.patience)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False, catch=(RuntimeError,))
    save_optuna_artifacts(study, study_dir)

    if args.trials_only:
        log.info("%s dev=%s: --trials-only, exiting after contributing trials (current best CV AUROC=%.4f). "
                  "The primary job for this (family, site) will do the final refit once its own view of "
                  "n_remaining reaches 0.", model_name, dev_site, study.best_value)
        return

    best_params = study.best_params
    if family == "MLP":
        best_params = dl_models.mlp_reconstruct_params(best_params)
    _, build_fn = DL_BUILDERS[family]
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    # --- OOF predictions for the audit trail (mirrors experiment_runner's internal-OOF file) ---
    folds = cv.make_folds(X_dev_raw, y_dev.values, args.n_folds, seed)
    oof_pred = np.full(len(y_dev), np.nan)
    for tr_idx, va_idx in folds:
        prep = make_preprocessor()
        X_tr = prep.fit_transform(X_dev_raw.iloc[tr_idx]).values.astype(np.float32)
        X_va = prep.transform(X_dev_raw.iloc[va_idx]).values.astype(np.float32)
        model = build_fn(X_tr.shape[1], best_params)
        model, _ = dl_models.train_torch_classifier(
            model, X_tr, y_dev.values[tr_idx], X_va, y_dev.values[va_idx], best_params,
            max_epochs=args.max_epochs, patience=args.patience, seed=seed,
        )
        oof_pred[va_idx] = dl_models.predict_proba_torch(model, X_va)
    fold_aucs = []
    for tr_idx, va_idx in folds:
        yv = y_dev.values[va_idx]
        if len(np.unique(yv)) > 1:
            fold_aucs.append(roc_auc_score(yv, oof_pred[va_idx]))
    cv_auroc_mean, cv_auroc_sd = float(np.nanmean(fold_aucs)), float(np.nanstd(fold_aucs))

    # --- final refit on full dev cohort (internal 90/10 early-stopping split only) ---
    prep_final = make_preprocessor()
    X_dev_final = prep_final.fit_transform(X_dev_raw).values.astype(np.float32)
    X_ext_final = prep_final.transform(X_ext_raw).values.astype(np.float32)
    X_tr, X_es, y_tr, y_es = train_test_split(X_dev_final, y_dev.values, test_size=0.1,
                                               random_state=seed, stratify=y_dev.values)
    model = build_fn(X_tr.shape[1], best_params)
    model, _ = dl_models.train_torch_classifier(model, X_tr, y_tr, X_es, y_es, best_params,
                                                 max_epochs=args.max_epochs, patience=args.patience, seed=seed)
    ext_prob = dl_models.predict_proba_torch(model, X_ext_final)

    _finish_and_save(model_name, family, "torch", model, prep_final, dev_site, ext_label,
                      X_dev_raw, y_dev, dev_ids, X_ext_raw, y_ext, ext_ids,
                      oof_pred, ext_prob, cv_auroc_mean, cv_auroc_sd, args, seed, storage,
                      f"Optuna-tuned {family} (PyTorch); {len(study.trials)} trials; CPU-only")


# ---------------------------------------------------------------------------
# TabNet (own training loop, pytorch-tabnet)
# ---------------------------------------------------------------------------
def build_tabnet_objective(X_dev_raw, y_dev, seed, n_folds, max_epochs, patience):
    folds = cv.make_folds(X_dev_raw, y_dev.values, n_folds, seed)

    def objective(trial: optuna.Trial):
        params = dl_models.tabnet_optuna_space(trial)
        aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            prep = make_preprocessor()
            X_tr = prep.fit_transform(X_dev_raw.iloc[tr_idx]).values.astype(np.float32)
            X_va = prep.transform(X_dev_raw.iloc[va_idx]).values.astype(np.float32)
            y_tr, y_va = y_dev.iloc[tr_idx], y_dev.iloc[va_idx]
            clf = dl_models.build_and_train_tabnet(X_tr, y_tr.values, X_va, y_va.values, params,
                                                     seed=seed, max_epochs=max_epochs, patience=patience)
            p_va = dl_models.tabnet_predict_proba(clf, X_va)
            if y_va.nunique() > 1:
                aucs.append(roc_auc_score(y_va, p_va))
            trial.report(float(np.mean(aucs)) if aucs else 0.0, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs)) if aucs else 0.0

    return objective


def run_tabnet(dev_site: str, ext_sites: list, args, seed: int, frames: dict):
    model_name = "TabNet"
    ext_label = "+".join(ext_sites)
    log.info("=== %s | dev=%s -> ext=%s (REDUCED) ===", model_name, dev_site, ext_label)

    df_dev = frames[dev_site]
    df_ext = pd.concat([frames[s] for s in ext_sites], ignore_index=True)
    X_dev_raw, y_dev, dev_ids = df_dev[FEATURE_COLS], df_dev[config.TARGET_COL], df_dev["sample_id"]
    X_ext_raw, y_ext, ext_ids = df_ext[FEATURE_COLS], df_ext[config.TARGET_COL], df_ext["sample_id"]

    study_dir = os.path.join(config.DIR_OPTUNA_DL, "reduced_" + dev_site, model_name)
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

    objective = build_tabnet_objective(X_dev_raw, y_dev, seed, args.n_folds, args.max_epochs, args.patience)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, n_jobs=1, show_progress_bar=False)
    save_optuna_artifacts(study, study_dir)

    if args.trials_only:
        log.info("%s dev=%s: --trials-only, exiting after contributing trials (current best CV AUROC=%.4f).",
                  model_name, dev_site, study.best_value)
        return

    best_params = dict(study.best_params)
    best_params["n_a"] = best_params["n_d"]
    log.info("%s best CV AUROC=%.4f params=%s", model_name, study.best_value, best_params)

    folds = cv.make_folds(X_dev_raw, y_dev.values, args.n_folds, seed)
    oof_pred = np.full(len(y_dev), np.nan)
    for tr_idx, va_idx in folds:
        prep = make_preprocessor()
        X_tr = prep.fit_transform(X_dev_raw.iloc[tr_idx]).values.astype(np.float32)
        X_va = prep.transform(X_dev_raw.iloc[va_idx]).values.astype(np.float32)
        clf = dl_models.build_and_train_tabnet(X_tr, y_dev.values[tr_idx], X_va, y_dev.values[va_idx],
                                                 best_params, seed=seed, max_epochs=args.max_epochs, patience=args.patience)
        oof_pred[va_idx] = dl_models.tabnet_predict_proba(clf, X_va)
    fold_aucs = []
    for tr_idx, va_idx in folds:
        yv = y_dev.values[va_idx]
        if len(np.unique(yv)) > 1:
            fold_aucs.append(roc_auc_score(yv, oof_pred[va_idx]))
    cv_auroc_mean, cv_auroc_sd = float(np.nanmean(fold_aucs)), float(np.nanstd(fold_aucs))

    prep_final = make_preprocessor()
    X_dev_final = prep_final.fit_transform(X_dev_raw).values.astype(np.float32)
    X_ext_final = prep_final.transform(X_ext_raw).values.astype(np.float32)
    X_tr, X_es, y_tr, y_es = train_test_split(X_dev_final, y_dev.values, test_size=0.1,
                                               random_state=seed, stratify=y_dev.values)
    clf = dl_models.build_and_train_tabnet(X_tr, y_tr, X_es, y_es, best_params, seed=seed,
                                             max_epochs=args.max_epochs, patience=args.patience)
    ext_prob = dl_models.tabnet_predict_proba(clf, X_ext_final)

    _finish_and_save(model_name, "TabNet", "tabnet", clf, prep_final, dev_site, ext_label,
                      X_dev_raw, y_dev, dev_ids, X_ext_raw, y_ext, ext_ids,
                      oof_pred, ext_prob, cv_auroc_mean, cv_auroc_sd, args, seed, storage,
                      f"Optuna-tuned TabNet (pytorch-tabnet); {len(study.trials)} trials; CPU-only")


# ---------------------------------------------------------------------------
# shared: metrics, prediction files, model bundle save, performance row
# ---------------------------------------------------------------------------
def _finish_and_save(model_name, family, model_kind, fitted_obj, preprocessor, dev_site, ext_label,
                      X_dev_raw, y_dev, dev_ids, X_ext_raw, y_ext, ext_ids,
                      oof_pred, ext_prob, cv_auroc_mean, cv_auroc_sd, args, seed, storage, notes):
    from utils import metrics, model_io

    ext_metrics = metrics.compute_metrics(y_ext.values, ext_prob)
    boot = metrics.bootstrap_ci(y_ext.values, ext_prob, metrics.DEFAULT_BOOTSTRAP_METRICS,
                                 n_boot=args.n_bootstrap, seed=seed)
    auroc_pt, auroc_lo, auroc_hi = boot["AUROC"]
    auprc_pt, auprc_lo, auprc_hi = boot["AUPRC"]

    true_col = f"true_{config.TARGET_COL}"
    experiment_id = f"{model_name}__dev-{dev_site}__ext-{ext_label}__REDUCED"
    pred_df = pd.DataFrame({"patient_id": ext_ids.values, true_col: y_ext.values,
                             "predicted_probability": ext_prob, "development_dataset": dev_site,
                             "external_dataset": ext_label, "model_name": model_name, "pipeline_name": "Universal"})
    pred_path = os.path.join(config.DIR_PREDICTIONS, f"{experiment_id}.csv")
    pred_df.to_csv(pred_path, index=False)

    oof_df = pd.DataFrame({"patient_id": dev_ids.values, true_col: y_dev.values,
                            "predicted_probability": oof_pred, "development_dataset": dev_site,
                            "external_dataset": f"{dev_site}_internal_OOF", "model_name": model_name,
                            "pipeline_name": "Universal"})
    oof_df.to_csv(os.path.join(config.DIR_PREDICTIONS, f"{experiment_id}__internal_OOF.csv"), index=False)

    artifact_dir = os.path.join(config.DIR_TRAINED_MODELS, ML_OR_DL, dev_site, model_name)
    model_io.save_bundle(artifact_dir, model_kind, fitted_obj, preprocessor, metadata={
        "model_name": model_name, "model_family": family, "ML_or_DL": ML_OR_DL,
        "development_dataset": dev_site, "external_test_dataset": ext_label,
        "raw_feature_columns": list(X_dev_raw.columns), "random_seed": seed,
        "target": config.TARGET_COL, "prediction_landmark": "ICU admission + 24h",
        "pipeline_name": "Universal", "external_AUROC": float(auroc_pt),
    })

    perf_row = {
        "model_name": model_name, "model_family": family, "ML_or_DL": ML_OR_DL,
        "pipeline_name": "Universal", "development_dataset": dev_site, "external_dataset": ext_label,
        "CV_AUROC_mean": cv_auroc_mean, "CV_AUROC_SD": cv_auroc_sd, "CV_AUPRC_mean": float("nan"),
        "external_AUROC": auroc_pt, "external_AUROC_CI_low": auroc_lo, "external_AUROC_CI_high": auroc_hi,
        "external_AUPRC": auprc_pt, "external_AUPRC_CI_low": auprc_lo, "external_AUPRC_CI_high": auprc_hi,
        "external_Brier": ext_metrics["Brier"], "external_logloss": ext_metrics["logloss"],
        "calibration_intercept": ext_metrics["calibration_intercept"], "calibration_slope": ext_metrics["calibration_slope"],
        "external_sensitivity": ext_metrics["sensitivity"], "external_specificity": ext_metrics["specificity"],
        "external_PPV": ext_metrics["PPV"], "external_NPV": ext_metrics["NPV"], "external_F1": ext_metrics["F1"],
        "external_N": ext_metrics["N"], "external_N_events": ext_metrics["N_events"],
        "number_of_predictors": X_dev_raw.shape[1], "number_of_parameters_if_applicable": float("nan"),
    }
    reporting.append_performance_rows([perf_row])
    log.info("%s dev=%s ext=%s external AUROC=%.4f (CI %.4f-%.4f)", model_name, dev_site, ext_label,
              auroc_pt, auroc_lo, auroc_hi)
    per_site_breakdown(pred_path, model_name, dev_site)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(DL_BUILDERS.keys()) + ["TabNet"],
                     choices=list(DL_BUILDERS.keys()) + ["TabNet"])
    ap.add_argument("--sites", nargs="+", default=SITES, choices=SITES)
    ap.add_argument("--n-trials", type=int, default=config.N_OPTUNA_TRIALS_DL)
    ap.add_argument("--n-folds", type=int, default=config.N_CV_FOLDS)
    ap.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--trials-only", action="store_true",
                     help="Contribute Optuna trials to the shared (family, site) study and exit -- lets several "
                          "'helper' jobs point at the SAME study to churn through the trial budget in parallel "
                          "(mirrors Stage 1's log/04_helper_FT_*.out pattern). Only a job WITHOUT this flag does "
                          "the final refit, once its own view of n_remaining reaches 0.")
    args = ap.parse_args()

    config.ensure_output_dirs()
    log.info("torch.cuda.is_available() = %s (expected False -- GPU intentionally not used, see module docstring)",
              torch.cuda.is_available())
    n_jobs = args.n_jobs or config.get_n_jobs()
    torch.set_num_threads(n_jobs)
    log.info("CPU-only: torch.set_num_threads(%d)", n_jobs)

    frames = {s: load_site(s) for s in SITES}

    for family in args.families:
        for dev_site in args.sites:
            ext_sites = [s for s in SITES if s != dev_site]
            try:
                if family == "TabNet":
                    run_tabnet(dev_site, ext_sites, args, args.seed, frames)
                else:
                    run_torch_family(family, dev_site, ext_sites, args, args.seed, frames)
            except Exception as e:
                log.error("FAILED: family=%s dev=%s: %s", family, dev_site, e)
                traceback.print_exc()
                reporting.append_failure_rows([{
                    "stage": "13_tune_dl_reduced", "model_name": family, "development_dataset": dev_site,
                    "error_type": type(e).__name__, "error_message": str(e)[:1000],
                    "resolution": "skipped this (family, site) combination; others continued",
                }])
                continue

    reporting.write_model_input_summary_md()
    log.info("13_tune_dl_reduced.py complete.")


if __name__ == "__main__":
    main()
