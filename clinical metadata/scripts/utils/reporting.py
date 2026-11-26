"""reporting.py — model-input manifest, markdown summaries, reproducibility audit."""
from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager

import pandas as pd

from . import config


@contextmanager
def _locked(path: str):
    """Exclusive advisory lock (fcntl.flock on a sibling .lock file) around a
    read-modify-write critical section on `path`. Needed because 04_tune_ml.py
    / 05_tune_dl.py may run as several concurrent SLURM jobs (e.g. one per DL
    family, spread across node1/node2 to use otherwise-idle CPUs) that all
    append to the same shared CSVs (model_performance_comparison.csv,
    model_input_manifest.csv, model_failures.csv, fold_preprocessing_audit.csv)
    -- without locking, two processes racing a read+append+write would silently
    drop one process's rows. fcntl advisory locks work across processes on any
    POSIX-compliant shared filesystem, including NFS, which is what node1/node2
    share here."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _atomic_write_csv(df: pd.DataFrame, path: str):
    """Write via temp file + os.replace (atomic on POSIX) so a concurrent
    reader (e.g. performance_row_exists(), or 06_evaluate_models.py) never
    observes a partially-written file, even without taking the lock itself."""
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


MANIFEST_PATH = os.path.join(config.DIR_REPORTS, "model_input_manifest.csv")

MANIFEST_COLUMNS = [
    "model_name", "model_family", "ML_or_DL", "development_dataset", "external_test_dataset",
    "input_feature_count_original", "input_feature_count_after_encoding",
    "numeric_missing_strategy", "categorical_missing_strategy", "missing_indicator", "scaling",
    "categorical_encoding", "native_NaN_handling", "class_imbalance_strategy",
    "hyperparameter_tuning_method", "CV_strategy", "target", "prediction_landmark", "notes",
]

MODEL_FAILURES_PATH = os.path.join(config.DIR_REPORTS, "model_failures.csv")
FAILURE_COLUMNS = ["stage", "model_name", "development_dataset", "error_type", "error_message", "resolution"]


def append_manifest_rows(rows: list[dict]):
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    with _locked(MANIFEST_PATH):
        if os.path.exists(MANIFEST_PATH):
            df_existing = pd.read_csv(MANIFEST_PATH)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        _atomic_write_csv(df_out, MANIFEST_PATH)


def append_failure_rows(rows: list[dict]):
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=FAILURE_COLUMNS)
    with _locked(MODEL_FAILURES_PATH):
        if os.path.exists(MODEL_FAILURES_PATH):
            df_existing = pd.read_csv(MODEL_FAILURES_PATH)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        _atomic_write_csv(df_out, MODEL_FAILURES_PATH)


PERFORMANCE_PATH = os.path.join(config.DIR_PERFORMANCE, "model_performance_comparison.csv")

PERFORMANCE_COLUMNS = [
    "model_name", "model_family", "ML_or_DL", "pipeline_name", "development_dataset", "external_dataset",
    "CV_AUROC_mean", "CV_AUROC_SD", "CV_AUPRC_mean",
    "external_AUROC", "external_AUROC_CI_low", "external_AUROC_CI_high",
    "external_AUPRC", "external_AUPRC_CI_low", "external_AUPRC_CI_high",
    "external_Brier", "external_logloss", "calibration_intercept", "calibration_slope",
    "external_sensitivity", "external_specificity", "external_PPV", "external_NPV", "external_F1",
    "external_N", "external_N_events",
    "number_of_predictors", "number_of_parameters_if_applicable",
]


def performance_row_exists(model_name: str, development_dataset: str, external_dataset: str) -> bool:
    """Used by 04_tune_ml.py/05_tune_dl.py to make resubmission idempotent:
    if a walltime-limited SLURM job is resubmitted after a (model, direction)
    combo already completed (Optuna study already at n_trials AND a
    performance row already recorded), skip re-running the final refit +
    external evaluation entirely, rather than silently appending a duplicate
    row to model_performance_comparison.csv / model_input_manifest.csv."""
    if not os.path.exists(PERFORMANCE_PATH):
        return False
    with _locked(PERFORMANCE_PATH):
        df = pd.read_csv(PERFORMANCE_PATH)
    if df.empty:
        return False
    mask = (
        (df.model_name == model_name)
        & (df.development_dataset == development_dataset)
        & (df.external_dataset == external_dataset)
    )
    return bool(mask.any())


def append_performance_rows(rows: list[dict]):
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)
    with _locked(PERFORMANCE_PATH):
        if os.path.exists(PERFORMANCE_PATH):
            df_existing = pd.read_csv(PERFORMANCE_PATH)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        _atomic_write_csv(df_out, PERFORMANCE_PATH)


def write_model_input_summary_md():
    path = os.path.join(config.DIR_REPORTS, "model_input_summary.md")
    if not os.path.exists(MANIFEST_PATH):
        text = "# Model Input Summary\n\nNo models have been registered in model_input_manifest.csv yet.\n"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    df = pd.read_csv(MANIFEST_PATH)
    lines = ["# Model Input Summary", "",
             "One row per model/pipeline combination that was actually fit. "
             "See `model_input_manifest.csv` for the full machine-readable table.", ""]
    for _, row in df.iterrows():
        lines.append(f"## {row['model_name']} ({row['ML_or_DL']}, dev={row['development_dataset']} "
                     f"-> ext={row['external_test_dataset']})")
        lines.append(f"- Family: {row['model_family']}")
        lines.append(f"- Features: {row['input_feature_count_original']} candidate -> "
                     f"{row['input_feature_count_after_encoding']} after encoding")
        lines.append(f"- Numeric missing strategy: {row['numeric_missing_strategy']}")
        lines.append(f"- Categorical missing strategy: {row['categorical_missing_strategy']}")
        lines.append(f"- Missing indicator: {row['missing_indicator']}; Scaling: {row['scaling']}; "
                     f"Native NaN handling: {row['native_NaN_handling']}")
        lines.append(f"- Class imbalance strategy: {row['class_imbalance_strategy']}")
        lines.append(f"- Tuning: {row['hyperparameter_tuning_method']}; CV: {row['CV_strategy']}")
        lines.append(f"- Target: {row['target']}; Landmark: {row['prediction_landmark']}")
        if pd.notna(row.get("notes")):
            lines.append(f"- Notes: {row['notes']}")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def write_reproducibility_audit(checks: list[tuple]):
    """checks: list of (item_number:int, description:str, passed:bool, detail:str)"""
    path = os.path.join(config.DIR_REPORTS, "reproducibility_audit.txt")
    lines = ["REPRODUCIBILITY AUDIT — ICU Clinical Mortality AI, Stage 1", "=" * 60, ""]
    n_fail = 0
    for num, desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            n_fail += 1
        lines.append(f"[{status}] {num}. {desc}")
        if detail:
            lines.append(f"       {detail}")
    lines.append("")
    lines.append(f"SUMMARY: {len(checks) - n_fail}/{len(checks)} checks passed.")
    if n_fail:
        lines.append(f"{n_fail} check(s) FAILED — see above. Fix before considering Stage 1 complete.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path, n_fail
