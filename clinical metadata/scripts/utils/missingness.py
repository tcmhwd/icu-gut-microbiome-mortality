"""missingness.py — missingness reports A-E (Section 18)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config
from .reporting import _atomic_write_csv, _locked

CANDIDATE_RAW_COLS = (
    config.NUMERIC_PREDICTORS
    + ["sex", "type"]
    + config.COMORBIDITY_PREDICTORS
    + config.ORGAN_SUPPORT_PREDICTORS
)


def missingness_table(frames: dict) -> pd.DataFrame:
    """frames: {'ICU': df, 'MIMIC': df} using ORIGINAL raw column names."""
    rows = []
    for dataset, df in frames.items():
        n_total = len(df)
        for var in CANDIDATE_RAW_COLS:
            n_missing = int(df[var].isna().sum())
            rows.append({
                "dataset": dataset,
                "variable": var,
                "N_total": n_total,
                "N_missing": n_missing,
                "percent_missing": round(100 * n_missing / n_total, 3) if n_total else np.nan,
                "N_nonmissing": n_total - n_missing,
            })
    return pd.DataFrame(rows)


def cross_dataset_missingness(missingness_after_landmark: pd.DataFrame) -> pd.DataFrame:
    icu = missingness_after_landmark[missingness_after_landmark.dataset == "ICU"].set_index("variable")
    mimic = missingness_after_landmark[missingness_after_landmark.dataset == "MIMIC"].set_index("variable")
    rows = []
    for var in CANDIDATE_RAW_COLS:
        icu_row = icu.loc[var]
        mimic_row = mimic.loc[var]
        diff = abs(icu_row.percent_missing - mimic_row.percent_missing)
        rows.append({
            "variable": var,
            "ICU_N": int(icu_row.N_total),
            "ICU_missing_N": int(icu_row.N_missing),
            "ICU_missing_percent": icu_row.percent_missing,
            "MIMIC_N": int(mimic_row.N_total),
            "MIMIC_missing_N": int(mimic_row.N_missing),
            "MIMIC_missing_percent": mimic_row.percent_missing,
            "absolute_missingness_difference": round(diff, 3),
        })
    out = pd.DataFrame(rows).sort_values("absolute_missingness_difference", ascending=False)
    return out.reset_index(drop=True)


def missingness_group_classification(cross_df: pd.DataFrame,
                                       low=10.0, high=40.0, big_diff=20.0) -> pd.DataFrame:
    """Descriptive-only grouping (Section 20). No automatic drop decisions."""
    def classify(row):
        icu_m, mimic_m, diff = row.ICU_missing_percent, row.MIMIC_missing_percent, row.absolute_missingness_difference
        if icu_m >= 99.9 or mimic_m >= 99.9:
            return "almost_unavailable_in_one_dataset"
        if diff >= big_diff:
            return "large_ICU_vs_MIMIC_missingness_discrepancy"
        if max(icu_m, mimic_m) >= high:
            return "high_missingness_in_both_datasets"
        if max(icu_m, mimic_m) >= low:
            return "moderate_but_similar_missingness"
        return "low_missingness_in_both_datasets"

    out = cross_df.copy()
    out["missingness_group"] = out.apply(classify, axis=1)
    return out


def missing_indicator_prevalence(predictors: dict) -> pd.DataFrame:
    """predictors: {'ICU': encoded_predictor_df, 'MIMIC': encoded_predictor_df}
    (from harmonize.encode_predictors, on the landmark cohort). Indicators are
    reported for numeric predictors only (categorical predictors in this
    dataset are fully observed; see categorical_levels.csv)."""
    rows = []
    numeric_clean_names = [
        col.strip().replace(" ", "_").replace("/", "_").replace(".", "_")
        for col in config.NUMERIC_PREDICTORS
    ]
    for dataset, df in predictors.items():
        n_total = len(df)
        for col in numeric_clean_names:
            n_missing = int(df[col].isna().sum())
            rows.append({
                "dataset": dataset,
                "variable": col,
                "missing_indicator_N": n_missing,
                "missing_indicator_percent": round(100 * n_missing / n_total, 3) if n_total else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. Fold-level preprocessing audit — appended to incrementally by 04/05
# ---------------------------------------------------------------------------
FOLD_AUDIT_PATH = os.path.join(config.DIR_MISSINGNESS, "fold_preprocessing_audit.csv")

FOLD_AUDIT_COLUMNS = [
    "experiment_id", "development_dataset", "model", "fold", "variable",
    "training_N", "training_missing_N", "validation_N", "validation_missing_N",
    "training_median_if_applicable", "number_imputed_training", "number_imputed_validation",
    "missing_indicator_used", "scaling_used", "native_missing_used",
]


def append_fold_audit(rows: list[dict]):
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=FOLD_AUDIT_COLUMNS)
    with _locked(FOLD_AUDIT_PATH):
        if os.path.exists(FOLD_AUDIT_PATH):
            df_existing = pd.read_csv(FOLD_AUDIT_PATH)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        _atomic_write_csv(df_out, FOLD_AUDIT_PATH)
