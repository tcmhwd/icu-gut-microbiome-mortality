"""harmonize.py — target harmonization, 24h landmark cohort, predictor encoding.

All logic here operates on the RAW loaded frame (utils.io_utils.load_raw) and
only ever reads it; nothing here writes back to ICU.csv / mimic.csv.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target harmonization
# ---------------------------------------------------------------------------
def harmonize_target(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add 'mortality_harmonized' (0/1, float w/ NaN for unmapped/missing).

    Any raw value not in config.TARGET_ENCODING[dataset] is left as NaN and
    counted, never guessed.
    """
    mapping = config.TARGET_ENCODING[dataset]
    raw = df[config.TARGET_RAW_COL]
    harmonized = raw.map(mapping)
    unmapped_mask = raw.notna() & harmonized.isna()
    if unmapped_mask.any():
        bad_values = raw[unmapped_mask].value_counts().to_dict()
        log.warning("dataset=%s: %d rows with unmapped %s values: %s",
                    dataset, unmapped_mask.sum(), config.TARGET_RAW_COL, bad_values)
    df = df.copy()
    df["mortality_harmonized"] = harmonized.astype("float")
    return df


def outcome_harmonization_report(df_icu: pd.DataFrame, df_mimic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in (("ICU", df_icu), ("MIMIC", df_mimic)):
        raw = df[config.TARGET_RAW_COL]
        harmonized = df["mortality_harmonized"]
        n_total = len(df)
        n_missing_raw = raw.isna().sum()
        n_unmapped = (raw.notna() & harmonized.isna()).sum()
        n_survived = int((harmonized == 0).sum())
        n_died = int((harmonized == 1).sum())
        n_excluded = int(harmonized.isna().sum())
        rows.append({
            "dataset": name,
            "raw_column": config.TARGET_RAW_COL,
            "raw_encoding_observed": sorted(raw.dropna().unique().tolist()),
            "n_total_rows": n_total,
            "n_missing_raw_label": int(n_missing_raw),
            "n_unmapped_raw_label": int(n_unmapped),
            "n_excluded_no_valid_label": n_excluded,
            "n_survived_target_0": n_survived,
            "n_died_target_1": n_died,
            "harmonized_mortality_rate_percent": round(100 * n_died / (n_survived + n_died), 3)
            if (n_survived + n_died) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 24-hour landmark eligibility
# ---------------------------------------------------------------------------
def compute_landmark_eligibility(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add 'los_days', 'los_missing', 'landmark_eligible', 'landmark_basis'.

    landmark_basis values:
      'verified_los_included'   : LOS present and >= LANDMARK_DAYS
      'verified_los_excluded'   : LOS present and < LANDMARK_DAYS
      'presumed_eligible_missing_los_death' : ICU.csv-only; LOS missing AND
          the harmonized target (config.TARGET_COL, currently hospital_mortality)
          == 1 -> included per documented user decision
          (see config.LOS_MISSING_DEATH_ASSUMPTION)
      'excluded_missing_los'    : LOS missing and not covered by the
          presumed-eligible rule above (missing LOS + survivor, or missing
          target) -> excluded, eligibility unverifiable
      'excluded_missing_target' : outcome could not be harmonized -> excluded
          regardless of LOS
    """
    df = df.copy()
    los = pd.to_numeric(df[config.LOS_COL], errors="coerce")
    df["los_days"] = los
    df["los_missing"] = los.isna()

    mortality = df["mortality_harmonized"]
    basis = pd.Series(index=df.index, dtype=object)
    eligible = pd.Series(False, index=df.index)

    has_target = mortality.notna()
    has_los = los.notna()
    meets_cutoff = los >= config.LANDMARK_DAYS

    # verified LOS
    basis[has_target & has_los & meets_cutoff] = "verified_los_included"
    eligible[has_target & has_los & meets_cutoff] = True
    basis[has_target & has_los & ~meets_cutoff] = "verified_los_excluded"

    # missing LOS
    missing_los_mask = has_target & ~has_los
    if dataset == "ICU":
        presumed_mask = missing_los_mask & (mortality == 1)
        basis[presumed_mask] = "presumed_eligible_missing_los_death"
        eligible[presumed_mask] = True
        other_missing = missing_los_mask & ~presumed_mask
        basis[other_missing] = "excluded_missing_los"
    else:
        basis[missing_los_mask] = "excluded_missing_los"

    # no valid target at all
    basis[~has_target] = "excluded_missing_target"

    df["landmark_basis"] = basis
    df["landmark_eligible"] = eligible
    return df


def landmark_cohort_summary_row(df: pd.DataFrame, dataset: str) -> dict:
    los = df["los_days"]
    mort = df["mortality_harmonized"]
    n_total = len(df)

    short = (los.notna()) & (los < config.LANDMARK_DAYS)
    long = df["landmark_eligible"]

    row = {
        "dataset": dataset,
        "original_N": n_total,
        "ICU_LOS_<24h_N": int(short.sum()),
        "ICU_LOS_<24h_percent": round(100 * short.sum() / n_total, 2) if n_total else np.nan,
        "ICU_LOS_<24h_survivors": int((short & (mort == 0)).sum()),
        "ICU_LOS_<24h_deaths": int((short & (mort == 1)).sum()),
        "ICU_LOS_>=24h_N": int(long.sum()),
        "ICU_LOS_>=24h_survivors": int((long & (mort == 0)).sum()),
        "ICU_LOS_>=24h_deaths": int((long & (mort == 1)).sum()),
        "mortality_rate_after_landmark": round(
            100 * (long & (mort == 1)).sum() / long.sum(), 3
        ) if long.sum() else np.nan,
        "missing_LOS_N": int(df["los_missing"].sum()),
        "presumed_eligible_missing_los_death_N": int((df["landmark_basis"] == "presumed_eligible_missing_los_death").sum()),
        "excluded_missing_los_N": int((df["landmark_basis"] == "excluded_missing_los").sum()),
        "excluded_missing_target_N": int((df["landmark_basis"] == "excluded_missing_target").sum()),
    }
    return row


def los_distribution_bins(df: pd.DataFrame) -> pd.DataFrame:
    """LOS-hour distribution around the cutoff, from rows with verified LOS only
    (LOS presumed-eligible rows have no numeric LOS and cannot be binned)."""
    los_hours = df.loc[df["los_days"].notna(), "los_days"] * 24.0
    bins = [-np.inf, 6, 12, 18, 24, 48, np.inf]
    labels = ["<6h", "6-12h", "12-18h", "18-24h", "24-48h", ">48h"]
    cats = pd.cut(los_hours, bins=bins, labels=labels, right=False)
    counts = cats.value_counts().reindex(labels).fillna(0).astype(int)
    return counts.rename_axis("los_hour_bin").reset_index(name="N")


# ---------------------------------------------------------------------------
# Predictor encoding (raw NaNs preserved; imputation happens per-CV-fold later)
# ---------------------------------------------------------------------------
def _to_numeric_checked(series: pd.Series, name: str, dataset: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    bad_mask = series.notna() & coerced.isna()
    if bad_mask.any():
        bad_values = series[bad_mask].unique().tolist()[:10]
        log.warning("dataset=%s variable=%s: %d non-numeric values coerced to NaN (examples: %s)",
                    dataset, name, bad_mask.sum(), bad_values)
    return coerced


def _map_binary_checked(series: pd.Series, mapping: dict, name: str, dataset: str) -> pd.Series:
    mapped = series.map(mapping)
    bad_mask = series.notna() & mapped.isna()
    if bad_mask.any():
        bad_values = series[bad_mask].unique().tolist()[:10]
        log.warning("dataset=%s variable=%s: %d unmapped categorical values -> NaN (examples: %s)",
                    dataset, name, bad_mask.sum(), bad_values)
    return mapped.astype("float")


def encode_predictors(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return a NEW dataframe containing only encoded candidate predictors
    (numeric dtype throughout; NaNs preserved for the missing-data pipeline
    to handle downstream). Column names are prefixed to make provenance
    unambiguous: raw column 'Cr min' -> 'Cr_min'.
    """
    out = pd.DataFrame(index=df.index)

    for col in config.NUMERIC_PREDICTORS:
        clean_name = _clean_colname(col)
        out[clean_name] = _to_numeric_checked(df[col], col, dataset)

    for col in ["sex"]:
        out["sex_male"] = _map_binary_checked(df[col], config.SEX_MAP, col, dataset)

    for col in ["type"]:
        out["type_surgical"] = _map_binary_checked(df[col], config.TYPE_MAP, col, dataset)

    for col in config.COMORBIDITY_PREDICTORS + config.ORGAN_SUPPORT_PREDICTORS:
        clean_name = _clean_colname(col)
        out[clean_name] = _map_binary_checked(df[col], config.YESNO_MAP, col, dataset)

    return out


def _clean_colname(col: str) -> str:
    return (
        col.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def build_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in config.SEVERITY_SCORE_COMPARATORS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def build_model_ready_frame(df: pd.DataFrame, dataset: str, id_col: str) -> dict:
    """Assemble the full harmonized, landmark-filtered, model-ready bundle for
    one dataset. Returns a dict of DataFrames (not yet imputed/scaled):
      'full'        - every row post target-harmonization, with landmark flags
      'landmark'    - rows where landmark_eligible == True
      'predictors'  - encoded candidate predictors for the landmark cohort
      'severity'    - severity-score comparators for the landmark cohort
      'target'      - harmonized target for the landmark cohort
    """
    df = harmonize_target(df, dataset)
    df = compute_landmark_eligibility(df, dataset)

    df["sample_id"] = dataset + "_" + df[id_col].astype(str)
    df["op_value"] = df[config.OP_COL]

    landmark = df.loc[df["landmark_eligible"]].reset_index(drop=True)
    predictors = encode_predictors(landmark, dataset)
    severity = build_severity_scores(landmark)
    target = landmark["mortality_harmonized"].astype(int).rename(config.TARGET_COL)

    return {
        "full": df,
        "landmark": landmark,
        "predictors": predictors,
        "severity": severity,
        "target": target,
    }
