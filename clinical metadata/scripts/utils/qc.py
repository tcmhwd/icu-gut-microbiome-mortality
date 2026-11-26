"""qc.py — numeric QC, implausible-value flags, categorical-level reports.

Everything here is *report only*: no values are silently modified, truncated,
or winsorized. If evidence later justifies correction, that must be a
separate, documented step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Plausibility bounds: (lower, upper, unit/description). Values strictly
# outside [lower, upper] are flagged. Bounds are generous clinical sanity
# limits, not clinical reference ranges.
PLAUSIBILITY_RULES = {
    "BT min": (25.0, 43.0, "body temperature, deg C"),
    "BT max": (25.0, 43.0, "body temperature, deg C"),
    "MAP min": (0.0, 250.0, "mean arterial pressure, mmHg"),
    "MAP max": (0.0, 250.0, "mean arterial pressure, mmHg"),
    "HR min": (0.0, 300.0, "heart rate, bpm"),
    "HR max": (0.0, 300.0, "heart rate, bpm"),
    "RR min": (0.0, 80.0, "respiratory rate, /min"),
    "RR max": (0.0, 80.0, "respiratory rate, /min"),
    "SBP_min": (0.0, 300.0, "systolic BP, mmHg"),
    "SBP_max": (0.0, 300.0, "systolic BP, mmHg"),
    "PH min": (6.5, 8.0, "arterial pH"),
    "PH max": (6.5, 8.0, "arterial pH"),
    "Na min": (100.0, 200.0, "sodium, mEq/L"),
    "Na max": (100.0, 200.0, "sodium, mEq/L"),
    "K min": (1.0, 10.0, "potassium, mEq/L"),
    "K max": (1.0, 10.0, "potassium, mEq/L"),
    "Cr min": (0.0, 30.0, "creatinine, mg/dL"),
    "Cr max": (0.0, 30.0, "creatinine, mg/dL"),
    "Hct min": (0.0, 70.0, "hematocrit, %"),
    "Hct max": (0.0, 70.0, "hematocrit, %"),
    "WBC min": (0.0, 200.0, "WBC, x10^3/uL"),
    "WBC max": (0.0, 200.0, "WBC, x10^3/uL"),
    "PLT min": (0.0, 2000.0, "platelets, x10^3/uL"),
    "PLT max": (0.0, 2000.0, "platelets, x10^3/uL"),
    "bilirubin min": (0.0, 50.0, "total bilirubin, mg/dL"),
    "bilirubin max": (0.0, 50.0, "total bilirubin, mg/dL"),
    "BUN min": (0.0, 200.0, "BUN, mg/dL"),
    "BUN max": (0.0, 200.0, "BUN, mg/dL"),
    "Glucose min": (0.0, 1500.0, "glucose, mg/dL"),
    "Glucose max": (0.0, 1500.0, "glucose, mg/dL"),
    "pCO2 min": (0.0, 150.0, "pCO2, mmHg"),
    "pCO2 max": (0.0, 150.0, "pCO2, mmHg"),
    "albumin min": (0.0, 6.0, "albumin, g/dL"),
    "albumin max": (0.0, 6.0, "albumin, g/dL"),
    "HCO3 min": (0.0, 60.0, "bicarbonate, mEq/L"),
    "HCO3 max": (0.0, 60.0, "bicarbonate, mEq/L"),
    "GCS.score": (3.0, 15.0, "Glasgow Coma Scale"),
    "Urine_output": (0.0, 20000.0, "urine output, mL/24h"),
    "age": (18.0, 120.0, "age, years"),
}


def numeric_qc_summary(raw_frames: dict) -> pd.DataFrame:
    """raw_frames: {'ICU': landmark_df, 'MIMIC': landmark_df} with ORIGINAL
    (raw, un-prefixed) column names still present."""
    rows = []
    for dataset, df in raw_frames.items():
        for col in config.NUMERIC_PREDICTORS:
            s = pd.to_numeric(df[col], errors="coerce")
            n_total = len(s)
            n_missing = int(s.isna().sum())
            desc = s.describe()
            rows.append({
                "dataset": dataset,
                "variable": col,
                "N": n_total,
                "missing_N": n_missing,
                "missing_percent": round(100 * n_missing / n_total, 2) if n_total else np.nan,
                "mean": desc.get("mean", np.nan),
                "SD": desc.get("std", np.nan),
                "median": s.median(),
                "Q1": s.quantile(0.25),
                "Q3": s.quantile(0.75),
                "min": desc.get("min", np.nan),
                "max": desc.get("max", np.nan),
                "n_unique": int(s.nunique(dropna=True)),
            })
    return pd.DataFrame(rows)


def outlier_qc(raw_frames: dict) -> pd.DataFrame:
    rows = []
    for dataset, df in raw_frames.items():
        for col, (lo, hi, desc) in PLAUSIBILITY_RULES.items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            n_total = int(s.notna().sum())
            flagged = s.dropna()
            flagged = flagged[(flagged < lo) | (flagged > hi)]
            n_flagged = len(flagged)
            rows.append({
                "dataset": dataset,
                "variable": col,
                "QC_rule": f"expected [{lo}, {hi}] ({desc})",
                "N_flagged": n_flagged,
                "percent_flagged": round(100 * n_flagged / n_total, 3) if n_total else np.nan,
                "example_values": sorted(flagged.unique().tolist())[:10],
            })
    return pd.DataFrame(rows)


def categorical_levels(raw_frames: dict) -> pd.DataFrame:
    cat_cols = ["sex", "type"] + config.COMORBIDITY_PREDICTORS + config.ORGAN_SUPPORT_PREDICTORS
    rows = []
    for dataset, df in raw_frames.items():
        for col in cat_cols:
            s = df[col].fillna("<blank/NaN>")
            counts = s.value_counts(dropna=False)
            n_total = len(s)
            for category, n in counts.items():
                rows.append({
                    "dataset": dataset,
                    "variable": col,
                    "category": category,
                    "N": int(n),
                    "percent": round(100 * n / n_total, 2) if n_total else np.nan,
                })
    return pd.DataFrame(rows)
