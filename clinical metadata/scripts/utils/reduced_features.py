"""reduced_features.py — the 29-feature reduced predictor set used by the
"reduced clinical AI" family of scripts (08b/12/13), built to match what is
actually available for the sibling microbiome project's 302 Gangnam ICU
patients (ICU_public/metadata/microbiome.clinical.csv), which has only a
SINGLE clinical snapshot per patient rather than the full ICU-stay min/max
vitals-labs time series ICU.csv/mimic.csv provide.

Single source of truth for both the ML (12_tune_ml_reduced.py) and DL
(13_tune_dl_reduced.py) reduced-model scripts, and for
09_train_reduced_clinical_model.py / 10_apply_reduced_model_to_microbiome_cohort.py
(kept independently there for now; not yet refactored to import this module).

Mapping notes (all explicit user decisions, 2026-08-14):
  MAP  <- mean(MAP_min, MAP_max) in training; proxy for microbiome side's MBP
  HR   <- mean(HR_min, HR_max) in training; proxy for microbiome side's PR
          (pulse rate), per explicit instruction "PR와 HR은 동일하게 취급해줘"
  bilirubin <- proxy for microbiome side's T.bil
  albumin   <- proxy for microbiome side's Alb
Excluded comorbidities (conceptually mismatched with the microbiome side's
nearest field, not used as silent proxies): CVD, ESRD, hematologic mal,
lung ds, PAOD, renal ds.
"""
from __future__ import annotations

import pandas as pd

# (reduced_name -> (raw_min_col_clean, raw_max_col_clean)) after
# harmonize._clean_colname() has already been applied by encode_predictors()
VITAL_LAB_MINMAX_PAIRS = {
    "BT": ("BT_min", "BT_max"),
    "MAP": ("MAP_min", "MAP_max"),
    "HR": ("HR_min", "HR_max"),
    "RR": ("RR_min", "RR_max"),
    "SBP": ("SBP_min", "SBP_max"),
    "Cr": ("Cr_min", "Cr_max"),
    "WBC": ("WBC_min", "WBC_max"),
    "PLT": ("PLT_min", "PLT_max"),
    "bilirubin": ("bilirubin_min", "bilirubin_max"),
    "albumin": ("albumin_min", "albumin_max"),
}

COMORBIDITIES_15 = [
    "cancer", "CHD", "connective_tissue_ds", "COPD", "dementia", "DM", "hemiplegia",
    "HTN", "leukemia", "liver_ds", "lymphoma", "MI", "moderate_severe_liver_ds",
    "ulcer", "transplantation",
]

REDUCED_NUMERIC_COLS = ["age"] + list(VITAL_LAB_MINMAX_PAIRS.keys()) + ["GCS_score"]
REDUCED_BINARY_COLS = ["sex_male", "ventilator"] + COMORBIDITIES_15
REDUCED_ALL_COLS = REDUCED_NUMERIC_COLS + REDUCED_BINARY_COLS  # 12 + 17 = 29


def build_reduced_predictors(encoded_predictors: pd.DataFrame) -> pd.DataFrame:
    """encoded_predictors: output of utils.harmonize.encode_predictors() (full
    NUMERIC_PREDICTORS/COMORBIDITY_PREDICTORS schema). Returns a NEW
    dataframe with just the 29 reduced columns, same index."""
    out = pd.DataFrame(index=encoded_predictors.index)
    out["age"] = encoded_predictors["age"]
    for name, (min_col, max_col) in VITAL_LAB_MINMAX_PAIRS.items():
        out[name] = encoded_predictors[[min_col, max_col]].mean(axis=1)
    out["GCS_score"] = encoded_predictors["GCS_score"]
    out["sex_male"] = encoded_predictors["sex_male"]
    out["ventilator"] = encoded_predictors["ventilator"]
    for c in COMORBIDITIES_15:
        out[c] = encoded_predictors[c]
    return out[REDUCED_ALL_COLS]
