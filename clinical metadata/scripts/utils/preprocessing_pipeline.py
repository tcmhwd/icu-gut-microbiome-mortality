"""preprocessing_pipeline.py — the two predetermined missing-data pipelines.

Pipeline A (Universal): training-fold median imputation + missingness
indicators (+ optional scaling), for penalized LR, sklearn ML that can't
handle NaN, and all DL models.

Pipeline B (Native-missing): numeric NaN preserved as-is for tree-boosting
models (CatBoost / LightGBM / XGBoost) that handle it natively.

Both pipelines are fit *exclusively* on the training fold passed to `.fit()`.
Nothing here ever looks at validation/external data during `.fit()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

NUMERIC_CLEAN_COLS = [
    c.strip().replace(" ", "_").replace("/", "_").replace(".", "_")
    for c in config.NUMERIC_PREDICTORS
]
BINARY_CLEAN_COLS = ["sex_male", "type_surgical"] + [
    c.strip().replace(" ", "_").replace("/", "_").replace(".", "_")
    for c in config.COMORBIDITY_PREDICTORS + config.ORGAN_SUPPORT_PREDICTORS
]
ALL_CLEAN_COLS = NUMERIC_CLEAN_COLS + BINARY_CLEAN_COLS


@dataclass
class UniversalPreprocessor:
    """Pipeline A. scale=True adds z-scoring of (imputed) numeric columns,
    fit on the training fold only — use for penalized LR / MLP / FT-Transformer
    numeric inputs. scale=False is used for tree-based sklearn models
    (e.g. RandomForest) that don't need scaling.
    """
    scale: bool = False
    numeric_cols: list = field(default_factory=lambda: list(NUMERIC_CLEAN_COLS))
    binary_cols: list = field(default_factory=lambda: list(BINARY_CLEAN_COLS))

    def fit(self, X: pd.DataFrame):
        self.numeric_medians_ = {c: X[c].median() for c in self.numeric_cols}
        self.binary_modes_ = {c: (X[c].mode().iloc[0] if not X[c].mode().empty else 0.0)
                               for c in self.binary_cols}
        if self.scale:
            imputed_numeric = X[self.numeric_cols].fillna(self.numeric_medians_)
            self.scaler_ = StandardScaler().fit(imputed_numeric.values)
        else:
            self.scaler_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_missing = X[self.numeric_cols].isna().astype(int).add_suffix("_missing")
        numeric_imputed = X[self.numeric_cols].fillna(self.numeric_medians_)
        if self.scale:
            scaled = self.scaler_.transform(numeric_imputed.values)
            numeric_imputed = pd.DataFrame(scaled, index=X.index, columns=self.numeric_cols)

        binary_missing = X[self.binary_cols].isna().astype(int).add_suffix("_missing")
        binary_imputed = X[self.binary_cols].fillna(self.binary_modes_)

        # single concat (rather than iterative column assignment) avoids
        # pandas' "highly fragmented DataFrame" perf warning at this width
        out = pd.concat([numeric_imputed, numeric_missing, binary_imputed, binary_missing], axis=1)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def audit_rows(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                    experiment_id: str, dataset: str, model: str, fold) -> list[dict]:
        rows = []
        for c in self.numeric_cols:
            rows.append({
                "experiment_id": experiment_id, "development_dataset": dataset,
                "model": model, "fold": fold, "variable": c,
                "training_N": len(X_train), "training_missing_N": int(X_train[c].isna().sum()),
                "validation_N": len(X_val), "validation_missing_N": int(X_val[c].isna().sum()),
                "training_median_if_applicable": self.numeric_medians_[c],
                "number_imputed_training": int(X_train[c].isna().sum()),
                "number_imputed_validation": int(X_val[c].isna().sum()),
                "missing_indicator_used": True,
                "scaling_used": self.scale,
                "native_missing_used": False,
            })
        for c in self.binary_cols:
            rows.append({
                "experiment_id": experiment_id, "development_dataset": dataset,
                "model": model, "fold": fold, "variable": c,
                "training_N": len(X_train), "training_missing_N": int(X_train[c].isna().sum()),
                "validation_N": len(X_val), "validation_missing_N": int(X_val[c].isna().sum()),
                "training_median_if_applicable": np.nan,
                "number_imputed_training": int(X_train[c].isna().sum()),
                "number_imputed_validation": int(X_val[c].isna().sum()),
                "missing_indicator_used": True,
                "scaling_used": False,
                "native_missing_used": False,
            })
        return rows


@dataclass
class NativeMissingPreprocessor:
    """Pipeline B. No imputation, no indicators, no scaling: numeric NaN is
    passed through untouched for boosting libraries with native NaN support
    (CatBoost / LightGBM / XGBoost). Fitting only records column lists for
    the audit trail — nothing is learned from data.
    """
    numeric_cols: list = field(default_factory=lambda: list(NUMERIC_CLEAN_COLS))
    binary_cols: list = field(default_factory=lambda: list(BINARY_CLEAN_COLS))

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.numeric_cols + self.binary_cols].copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transform(X)

    def audit_rows(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                    experiment_id: str, dataset: str, model: str, fold) -> list[dict]:
        rows = []
        for c in self.numeric_cols + self.binary_cols:
            rows.append({
                "experiment_id": experiment_id, "development_dataset": dataset,
                "model": model, "fold": fold, "variable": c,
                "training_N": len(X_train), "training_missing_N": int(X_train[c].isna().sum()),
                "validation_N": len(X_val), "validation_missing_N": int(X_val[c].isna().sum()),
                "training_median_if_applicable": np.nan,
                "number_imputed_training": 0,
                "number_imputed_validation": 0,
                "missing_indicator_used": False,
                "scaling_used": False,
                "native_missing_used": True,
            })
        return rows
