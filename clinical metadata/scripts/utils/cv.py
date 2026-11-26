"""cv.py — cross-validation splitting.

Neither ICU.csv nor mimic.csv provides a patient-level identifier distinct
from the per-row admission id ('No' is unique per row in both files, verified
nunique(No) == nrows). Repeated ICU stays by the same patient therefore
cannot be detected from the source data, so plain stratified K-fold (on the
outcome) is used rather than patient-grouped CV. This is documented, not
assumed silently — see clinical_ai/reports/dataset_summary.csv notes and
scripts/utils/config.py.
"""
from __future__ import annotations

from sklearn.model_selection import StratifiedKFold

from . import config


def get_cv_splitter(n_splits: int = config.N_CV_FOLDS, seed: int = config.RANDOM_SEED):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_folds(X, y, n_splits: int = config.N_CV_FOLDS, seed: int = config.RANDOM_SEED):
    splitter = get_cv_splitter(n_splits, seed)
    return list(splitter.split(X, y))
