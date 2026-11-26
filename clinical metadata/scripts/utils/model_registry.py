"""model_registry.py — ML algorithm family definitions shared by PyCaret
screening (03) and Optuna tuning (04): pycaret ids, family groups (for
diversity-aware candidate selection), sklearn/boosting estimator builders,
and Optuna search spaces.
"""
from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def _lr_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _lr_build(params, seed, n_jobs):
    return LogisticRegression(
        C=params["C"], class_weight=params["class_weight"],
        penalty="l2", solver="lbfgs", max_iter=3000, random_state=seed,
    )


def _rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _rf_build(params, seed, n_jobs):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
        class_weight=params["class_weight"], random_state=seed, n_jobs=n_jobs,
    )


def _et_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _et_build(params, seed, n_jobs):
    return ExtraTreesClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
        class_weight=params["class_weight"], random_state=seed, n_jobs=n_jobs,
    )


def _gbc_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _gbc_build(params, seed, n_jobs):
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        random_state=seed,
    )


def _ada_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2.0, log=True),
    }


def _ada_build(params, seed, n_jobs):
    return AdaBoostClassifier(
        n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
        random_state=seed,
    )


def _dt_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _dt_build(params, seed, n_jobs):
    return DecisionTreeClassifier(
        max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"], random_state=seed,
    )


def _knn_space(trial):
    return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 100)}


def _knn_build(params, seed, n_jobs):
    return KNeighborsClassifier(n_neighbors=params["n_neighbors"], n_jobs=n_jobs)


def _nb_space(trial):
    return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}


def _nb_build(params, seed, n_jobs):
    return GaussianNB(var_smoothing=params["var_smoothing"])


def _catboost_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
    }


def _catboost_build(params, seed, n_jobs):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=params["iterations"], depth=params["depth"],
        learning_rate=params["learning_rate"], l2_leaf_reg=params["l2_leaf_reg"],
        auto_class_weights=params["auto_class_weights"],
        random_seed=seed, thread_count=n_jobs, verbose=False, allow_writing_files=False,
    )


def _lightgbm_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
    }


def _lightgbm_build(params, seed, n_jobs):
    from lightgbm import LGBMClassifier
    p = dict(params)
    return LGBMClassifier(
        **p, random_state=seed, n_jobs=n_jobs, verbosity=-1,
    )


def _xgboost_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight_mode": trial.suggest_categorical("scale_pos_weight_mode", ["balanced", "none"]),
    }


def _xgboost_build(params, seed, n_jobs, y_train=None):
    from xgboost import XGBClassifier
    p = {k: v for k, v in params.items() if k != "scale_pos_weight_mode"}
    spw = 1.0
    if params.get("scale_pos_weight_mode") == "balanced" and y_train is not None:
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        spw = n_neg / n_pos
    return XGBClassifier(
        **p, scale_pos_weight=spw, random_state=seed, n_jobs=n_jobs,
        eval_metric="logloss", use_label_encoder=False, tree_method="hist",
    )


# family_group is used for diversity-aware selection (avoid 2-3 near-duplicates)
REGISTRY = {
    "lr": {"pycaret_id": "lr", "family_group": "linear", "supports_native_nan": False,
           "space": _lr_space, "build": _lr_build, "display": "Penalized Logistic Regression"},
    "rf": {"pycaret_id": "rf", "family_group": "tree_bagging", "supports_native_nan": False,
           "space": _rf_space, "build": _rf_build, "display": "Random Forest"},
    "et": {"pycaret_id": "et", "family_group": "tree_bagging", "supports_native_nan": False,
           "space": _et_space, "build": _et_build, "display": "Extra Trees"},
    "gbc": {"pycaret_id": "gbc", "family_group": "boosting", "supports_native_nan": False,
            "space": _gbc_space, "build": _gbc_build, "display": "Gradient Boosting (sklearn)"},
    "ada": {"pycaret_id": "ada", "family_group": "boosting", "supports_native_nan": False,
            "space": _ada_space, "build": _ada_build, "display": "AdaBoost"},
    "dt": {"pycaret_id": "dt", "family_group": "tree_single", "supports_native_nan": False,
           "space": _dt_space, "build": _dt_build, "display": "Decision Tree"},
    "knn": {"pycaret_id": "knn", "family_group": "instance", "supports_native_nan": False,
            "space": _knn_space, "build": _knn_build, "display": "k-Nearest Neighbors"},
    "nb": {"pycaret_id": "nb", "family_group": "bayes", "supports_native_nan": False,
           "space": _nb_space, "build": _nb_build, "display": "Gaussian Naive Bayes"},
    "catboost": {"pycaret_id": "catboost", "family_group": "boosting_native_nan", "supports_native_nan": True,
                 "space": _catboost_space, "build": _catboost_build, "display": "CatBoost"},
    "lightgbm": {"pycaret_id": "lightgbm", "family_group": "boosting_native_nan", "supports_native_nan": True,
                 "space": _lightgbm_space, "build": _lightgbm_build, "display": "LightGBM"},
    "xgboost": {"pycaret_id": "xgboost", "family_group": "boosting_native_nan", "supports_native_nan": True,
                "space": _xgboost_space, "build": _xgboost_build, "display": "XGBoost"},
}

# Excluded from the PyCaret compare_models() zoo up front for documented,
# non-scientific (pure runtime feasibility) reasons at N ~ 65k-76k rows x
# 5-fold CV on a shared CPU node: 'svm' (linear SVM w/ prob calibration) and
# 'rbfsvm' (kernel SVM, super-linear in N) do not scale to this N in
# reasonable wall time; 'gpc' (Gaussian Process) is O(N^3); 'mlp' here would
# duplicate the dedicated DL-MLP stage (Section 31); 'qda'/'lda' are unstable
# with many correlated/near-collinear binary predictors at this width.
SCREENING_INCLUDE = ["lr", "rf", "et", "gbc", "ada", "dt", "knn", "nb", "catboost", "lightgbm", "xgboost"]
SCREENING_EXCLUDED_WITH_REASON = {
    "svm": "linear SVM with probability calibration does not scale to N~65-76k in reasonable wall time",
    "rbfsvm": "kernel SVM training is super-linear in N; infeasible at this scale on a shared CPU node",
    "gpc": "Gaussian Process classifier is O(N^3); infeasible at N~65-76k",
    "mlp": "duplicates the dedicated tabular-MLP Optuna stage (05_tune_dl.py)",
    "qda": "unstable/ill-conditioned with many correlated near-binary predictors at this width",
    "lda": "redundant with penalized logistic regression as the linear reference model",
    "dummy": "not a meaningful clinical comparator",
    "ridge": "redundant with penalized logistic regression (lr) already in the screening set",
}
