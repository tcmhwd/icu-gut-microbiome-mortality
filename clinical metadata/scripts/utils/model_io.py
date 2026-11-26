"""model_io.py — single-call save/load for a fully-trained (model +
preprocessing pipeline) bundle, so any tuned ML or DL model can be reloaded
and used for prediction on new raw data with ONE function call, without
manually reassembling separate files.

Extends Section 46 ("a trained model without its preprocessing pipeline is
not a valid saved model") per user request 2026-08-12: bundle the model,
preprocessor, and the exact raw feature list it was trained on into one
artifact, with a `LoadedModel.predict_proba(raw_df)` convenience API.

Used by utils/experiment_runner.py (save side, both 04_tune_ml.py and
05_tune_dl.py) and available standalone for later loading:

    from utils.model_io import load_bundle, list_available_models
    list_available_models()                    # browse what's saved
    m = load_bundle("clinical_ai/trained_models/ML/ICU/catboost")
    probs = m.predict_proba(some_raw_dataframe)  # raw candidate-predictor columns
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config


class LoadedModel:
    """Returned by load_bundle(). `.predict_proba(df)` takes a DataFrame
    containing (at least) the raw candidate-predictor columns the model was
    trained on (see `.metadata["raw_feature_columns"]`) and returns
    P(target=1), applying the saved preprocessing pipeline internally --
    the caller never needs to touch the preprocessor directly."""

    def __init__(self, model_kind: str, model_obj, preprocessor, metadata: dict):
        self.model_kind = model_kind  # 'sklearn' | 'torch' | 'tabnet'
        self.model_obj = model_obj
        self.preprocessor = preprocessor
        self.metadata = metadata

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        cols = self.metadata["raw_feature_columns"]
        X = self.preprocessor.transform(X_raw[cols])
        if self.model_kind == "sklearn":
            return self.model_obj.predict_proba(X.values)[:, 1]
        if self.model_kind == "torch":
            from . import dl_models
            return dl_models.predict_proba_torch(self.model_obj, X.values.astype(np.float32))
        if self.model_kind == "tabnet":
            from . import dl_models
            return dl_models.tabnet_predict_proba(self.model_obj, X.values.astype(np.float32))
        raise ValueError(f"unknown model_kind '{self.model_kind}'")

    def __repr__(self):
        m = self.metadata
        return (f"<LoadedModel {m.get('model_name')} ({m.get('ML_or_DL')}) "
                f"dev={m.get('development_dataset')} ext={m.get('external_test_dataset')} "
                f"kind={self.model_kind}>")


def save_bundle(artifact_dir: str, model_kind: str, model_obj, preprocessor, metadata: dict):
    """model_kind:
      'sklearn' -- any fitted sklearn-API estimator (LogisticRegression,
                   RandomForest, CatBoost/LightGBM/XGBoost classifiers, ...)
      'torch'   -- a fitted torch.nn.Module (MLP, FTTransformer) -- saved
                   WHOLE (architecture + weights), not just a state_dict, so
                   loading needs no separate hyperparameter reconstruction.
      'tabnet'  -- a fitted pytorch_tabnet.tab_model.TabNetClassifier --
                   uses TabNet's own native save/load (a zip of network +
                   config), which is more robust than raw-pickling it.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    if model_kind == "tabnet":
        model_obj.save_model(os.path.join(artifact_dir, "tabnet_model"))
        joblib.dump(preprocessor, os.path.join(artifact_dir, "preprocessor.joblib"))
        with open(os.path.join(artifact_dir, "metadata.json"), "w") as fh:
            json.dump({**metadata, "model_kind": "tabnet"}, fh, indent=2)
    else:
        bundle = {"model_kind": model_kind, "model": model_obj,
                  "preprocessor": preprocessor, "metadata": metadata}
        joblib.dump(bundle, os.path.join(artifact_dir, "bundle.joblib"))
        # metadata.json duplicated in plain-text form too, so list_available_models()
        # (and a curious human) can browse without unpickling the model itself.
        with open(os.path.join(artifact_dir, "metadata.json"), "w") as fh:
            json.dump({**metadata, "model_kind": model_kind}, fh, indent=2)


def load_bundle(artifact_dir: str) -> LoadedModel:
    bundle_path = os.path.join(artifact_dir, "bundle.joblib")
    if os.path.exists(bundle_path):
        b = joblib.load(bundle_path)
        return LoadedModel(b["model_kind"], b["model"], b["preprocessor"], b["metadata"])

    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            metadata = json.load(fh)
        if metadata.get("model_kind") == "tabnet":
            from pytorch_tabnet.tab_model import TabNetClassifier
            clf = TabNetClassifier()
            clf.load_model(os.path.join(artifact_dir, "tabnet_model.zip"))
            preprocessor = joblib.load(os.path.join(artifact_dir, "preprocessor.joblib"))
            return LoadedModel("tabnet", clf, preprocessor, metadata)

    raise FileNotFoundError(f"No saved model bundle found under {artifact_dir!r} "
                             f"(expected bundle.joblib or metadata.json+tabnet_model.zip)")


def list_available_models(base_dir: str = None) -> pd.DataFrame:
    """Scan clinical_ai/trained_models/ and list every saved bundle with its
    key metadata (without unpickling the model itself), so you can browse
    what's available before calling load_bundle() on a specific artifact_dir."""
    base_dir = base_dir or config.DIR_TRAINED_MODELS
    rows = []
    for ml_or_dl in ("ML", "DL"):
        d = os.path.join(base_dir, ml_or_dl)
        if not os.path.isdir(d):
            continue
        for dev in sorted(os.listdir(d)):
            dev_dir = os.path.join(d, dev)
            if not os.path.isdir(dev_dir):
                continue
            for model_name in sorted(os.listdir(dev_dir)):
                artifact_dir = os.path.join(dev_dir, model_name)
                meta_path = os.path.join(artifact_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as fh:
                    meta = json.load(fh)
                rows.append({"artifact_dir": artifact_dir, **meta})
    if not rows:
        return pd.DataFrame(columns=["artifact_dir", "model_name", "ML_or_DL",
                                      "development_dataset", "external_test_dataset", "model_kind"])
    return pd.DataFrame(rows)
