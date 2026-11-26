"""
10_apply_reduced_model_to_microbiome_cohort.py — applies EVERY trained
reduced-feature clinical AI model (ML_reduced: lr/catboost/et; DL_reduced:
MLP/FTTransformer/TabNet; each x dev-site in {Gangnam,Sinchon,MIMIC}, see
08b_preprocess_reduced_by_site.py / 12_tune_ml_reduced.py /
13_tune_dl_reduced.py) to the 302 Gangnam ICU patients in the sibling
microbiome project, producing one per-patient clinical-risk probability per
model for later Phase X incremental-value analysis (independent development,
combined only at that explicit integration stage).

Since our application cohort is 100% Gangnam Severance, the Gangnam-dev
models are the closest domain match, but Sinchon-dev/MIMIC-dev models'
generalization TO Gangnam is already measured directly (their
"Gangnam" row in reports/reduced_{ml,dl}_per_site_breakdown_*.csv, written by
12_tune_ml_reduced.py/13_tune_dl_reduced.py) -- this script does not itself
pick a single "winning" model; it applies all of them and leaves selection
for the Phase X aggregation step, informed by those external-performance
numbers.

Source of clinical data: ICU_public/metadata/microbiome.clinical.csv (one
row per admission, joined to the microbiome discovery cohort via
KCDC_no <-> kcdc_no, verified 302/302 exact match). This file is NEVER the
same file the reduced models were TRAINED on (ICU.csv/mimic.csv) -- it is a
separate, independently-collected clinical snapshot for the microbiome
patients specifically, matched to the same 29-feature schema by name/unit
correspondence (see utils/reduced_features.py docstring for the full mapping
rationale, including the PR->HR and MBP->MAP proxy decisions -- both
explicit user instructions, 2026-08-14).

Uses ONLY `model_io.load_bundle(...).predict_proba()` -- no retraining, no
refitting, no threshold tuning against this cohort (would be leakage of the
integration stage into model development).

Output:
  clinical_ai/predictions/reduced_model_microbiome_cohort_predictions.csv
  columns: patient_id_clinical, kcdc_no, n_features_missing,
           clinical_AI_prob_{model}_{site}dev  (one per discovered model bundle)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import config, model_io  # noqa: E402

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
CLINICAL_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")
DISCOVERY_CSV = os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv")
OUT_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")

COMORBIDITY_RENAME = {
    "connective.tissue.ds": "connective_tissue_ds",
    "liver.ds": "liver_ds",
    "moderate.to.severe.liver.ds": "moderate_severe_liver_ds",
}
DIRECT_COMORBIDITIES = ["cancer", "CHD", "COPD", "dementia", "DM", "hemiplegia", "HTN",
                         "leukemia", "lymphoma", "MI", "ulcer", "transplantation"]


def find_reduced_model_dirs():
    """Discover every trained_models/{ML_reduced,DL_reduced}/{site}/{model}
    bundle that has completed (has a metadata.json), regardless of which of
    the still-running 12_tune_ml_reduced.py/13_tune_dl_reduced.py SLURM jobs
    produced it -- so this script can be safely re-run at any point while
    jobs are still in flight and simply pick up whatever is done so far."""
    found = []
    for ml_or_dl in ("ML_reduced", "DL_reduced"):
        base = os.path.join(config.DIR_TRAINED_MODELS, ml_or_dl)
        if not os.path.isdir(base):
            continue
        for site in sorted(os.listdir(base)):
            site_dir = os.path.join(base, site)
            if not os.path.isdir(site_dir):
                continue
            for model_name in sorted(os.listdir(site_dir)):
                artifact_dir = os.path.join(site_dir, model_name)
                if os.path.exists(os.path.join(artifact_dir, "metadata.json")):
                    found.append((ml_or_dl, site, model_name, artifact_dir))
    return found


def build_mapped_X(clin: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=clin.index)
    X["age"] = clin["age"]
    X["sex_male"] = clin["sex"].map({"M": 1, "F": 0})
    X["BT"] = clin["BT"]
    X["MAP"] = clin["MBP"]          # proxy: mean BP as MAP
    X["HR"] = clin["PR"]            # proxy: pulse rate as HR (explicit user instruction, 2026-08-14)
    X["RR"] = clin["RR"]
    X["SBP"] = clin["SBP"]
    X["GCS_score"] = clin["GCS"]
    X["Cr"] = clin["Cr"]
    X["WBC"] = clin["WBC"]
    X["PLT"] = clin["PLT"]
    X["bilirubin"] = clin["T.bil"]
    X["albumin"] = clin["Alb"]
    X["ventilator"] = clin["ventilator"]
    for c in DIRECT_COMORBIDITIES:
        X[c] = clin[c]
    for src, dst in COMORBIDITY_RENAME.items():
        X[dst] = clin[src]
    return X


def main():
    clin = pd.read_csv(CLINICAL_CSV)
    disc = pd.read_csv(DISCOVERY_CSV)

    disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
    clin = clin.merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner")
    print(f"[10] joined {len(clin)}/{len(disc)} discovery-cohort patients to microbiome.clinical.csv rows")
    assert len(clin) == len(disc), "expected exact 302/302 join -- investigate before proceeding"

    X = build_mapped_X(clin)
    out = pd.DataFrame({
        "patient_id_clinical": clin["patient_id_clinical"].values,
        "kcdc_no": clin["KCDC_no"].values,
    })
    out["n_features_missing"] = X.isna().sum(axis=1).values

    model_dirs = find_reduced_model_dirs()
    if not model_dirs:
        print("[10] No completed reduced-model bundles found yet -- nothing to apply. "
              "Re-run once 12_tune_ml_reduced.py/13_tune_dl_reduced.py jobs have produced at least one.")
        return
    print(f"[10] found {len(model_dirs)} completed model bundle(s): "
          + ", ".join(f"{s}/{m}" for _, s, m, _ in model_dirs))

    applied, failed = 0, []
    for ml_or_dl, site, model_name, artifact_dir in model_dirs:
        col = f"clinical_AI_prob_{model_name}_{site}dev"
        try:
            m = model_io.load_bundle(artifact_dir)
            cols = m.metadata["raw_feature_columns"]
            missing_cols = [c for c in cols if c not in X.columns]
            if missing_cols:
                raise ValueError(f"model expects columns not present in mapped X: {missing_cols}")
            probs = m.predict_proba(X[cols])
            out[col] = np.round(probs, 5)
            print(f"[10] {ml_or_dl}/{site}/{model_name}: applied to {len(out)} patients "
                  f"(mean={probs.mean():.3f}, min={probs.min():.3f}, max={probs.max():.3f})")
            applied += 1
        except Exception as e:
            print(f"[10] FAILED {ml_or_dl}/{site}/{model_name}: {e}")
            failed.append({"ml_or_dl": ml_or_dl, "site": site, "model_name": model_name, "error": str(e)})

    # mark the PRIMARY column, if 15_select_best_reduced_model.py has run
    selected_path = os.path.join(config.DIR_REPORTS, "selected_reduced_model.json")
    if os.path.exists(selected_path):
        with open(selected_path) as fh:
            sel = json.load(fh)
        primary_col = sel.get("selected_prediction_column")
        if primary_col in out.columns:
            out["clinical_AI_probability_PRIMARY"] = out[primary_col]
            print(f"[10] PRIMARY model = {primary_col} "
                  f"({'PROVISIONAL' if sel.get('selection_is_provisional') else 'FINAL'} selection, "
                  f"Gangnam-relevant AUROC={sel.get('selected_Gangnam_relevant_AUROC'):.4f})")
        else:
            print(f"[10] selected_reduced_model.json points at '{primary_col}', which is not yet "
                  f"among the applied columns -- PRIMARY not marked this run.")

    os.makedirs(config.DIR_PREDICTIONS, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[10] wrote {OUT_CSV} ({out.shape[0]} rows x {out.shape[1]} cols); "
          f"{applied} model(s) applied, {len(failed)} failed")
    if failed:
        fail_path = os.path.join(config.DIR_REPORTS, "reduced_model_apply_failures.json")
        with open(fail_path, "w") as fh:
            json.dump(failed, fh, indent=2)
        print(f"[10] wrote failure details to {fail_path}")
    print(f"[10] feature completeness (N missing of {X.shape[1]} reduced features): "
          f"{out['n_features_missing'].describe().to_dict()}")


if __name__ == "__main__":
    main()
