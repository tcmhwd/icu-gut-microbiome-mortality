#!/usr/bin/env python3
"""01_inspect_and_preprocess.py

Dataset inspection, outcome harmonization, 24h landmark cohort construction,
leakage audit, numeric/categorical QC, and model-ready frame export.

Reads:  ICU.csv, mimic.csv (read-only, never modified)
Writes: clinical_ai/preprocessing/*, clinical_ai/reports/dataset_summary.csv,
        clinical_ai/reports/software_versions.txt,
        clinical_ai/preprocessing/model_ready_{ICU,MIMIC}.parquet
        (feature matrices consumed by every later script)

Safe to run directly on the login node (fast pandas-only operations on
~85k/~65k rows; no heavy compute). See slurm/01_preprocess.sh for the SLURM
wrapper (bundles 01 + 02).
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, harmonize, io_utils, qc, reporting  # noqa: E402

log = io_utils.setup_logging("01_inspect_and_preprocess")


def write_software_versions():
    lines = [f"Generated: {pd.Timestamp.now()}", f"Python: {platform.python_version()}",
             f"Platform: {platform.platform()}", ""]
    pkgs = ["pandas", "numpy", "sklearn", "pycaret", "optuna", "torch", "catboost",
            "lightgbm", "xgboost", "pytorch_tabnet", "shap", "imblearn"]
    for pkg in pkgs:
        try:
            m = importlib.import_module(pkg)
            lines.append(f"{pkg}: {getattr(m, '__version__', 'unknown')}")
        except Exception as e:
            lines.append(f"{pkg}: NOT INSTALLED ({e})")
    try:
        import torch
        lines.append(f"torch.cuda available: {torch.cuda.is_available()}")
        lines.append(f"torch CUDA build version: {torch.version.cuda}")
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        lines.append(f"nvidia-smi -L: {out.stdout.strip() or out.stderr.strip()}")
    except Exception as e:
        lines.append(f"nvidia-smi -L: unavailable ({e})")
    lines.append(f"conda env expected: {config.CONDA_ENV}")
    lines.append(f"RANDOM_SEED: {config.RANDOM_SEED}")
    path = io_utils.save_text("\n".join(lines) + "\n",
                               os.path.join(config.DIR_REPORTS, "software_versions.txt"))
    log.info("Wrote %s", path)


def build_leakage_manifest() -> pd.DataFrame:
    rows = []

    def add(variable, dataset, excluded, reason, timing_concern, notes=""):
        rows.append({"variable": variable, "dataset": dataset, "excluded": excluded,
                     "reason": reason, "timing_concern": timing_concern, "notes": notes})

    add(config.TARGET_RAW_COL, "BOTH", True,
        "this IS the target/outcome variable (Section 4, corrected 2026-08-12: hospital_mortality, "
        "not ICU_mortality as originally specified) -- excluded from the predictor matrix by "
        "construction, not a leakage finding",
        "n/a -- target itself", "see config.TARGET_COL / config.TARGET_ENCODING")
    for v in config.OUTCOME_LEAKAGE_COLS:
        add(v, "BOTH", True, "outcome-derived / alternative mortality label",
            "reflects information at or after discharge/death, not at ICU+24h", "")
    for v in config.CULTURE_LEAKAGE_COLS:
        add(v, "BOTH", True, "culture/infection result may post-date the 24h landmark",
            "no confirmed first-24h-only timing for all fields in this group", "")
    for v in config.ANTIBIOTIC_LEAKAGE_COLS:
        add(v, "BOTH", True, "antibiotic exposure timing not confirmed to be within first 24h",
            "possible full-stay exposure; conservatively excluded", "")
    for v in config.EXPLICITLY_EXCLUDED_PREDICTORS:
        add(v, "BOTH", True, "explicitly excluded per study design (given)",
            "n/a or future/pre-ICU information", "user-specified exclusion, Section 8")
    add(config.LOS_COL, "BOTH", True, "landmark eligibility variable, never a predictor (future information)",
        "defines the prediction landmark itself -> using it as a predictor is leakage by construction",
        "see config.py for full LOS semantics discussion and the ICU.csv missing-LOS-at-death handling")
    for v in config.SEVERITY_SCORE_COMPARATORS:
        add(v, "BOTH", True, "reserved as a conventional-score comparator, not an AI predictor",
            "n/a", "Section 9; compared separately in 06_evaluate_models.py")
    for v in config.SEVERITY_SCORES_ICU_ONLY_EXCLUDED:
        add(v, "ICU", True, "100% missing in mimic.csv; not part of the requested comparator set",
            "n/a", "cannot support bidirectional comparison")
    add(config.OP_COL, "BOTH", True, "retained in harmonized dataset but never used as an AI predictor (given)",
        "n/a", "Section 10")
    for v in config.DATASET_IDENTIFYING_EXCLUDED:
        reason = {
            "hospital": "raw site/source identifier",
            "year": "ICU.csv years 2006-2020 vs mimic.csv de-identified years 2110-2213 are "
                    "entirely non-overlapping -> would let a model learn dataset identity instead of physiology",
            "icu_type": "vocabulary is completely disjoint/institution-specific between datasets and not "
                        "part of the Section 11 candidate predictor list",
        }[v]
        add(v, "BOTH", True, reason, "harms cross-dataset transportability if included", "Section 7")
    add("hospital", "BOTH", True, "raw identifier", "n/a", "duplicate entry retained for completeness")
    add("No", "BOTH", True, "raw row/admission identifier", "n/a", "used only to build sample_id for predictions")
    for v in config.MIMIC_PROVENANCE_COLS:
        add(v, "MIMIC", True, "MIMIC-IV extraction provenance columns", "n/a", "identifiers, not clinical data")

    for v in config.ALL_CANDIDATE_PREDICTORS:
        add(v, "BOTH", False, "candidate AI predictor (Section 11)", "n/a", "retained")

    df = pd.DataFrame(rows).drop_duplicates(subset=["variable", "dataset"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    config.ensure_output_dirs()
    write_software_versions()

    log.info("Loading raw ICU.csv and mimic.csv (read-only)...")
    raw = {"ICU": io_utils.load_raw("ICU"), "MIMIC": io_utils.load_raw("MIMIC")}
    id_cols = {"ICU": config.ID_COL_ICU, "MIMIC": config.ID_COL_MIMIC}

    log.info("Harmonizing outcome and computing 24h landmark eligibility...")
    bundles = {}
    for name in config.DATASETS:
        bundles[name] = harmonize.build_model_ready_frame(raw[name], name, id_cols[name])

    # --- Section 4: outcome harmonization report ---
    harmon_report = harmonize.outcome_harmonization_report(bundles["ICU"]["full"], bundles["MIMIC"]["full"])
    io_utils.save_csv(harmon_report, os.path.join(config.DIR_PREPROCESSING, "outcome_harmonization_report.csv"))
    log.info("Outcome harmonization:\n%s", harmon_report.to_string(index=False))

    # --- Section 6: landmark cohort summary + LOS distribution ---
    summary_rows = [harmonize.landmark_cohort_summary_row(bundles[n]["full"], n) for n in config.DATASETS]
    landmark_summary = pd.DataFrame(summary_rows)
    io_utils.save_csv(landmark_summary, os.path.join(config.DIR_PREPROCESSING, "landmark_cohort_summary.csv"))
    log.info("Landmark cohort summary:\n%s", landmark_summary.to_string(index=False))

    los_bins = []
    for n in config.DATASETS:
        b = harmonize.los_distribution_bins(bundles[n]["full"])
        b.insert(0, "dataset", n)
        los_bins.append(b)
    io_utils.save_csv(pd.concat(los_bins, ignore_index=True),
                       os.path.join(config.DIR_PREPROCESSING, "los_distribution_bins.csv"))

    # --- Section 12: leakage exclusion manifest ---
    leakage_manifest = build_leakage_manifest()
    io_utils.save_csv(leakage_manifest, os.path.join(config.DIR_PREPROCESSING, "leakage_exclusion_manifest.csv"))
    log.info("Leakage manifest: %d variables documented (%d excluded, %d retained as predictors)",
              len(leakage_manifest), int(leakage_manifest.excluded.sum()), int((~leakage_manifest.excluded).sum()))

    # --- Section 21/22: numeric QC, outlier QC, categorical levels (on landmark cohorts) ---
    landmark_raw = {n: bundles[n]["landmark"] for n in config.DATASETS}
    numeric_qc = qc.numeric_qc_summary(landmark_raw)
    io_utils.save_csv(numeric_qc, os.path.join(config.DIR_PREPROCESSING, "numeric_qc_summary.csv"))

    outlier_qc = qc.outlier_qc(landmark_raw)
    io_utils.save_csv(outlier_qc, os.path.join(config.DIR_PREPROCESSING, "outlier_qc.csv"))
    n_flag_total = outlier_qc.N_flagged.sum()
    if n_flag_total:
        log.warning("Outlier QC flagged %d implausible values total across variables (see outlier_qc.csv). "
                    "NOT modified/truncated -- reported only.", n_flag_total)

    categorical_levels = qc.categorical_levels(landmark_raw)
    io_utils.save_csv(categorical_levels, os.path.join(config.DIR_PREPROCESSING, "categorical_levels.csv"))

    # --- Section 13: patient-level leakage note ---
    patient_note = (
        "No patient-level identifier distinct from the per-row admission id ('No') exists in "
        "either source file. Verified: nunique(No) == nrows in both ICU.csv and mimic.csv, so "
        "repeated ICU stays by the same patient cannot be detected from the available data. "
        "Each row is therefore treated as one independent ICU stay/admission; CV uses plain "
        "stratified K-fold (utils/cv.py), not patient-grouped CV."
    )
    log.info(patient_note)

    # --- Section 23: dataset summary ---
    ds_rows = []
    for n in config.DATASETS:
        full = bundles[n]["full"]
        landmark = bundles[n]["landmark"]
        ds_rows.append({
            "dataset": n,
            "original_rows": len(full),
            "unique_patients": "unknown (no patient-level identifier in source data)",
            "rows_after_valid_target": int(full["mortality_harmonized"].notna().sum()),
            "rows_after_24h_landmark": len(landmark),
            "survivors": int((landmark["mortality_harmonized"] == 0).sum()),
            # NOTE: column named 'deaths'/'mortality_percent' (not 'ICU_deaths'/
            # 'ICU_mortality_percent' as in the original brief's Section 23) because
            # the target was corrected 2026-08-12 to hospital_mortality; see config.py.
            "deaths": int((landmark["mortality_harmonized"] == 1).sum()),
            "mortality_percent": round(100 * (landmark["mortality_harmonized"] == 1).mean(), 3),
            "number_candidate_predictors": len(config.ALL_CANDIDATE_PREDICTORS),
            "patient_level_note": patient_note,
        })
    io_utils.save_csv(pd.DataFrame(ds_rows), os.path.join(config.DIR_REPORTS, "dataset_summary.csv"))

    # --- model-ready frames for all downstream scripts ---
    for n in config.DATASETS:
        out = bundles[n]["predictors"].copy()
        out[config.TARGET_COL] = bundles[n]["target"].values
        for c in config.SEVERITY_SCORE_COMPARATORS:
            out[f"severity__{c}"] = bundles[n]["severity"][c].values
        out["sample_id"] = bundles[n]["landmark"]["sample_id"].values
        out["op_value"] = bundles[n]["landmark"]["op_value"].values
        out["landmark_basis"] = bundles[n]["landmark"]["landmark_basis"].values
        path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_{n}.parquet")
        out.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows, %d columns)", path, len(out), len(out.columns))

    # placeholder manifest/summary so paths exist even before 04/05/06 run
    reporting.append_manifest_rows([])
    if not os.path.exists(reporting.MANIFEST_PATH):
        pd.DataFrame(columns=reporting.MANIFEST_COLUMNS).to_csv(reporting.MANIFEST_PATH, index=False)
    reporting.write_model_input_summary_md()

    log.info("01_inspect_and_preprocess.py complete.")


if __name__ == "__main__":
    main()
