#!/usr/bin/env python3
"""08b_preprocess_reduced_by_site.py

Builds reduced-feature (29-column, see utils/reduced_features.py),
SITE-SPLIT model-ready frames for the "reduced clinical AI" family of
scripts (12_tune_ml_reduced.py / 13_tune_dl_reduced.py), additional to and
never modifying Stage 1's own 01_inspect_and_preprocess.py output.

Motivation (user question, 2026-08-14): the sibling microbiome project's 302
clinical-data patients are ALL from Gangnam Severance specifically, but
ICU.csv's own 'hospital' column mixes TWO sites: S=Sinchon/Severance main
campus (61,589 rows) and G=Gangnam Severance (23,557 rows). A reduced model
trained on the two sites pooled together (as plain 'ICU' dev, script 09) may
not transport well to Gangnam alone if the two sites differ systematically --
this mirrors an already-found cohort-confound bug in the sibling microbiome
project (all-cohort clustering trivially recovering cohort identity instead
of phenotype, semantic_microbiome/reports/no_leakage_audit.txt Known Issue #3).
So here we split ICU.csv by 'hospital' BEFORE landmark filtering and treat
Gangnam and Sinchon as two separate development/evaluation cohorts, in
addition to MIMIC -- 3 site cohorts total.

Site labels used throughout (dev_dataset / sample_id prefix):
  'Gangnam' -- ICU.csv rows with hospital == 'G'
  'Sinchon' -- ICU.csv rows with hospital == 'S'
  'MIMIC'   -- mimic.csv, unchanged

Target-encoding and 24h-landmark rules are applied per the SOURCE FILE
('ICU' encoding/landmark rule for both Gangnam and Sinchon, since both are
subsets of the same raw ICU.csv; 'MIMIC' rule for MIMIC) -- utils.harmonize
functions are called with dataset='ICU'/'MIMIC' for correctness, and the
site label is only substituted afterwards for naming/sample_id/output
purposes. See utils/harmonize.py for why this distinction matters
(TARGET_ENCODING and the ICU-only missing-LOS-at-death presumed-eligible
rule are both keyed by the raw source file, not by hospital site).

Writes: clinical_ai/preprocessing/model_ready_reduced_{Gangnam,Sinchon,MIMIC}.parquet
Safe to run directly on the login node (pandas-only, ~85k/~65k rows).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, harmonize, io_utils, reduced_features  # noqa: E402

log = io_utils.setup_logging("08b_preprocess_reduced_by_site")


def build_site_frame(raw_subset: pd.DataFrame, source_dataset: str, site_label: str) -> pd.DataFrame:
    """source_dataset: 'ICU' or 'MIMIC' -- controls TARGET_ENCODING / landmark
    rule (must match the file the rows actually came from).
    site_label: 'Gangnam' / 'Sinchon' / 'MIMIC' -- used for sample_id prefix
    and as the returned frame's identity for downstream scripts."""
    id_col = config.ID_COL_ICU if source_dataset == "ICU" else config.ID_COL_MIMIC

    df = harmonize.harmonize_target(raw_subset, source_dataset)
    df = harmonize.compute_landmark_eligibility(df, source_dataset)
    landmark = df.loc[df["landmark_eligible"]].reset_index(drop=True)

    encoded = harmonize.encode_predictors(landmark, source_dataset)
    reduced = reduced_features.build_reduced_predictors(encoded)
    reduced[config.TARGET_COL] = landmark["mortality_harmonized"].astype(int).values
    reduced["sample_id"] = site_label + "_" + landmark[id_col].astype(str)
    reduced["site"] = site_label
    return reduced


def main():
    config.ensure_output_dirs()

    log.info("Loading raw ICU.csv and mimic.csv (read-only)...")
    icu_raw = io_utils.load_raw("ICU")
    mimic_raw = io_utils.load_raw("MIMIC")

    site_counts = icu_raw["hospital"].value_counts()
    log.info("ICU.csv hospital site counts: %s", site_counts.to_dict())
    assert set(site_counts.index) == {"S", "G"}, f"unexpected hospital values: {site_counts.index.tolist()}"

    gangnam_raw = icu_raw.loc[icu_raw["hospital"] == "G"].reset_index(drop=True)
    sinchon_raw = icu_raw.loc[icu_raw["hospital"] == "S"].reset_index(drop=True)

    frames = {
        "Gangnam": build_site_frame(gangnam_raw, "ICU", "Gangnam"),
        "Sinchon": build_site_frame(sinchon_raw, "ICU", "Sinchon"),
        "MIMIC": build_site_frame(mimic_raw, "MIMIC", "MIMIC"),
    }

    summary_rows = []
    for site, fr in frames.items():
        path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_reduced_{site}.parquet")
        fr.to_parquet(path, index=False)
        n_events = int(fr[config.TARGET_COL].sum())
        log.info("Wrote %s (%d rows, %d cols, %d events [%.1f%%])",
                  path, len(fr), len(fr.columns), n_events, 100 * n_events / len(fr))
        summary_rows.append({"site": site, "N_landmark_eligible": len(fr), "N_events": n_events,
                              "mortality_percent": round(100 * n_events / len(fr), 3),
                              "N_reduced_features": len(reduced_features.REDUCED_ALL_COLS)})

    io_utils.save_csv(pd.DataFrame(summary_rows),
                       os.path.join(config.DIR_REPORTS, "reduced_site_split_summary.csv"))
    log.info("08b_preprocess_reduced_by_site.py complete.")


if __name__ == "__main__":
    main()
