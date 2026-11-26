#!/usr/bin/env python3
"""02_missingness_analysis.py

Missingness tracking, reports A-D (Section 18):
  A. raw candidate-feature missingness (before 24h landmark exclusion)
  B. landmark-cohort missingness (after 24h landmark exclusion)
  C. cross-dataset missingness comparison (+ descriptive grouping, Section 20)
  D. missing-indicator prevalence (on the landmark/model-ready cohort)

(Report E, the fold-level preprocessing audit, is produced incrementally by
04_tune_ml.py / 05_tune_dl.py as models are actually fit.)

Requires 01_inspect_and_preprocess.py to have been run first (reads its
model_ready_*.parquet outputs plus the raw CSVs). Safe to run on the login
node (fast pandas-only operations).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, harmonize, io_utils, missingness  # noqa: E402

log = io_utils.setup_logging("02_missingness_analysis")


def main():
    config.ensure_output_dirs()

    log.info("Loading raw data and recomputing harmonization/landmark flags...")
    raw = {"ICU": io_utils.load_raw("ICU"), "MIMIC": io_utils.load_raw("MIMIC")}
    id_cols = {"ICU": config.ID_COL_ICU, "MIMIC": config.ID_COL_MIMIC}
    bundles = {n: harmonize.build_model_ready_frame(raw[n], n, id_cols[n]) for n in config.DATASETS}

    # A. raw candidate-feature missingness (full frame, pre-landmark; still
    # restricted to rows with a valid harmonized target since undefined-target
    # rows never enter modeling regardless of LOS)
    pre_landmark_frames = {
        n: bundles[n]["full"].loc[bundles[n]["full"]["mortality_harmonized"].notna()]
        for n in config.DATASETS
    }
    missingness_raw = missingness.missingness_table(pre_landmark_frames)
    io_utils.save_csv(missingness_raw, os.path.join(config.DIR_MISSINGNESS, "missingness_raw.csv"))
    log.info("Wrote missingness_raw.csv (%d rows)", len(missingness_raw))

    # B. landmark-cohort missingness
    landmark_frames = {n: bundles[n]["landmark"] for n in config.DATASETS}
    missingness_after = missingness.missingness_table(landmark_frames)
    io_utils.save_csv(missingness_after, os.path.join(config.DIR_MISSINGNESS, "missingness_after_landmark.csv"))
    log.info("Wrote missingness_after_landmark.csv (%d rows)", len(missingness_after))

    # C. cross-dataset comparison (+ descriptive grouping)
    cross = missingness.cross_dataset_missingness(missingness_after)
    cross_grouped = missingness.missingness_group_classification(cross)
    io_utils.save_csv(cross_grouped, os.path.join(config.DIR_MISSINGNESS, "missingness_cross_dataset.csv"))
    log.info("Top 10 variables by absolute cross-dataset missingness difference:\n%s",
              cross_grouped.head(10).to_string(index=False))

    group_counts = cross_grouped["missingness_group"].value_counts()
    log.info("Missingness group counts (descriptive only, no auto-drop):\n%s", group_counts.to_string())

    # D. missing-indicator prevalence (predictors as encoded for modeling)
    predictors = {n: bundles[n]["predictors"] for n in config.DATASETS}
    indicator_prev = missingness.missing_indicator_prevalence(predictors)
    io_utils.save_csv(indicator_prev, os.path.join(config.DIR_MISSINGNESS, "missing_indicator_prevalence.csv"))
    log.info("Wrote missing_indicator_prevalence.csv (%d rows)", len(indicator_prev))

    # Flag variables entirely unavailable in one dataset for visibility
    unusable = cross_grouped[cross_grouped["missingness_group"] == "almost_unavailable_in_one_dataset"]
    if len(unusable):
        log.warning("Variables effectively unavailable in one dataset (technically unusable there):\n%s",
                    unusable[["variable", "ICU_missing_percent", "MIMIC_missing_percent"]].to_string(index=False))
    else:
        log.info("No candidate predictor is entirely missing in either dataset.")

    log.info("02_missingness_analysis.py complete.")


if __name__ == "__main__":
    main()
