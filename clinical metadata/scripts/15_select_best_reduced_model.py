#!/usr/bin/env python3
"""15_select_best_reduced_model.py

Step 2 of the post-training pipeline: reads
clinical_ai/reports/reduced_model_comparison.csv (script 14) and selects the
PRIMARY reduced clinical AI model for Phase X integration -- the one with
the highest Gangnam_relevant_AUROC (see script 14 docstring for the
definition and rationale; deliberately based on the large ICU.csv/mimic.csv
evidence, not on the 302-patient cohort, to avoid circularity).

This does not delete/hide the other candidates -- 10_apply_reduced_model_to_
microbiome_cohort.py still applies and keeps ALL of them; this script only
writes the SELECTION DECISION with its full justification, for a later
script (16) to mark one column as PRIMARY and to build the Phase X C0
baseline on.

Ties (AUROC within 0.005) are broken toward simpler/more interpretable
models in this preference order: lr > et > catboost > MLP > TabNet >
FTTransformer -- an explicit, documented rule rather than an arbitrary
sort-order artifact.

Outputs:
  clinical_ai/reports/selected_reduced_model.json
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("15_select_best_reduced_model")

COMPARISON_CSV = os.path.join(config.DIR_REPORTS, "reduced_model_comparison.csv")
OUT_JSON = os.path.join(config.DIR_REPORTS, "selected_reduced_model.json")

TIE_BREAK_PREFERENCE = ["lr", "et", "catboost", "MLP", "TabNet", "FTTransformer"]
TIE_MARGIN = 0.005
EXPECTED_TOTAL_RUNS = 18  # 6 algorithms x 3 dev-sites


def main():
    if not os.path.exists(COMPARISON_CSV):
        log.warning("%s not found -- run 14_aggregate_reduced_model_performance.py first.", COMPARISON_CSV)
        return
    df = pd.read_csv(COMPARISON_CSV).dropna(subset=["Gangnam_relevant_AUROC"])
    if df.empty:
        log.warning("No rows with a usable Gangnam_relevant_AUROC yet.")
        return

    n_done = len(df)
    if n_done < EXPECTED_TOTAL_RUNS:
        log.warning("Only %d/%d expected reduced-model runs completed so far -- selection below is "
                     "PROVISIONAL and will be re-evaluated when this script is re-run after more finish.",
                     n_done, EXPECTED_TOTAL_RUNS)

    df = df.sort_values("Gangnam_relevant_AUROC", ascending=False).reset_index(drop=True)
    top_auroc = df.loc[0, "Gangnam_relevant_AUROC"]
    tied = df[df["Gangnam_relevant_AUROC"] >= top_auroc - TIE_MARGIN].copy()

    if len(tied) > 1:
        tied["_pref_rank"] = tied["model_name"].apply(
            lambda m: TIE_BREAK_PREFERENCE.index(m) if m in TIE_BREAK_PREFERENCE else 999)
        tied = tied.sort_values("_pref_rank")
        winner = tied.iloc[0]
        tie_note = (f"{len(tied)} model(s) within {TIE_MARGIN} AUROC of the top score "
                     f"({sorted(tied['model_name'] + '/' + tied['development_dataset'])}); "
                     f"tie-broken toward simpler model per documented preference order.")
    else:
        winner = df.iloc[0]
        tie_note = "No tie (clear top performer)."

    selection = {
        "generated": pd.Timestamp.now().isoformat(),
        "n_runs_completed": int(n_done), "n_runs_expected": EXPECTED_TOTAL_RUNS,
        "selection_is_provisional": bool(n_done < EXPECTED_TOTAL_RUNS),
        "selected_model_name": winner["model_name"], "selected_ML_or_DL": winner["ML_or_DL"],
        "selected_development_dataset": winner["development_dataset"],
        "selected_prediction_column": f"clinical_AI_prob_{winner['model_name']}_{winner['development_dataset']}dev",
        "selected_Gangnam_relevant_AUROC": float(winner["Gangnam_relevant_AUROC"]),
        "selected_Gangnam_relevant_AUROC_basis": winner.get("Gangnam_relevant_AUROC_basis", ""),
        "tie_break_note": tie_note,
        "selection_rule": "highest Gangnam_relevant_AUROC (see 14_aggregate_reduced_model_performance.py); "
                           "ties broken toward simpler models (lr > et > catboost > MLP > TabNet > FTTransformer)",
        "full_ranking": df[["model_name", "ML_or_DL", "development_dataset", "Gangnam_relevant_AUROC",
                             "Gangnam_relevant_AUROC_basis"]].to_dict(orient="records"),
    }
    os.makedirs(config.DIR_REPORTS, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(selection, fh, indent=2, default=str)
    log.info("SELECTED: %s (dev=%s), Gangnam-relevant AUROC=%.4f [%s]. %s",
              selection["selected_model_name"], selection["selected_development_dataset"],
              selection["selected_Gangnam_relevant_AUROC"], selection["selected_Gangnam_relevant_AUROC_basis"],
              "(PROVISIONAL -- not all runs done yet)" if selection["selection_is_provisional"] else "(FINAL)")
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
