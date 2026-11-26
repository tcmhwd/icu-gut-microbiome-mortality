#!/usr/bin/env python3
"""14_aggregate_reduced_model_performance.py

Step 1 of the post-training pipeline (run after 12_tune_ml_reduced.py /
13_tune_dl_reduced.py finish): aggregates clinical_ai/performance/
model_performance_comparison.csv (ML_reduced + DL_reduced rows only) with
the per-site external-breakdown CSVs (reduced_{ml,dl}_per_site_breakdown_*)
into ONE comparison table, adding a single "Gangnam_relevant_AUROC" column --
since the application cohort (sibling microbiome project's 302 patients) is
100% Gangnam Severance, this is the number that actually matters for model
selection (15_select_best_reduced_model.py), defined as:
  - for a Gangnam-dev model: its own CV_AUROC_mean (5-fold internal CV,
    out-of-fold -- an unbiased estimate of Gangnam performance, since the
    model was fit on Gangnam data)
  - for a Sinchon-dev or MIMIC-dev model: the "Gangnam" row of its own
    per-site external breakdown (truly held-out -- the model never saw
    Gangnam during training or tuning)
This keeps model SELECTION (script 15) based on the large, independent
ICU.csv/mimic.csv-derived evidence, not on the 302-patient cohort itself
(that comparison is done separately, descriptively, in
16_phase_x_c0_baseline_eval.py, precisely to avoid select-and-evaluate-on-
the-same-302-patients circularity).

Safe to re-run at any point while 12/13 are still in flight -- picks up
whatever rows exist so far.

Outputs:
  clinical_ai/reports/reduced_model_comparison.csv
  clinical_ai/reports/reduced_model_comparison.md
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("14_aggregate_reduced_model_performance")


def df_to_md_table(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (the 'gpt' conda env lacks the
    optional 'tabulate' package that pandas.DataFrame.to_markdown() needs)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)

PERF_PATH = os.path.join(config.DIR_PERFORMANCE, "model_performance_comparison.csv")
OUT_CSV = os.path.join(config.DIR_REPORTS, "reduced_model_comparison.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "reduced_model_comparison.md")


def load_breakdown(ml_or_dl_prefix: str, model_name: str, dev_site: str) -> pd.DataFrame | None:
    path = os.path.join(config.DIR_PERFORMANCE,
                         f"reduced_{ml_or_dl_prefix}_per_site_breakdown_{model_name}_dev-{dev_site}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    if not os.path.exists(PERF_PATH):
        log.warning("%s does not exist yet -- no reduced-model results to aggregate.", PERF_PATH)
        return
    perf = pd.read_csv(PERF_PATH)
    red = perf[perf["ML_or_DL"].isin(["ML_reduced", "DL_reduced"])].copy()
    if red.empty:
        log.warning("No ML_reduced/DL_reduced rows in %s yet.", PERF_PATH)
        return

    rows = []
    for _, r in red.iterrows():
        prefix = "ml" if r["ML_or_DL"] == "ML_reduced" else "dl"
        if r["development_dataset"] == "Gangnam":
            gangnam_relevant_auroc = r["CV_AUROC_mean"]
            gangnam_relevant_basis = "own_internal_CV_AUROC (5-fold OOF, dev=Gangnam)"
        else:
            bd = load_breakdown(prefix, r["model_name"], r["development_dataset"])
            if bd is None:
                gangnam_relevant_auroc = float("nan")
                gangnam_relevant_basis = "MISSING (per-site breakdown file not found)"
            else:
                gm = bd.loc[bd["site"] == "Gangnam"]
                if gm.empty:
                    gangnam_relevant_auroc = float("nan")
                    gangnam_relevant_basis = "MISSING (no Gangnam row in breakdown)"
                else:
                    gangnam_relevant_auroc = float(gm["AUROC"].iloc[0])
                    gangnam_relevant_basis = f"external_AUROC_on_Gangnam (dev={r['development_dataset']}, held-out)"
        rows.append({
            "model_name": r["model_name"], "ML_or_DL": r["ML_or_DL"], "development_dataset": r["development_dataset"],
            "external_dataset": r["external_dataset"], "CV_AUROC_mean": r["CV_AUROC_mean"],
            "external_AUROC_pooled": r["external_AUROC"],
            "external_AUROC_pooled_CI": f"{r['external_AUROC_CI_low']:.3f}-{r['external_AUROC_CI_high']:.3f}",
            "calibration_slope": r["calibration_slope"], "external_Brier": r["external_Brier"],
            "Gangnam_relevant_AUROC": gangnam_relevant_auroc, "Gangnam_relevant_AUROC_basis": gangnam_relevant_basis,
        })

    out = pd.DataFrame(rows).sort_values("Gangnam_relevant_AUROC", ascending=False).reset_index(drop=True)
    io_utils.save_csv(out, OUT_CSV)
    log.info("wrote %s (%d rows)", OUT_CSV, len(out))

    md = ["# Reduced clinical AI model comparison (ranked by Gangnam-relevant AUROC)", "",
          f"{len(out)} of 18 possible (6 algorithms x 3 dev-sites) model runs completed so far. "
          "Ranking column (`Gangnam_relevant_AUROC`) uses each model's OWN internal CV AUROC when "
          "dev=Gangnam, or its held-out external AUROC on the Gangnam subset when dev=Sinchon/MIMIC "
          "(see script docstring) -- this is evidence from the large ICU.csv/mimic.csv cohorts, "
          "NOT from the 302-patient microbiome cohort (kept separate, see 16_phase_x_c0_baseline_eval.py).",
          "", df_to_md_table(out)]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))
    log.info("wrote %s", OUT_MD)
    log.info("\n%s", out.to_string(index=False))


if __name__ == "__main__":
    main()
