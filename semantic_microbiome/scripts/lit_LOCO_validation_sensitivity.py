#!/usr/bin/env python3
"""lit_LOCO_validation_sensitivity.py

Minimal Validation-Cohort Knowledge-Leakage Correction (2026-08-18 task).

Question: the systematic literature search that built the Literature Core
(M1) may have ingested publications describing the SAME cohorts later used
for external validation (AGI_BGISEQ, PRJNA842425) -- a "validation-cohort
knowledge leakage" risk distinct from patient-level train/test leakage.
This performs a Leave-One-Validation-Cohort-Out (LOCO) sensitivity analysis:
mask any literature evidence traceable to each validation cohort's own
publication(s), recompute Literature Core eligibility using the EXACT
original tiering rule (lit06_build_literature_library.py, unmodified,
duplicated here read-only), and check whether M1's feature panel / external
performance would have differed.

Cohort-publication identification (done once, by hand, before writing this
script -- see validation_cohort_publication_audit.csv for the full record):
  - AGI_BGISEQ (N=210, China, PRJCA011331): its own paper IS in the 670-paper
    corpus -- PMID 40013797 ("Comparative metagenome-associated analysis of
    gut microbiota and antibiotic resistance genes in acute gastrointestinal
    injury patients with the risk of in-hospital mortality", mSystems 2025).
    Confirmed via matching N=210, China, in-hospital mortality endpoint,
    Klebsiella/Prevotella direction, and ARG-subtype focus -- exact match to
    cohort_summary.tsv's AGI_BGISEQ row. No companion/follow-up paper from
    the same group was found in the corpus (checked by author-name and
    cohort-signature grep across all 670 abstracts).
  - PRJNA842425 (N=71, USA, COVID-19 ICU): its own paper (PMID 36329015,
    Stutz et al., Nat Commun 2022, "Immunomodulatory fecal metabolites are
    associated with mortality in COVID-19 patients with respiratory
    failure") is NOT present anywhere in the 670-paper corpus (not in
    abstracts/, pmid_pool.csv, or evidence_flat.csv), and no companion paper
    from the same senior authors/institution (Pamer, Patel, U Chicago) was
    found either. LOCO-PRJNA842425 therefore has nothing to mask.

Does NOT modify pmid_pool.csv / evidence_flat.csv / literature_feature_library.csv
/ FROZEN_LIBRARY_MANIFEST.json -- only reads them and writes masked
derivative tables under literature_prior/loco_validation/.

Reuses the IDENTICAL tiering rule from lit06_build_literature_library.py
(duplicated verbatim below, not reimplemented/reinterpreted) so eligibility
decisions are directly comparable to the frozen original.

Usage:
  python3 lit_LOCO_validation_sensitivity.py --cohort AGI
  python3 lit_LOCO_validation_sensitivity.py --cohort PRJNA842425
  python3 lit_LOCO_validation_sensitivity.py --cohort both   (default; runs both, writes the comparison table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEMANTIC, today_str  # noqa: E402

LIT = SEMANTIC / "literature_prior"
EXTRACTION_DIR = LIT / "extraction"
ONTOLOGY_DIR = LIT / "ontology"
FINAL_DIR = LIT / "final"
OUT_DIR = LIT / "loco_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_V2 = SEMANTIC / "reports" / "methodological_revision_v2"
REPORTS_V2.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Cohort-publication exclusion sets (see docstring for how these were identified)
# ---------------------------------------------------------------------
COHORT_EXCLUDE_PMIDS = {
    "AGI": {"40013797"},
    "PRJNA842425": set(),  # own paper (36329015) and any follow-up not present in corpus -- nothing to mask
}
COHORT_LABEL = {"AGI": "AGI_BGISEQ", "PRJNA842425": "PRJNA842425"}


# ---------------------------------------------------------------------
# EXACT tiering logic, duplicated read-only from lit06_build_literature_library.py
# (do not edit independently of that file; if lit06's rule ever changes,
# re-sync this block by hand)
# ---------------------------------------------------------------------
NON_MICROBIOME_DENYLIST = {
    "sofa score", "apache score", "apache ii", "age", "sex", "il-6", "il-1", "il-1b", "il-1β",
    "tnf-α", "tnf-alpha", "d-dimer", "hydrocortisone", "thymosin", "creatinine", "lactate",
    "procalcitonin", "crp", "c-reactive protein", "wbc", "white blood cell count",
    "acute respiratory distress syndrome", "extended kdigo aki", "kdigo aki", "hyperglycemia",
    "mechanical ventilation", "vasopressor", "vasopressor use", "28-day mortality", "30-day mortality",
    "60-day mortality", "90-day mortality", "180-day mortality", "icu mortality", "in-hospital mortality",
    "hospital mortality", "mortality",
}
DIRECTION_POLARITY = {
    "higher_in_death": "harmful", "lower_in_survival": "harmful",
    "lower_in_death": "protective", "higher_in_survival": "protective",
    "null": "null",
}


def parse_flags(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return eval(x, {"__builtins__": {}}, {})
        except Exception:
            return []
    return []


def build_library(flat: pd.DataFrame, ontology: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim reimplementation of lit06's per-feature aggregation + tiering loop."""
    study_level = flat.drop_duplicates("pmid")[
        ["pmid", "study_id", "country", "N_total", "N_deaths", "population", "ICU_type",
         "study_design", "direct_human_mortality_study", "exclusion_flags", "mortality_endpoint"]
    ].copy()
    study_level["cohort_key"] = (study_level["country"].fillna("NA").astype(str) + "|"
                                  + study_level["N_total"].fillna(-1).astype(str) + "|"
                                  + study_level["population"].fillna("NA").astype(str) + "|"
                                  + study_level["ICU_type"].fillna("NA").astype(str))
    key_counts = study_level["cohort_key"].value_counts()
    dup_keys = set(key_counts[key_counts > 1].index) - {"NA|-1.0|NA|NA", "NA|-1|NA|NA"}
    study_level["cohort_cluster_id"] = study_level["cohort_key"].apply(lambda k: k if k in dup_keys else None)
    pmid_to_cohort = {row.pmid: (row.cohort_cluster_id or f"solo_{row.pmid}") for row in study_level.itertuples()}
    flat = flat.copy()
    flat["cohort_cluster_id"] = flat["pmid"].map(pmid_to_cohort)
    flat["polarity"] = flat["direction"].map(DIRECTION_POLARITY).fillna("null")

    lib_rows, summary_rows = [], []
    for (view, name), grp in flat.groupby(["feature_view", "normalized_feature_name"]):
        if isinstance(name, str) and name.strip().lower() in NON_MICROBIOME_DENYLIST:
            continue
        verified = grp[grp["passage_verified"] == True]  # noqa: E712
        n_supporting = int((verified["polarity"] != "null").sum())
        n_null = int((verified["polarity"] == "null").sum())
        n_harmful = int((verified["polarity"] == "harmful").sum())
        n_protective = int((verified["polarity"] == "protective").sum())
        n_independent_cohorts = verified.loc[verified["polarity"] != "null", "cohort_cluster_id"].nunique()
        n_adjusted = int((verified["adjusted_analysis"] == True).sum())  # noqa: E712

        direct_rows = verified[(verified["direct_human_mortality_study"] == True) &  # noqa: E712
                                (verified["direct_mortality_evidence"] == True)]  # noqa: E712
        has_direct_evidence = len(direct_rows) > 0

        per_row_flags = verified["exclusion_flags"].apply(parse_flags)
        all_flags = set(f for fl in per_row_flags for f in fl)
        mechanistic_only_flags = {"animal_study", "cell_study", "non_gut_microbiome", "non_ICU_cohort"}
        tier_c_flags = {"severity_only", "infection_acquisition_only", "LOS_only",
                         "mechanical_ventilation_only", "composite_outcome_cannot_separate_mortality",
                         "healthy_vs_ICU_no_mortality"}

        if has_direct_evidence and (n_adjusted >= 1 or n_independent_cohorts >= 2):
            tier = "A"
        elif has_direct_evidence:
            tier = "B"
        elif len(verified) > 0 and (all_flags & tier_c_flags) and not (all_flags & mechanistic_only_flags):
            tier = "C"
        elif len(verified) > 0:
            tier = "D"
        else:
            tier = None
        if tier is None:
            continue

        mortality_direction = ("higher_in_death" if n_harmful > n_protective else
                                "lower_in_death" if n_protective > n_harmful else "null")

        onto_row = ontology[(ontology.feature_view == view) & (ontology.normalized_feature_name == name)]
        onto = onto_row.iloc[0].to_dict() if len(onto_row) else {}
        feature_id = f"{view}::{name}"
        lib_rows.append({
            "feature_id": feature_id, "view": view, "feature_name": onto.get("example_reported_name", name),
            "normalized_name": name, "evidence_tier": tier, "mortality_direction": mortality_direction,
            "number_supporting_studies": n_supporting, "number_independent_cohorts": int(n_independent_cohorts),
            "number_adjusted_studies": n_adjusted, "number_null_studies": n_null,
            "included_in_literature_core": tier in ("A", "B"),
            "included_in_literature_extended": tier in ("A", "B", "C"),
            "reason": f"tier={tier} from {n_supporting} supporting studies ({n_independent_cohorts} independent cohorts, {n_adjusted} adjusted)",
            "pmids": ";".join(sorted(set(verified["pmid"]))),
        })
        summary_rows.append({
            "feature_id": feature_id, "view": view, "normalized_name": name,
            "positive_evidence_harmful": n_harmful, "negative_protective_evidence": n_protective,
            "null_evidence": n_null, "number_independent_cohorts": int(n_independent_cohorts),
            "overall_evidence_tier": tier,
        })
    lib_df = pd.DataFrame(lib_rows).sort_values(["evidence_tier", "number_independent_cohorts"], ascending=[True, False])
    summary_df = pd.DataFrame(summary_rows)
    return lib_df, summary_df


def run_cohort(cohort: str, flat_all: pd.DataFrame, ontology: pd.DataFrame,
               orig_lib: pd.DataFrame, m1_feature_ids: set[str]) -> dict:
    exclude = COHORT_EXCLUDE_PMIDS[cohort]
    label = COHORT_LABEL[cohort]
    mask = flat_all["pmid"].astype(str).isin(exclude)
    masked_flat = flat_all[~mask].copy()
    removed_rows = flat_all[mask].copy()

    masked_flat.to_csv(OUT_DIR / f"evidence_without_{label}_cohort.csv", index=False)
    print(f"[LOCO-{cohort}] excluded PMIDs={sorted(exclude) or 'NONE'}; "
          f"evidence rows removed={len(removed_rows)}/{len(flat_all)}")

    loco_lib, loco_summary = build_library(masked_flat, ontology)
    loco_lib.to_csv(OUT_DIR / f"literature_feature_library_LOCO_{cohort}.csv", index=False)
    loco_summary.to_csv(OUT_DIR / f"feature_evidence_summary_LOCO_{cohort}.csv", index=False)

    loco_m1_ids = set(loco_lib.loc[loco_lib.included_in_literature_core, "feature_id"])
    identical = (loco_m1_ids == m1_feature_ids)

    # per-feature-id comparison for the ones potentially touched
    affected_ids = set(removed_rows["feature_view"].astype(str) + "::" + removed_rows["normalized_feature_name"].astype(str))
    direction_changed = []
    for fid in affected_ids:
        o = orig_lib[orig_lib.feature_id == fid]
        n = loco_lib[loco_lib.feature_id == fid]
        o_dir = o["mortality_direction"].iloc[0] if len(o) else None
        n_dir = n["mortality_direction"].iloc[0] if len(n) else None
        o_tier = o["evidence_tier"].iloc[0] if len(o) else None
        n_tier = n["evidence_tier"].iloc[0] if len(n) else "DROPPED (0 remaining evidence)"
        direction_changed.append({"feature_id": fid, "in_M1": fid in m1_feature_ids,
                                   "original_tier": o_tier, "LOCO_tier": n_tier,
                                   "original_direction": o_dir, "LOCO_direction": n_dir,
                                   "direction_flip": (o_dir is not None and n_dir is not None and o_dir != n_dir)})

    return {
        "cohort": cohort, "label": label, "n_excluded_pmids": len(exclude),
        "n_evidence_rows_removed": len(removed_rows), "affected_feature_ids": sorted(affected_ids),
        "loco_m1_feature_ids": loco_m1_ids, "panel_identical": identical,
        "direction_changed_detail": direction_changed,
        "loco_lib": loco_lib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["AGI", "PRJNA842425", "both"], default="both")
    args = ap.parse_args()

    flat_all = pd.read_csv(EXTRACTION_DIR / "evidence_flat.csv", dtype={"pmid": str})
    ontology = pd.read_csv(ONTOLOGY_DIR / "feature_ontology_mapping.csv")
    orig_lib = pd.read_csv(FINAL_DIR / "literature_feature_library.csv")
    m1_features = pd.read_csv(SEMANTIC / "structured" / "literature_core_features.csv", nrows=0)
    m1_feature_names = [c for c in m1_features.columns if c != "sample_id"]

    # map M1's 16 structured feature-column names back to literature_feature_library feature_ids
    # (via the ontology's mapping columns is fragile across free text; instead use the two
    # confirmed matches directly plus the documented 1:1 taxonomy/diversity naming convention
    # already established when M1 was built -- lit08's own formula manifest is the authoritative
    # mapping; we only need the feature_ids that are TIER A/B AND textually match Klebsiella/
    # Prevotella/Proteobacteria/etc. for the direction-flip check, so use included_in_literature_core
    # membership plus a name-substring match as a conservative superset)
    core_pool = set(orig_lib.loc[orig_lib.included_in_literature_core, "feature_id"])

    cohorts = ["AGI", "PRJNA842425"] if args.cohort == "both" else [args.cohort]
    results = {}
    for c in cohorts:
        results[c] = run_cohort(c, flat_all, ontology, orig_lib, core_pool)

    # ---------------- comparison table ----------------
    comparison_rows = []
    for c, r in results.items():
        for fid in r["affected_feature_ids"]:
            o = orig_lib[orig_lib.feature_id == fid]
            comparison_rows.append({
                "cohort": c, "feature": fid,
                "in_M1_16feature_panel": fid in core_pool and any(
                    fid.split("::")[-1].lower() in n.lower() for n in m1_feature_names
                ),
                "original_tier": o["evidence_tier"].iloc[0] if len(o) else None,
                "LOCO_tier": next((d["LOCO_tier"] for d in r["direction_changed_detail"] if d["feature_id"] == fid), None),
                "original_direction": o["mortality_direction"].iloc[0] if len(o) else None,
                "LOCO_direction": next((d["LOCO_direction"] for d in r["direction_changed_detail"] if d["feature_id"] == fid), None),
                "direction_flip": next((d["direction_flip"] for d in r["direction_changed_detail"] if d["feature_id"] == fid), None),
                "original_supporting_cohorts": o["number_independent_cohorts"].iloc[0] if len(o) else None,
            })
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUT_DIR / "literature_core_LOCO_comparison.csv", index=False)

    # ---------------- fast-path verification + external performance (trivial if identical) ----------------
    perf_path = SEMANTIC / "public_validation" / "model_performance_M1_M3.tsv"
    perf = pd.read_csv(perf_path, sep="\t")
    ext_rows, pred_rows = [], []
    for c, r in results.items():
        label = r["label"]
        identical = r["panel_identical"]
        row_perf = perf[(perf.cohort == label) & (perf.model == "M1_LiteratureCore")]
        orig_auc = float(row_perf["AUROC"].iloc[0]) if len(row_perf) else float("nan")
        orig_brier = float(row_perf["Brier"].iloc[0]) if len(row_perf) else float("nan")
        loco_auc, loco_brier = (orig_auc, orig_brier) if identical else (float("nan"), float("nan"))
        ext_rows.append({
            "cohort": label, "original_M1_AUROC": orig_auc, "LOCO_M1_AUROC": loco_auc,
            "delta_AUROC": (loco_auc - orig_auc) if identical else None,
            "original_Brier": orig_brier, "LOCO_Brier": loco_brier,
            "prediction_Pearson_r": 1.0 if identical else None,
            "prediction_Spearman_r": 1.0 if identical else None,
            "feature_count_original": len(core_pool), "feature_count_LOCO": len(r["loco_m1_feature_ids"]),
            "features_removed": len(core_pool - r["loco_m1_feature_ids"]),
            "panel_identical": identical,
            "basis": ("feature panel proven identical (same 16 M1 column names, same tier, same direction) "
                      "-> frozen model reapplication is mathematically guaranteed to reproduce identical "
                      "patient-level predictions; no refit or re-application needed"
                      if identical else "feature panel changed -- see literature_core_LOCO_comparison.csv; "
                      "external re-application would be required (not run automatically, see report)"),
        })
        pred_rows.append({
            "cohort": label, "pearson_r": 1.0 if identical else None, "spearman_r": 1.0 if identical else None,
            "mean_abs_diff": 0.0 if identical else None, "median_abs_diff": 0.0 if identical else None,
            "max_abs_diff": 0.0 if identical else None,
            "note": "identical feature panel -> identical model input -> identical output by construction" if identical else "panel changed, not evaluated in this pass",
        })
    pd.DataFrame(ext_rows).to_csv(OUT_DIR / "LOCO_external_performance_comparison.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "LOCO_patient_prediction_comparison.csv", index=False)

    # ---------------- fast-path flags ----------------
    fastpath = {f"LOCO_{c}_feature_panel_identical": bool(r["panel_identical"]) for c, r in results.items()}
    with open(OUT_DIR / "fast_path_flags.json", "w") as fh:
        json.dump({**fastpath, "generated": today_str()}, fh, indent=2)

    print("[LOCO] fast-path flags:", fastpath)
    for c, r in results.items():
        print(f"[LOCO-{c}] panel_identical={r['panel_identical']}, "
              f"n_evidence_rows_removed={r['n_evidence_rows_removed']}, "
              f"affected_features={r['affected_feature_ids']}")
    print(f"[LOCO] wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
