"""
lit06_build_literature_library.py — Phase I / Gate 1, step 6: duplicate
cohort detection (spec sec. 13), evidence tiering (sec. 20), and
assembly of the final literature feature library + evidence summary +
manual review queue (sec. 31-33).

hospital_mortality (Gangnam) is NOT read here -- tiering uses only
literature-reported properties (adjusted analysis, independent cohorts,
direct-mortality-evidence flag, exclusion flags), never our own data.

Cohort duplicate detection (heuristic, best-effort per spec -- "attempt
to detect", not guaranteed): papers are grouped into the same
cohort_cluster_id when they report the same country + N_total + ICU_type
+ population (a same-country + identical-N coincidence across two
different unrelated studies is possible but unlikely; every grouping
decision is written to cohort_registry.csv for audit, nothing is merged
silently).

Evidence tiering (sec. 20), computed per unique (feature_view,
normalized_feature_name), aggregating across all supporting papers:
  TIER A: direct human ICU mortality evidence AND (>=1 adjusted study OR
          >=2 independent, non-duplicate cohorts)
  TIER B: direct human ICU mortality evidence, single cohort/unadjusted
  TIER C: human critical-illness evidence (severity/infection/organ
          failure/dysbiosis) but no convincing direct mortality evidence
  TIER D: mechanistic only (animal/in_vitro/non-ICU exclusion flags, or
          no qualifying human evidence at all)

Only passage-verified rows (lit04's passage_verified=True) count toward
tier A/B/C evidence counts -- unverified rows are kept in the full table
(never silently dropped, sec. 16) but routed to manual_review_queue.csv
and excluded from tier computation until a human confirms them.

Outputs:
  literature_prior/extraction/cohort_registry.csv
  literature_prior/final/literature_feature_library.csv
  literature_prior/final/feature_evidence_summary.csv
  literature_prior/final/manual_review_queue.csv
"""
from __future__ import annotations

import pandas as pd

from common import SEMANTIC, today_str

EXTRACTION_DIR = SEMANTIC / "literature_prior" / "extraction"
ONTOLOGY_DIR = SEMANTIC / "literature_prior" / "ontology"
FINAL_DIR = SEMANTIC / "literature_prior" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# QC filter caught on review of the first freeze attempt (2026-08-13): the
# lit04 LLM extraction occasionally pulled in clinical covariates/outcomes
# that happened to appear inside a paper's adjusted-mortality-model
# sentence (e.g. "...after adjusting for SOFA score and age...") as if they
# were microbiome findings, and a few papers' "other"-view rows were the
# mortality endpoint itself (e.g. "28-day mortality") rather than a
# microbiome feature. This is a text/construct-validity filter applied
# uniformly BEFORE any Gangnam data is touched -- not tuned on Gangnam
# performance, so it does not violate the sec. 34 freeze rule.
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

# ---------------------------------------------------------------------
# 1. cohort duplicate detection (paper-level, heuristic, sec. 13)
# ---------------------------------------------------------------------
flat = pd.read_csv(EXTRACTION_DIR / "evidence_flat.csv", dtype={"pmid": str})
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
study_level["cohort_cluster_id"] = study_level["cohort_key"].apply(
    lambda k: k if k in dup_keys else None)
study_level["possible_duplicate_cohort"] = study_level["cohort_cluster_id"].notna()

cohort_registry = study_level[study_level["possible_duplicate_cohort"]].sort_values("cohort_cluster_id")
cohort_registry.to_csv(EXTRACTION_DIR / "cohort_registry.csv", index=False)
print(f"[lit06] cohort dedup: {len(cohort_registry)}/{len(study_level)} papers flagged as possibly sharing a "
      f"cohort with >=1 other paper ({study_level['cohort_cluster_id'].nunique()} candidate clusters) "
      f"-> {EXTRACTION_DIR}/cohort_registry.csv")

pmid_to_cohort = {row.pmid: (row.cohort_cluster_id or f"solo_{row.pmid}") for row in study_level.itertuples()}
flat["cohort_cluster_id"] = flat["pmid"].map(pmid_to_cohort)
flat["polarity"] = flat["direction"].map(DIRECTION_POLARITY).fillna("null")

# ---------------------------------------------------------------------
# 2. per-feature aggregation + tiering
# ---------------------------------------------------------------------
ontology = pd.read_csv(ONTOLOGY_DIR / "feature_ontology_mapping.csv")

manual_review_rows = []
lib_rows = []
summary_rows = []

n_denylisted = 0
for (view, name), grp in flat.groupby(["feature_view", "normalized_feature_name"]):
    if isinstance(name, str) and name.strip().lower() in NON_MICROBIOME_DENYLIST:
        n_denylisted += 1
        continue
    verified = grp[grp["passage_verified"] == True]  # noqa: E712
    unverified = grp[grp["passage_verified"] != True]  # noqa: E712
    for _, r in unverified.iterrows():
        manual_review_rows.append({**r.to_dict(), "review_reason": "evidence_passage did not verify against source text"})

    n_supporting = int((verified["polarity"] != "null").sum())
    n_null = int((verified["polarity"] == "null").sum())
    n_harmful = int((verified["polarity"] == "harmful").sum())
    n_protective = int((verified["polarity"] == "protective").sum())
    n_independent_cohorts = verified.loc[verified["polarity"] != "null", "cohort_cluster_id"].nunique()
    n_adjusted = int((verified["adjusted_analysis"] == True).sum())  # noqa: E712

    direct_rows = verified[(verified["direct_human_mortality_study"] == True) &  # noqa: E712
                            (verified["direct_mortality_evidence"] == True)]  # noqa: E712
    has_direct_evidence = len(direct_rows) > 0

    # exclusion-flag-driven human-critical-illness-but-not-mortality signal (Tier C)
    def parse_flags(x):
        if isinstance(x, str) and x.startswith("["):
            try:
                return eval(x, {"__builtins__": {}}, {})
            except Exception:
                return []
        return []
    per_row_flags = verified["exclusion_flags"].apply(parse_flags)
    all_flags = set(f for fl in per_row_flags for f in fl)
    mechanistic_only_flags = {"animal_study", "cell_study", "non_gut_microbiome", "non_ICU_cohort"}
    tier_c_flags = {"severity_only", "infection_acquisition_only", "LOS_only",
                     "mechanical_ventilation_only", "composite_outcome_cannot_separate_mortality",
                     "healthy_vs_ICU_no_mortality"}
    rows_all_mechanistic = [bool(fl) and all(f in mechanistic_only_flags for f in fl) for fl in per_row_flags]
    is_mechanistic_only = len(verified) > 0 and all(rows_all_mechanistic) and not has_direct_evidence

    if has_direct_evidence and (n_adjusted >= 1 or n_independent_cohorts >= 2):
        tier = "A"
    elif has_direct_evidence:
        tier = "B"
    elif len(verified) > 0 and (all_flags & tier_c_flags) and not (all_flags & mechanistic_only_flags):
        tier = "C"
    elif len(verified) > 0:
        tier = "D"
    else:
        tier = None  # only unverified evidence exists -> not usable, goes to manual review only

    if tier is None:
        continue

    if n_harmful > 0 and n_protective > 0:
        consistency = "inconsistent"
    elif n_harmful >= 2 and n_protective == 0:
        consistency = "consistent_harmful"
    elif n_protective >= 2 and n_harmful == 0:
        consistency = "consistent_protective"
    elif n_harmful == 1 and n_protective == 0:
        consistency = "mostly_harmful"
    elif n_protective == 1 and n_harmful == 0:
        consistency = "mostly_protective"
    else:
        consistency = "insufficient"

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
        "number_conflicting_studies": int(n_harmful > 0 and n_protective > 0),
        "adult_ICU_support": bool((verified["population"].isin(["adult", "mixed"])).any()),
        "pediatric_support": bool((verified["population"] == "pediatric").any()),
        "endpoint_types": ";".join(sorted(set(verified["mortality_endpoint"].dropna().astype(str)))),
        "MetaPhlAn_mapping": onto.get("MetaPhlAn_mapping"), "HUMAnN_mapping": onto.get("HUMAnN_mapping"),
        "RGI_mapping": onto.get("RGI_mapping"), "MAG_mapping": onto.get("MAG_mapping"),
        "eggNOG_mapping": onto.get("eggNOG_mapping"), "antiSMASH_mapping": onto.get("antiSMASH_mapping"),
        "VFDB_mapping": onto.get("VFDB_mapping"),
        "exact_computation_rule": None,  # filled in Phase III (script lit08+) when Core/Extended scores are defined
        "included_in_literature_core": tier in ("A", "B"),
        "included_in_literature_extended": tier in ("A", "B", "C"),
        "reason": f"tier={tier} from {n_supporting} supporting studies ({n_independent_cohorts} independent cohorts, {n_adjusted} adjusted)",
        "pmids": ";".join(sorted(set(verified["pmid"]))),
    })

    summary_rows.append({
        "feature_id": feature_id, "view": view, "normalized_name": name,
        "positive_evidence_harmful": n_harmful, "negative_protective_evidence": n_protective,
        "null_evidence": n_null, "conflicting_evidence": int(n_harmful > 0 and n_protective > 0),
        "number_independent_cohorts": int(n_independent_cohorts), "overall_evidence_tier": tier,
        "consistency_assessment": consistency,
    })

lib_df = pd.DataFrame(lib_rows).sort_values(["evidence_tier", "number_independent_cohorts"], ascending=[True, False])
lib_df.to_csv(FINAL_DIR / "literature_feature_library.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FINAL_DIR / "feature_evidence_summary.csv", index=False)

review_df = pd.DataFrame(manual_review_rows)
review_df.to_csv(FINAL_DIR / "manual_review_queue.csv", index=False)

print(f"[lit06] {n_denylisted} candidate (view, name) groups excluded as non-microbiome clinical covariates/outcomes (NON_MICROBIOME_DENYLIST)")
print(f"[lit06] literature_feature_library: {len(lib_df)} features -- tier counts: {lib_df.evidence_tier.value_counts().to_dict()}")
print(f"[lit06] literature_core (Tier A/B) candidate features: {int(lib_df.included_in_literature_core.sum())}")
print(f"[lit06] literature_extended (Tier A/B/C) candidate features: {int(lib_df.included_in_literature_extended.sum())}")
print(f"[lit06] manual_review_queue: {len(review_df)} rows (unverified evidence passages)")
print(f"[lit06] wrote {FINAL_DIR}/")
