"""
lit07_freeze_literature_library.py — Phase III / GATE 2 (spec sec. 34,
81): freeze the literature feature library BEFORE any Gangnam
hospital_mortality is examined for feature selection or model
optimization.

This script reads NOTHING from harmonized/, structured/, or any Gangnam
clinical file -- it only hashes/summarizes what lit01-lit06 already
produced, and writes the freeze manifest. After this point, per spec
sec. 34: "do not modify literature features because they perform poorly
or well in Gangnam." Any later change to search queries, extraction
prompts, or tiering rules must be a NEW, separately-dated version, never
a silent edit of this frozen manifest.

Outputs:
  literature_prior/final/FROZEN_LIBRARY_MANIFEST.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from common import SEMANTIC, today_str

SEARCH_DIR = SEMANTIC / "literature_prior" / "search"
EXTRACTION_DIR = SEMANTIC / "literature_prior" / "extraction"
ONTOLOGY_DIR = SEMANTIC / "literature_prior" / "ontology"
FINAL_DIR = SEMANTIC / "literature_prior" / "final"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    search_manifest = pd.read_csv(SEARCH_DIR / "search_manifest.csv")
    pmid_pool = pd.read_csv(SEARCH_DIR / "pmid_pool.csv", dtype={"pmid": str})
    extraction_manifest = json.loads((EXTRACTION_DIR / "extraction_manifest.json").read_text())
    lib_df = pd.read_csv(FINAL_DIR / "literature_feature_library.csv")
    summary_df = pd.read_csv(FINAL_DIR / "feature_evidence_summary.csv")
    review_df = pd.read_csv(FINAL_DIR / "manual_review_queue.csv")

    frozen = {
        "freeze_timestamp": today_str(),
        "freeze_gate": "GATE_2 (spec sec. 81) -- literature library frozen BEFORE any Gangnam hospital_mortality use",
        "pipeline_scripts": ["lit01_search_pubmed.py", "lit02_fetch_articles.py", "lit03_build_rag_corpus.py",
                              "lit04_extract_evidence.py", "lit05_normalize_features.py",
                              "lit06_build_literature_library.py", "lit07_freeze_literature_library.py"],
        "search": {
            "search_date": search_manifest["search_date"].iloc[0],
            "n_query_attempts": len(search_manifest),
            "n_unique_pmids_pooled": len(pmid_pool),
            "query_groups": search_manifest["query_group"].unique().tolist(),
        },
        "extraction": extraction_manifest,
        "literature_feature_library": {
            "n_features_total": len(lib_df),
            "tier_counts": lib_df["evidence_tier"].value_counts().to_dict(),
            "n_literature_core_candidates": int(lib_df["included_in_literature_core"].sum()),
            "n_literature_extended_candidates": int(lib_df["included_in_literature_extended"].sum()),
            "feature_ids": lib_df["feature_id"].tolist(),
        },
        "manual_review_queue_size": len(review_df),
        "known_limitations": [
            "Tier C (human critical-illness evidence without direct mortality) is currently EMPTY (0 features) "
            "-- the LLM extraction rarely emitted the specific exclusion_flags (severity_only/infection_"
            "acquisition_only/etc.) needed to distinguish Tier C from Tier D; most non-direct-mortality papers "
            "were instead flagged with mechanistic-only reasons (animal/cell/non-ICU). As a result, "
            "'literature_extended' == 'literature_core' (69 features each) at this freeze -- Extended adds "
            "nothing new yet. This is reported transparently rather than silently forcing a Tier C bucket.",
            "No 'genome_resolved' (MAG/virulence/BGC) feature rows were extracted by lit04 at all -- despite "
            "lit01's view_genome_resolved search queries surfacing 159 candidate PMIDs, the LLM extraction step "
            "classified their reportable findings under other feature_view categories or found nothing passage-"
            "verifiable. Consequently VFDB_mapping and antiSMASH_mapping are UNMAPPABLE for every feature in "
            "this freeze. This mirrors the earlier ad-hoc finding (scripts 34/35, 2026-08-13) that literature "
            "directly tying MAG-level resistome/virulence/BGC findings to ICU mortality is sparse.",
            "Cohort duplicate detection (sec. 13) is a coarse heuristic (country+N_total+population+ICU_type "
            "exact match) -- it is a conservative flag for manual review, not a validated declaration of shared "
            "cohorts.",
            f"{len(review_df)} feature-evidence rows had an evidence_passage that did not verify against the "
            "source text and were excluded from tier computation; they remain visible in manual_review_queue.csv.",
        ],
        "file_hashes": {
            str(p.relative_to(SEMANTIC)): file_sha256(p) for p in [
                SEARCH_DIR / "search_manifest.csv", SEARCH_DIR / "pmid_pool.csv",
                EXTRACTION_DIR / "evidence_flat.csv", ONTOLOGY_DIR / "feature_ontology_mapping.csv",
                FINAL_DIR / "literature_feature_library.csv", FINAL_DIR / "feature_evidence_summary.csv",
                FINAL_DIR / "manual_review_queue.csv",
            ]
        },
    }

    with open(FINAL_DIR / "FROZEN_LIBRARY_MANIFEST.json", "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"[lit07] GATE 2 PASSED: literature library frozen at {frozen['freeze_timestamp']}")
    print(f"[lit07] {frozen['literature_feature_library']['n_features_total']} features, "
          f"tiers={frozen['literature_feature_library']['tier_counts']}")
    print(f"[lit07] literature_core candidates: {frozen['literature_feature_library']['n_literature_core_candidates']}")
    print(f"[lit07] wrote {FINAL_DIR}/FROZEN_LIBRARY_MANIFEST.json")
    print("[lit07] From this point forward: literature features must NOT be modified based on Gangnam performance.")


if __name__ == "__main__":
    main()
