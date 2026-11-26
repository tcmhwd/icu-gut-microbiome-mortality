"""
lit08_build_literature_core_extended.py — Phase III (spec sec. 35-36):
build the Literature Core and Literature Extended representations from
the FROZEN literature_feature_library.csv (lit07). This is the first
script in the pipeline that touches Gangnam/public microbiome VALUES --
but still strictly outcome-blind: hospital_mortality is never read here,
and every formula below was fixed by the literature tiering alone
(lit06), not by checking what predicts Gangnam mortality.

*** BUG FOUND & FIXED 2026-08-14 (during lit13, second pipeline pass) ***
The first version of this script indexed Gangnam rows by
`sample_metadata.tsv`'s own "sample_id" column, which for the ICU
dataset is an INCONSISTENT MIX of two schemes (some rows already show
the clinical ID like "ICUxxxx", most show the raw "<KCDC_no>_WM" form
like "A0824Gxxxx_WM" -- documented in common.py's CLINICAL_CSV comment).
Only 110/302 discovery-cohort patient_id_clinical values happened to
match that column directly; the other 192/302 silently became all-NaN
rows once anything downstream did `.reindex(y.index)` (y.index =
patient_id_clinical from discovery_cohort_primary.csv). This corrupted
~64% of Gangnam patients across lit08/lit10/lit11/lit12 without any
error until lit13's PCA call finally raised on the resulting NaNs.

Fix: build Gangnam values using `harmonized/sample_id_mapping.csv`
(patient_id_clinical <-> the ACTUAL raw column name per view -- the
same mapping the already-validated scripts 17/25/33/35 use), then
RENAME each raw table's Gangnam columns to patient_id_clinical before
any lookup. Public-cohort columns are untouched (sample_id was already
reliable there, as confirmed in scripts 25/33/35).

Core-inclusion rule (operationalizes spec sec. 35's "features measurable
across Gangnam and public cohorts, do not include dozens of redundant
taxa"): a Tier A/B literature feature enters Core only if it has a
literal, non-approximated mapping (EXACT/PARENT_CHILD/SYNONYM) to
MetaPhlAn, MAG(GTDB), HUMAnN, or RGI in feature_ontology_mapping.csv.
This single rule is also what filters out every non-microbiome item that
slipped through lit04/lit06 (survival rates, GCS score, bile acid labs,
etc. all map to nothing in our data, so they cannot enter Core by
construction -- no separate denylist needed here).

Individual Core features (11, all Tier A/B, EXACT/PARENT_CHILD mapped):
  alpha diversity (Shannon, species-level) [protective]
  Enterococcus, Klebsiella, Escherichia (genus)          [harmful]
  Faecalibacterium, Prevotella (genus)                    [protective]
  Bacteroides (genus), Proteobacteria (phylum)             [harmful, as literature-reported]
  Christensenellaceae, Erysipelotrichaceae (family)        [harmful]
  Clostridium_leptum (species)                             [harmful]
  Bacteroidetes/Firmicutes ratio (phylum)                  [harmful]

Combined biological scores (formulas fixed BEFORE any Gangnam outcome
check, per spec sec. 35's suggested examples):
  Literature_pathobiont_burden = Enterococcus + Klebsiella + Escherichia (genus relabund sum)
  Literature_obligate_anaerobe_preservation = Faecalibacterium + Prevotella (genus relabund sum)
  Literature_facultative_to_obligate_anaerobe_balance = (pathobiont+eps) / (anaerobe_preservation+eps)
  Genome_resolved_pathobiont_MAG_burden = sum of representative-MAG relative abundance (CoverM,
      pooled ICU+AGI+non_AGI catalog) for MAGs whose GTDB genus in {Enterococcus, Klebsiella,
      Escherichia} -- the assembly/MAG-level realization of the SAME literature-supported
      pathobiont concept as a cross-check against the read-based (MetaPhlAn) version, per the
      user's explicit request (2026-08-13 chat) to also verify MAG-derived indicators.

Literature Extended (sec. 36) additionally includes:
  Literature_ARG_burden_broad = total MAG-level RGI drug-class abundance (summed across all 37
      drug classes). NOTE: no individual Tier A/B feature mapped cleanly to a specific RGI drug
      class (see FROZEN_LIBRARY_MANIFEST known_limitations), so this is included as a broad,
      literature-MOTIVATED-but-not-literature-SPECIFIC mechanistic module (spec sec. 24's generic
      "total ARG abundance" candidate), clearly separated from the literature-anchored Core scores.
  (Tier C is currently empty -- see FROZEN_LIBRARY_MANIFEST -- so no Tier C modules exist to add.)

Outputs (computed for the 302 Gangnam discovery patients (patient_id_clinical)
+ all public cohort samples (raw sample_id) -- using the exact same formula
for both, since public outcomes are not examined here at all, only
microbiome values):
  structured/literature_core_features.csv
  structured/literature_extended_features.csv
  literature_prior/final/literature_core_extended_formulas.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import SEMANTIC, ICU_PUBLIC, HARMONIZED_DIR, METAPHLAN_DIR, today_str

STRUCTURED_DIR = SEMANTIC / "structured"
FINAL_DIR = SEMANTIC / "literature_prior" / "final"
MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
MAG_SUMMARY = MERGE_OUTPUT / "mag_summary"

cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
GANGNAM_IDS = cohort.index.tolist()

sm = pd.read_csv(MERGE_OUTPUT / "sample_metadata.tsv", sep="\t", dtype=str)
PUBLIC_IDS = sm.loc[sm.dataset != "ICU", "sample_id"].tolist()
ALL_IDS = GANGNAM_IDS + PUBLIC_IDS  # patient_id_clinical for Gangnam, raw sample_id for public -- non-colliding namespaces

sid_map = pd.read_csv(HARMONIZED_DIR / "sample_id_mapping.csv")


def gangnam_rename_dict(view_name):
    m = sid_map[(sid_map.view == view_name) & (sid_map.match_method != "unmatched")]
    m = m[m.patient_id_clinical.isin(set(GANGNAM_IDS))]
    d = dict(zip(m.sample_id_in_view, m.patient_id_clinical))
    print(f"[lit08]   {view_name}: {len(d)}/{len(GANGNAM_IDS)} Gangnam patients have a mapping")
    return d


def rename_gangnam_index(df, view_name):
    """df must have SAMPLE identifiers as its INDEX. Renames the Gangnam
    subset from raw sample_id_in_view to patient_id_clinical; public rows
    untouched."""
    return df.rename(index=gangnam_rename_dict(view_name))


def rename_gangnam_columns(df, view_name):
    """df must have SAMPLE identifiers as its COLUMNS."""
    return df.rename(columns=gangnam_rename_dict(view_name))


# ---------------------------------------------------------------------
# read-based (MetaPhlAn) individual Core features
# ---------------------------------------------------------------------
LEVELS = {}
for level in ["species", "genus", "family", "phylum"]:
    raw = pd.read_csv(METAPHLAN_DIR / f"metaphlan3_{level}_merged.tsv", sep="\t", index_col=0).T  # .T -> index=sample, columns=clade
    LEVELS[level] = rename_gangnam_index(raw, "MetaPhlAn")


def find_col(level, leaf_name):
    hits = [c for c in LEVELS[level].columns if c.lower().split("|")[-1].split("__")[-1].replace("_", " ") == leaf_name.lower()]
    return hits[0] if hits else None


alpha_shannon = LEVELS["species"].apply(lambda r: (lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum()))(r.values / 100.0), axis=1)

genus_targets = {"Enterococcus": "harmful", "Klebsiella": "harmful", "Escherichia": "harmful",
                  "Faecalibacterium": "protective", "Prevotella": "protective", "Bacteroides": "harmful"}
family_targets = {"Christensenellaceae": "harmful", "Erysipelotrichaceae": "harmful"}
phylum_targets = {"Proteobacteria": "harmful"}
species_targets = {"Clostridium leptum": "harmful"}

core = pd.DataFrame(index=ALL_IDS)
core["Literature_alpha_diversity_shannon"] = alpha_shannon.reindex(ALL_IDS)

for name in genus_targets:
    col = find_col("genus", name)
    core[f"Literature_{name}_genus_relabund"] = LEVELS["genus"][col].reindex(ALL_IDS) if col else np.nan

for name in family_targets:
    col = find_col("family", name)
    core[f"Literature_{name}_family_relabund"] = LEVELS["family"][col].reindex(ALL_IDS) if col else np.nan

for name in phylum_targets:
    col = find_col("phylum", name)
    core[f"Literature_{name}_phylum_relabund"] = LEVELS["phylum"][col].reindex(ALL_IDS) if col else np.nan

for name in species_targets:
    col = find_col("species", name)
    core[f"Literature_{name.replace(' ', '_')}_species_relabund"] = LEVELS["species"][col].reindex(ALL_IDS) if col else np.nan

firm_col, bact_col = find_col("phylum", "Firmicutes"), find_col("phylum", "Bacteroidetes")
core["Literature_Bacteroidetes_Firmicutes_ratio"] = (
    (LEVELS["phylum"][bact_col] + 1e-6) / (LEVELS["phylum"][firm_col] + 1e-6)
).reindex(ALL_IDS) if (firm_col and bact_col) else np.nan

# ---------------------------------------------------------------------
# combined read-based scores (formula fixed here, before any outcome check)
# ---------------------------------------------------------------------
EPS = 1e-6
core["Literature_pathobiont_burden"] = (core[[f"Literature_{n}_genus_relabund" for n in ["Enterococcus", "Klebsiella", "Escherichia"]]].sum(axis=1, skipna=True))
core["Literature_obligate_anaerobe_preservation"] = (core[[f"Literature_{n}_genus_relabund" for n in ["Faecalibacterium", "Prevotella"]]].sum(axis=1, skipna=True))
core["Literature_facultative_to_obligate_anaerobe_balance"] = (
    (core["Literature_pathobiont_burden"] + EPS) / (core["Literature_obligate_anaerobe_preservation"] + EPS))

# ---------------------------------------------------------------------
# genome-resolved (MAG-level) pathobiont burden -- same biological
# concept as Literature_pathobiont_burden, computed from the ASSEMBLED/
# BINNED representative-MAG catalog instead of read-based MetaPhlAn, per
# user's explicit request to verify MAG-derived indicators too.
# ---------------------------------------------------------------------
gtdb = pd.read_csv(MAG_SUMMARY / "gtdb" / "gtdb_mag_taxonomy.tsv", sep="\t")
pathobiont_genera = {"Enterococcus", "Klebsiella", "Escherichia"}
pathobiont_mags = set(gtdb.loc[gtdb["genus"].isin(pathobiont_genera), "MAG"])
print(f"[lit08] {len(pathobiont_mags)} representative MAGs match literature pathobiont genera {pathobiont_genera}")

coverm_raw = pd.read_csv(MAG_SUMMARY / "coverm" / "coverm_abundance_matrix.tsv", sep="\t", index_col="MAG")
coverm_raw = coverm_raw.drop(index="unmapped", errors="ignore")
coverm = rename_gangnam_columns(coverm_raw, "MAG_abundance_CoverM")  # index=MAG, columns=sample already
mag_pathobiont_burden = coverm.loc[coverm.index.isin(pathobiont_mags)].sum(axis=0)
core["Genome_resolved_pathobiont_MAG_burden"] = mag_pathobiont_burden.reindex(ALL_IDS)

core.index.name = "sample_id"
core.to_csv(STRUCTURED_DIR / "literature_core_features.csv")
print(f"[lit08] Literature Core: {core.shape[1]} features x {core.shape[0]} samples "
      f"({len(GANGNAM_IDS)} Gangnam discovery + {len(PUBLIC_IDS)} public, outcome-blind)")
print(f"[lit08]   coverage (non-NaN, any cohort): \n{core.notna().sum().to_string()}")
n_gangnam_complete = core.reindex(GANGNAM_IDS).notna().all(axis=1).sum()
print(f"[lit08]   Gangnam patients with ALL core features present: {n_gangnam_complete}/{len(GANGNAM_IDS)}")

# ---------------------------------------------------------------------
# Literature Extended = Core + broad ARG burden mechanistic module
# ---------------------------------------------------------------------
rgi_mag_raw = pd.read_csv(MAG_SUMMARY / "rgi" / "rgi_drug_class_per_sample.tsv", sep="\t", index_col="feature")
rgi_mag = rename_gangnam_columns(rgi_mag_raw, "MAG_RGI")  # index=feature, columns=sample already
extended = core.copy()
extended["Literature_ARG_burden_broad_MAGlevel"] = rgi_mag.sum(axis=0).reindex(ALL_IDS)
extended.to_csv(STRUCTURED_DIR / "literature_extended_features.csv")
print(f"[lit08] Literature Extended: {extended.shape[1]} features (Core + 1 broad ARG burden module)")

# ---------------------------------------------------------------------
# formula manifest
# ---------------------------------------------------------------------
formulas = {
    "generated": today_str(),
    "bugfix_note": "2026-08-14: Gangnam columns now renamed via harmonized/sample_id_mapping.csv "
                    "(patient_id_clinical) instead of raw sample_metadata.tsv sample_id (was a mixed "
                    "scheme, silently NaN'd 192/302 Gangnam patients in the first version).",
    "core_inclusion_rule": "Tier A/B literature feature AND has EXACT/PARENT_CHILD/SYNONYM mapping "
                            "to MetaPhlAn, MAG(GTDB), HUMAnN, or RGI in feature_ontology_mapping.csv",
    "individual_core_features": {c: "literal MetaPhlAn relative abundance / computed diversity index"
                                  for c in core.columns if c not in
                                  ["Literature_pathobiont_burden", "Literature_obligate_anaerobe_preservation",
                                   "Literature_facultative_to_obligate_anaerobe_balance", "Genome_resolved_pathobiont_MAG_burden"]},
    "combined_scores": {
        "Literature_pathobiont_burden": "sum(genus_relabund[Enterococcus, Klebsiella, Escherichia])",
        "Literature_obligate_anaerobe_preservation": "sum(genus_relabund[Faecalibacterium, Prevotella])",
        "Literature_facultative_to_obligate_anaerobe_balance": "(pathobiont_burden + 1e-6) / (obligate_anaerobe_preservation + 1e-6)",
        "Genome_resolved_pathobiont_MAG_burden": "sum(CoverM_MAG_relabund[MAG in representative catalog AND GTDB genus in {Enterococcus,Klebsiella,Escherichia}])",
        "Literature_ARG_burden_broad_MAGlevel": "sum(MAG-level RGI drug class relabund, all 37 classes) -- Extended only, not literature-tier-anchored",
    },
    "n_core_features": int(core.shape[1]), "n_extended_features": int(extended.shape[1]),
    "pathobiont_mag_count": len(pathobiont_mags),
    "n_gangnam": len(GANGNAM_IDS), "n_public": len(PUBLIC_IDS),
    "n_gangnam_complete_core": int(n_gangnam_complete),
}
with open(FINAL_DIR / "literature_core_extended_formulas.json", "w") as f:
    json.dump(formulas, f, indent=2)
print(f"[lit08] wrote {STRUCTURED_DIR}/literature_core_features.csv, literature_extended_features.csv, "
      f"{FINAL_DIR}/literature_core_extended_formulas.json")
