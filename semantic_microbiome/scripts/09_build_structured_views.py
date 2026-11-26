"""
09_build_structured_views.py — Phase 2: structured biological representation

For each of the 8 microbiome views, restrict to the Gangnam day0-3 primary
cohort (harmonized/discovery_cohort_primary.csv, N=302) and produce a
QC-filtered, normalized, patient x feature table under
semantic_microbiome/structured/. Outcome-blind throughout: mortality is
NEVER read or used to choose/filter/threshold any feature here (confirmed
by construction — this script never imports the mortality column).

Prevalence filter: keep a feature if present (>0) in >= common.PREVALENCE_THRESHOLD
(10%) of the view's available cohort samples (spec sec. 29's "QC ->
prevalence/technical filtering" step). Thresholds are fixed a priori, not
tuned against outcome.

Intended to run under SLURM (loads full matrices for several views).

Outputs (semantic_microbiome/structured/):
  taxonomy_species_filtered.tsv, taxonomy_genus_filtered.tsv
  taxonomy_alpha_diversity.tsv          (computed BEFORE prevalence filtering)
  humann_pathway_filtered.tsv
  mag_abundance_filtered.tsv
  contig_rgi_drugclass_filtered.tsv, contig_rgi_rpkm_drugclass_filtered.tsv
  mag_rgi_drugclass_filtered.tsv, mag_rgi_burden_summary.tsv
  vfdb_category_filtered.tsv, vfdb_burden_summary.tsv
  antismash_bgc_types_filtered.tsv, antismash_burden_summary.tsv  (288-sample denominator)
  eggnog_kegg_ko_filtered.tsv, eggnog_cog_filtered.tsv            (count-based, NOT abundance-weighted -- see comments)
  manifest.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    METAPHLAN_DIR, HUMANN_DIR, COVERM_DIR, EGGNOG_DIR, ANTISMASH_DIR,
    RGI_MAG_DIR, RGI_CONTIG_DIR, RPKM_RGI_DIR, VFDB_DIR,
    HARMONIZED_DIR, STRUCTURED_DIR, PREVALENCE_THRESHOLD,
    load_cohort_matrix, prevalence_filter,
)

mapping = pd.read_csv(HARMONIZED_DIR / "sample_id_mapping.csv")
cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv")
# NOTE: hospital_mortality is intentionally NOT loaded into any variable
# below other than this cohort frame, which is used only to know WHICH
# patients/sample_ids are eligible -- never to filter/weight features.

manifest_rows = []


def log(view, n_patients, n_before, n_after, dropped, normalization, notes=""):
    manifest_rows.append({
        "view": view, "n_patients": n_patients, "features_before_filter": n_before,
        "features_after_filter": n_after, "features_dropped_low_prevalence": dropped,
        "prevalence_threshold": PREVALENCE_THRESHOLD, "normalization": normalization,
        "notes": notes,
    })
    print(f"[09] {view}: patients={n_patients} features {n_before}->{n_after} (dropped {dropped}) norm={normalization}")


# ---------------------------------------------------------------------
# 1. Taxonomy (MetaPhlAn) + alpha diversity
# ---------------------------------------------------------------------
for level, fname in [("species", "metaphlan3_species_merged.tsv"), ("genus", "metaphlan3_genus_merged.tsv")]:
    mat = load_cohort_matrix("MetaPhlAn", METAPHLAN_DIR / fname, "clade_name", mapping, cohort)
    n_before = mat.shape[1]
    if level == "species":
        prop = mat / 100.0
        shannon = -(prop * np.log(prop.where(prop > 0))).sum(axis=1, skipna=True)
        simpson = 1 - (prop ** 2).sum(axis=1)
        richness = (mat > 0).sum(axis=1)
        alpha = pd.DataFrame({"shannon": shannon, "simpson": simpson, "richness_species": richness})
        alpha.index.name = "patient_id"
        alpha.to_csv(STRUCTURED_DIR / "taxonomy_alpha_diversity.tsv", sep="\t")
    filt, n_after, n_dropped = prevalence_filter(mat)
    filt.index.name = "patient_id"
    filt.to_csv(STRUCTURED_DIR / f"taxonomy_{level}_filtered.tsv", sep="\t")
    log(f"MetaPhlAn_{level}", mat.shape[0], n_before, n_after, n_dropped,
        "relative abundance (%, MetaPhlAn native scale, unfiltered rows renormalized NOT applied)",
        "alpha diversity computed pre-filter on species level" if level == "species" else "")

# ---------------------------------------------------------------------
# 2. HUMAnN3 pathway abundance
# ---------------------------------------------------------------------
mat = load_cohort_matrix("HUMAnN3", HUMANN_DIR / "humann_pathabundance_wide.unstratified.tsv", "# Pathway", mapping, cohort)
special = [c for c in mat.columns if c in ("UNMAPPED", "UNINTEGRATED")]
mat_bio = mat.drop(columns=special)
n_before = mat_bio.shape[1]
sample_total = mat_bio.sum(axis=1)
rel = mat_bio.div(sample_total.replace(0, np.nan), axis=0).fillna(0.0)
filt, n_after, n_dropped = prevalence_filter(mat_bio)  # prevalence on raw RPK presence, values kept as relative
filt_rel = rel[filt.columns]
filt_rel.index.name = "patient_id"
filt_rel.to_csv(STRUCTURED_DIR / "humann_pathway_filtered.tsv", sep="\t")
log("HUMAnN3_pathway", mat.shape[0], n_before, n_after, n_dropped,
    "relative proportion of per-sample RPK sum, UNMAPPED/UNINTEGRATED excluded before normalizing",
    f"UNMAPPED/UNINTEGRATED columns dropped: {special}")

# ---------------------------------------------------------------------
# 3. MAG abundance (CoverM)
# ---------------------------------------------------------------------
mat = load_cohort_matrix("MAG_abundance_CoverM", COVERM_DIR / "coverm_abundance_matrix.tsv", "MAG", mapping, cohort)
if "unmapped" in mat.columns:
    mat = mat.drop(columns=["unmapped"])
n_before = mat.shape[1]
filt, n_after, n_dropped = prevalence_filter(mat)
filt.index.name = "patient_id"
filt.to_csv(STRUCTURED_DIR / "mag_abundance_filtered.tsv", sep="\t")
log("MAG_abundance_CoverM", mat.shape[0], n_before, n_after, n_dropped,
    "relative abundance (%, CoverM native scale)", "'unmapped' pseudo-row excluded")

# ---------------------------------------------------------------------
# 4. Resistome: contig-RGI (raw + RPKM), MAG-RGI
# ---------------------------------------------------------------------
def structure_feature_matrix(view, path, out_name, normalization, patient_burden_name=None):
    mat = load_cohort_matrix(view, path, "feature", mapping, cohort)
    n_before = mat.shape[1]
    filt, n_after, n_dropped = prevalence_filter(mat)
    filt.index.name = "patient_id"
    filt.to_csv(STRUCTURED_DIR / out_name, sep="\t")
    log(view + f"[{out_name}]", mat.shape[0], n_before, n_after, n_dropped, normalization)
    if patient_burden_name:
        burden = mat.sum(axis=1).to_frame("total_burden_raw_count")
        burden.index.name = "patient_id"
        burden.to_csv(STRUCTURED_DIR / patient_burden_name, sep="\t")
    return mat


structure_feature_matrix("contig_RGI", RGI_CONTIG_DIR / "rgi_contig_drug_class_per_sample.tsv",
                          "contig_rgi_drugclass_filtered.tsv", "raw count (per-sample gene hits)")
structure_feature_matrix("RPKM_RGI_contig", RPKM_RGI_DIR / "rpkm_rgi_drug_class_per_sample.tsv",
                          "contig_rgi_rpkm_drugclass_filtered.tsv", "RPKM (sequencing-depth normalized)")
mag_rgi_mat = structure_feature_matrix("MAG_RGI", RGI_MAG_DIR / "rgi_drug_class_per_sample.tsv",
                                        "mag_rgi_drugclass_filtered.tsv",
                                        "raw count, NOT abundance-weighted (representative-MAG hit counts)",
                                        patient_burden_name="mag_rgi_burden_summary.tsv")

# ---------------------------------------------------------------------
# 5. Virulence (VFDB)
# ---------------------------------------------------------------------
structure_feature_matrix("VFDB", VFDB_DIR / "vfdb_category_per_sample.tsv",
                          "vfdb_category_filtered.tsv",
                          "raw count, NOT abundance-weighted (representative-MAG hit counts)",
                          patient_burden_name="vfdb_burden_summary.tsv")

# ---------------------------------------------------------------------
# 6. antiSMASH BGC — reindexed to the 288-sample denominator (Phase 1 caveat)
# ---------------------------------------------------------------------
m_total = mapping[(mapping.view == "antiSMASH") & (mapping.match_method != "unmatched")]
m_total = m_total[m_total.patient_id_clinical.isin(set(cohort["patient_id_clinical"]))]
full_patient_ids = sorted(set(m_total["patient_id_clinical"]))  # 288 patients, per bgc_total denominator

mat = load_cohort_matrix("antiSMASH", ANTISMASH_DIR / "antismash_bgc_types_per_sample.tsv", "feature", mapping, cohort)
mat = mat.reindex(full_patient_ids, fill_value=0.0)  # add back the one sample with no detected BGC as an all-zero row
n_before = mat.shape[1]
filt, n_after, n_dropped = prevalence_filter(mat)
filt.index.name = "patient_id"
filt.to_csv(STRUCTURED_DIR / "antismash_bgc_types_filtered.tsv", sep="\t")
burden = mat.sum(axis=1).to_frame("total_bgc_count")
burden.index.name = "patient_id"
burden.to_csv(STRUCTURED_DIR / "antismash_burden_summary.tsv", sep="\t")
log("antiSMASH", mat.shape[0], n_before, n_after, n_dropped, "raw BGC-type count",
    "reindexed to 288-sample denominator (antismash_bgc_total_per_sample.tsv); "
    "one sample with no detected BGC added back as an explicit all-zero row (true zero result, see Phase1 caveat)")

# ---------------------------------------------------------------------
# 7. MAG function (eggNOG) — count-based, NOT abundance-weighted (documented)
# ---------------------------------------------------------------------
for feat, fname in [("kegg_ko", "eggnog_kegg_ko_per_sample.tsv"), ("cog", "eggnog_cog_per_sample.tsv")]:
    structure_feature_matrix("eggNOG", EGGNOG_DIR / fname, f"eggnog_{feat}_filtered.tsv",
                              "raw count, NOT abundance-weighted (see Phase1 reports/data_inventory.csv comments; "
                              "a true abundance-weighted version would need EggNOG_result/*.emapper.annotations "
                              "+ coverm_abundance_matrix.tsv recombination, out of scope for this pass)")

# ---------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------
pd.DataFrame(manifest_rows).to_csv(STRUCTURED_DIR / "manifest.csv", index=False)
print(f"[09] wrote structured/manifest.csv ({len(manifest_rows)} entries)")
