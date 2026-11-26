"""
02_harmonize_sample_ids.py

Builds the master sample-ID mapping table across clinical metadata and all
per-view feature matrices under merge/output/, per Phase-1 spec sec. 15/16.

ID scheme found during audit (documented, not assumed — see also the
docstring in 03_define_discovery_cohort.py for the full column audit):
  - CLINICAL_CSV (Gangnam) has FIVE identifier-like columns per row
    (Total/No/list/KCDC_no/ID/sample_list). Of these:
      * `No` is the reliable "ICU####" pseudonymized patient code, unique
        and populated for all 430 rows. Used here as the canonical
        `patient_id`.
      * `KCDC_no` (e.g. "A0824Gxxxx") is the sequencing-batch id, unique and
        populated for all 430 rows. Used here as the sample-matrix join key.
      * `ID` holds a distinct raw numeric value (e.g. "<7-digit number, redacted>") that looks
        like a hospital chart/record number (potential PHI) — NEVER used.
      * `list`/`sample_list` are UNRELIABLE: for 180/430 rows they mirror
        `No` (ICU####), but for the other 250/430 rows they instead hold the
        KCDC_no-style id — i.e. they silently switch schemes row-to-row.
        NEVER used as a join key.
  - merge/output/* view matrices key columns on a `sample_id` that is USUALLY
    the KCDC_no with a "_WM" suffix appended (e.g. "A0824Gxxxx_WM"), but for
    a subset of samples the column is instead the bare `No` scheme with NO
    suffix (e.g. "ICUxxxx"). Both forms co-occur in the same file (confirmed
    in rgi_drug_class_per_sample.tsv, rgi_contig_drug_class_per_sample.tsv).
  - AGI/non_AGI sample_ids in merge/output are the public run/sample
    accessions (CRR######, SRR########) — these are matched against
    AGI_public.csv (Run) and publicdata.csv (run_accession) respectively in
    script 07, not here (this script covers the Gangnam/ICU layer).

Matching rule (explicit, applied in this priority order; never guessed):
  1. strip "_WM" suffix from the view's sample_id -> compare to CLINICAL_CSV.KCDC_no
  2. if no match, compare the raw (unstripped) sample_id -> compare to CLINICAL_CSV.No
  3. otherwise: unmatched

Outputs:
  harmonized/sample_id_mapping.csv       (long format: one row per
                                           view x sample_id, with match info)
  reports/sample_id_harmonization_qc.csv (per-view match-rate summary,
                                           spec sec. 16 fields)
  reports/unmatched_sample_ids.csv       (listed separately, never discarded)
"""
from __future__ import annotations

import pandas as pd

from common import (
    CLINICAL_CSV, SAMPLE_METADATA_TSV, HARMONIZED_DIR, REPORTS_DIR,
    METAPHLAN_DIR, HUMANN_DIR, COVERM_DIR, EGGNOG_DIR, ANTISMASH_DIR,
    RGI_MAG_DIR, RGI_CONTIG_DIR, RPKM_RGI_DIR, VFDB_DIR, GTDB_DIR,
    strip_wm_suffix,
)

clin = pd.read_csv(CLINICAL_CSV, dtype=str)
kcdc_to_patient = dict(zip(clin["KCDC_no"], clin["No"]))
id_to_patient = dict(zip(clin["No"], clin["No"]))  # No is its own patient key (ICU#### scheme)

sm = pd.read_csv(SAMPLE_METADATA_TSV, sep="\t", dtype=str)
sm_icu = sm[sm["dataset"] == "ICU"]
sm_id_to_row = sm_icu.set_index("sample_id").to_dict("index")


def header_cols(path, sep="\t"):
    with open(path) as f:
        return f.readline().rstrip("\n").split(sep)


VIEWS = {
    "MetaPhlAn": (METAPHLAN_DIR / "metaphlan3_species_merged.tsv", "clade_name"),
    "HUMAnN3": (HUMANN_DIR / "humann_pathabundance_wide.unstratified.tsv", "# Pathway"),
    "MAG_abundance_CoverM": (COVERM_DIR / "coverm_abundance_matrix.tsv", "MAG"),
    "eggNOG": (EGGNOG_DIR / "eggnog_cog_per_sample.tsv", "feature"),
    "antiSMASH": (ANTISMASH_DIR / "antismash_bgc_total_per_sample.tsv", "sample"),  # sample-row oriented
    "MAG_RGI": (RGI_MAG_DIR / "rgi_drug_class_per_sample.tsv", "feature"),
    "contig_RGI": (RGI_CONTIG_DIR / "rgi_contig_drug_class_per_sample.tsv", "feature"),
    "RPKM_RGI_contig": (RPKM_RGI_DIR / "rpkm_rgi_drug_class_per_sample.tsv", "feature"),
    "VFDB": (VFDB_DIR / "vfdb_category_per_sample.tsv", "feature"),
    "GTDB_species_abundance": (GTDB_DIR / "gtdb_species_abundance_matrix.tsv", "species"),
}

long_rows = []
qc_rows = []

for view_name, (path, id_col) in VIEWS.items():
    if view_name == "antiSMASH":
        # sample-row oriented: first column IS the sample id list
        col = pd.read_csv(path, sep="\t", dtype=str)["sample"].tolist()
    else:
        col = [c for c in header_cols(path) if c != id_col]

    n_source = len(col)
    n_matched_kcdc = n_matched_id = n_unmatched = 0
    for sample_id in col:
        stripped = strip_wm_suffix(sample_id)
        if stripped in kcdc_to_patient:
            patient_id = kcdc_to_patient[stripped]
            method = "kcdc_no_strip_wm"
            n_matched_kcdc += 1
        elif sample_id in id_to_patient:
            patient_id = id_to_patient[sample_id]
            method = "id_raw"
            n_matched_id += 1
        else:
            patient_id = None
            method = "unmatched"
            n_unmatched += 1
        long_rows.append({
            "view": view_name, "sample_id_in_view": sample_id,
            "matched_kcdc_no_or_id": stripped if method == "kcdc_no_strip_wm" else sample_id,
            "patient_id_clinical": patient_id, "match_method": method,
        })

    n_matched = n_matched_kcdc + n_matched_id
    qc_rows.append({
        "view": view_name, "source_file": str(path),
        "n_unique_source_ids": n_source,
        "n_matched": n_matched, "n_unmatched": n_unmatched,
        "match_pct": round(100 * n_matched / n_source, 1) if n_source else float("nan"),
        "n_matched_via_kcdc_no": n_matched_kcdc, "n_matched_via_id": n_matched_id,
        "n_duplicate_source_ids": n_source - len(set(col)),
    })

mapping_df = pd.DataFrame(long_rows)
mapping_df.to_csv(HARMONIZED_DIR / "sample_id_mapping.csv", index=False)

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(REPORTS_DIR / "sample_id_harmonization_qc.csv", index=False)

unmatched = mapping_df[mapping_df["match_method"] == "unmatched"]
unmatched.to_csv(REPORTS_DIR / "unmatched_sample_ids.csv", index=False)

# many-to-one check: does any single sample_id map into >1 distinct patient
# across duplicated header entries within the same file (data integrity)?
dup_check = (
    mapping_df.dropna(subset=["patient_id_clinical"])
    .groupby(["view", "sample_id_in_view"])["patient_id_clinical"].nunique()
)
n_ambiguous = int((dup_check > 1).sum())

print("[02] per-view match rates:")
print(qc_df[["view", "n_unique_source_ids", "n_matched", "n_unmatched", "match_pct"]].to_string(index=False))
print(f"[02] ambiguous (sample_id -> >1 patient) rows: {n_ambiguous}")
print(f"[02] total unmatched rows written to reports/unmatched_sample_ids.csv: {len(unmatched)}")
