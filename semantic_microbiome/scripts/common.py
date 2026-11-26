"""
Shared configuration and helpers for the semantic_microbiome Phase-1 audit pipeline.

All paths are absolute and documented here so every downstream script uses the
same authoritative sources. Nothing in this pipeline writes to, renames, or
deletes any existing raw/analysis file — outputs only go under
semantic_microbiome/{inventory,harmonized,qc,reports,structured}/.

Random seed: 2026 (per project instructions). Most Phase-1 operations are
deterministic and do not actually consume randomness.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

SEED = 2026

# Phase 2 (structured representation) defaults. Outcome-blind: never
# adjusted based on mortality. Documented here so every structuring script
# uses the same rule.
PREVALENCE_THRESHOLD = 0.10  # keep a feature if present (>0) in >=10% of a view's cohort samples

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------
ICU_PUBLIC = Path("/home/tcmhwd/microbiome/ICU_public")
MICROBIOME = Path("/home/tcmhwd/microbiome")
SEMANTIC = ICU_PUBLIC / "semantic_microbiome"

INVENTORY_DIR = SEMANTIC / "inventory"
HARMONIZED_DIR = SEMANTIC / "harmonized"
QC_DIR = SEMANTIC / "qc"
REPORTS_DIR = SEMANTIC / "reports"
STRUCTURED_DIR = SEMANTIC / "structured"
CLUSTERING_DIR = SEMANTIC / "clustering"
FIGURES_DIR = CLUSTERING_DIR / "figures"
EMBEDDINGS_DIR = SEMANTIC / "embeddings"

for d in (INVENTORY_DIR, HARMONIZED_DIR, QC_DIR, REPORTS_DIR, STRUCTURED_DIR, CLUSTERING_DIR, FIGURES_DIR, EMBEDDINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str, dpi: int = 200, figures_dir: Path = None) -> None:
    """Save a matplotlib figure as BOTH .svg and .jpg (per user request
    2026-08-13: '모든 figure는 svg, jpg 2가지 방식으로'). Defaults to
    clustering/figures/; pass figures_dir= for a different run's figures
    (e.g. clustering/core_only/figures/)."""
    d = figures_dir or FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(d / f"{name}.jpg", format="jpg", dpi=dpi, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Authoritative source files
# ---------------------------------------------------------------------------
# Gangnam Severance clinical/sample metadata. Confirmed with the user
# (2026-08-13 chat) to be the current authoritative file. NOTE: an older file
# exists at ICU_PUBLIC/microbiome.clinical.csv (mtime 2026-07-13, 490KB, 549
# rows via a different lineage) that DIFFERS from this one (mtime 2026-08-13,
# 236KB, 430 rows). The old root-level copy is NOT used by this pipeline and
# should be treated as superseded/stale for semantic_microbiome purposes.
CLINICAL_CSV = ICU_PUBLIC / "metadata" / "microbiome.clinical.csv"
CLINICAL_CSV_STALE_COPY = ICU_PUBLIC / "microbiome.clinical.csv"

PUBLICDATA_CSV = ICU_PUBLIC / "metadata" / "publicdata.csv"   # non-AGI (EBI/SRA)
AGI_PUBLIC_CSV = ICU_PUBLIC / "metadata" / "AGI_public.csv"   # AGI (CNCB/NGDC, BGISEQ)
# Added by user 2026-08-13: per-BioProject clinical/outcome sheets. Sheet
# "PRJNA842425" has patient_ID (== publicdata.csv sample_alias, "covid_###")
# + mortality (0/1) + mortality_label (alive/deceased) -- confirms the
# 39-survivor/32-death split already present in merge/output/sample_metadata.tsv.
# Sheet "PRJCA011331" (== AGI's BioProject) is present but EMPTY -- AGI
# mortality still comes from AGI_public.csv Title S###/D### parsing only.
# Sheets "PRJNA1100796"/"PRJNA1102991" are supplementary description tables,
# not needed (user's rule: both studies are 100% death).
METADATA_XLSX = ICU_PUBLIC / "metadata" / "metadata.xlsx"

# Already-merged, cross-cohort (ICU+AGI+non_AGI) per-view feature matrices,
# built and maintained by the separate merge/ pipeline (merge/scripts/*).
# Reused here as the raw input for the 8-view inventory per user decision
# (2026-08-13 chat: "merge/output/ 재사용 (권장)"). We do NOT reuse anything
# under merge/downstream_analysis/ (a different, outcome-aware R pipeline).
MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
SAMPLE_METADATA_TSV = MERGE_OUTPUT / "sample_metadata.tsv"  # dataset/patient_id/mortality registry, 887 rows

METAPHLAN_DIR = MERGE_OUTPUT / "metaphlan3"
HUMANN_DIR = MERGE_OUTPUT / "humann3"
MAG_SUMMARY_DIR = MERGE_OUTPUT / "mag_summary"
COVERM_DIR = MAG_SUMMARY_DIR / "coverm"
EGGNOG_DIR = MAG_SUMMARY_DIR / "eggnog"
ANTISMASH_DIR = MAG_SUMMARY_DIR / "antismash"
RGI_MAG_DIR = MAG_SUMMARY_DIR / "rgi"
RGI_CONTIG_DIR = MAG_SUMMARY_DIR / "rgi_contig"
RPKM_RGI_DIR = MAG_SUMMARY_DIR / "rpkm_rgi"
VFDB_DIR = MAG_SUMMARY_DIR / "vfdb"
GTDB_DIR = MAG_SUMMARY_DIR / "gtdb"
R_READY_DIR = MERGE_OUTPUT / "R_ready"

# Raw eggNOG annotation TSVs (per-sample, pre-pivot) — kept as reference to
# the authoritative raw annotation source per project instructions (sec. 9).
EGGNOG_RAW_DIR = ICU_PUBLIC / "EggNOG_result"

# dRep dereplication mapping (cohort-wide: ICU+AGI+non_AGI MAGs pooled).
DREP_DIR = ICU_PUBLIC / "cohort_drep"
DREP_CDB_CSV = DREP_DIR / "data_tables" / "Cdb.csv"   # genome -> secondary_cluster (all MAGs)
DREP_WDB_CSV = DREP_DIR / "data_tables" / "Wdb.csv"   # genome -> cluster (winners = representatives)
DREP_SDB_CSV = DREP_DIR / "data_tables" / "Sdb.csv"

# Suspected prompt-injection file. Left untouched per user decision
# (2026-08-13 chat: "그대로 두기"). Never read as instructions by this pipeline.
SUSPECT_INJECTION_FILE = (
    ICU_PUBLIC / "merge" / "downstream_analysis" / "URGENT_COHORT_CORRECTION_20260715.md"
)

# ---------------------------------------------------------------------------
# ID transformation rules (documented explicitly per sec. 15/16 requirement)
# ---------------------------------------------------------------------------
# merge/output/* per-view matrices use a "sample_id" that is usually the
# clinical KCDC_no with a "_WM" suffix appended (e.g. clinical KCDC_no
# "A0824Gxxxx" <-> merge sample_id "A0824Gxxxx_WM"). Some ICU sample_ids in
# sample_metadata.tsv instead follow the clinical "ID" scheme (e.g. "ICUxxxx")
# with no suffix. Both are tried, in that order; anything left over is
# reported unmatched rather than guessed.
def strip_wm_suffix(sample_id: str) -> str:
    return sample_id[:-3] if sample_id.endswith("_WM") else sample_id


def today_str() -> str:
    return datetime.date.today().isoformat()


def run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def load_cohort_matrix(view_name: str, path: Path, id_col: str, mapping_df, cohort_df, sep: str = "\t"):
    """
    Load a merge/output/* feature x sample matrix, restrict columns to the
    Gangnam primary cohort's matched sample_ids for `view_name` (per
    harmonized/sample_id_mapping.csv), and return (patients x features)
    with patient_id_clinical as the index. Values coerced numeric, NaN->0.
    Only used for Phase 2 (structured representation) scripts.
    """
    import pandas as pd

    m = mapping_df[(mapping_df.view == view_name) & (mapping_df.match_method != "unmatched")]
    m = m[m.patient_id_clinical.isin(set(cohort_df["patient_id_clinical"]))]
    sid_to_patient = dict(zip(m["sample_id_in_view"], m["patient_id_clinical"]))
    sample_ids = list(sid_to_patient.keys())

    df = pd.read_csv(path, sep=sep, usecols=lambda c: c in set(sample_ids) | {id_col}, dtype={id_col: str})
    df = df.set_index(id_col).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df.rename(columns=sid_to_patient)  # sample_id -> patient_id_clinical
    return df.T  # patients x features


def prevalence_filter(patients_by_features, threshold: float = PREVALENCE_THRESHOLD):
    """Keep columns (features) detected (>0) in >= threshold fraction of rows (patients)."""
    n = len(patients_by_features)
    keep = (patients_by_features > 0).sum(axis=0) >= (threshold * n)
    return patients_by_features.loc[:, keep], int(keep.sum()), int((~keep).sum())


def gangnam_id_map(view_name: str, gangnam_ids=None, min_resolve_frac: float = 0.95):
    """Return {raw_sample_id_in_view -> patient_id_clinical} for one merge/
    output/* view, via harmonized/sample_id_mapping.csv -- the CANONICAL
    Gangnam id-mapping (this is what lit08_build_literature_core_extended.py
    uses). Added 2026-09-02 (P1 corrective pass, fable-auditor blast-radius
    audit item: "Gangnam id-mapping divergence") as a SHARED helper for the
    pattern that was independently reimplemented (correctly by lit08, but
    NOT applied at all -- silently reindexing raw tables by
    patient_id_clinical instead -- by at least one other script, lit29,
    fixed in this same pass) in >=6 scripts across the pipeline. NOT wired
    into any existing script by this pass (too risky mid-correction); new
    scripts should prefer this over a private reimplementation.

    Raw merge/output/* tables key most ICU samples by a "*_WM"-style id;
    only ~110/302 Gangnam patient_id_clinical values match a raw column
    directly. Renaming raw columns via this map BEFORE any reindex avoids
    silently NaN-ing (and, if summed without min_count=1, zeroing) the
    other ~192 patients.

    gangnam_ids: optional iterable to restrict/validate against (e.g.
    harmonized/discovery_cohort_primary.csv's patient_id_clinical column);
    if given, asserts at least `min_resolve_frac` of them resolve to a raw
    id for this view (raises AssertionError otherwise -- catches a view
    name typo or an unexpectedly sparse view before it silently zeroes
    most of the cohort, exactly the failure mode this helper exists to
    prevent).
    """
    import pandas as pd

    sid_map = pd.read_csv(HARMONIZED_DIR / "sample_id_mapping.csv")
    m = sid_map[(sid_map.view == view_name) & (sid_map.match_method != "unmatched")]
    if gangnam_ids is not None:
        gangnam_ids = list(gangnam_ids)
        m = m[m.patient_id_clinical.isin(set(gangnam_ids))]
        resolved_frac = len(m) / max(len(gangnam_ids), 1)
        assert resolved_frac >= min_resolve_frac, (
            f"gangnam_id_map({view_name!r}): only {len(m)}/{len(gangnam_ids)} "
            f"({resolved_frac:.1%}) requested Gangnam ids resolved a raw sample id "
            f"(required >= {min_resolve_frac:.0%}) -- check the view name and "
            f"harmonized/sample_id_mapping.csv coverage before proceeding"
        )
    return dict(zip(m.sample_id_in_view, m.patient_id_clinical))


def normalized_humann_pathway_block(ids, frozen_cols=None):
    """Build a HUMAnN3 pathway block normalized EXACTLY like
    09_build_structured_views.py's internal (Gangnam) block: drop
    UNMAPPED/UNINTEGRATED, divide each sample's remaining pathway RPK by
    that sample's own sum (rows sum to ~1.0), reindex to `frozen_cols`
    (defaults to the full frozen 429-column list from
    structured/humann_pathway_filtered.tsv).

    Added 2026-09-02 (P1 corrective pass, fable-auditor blast-radius audit
    item 21/P1: "Function-view external HUMAnN3 normalization bug"). Mirrors
    lit43_view_external_normalization_fix.py's transform line-for-line (that
    script mirrors it because 09_build_structured_views.py executes
    top-level on import and cannot be imported directly). Every script that
    built an EXTERNAL HUMAnN3 block from raw RPK instead of this transform
    scored frozen models >2e5x out of their training scale (lit15, lit18,
    lit25, lit26, lit28, lit29-adjacent/lit30/lit31 all independently
    reimplemented the "frozen Gangnam columns x raw external values"
    pattern without it -- see revision_log.md issue 21/P1 and this P1 pass's
    lit46/lit47/lit48 corrections). NOT wired into any existing script by
    this pass (too risky mid-correction); new scripts should call this
    instead of reimplementing the raw-RPK read.

    ids: sample ids to build the block for (Gangnam patient_id_clinical or
    raw public sample_id -- this function does not do any id-renaming;
    combine with gangnam_id_map() above first if Gangnam ids are needed and
    the raw table's columns are still raw sample ids).
    """
    import numpy as np
    import pandas as pd

    if frozen_cols is None:
        frozen_cols = pd.read_csv(STRUCTURED_DIR / "humann_pathway_filtered.tsv", sep="\t",
                                   index_col="patient_id", nrows=0).columns.tolist()
    raw = pd.read_csv(HUMANN_DIR / "humann_pathabundance_wide.unstratified.tsv", sep="\t", index_col="# Pathway")
    special = [c for c in raw.index if c in ("UNMAPPED", "UNINTEGRATED")]
    raw_bio = raw.drop(index=special)
    sample_total = raw_bio.sum(axis=0)
    rel = raw_bio.div(sample_total.replace(0, np.nan), axis=1).fillna(0.0).T  # sample x pathway, sums to ~1.0
    return rel.reindex(index=list(ids), columns=frozen_cols)


def write_software_versions(path: Path) -> None:
    import numpy
    import pandas

    lines = [
        f"generated: {datetime.datetime.now().isoformat()}",
        f"python: {sys.version}",
        f"pandas: {pandas.__version__}",
        f"numpy: {numpy.__version__}",
        f"seed: {SEED}",
        "",
        "-- tool/database versions identifiable from repo (best-effort) --",
    ]
    path.write_text("\n".join(lines) + "\n")
