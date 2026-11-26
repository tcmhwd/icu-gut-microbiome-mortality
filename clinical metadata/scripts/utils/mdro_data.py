"""mdro_data.py — shared data-loading/definitions for the MDRO surveillance
analysis (scripts 51/52, user request 2026-09-02, "Task B unblocked").

Population: the 430 patients in `metadata/microbiome.clinical.csv` that
already link to the 466-patient association-arm cohort (established in
`phase_x_surveillance_feasibility.md`, script 50). This is NOT the file
`clinical metadata/ICU.csv` (confirmed unlinkable, script 50) -- it is the
file CLAUDE.md's "patient_id minus ICU_ prefix -> ID" join note actually
describes, and the same file that already supplies SOFA throughout Phase X.

TEMPORAL ANCHOR (data-owner-confirmed, 2026-09-02, not documented in the
file itself -- recorded here per that instruction): `post.*` flags are
STRICTLY AFTER the microbiome sample collection date; `pre.*` flags are the
same day or before. This is what makes a genuine prediction analysis
(pre-negative -> post-positive = new acquisition, observed strictly after
the admission gut sample) possible.

VRE DUPLICATE-COLUMN RECONCILIATION (raw CSV literally has 2 columns each
named `pre.VRE`/`post.VRE`; pandas auto-suffixes the second occurrence
`.1`). Resolved computationally, not guessed:
  - `pre.VRE`/`post.VRE` (first occurrence): pre.VRE is 0 for all 430
    patients; post.VRE reproduces the `VREFM` summary column EXACTLY
    (430/430 match) -- this is the E. FAECIUM-SPECIFIC VRE definition.
  - `pre.VRE.1`/`post.VRE.1` (second occurrence, pandas-suffixed): reproduces
    the `VRE` summary column EXACTLY (430/430 match, i.e. VRE == pre.VRE.1 |
    post.VRE.1) -- this is the BROADER/any-Enterococcus VRE definition.
  General pattern confirmed across ALL organism blocks in this file: the
  un-prefixed summary column (CRE, CDI, CRKP, CRAB, CRPA, thirdGCEC,
  MRSA.inf, VRE, VREFM) == pre.X | post.X exactly (430/430 match for every
  one of the 7 non-VRE organisms checked). This independently corroborates
  the VRE reconciliation above (same pattern holds for the column literally
  named `VRE`).
  DECISION: use the broader `pre.VRE.1`/`post.VRE.1` ("VRE_broad") pair as
  the primary Enterococcus/VRE definition -- it is the one whose summary
  column is literally named `VRE`, and it clears the >=10-acquisition-event
  bar (11) while VREFM (4) does not. VREFM is reported alongside as an
  explicit alternate/secondary definition, never silently dropped.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
MC_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")
GENUS_TSV = os.path.join(MICROBIOME_BASE, "merge", "output", "metaphlan3", "metaphlan3_genus_merged.tsv")
RGI_CONTIG_ARO_TSV = os.path.join(MICROBIOME_BASE, "merge", "output", "mag_summary", "rgi_contig",
                                   "rgi_contig_aro_per_sample.tsv")
RGI_MAG_ARO_TSV = os.path.join(MICROBIOME_BASE, "merge", "output", "mag_summary", "rgi", "rgi_aro_per_sample.tsv")
RGI_CONTIG_FAMILY_TSV = os.path.join(MICROBIOME_BASE, "merge", "output", "mag_summary", "rgi_contig",
                                      "rgi_contig_amr_gene_family_per_sample.tsv")
RGI_MAG_FAMILY_TSV = os.path.join(MICROBIOME_BASE, "merge", "output", "mag_summary", "rgi",
                                   "rgi_amr_gene_family_per_sample.tsv")

# ---------------------------------------------------------------------------
# organism -> (pre column, post column, summary column) after VRE reconciliation
# ---------------------------------------------------------------------------
ORGANISM_PREPOST = {
    "CRE":        ("pre.CRE", "post.CRE", "CRE"),
    "CRKP":       ("pre.CRKP", "post.CRKP", "CRKP"),
    "thirdGCEC":  ("pre.thirdGCEC", "post.thirdGCEC", "thirdGCEC"),
    "CRPA":       ("pre.CRPA", "post.CRPA", "CRPA"),
    "MRSA":       ("pre.MRSA", "post.MRSA", "MRSA.inf"),
    "VRE":        ("pre.VRE.1", "post.VRE.1", "VRE"),          # broad definition, PRIMARY
    "VREFM":      ("pre.VRE", "post.VRE", "VREFM"),             # E. faecium-specific, SECONDARY
}

MIN_ACQUISITION_EVENTS = 10  # task's own stated bar (pre=0/post=1 count)

# ---------------------------------------------------------------------------
# organism -> corresponding gut genus (read-based, MetaPhlAn3 genus-level %)
# ---------------------------------------------------------------------------
ORGANISM_GENUS = {
    "CRKP": ["Klebsiella"],
    "thirdGCEC": ["Escherichia"],
    "CRE": ["__ENTEROBACTERALES_ALL__"],   # composite: all g__ clades under o__Enterobacterales
    "VRE": ["Enterococcus"],
    "VREFM": ["Enterococcus"],
    "MRSA": ["Staphylococcus"],
    "CRPA": ["Pseudomonas"],
}
TAXONOMY_THRESHOLDS_PCT = [0, 10, 30]  # 0 = presence/absence at profiler detection limit

# ---------------------------------------------------------------------------
# organism -> resistome gene definition, per the task's explicit mapping.
# Two levels used: AMR-GENE-FAMILY (CARD-curated grouping, used where the
# whole family is phenotype-relevant) and raw ARO allele name (used for
# vanA/vanB and mecA/mecC exactly as the task named them, and for CTX-M/
# SHV/TEM where family-level would over-include narrow-spectrum alleles).
# ---------------------------------------------------------------------------
CARBAPENEMASE_FAMILIES = ["KPC beta-lactamase", "NDM beta-lactamase", "OXA-48-like beta-lactamase",
                          "VIM beta-lactamase", "IMP beta-lactamase"]
# VIM/IMP beta-lactamase families are NOT even present among the 151/819
# families/alleles detected in ANY sample in this cohort (checked, not
# assumed) -- reported as zero-detection, not silently omitted.

MECA_MECC_ARO = ["mecA", "mecC"]  # mecC never detected in this cohort (checked)
VAN_ARO = ["vanA", "vanB"]        # task's explicit gene set (not vanC/D/G, which are often intrinsic/non-VRE)


def _esbl_aro_names(all_feature_names: list[str]) -> list[str]:
    """CTX-M (any allele) + SHV/TEM EXCLUDING the ancestral narrow-spectrum
    reference alleles SHV-1/TEM-1 (well-established convention: virtually all
    numbered SHV/TEM derivatives beyond -1 are ESBL variants; -1 itself is
    the non-ESBL ancestral penicillinase). CARD's AMR-Gene-Family level does
    NOT distinguish ESBL vs narrow-spectrum within SHV/TEM, which is why this
    needs allele-level matching rather than the family table used for
    carbapenemases."""
    out = []
    for f in all_feature_names:
        if re.match(r"^CTX-M-\d+$", f):
            out.append(f)
        elif re.match(r"^(SHV|TEM)-\d+$", f) and f not in ("SHV-1", "TEM-1"):
            out.append(f)
    return sorted(out)


ORGANISM_RESISTOME_LEVEL = {
    # organism -> ("family" or "aro", list-or-None [None = compute via _esbl_aro_names])
    "CRE": ("family", CARBAPENEMASE_FAMILIES),
    "CRKP": ("family", CARBAPENEMASE_FAMILIES),
    "CRPA": ("family", CARBAPENEMASE_FAMILIES),  # EXTENSION beyond task's explicit list -- see caveat below
    "thirdGCEC": ("aro", None),  # resolved via _esbl_aro_names at load time
    "VRE": ("aro", VAN_ARO),
    "VREFM": ("aro", VAN_ARO),
    "MRSA": ("aro", MECA_MECC_ARO),
}
CRPA_GENE_CAVEAT = (
    "The task brief's explicit gene-family mapping (carbapenemase->CRE/CRKP, ESBL->thirdGCEC, "
    "van->VRE, mecA/mecC->MRSA) did not specify a gene set for CRPA. Carbapenem-resistant "
    "P. aeruginosa is PREDOMINANTLY chromosomally mediated (efflux pump upregulation, OprD porin "
    "loss) rather than horizontally-acquired-carbapenemase-mediated in most strains, so the same "
    "carbapenemase-family marker used for CRE/CRKP is applied here as a reasonable extension but is "
    "expected to have LOW SENSITIVITY specifically for CRPA (most true CRPA cases will be "
    "resistome-gene-negative by this marker, since their resistance mechanism is not a detectable "
    "acquired gene at all). Reported for completeness, not as a validated CRPA-specific marker."
)


def reconcile_vre_and_verify(mc: pd.DataFrame, log=None) -> dict:
    """Verify the VRE duplicate-column resolution documented in the module
    docstring, computationally, on the actual loaded frame. Returns a dict of
    the verification facts (used in the report). Raises if the expected
    pattern does not hold (never silently assumes)."""
    vpre = pd.to_numeric(mc["pre.VRE"], errors="coerce")
    vpost = pd.to_numeric(mc["post.VRE"], errors="coerce")
    vpre1 = pd.to_numeric(mc["pre.VRE.1"], errors="coerce")
    vpost1 = pd.to_numeric(mc["post.VRE.1"], errors="coerce")
    vrefm = pd.to_numeric(mc["VREFM"], errors="coerce")
    vre = pd.to_numeric(mc["VRE"], errors="coerce")

    facts = {
        "pre_VRE_all_zero": bool((vpre == 0).all()),
        "post_VRE_eq_VREFM_n_match": int((vpost == vrefm).sum()),
        "n_total": len(mc),
        "prepost1_or_eq_VRE_n_match": int((((vpre1 == 1) | (vpost1 == 1)).astype(int) == vre.fillna(0).astype(int)).sum()),
        "n_pre_VRE_broad_pos": int((vpre1 == 1).sum()),
        "n_post_VRE_broad_pos": int((vpost1 == 1).sum()),
        "n_pre_VREFM_pos": int((vpre == 1).sum()),
        "n_post_VREFM_pos": int((vpost == 1).sum()),
    }
    ok = (facts["post_VRE_eq_VREFM_n_match"] == facts["n_total"]
          and facts["prepost1_or_eq_VRE_n_match"] == facts["n_total"]
          and facts["pre_VRE_all_zero"])
    if not ok:
        raise AssertionError(f"VRE duplicate-column reconciliation FAILED verification: {facts}")
    if log:
        log.info("VRE duplicate-column reconciliation verified: %s", facts)
    return facts


def load_clinical() -> pd.DataFrame:
    mc = pd.read_csv(MC_CSV, dtype=str)
    return mc


def acquisition_event_table(mc: pd.DataFrame) -> pd.DataFrame:
    """pre=0/post=1 (new acquisition) counts, plus the other 3 quadrants,
    for every organism in ORGANISM_PREPOST."""
    rows = []
    for organism, (pre_c, post_c, summ_c) in ORGANISM_PREPOST.items():
        vpre = pd.to_numeric(mc[pre_c], errors="coerce")
        vpost = pd.to_numeric(mc[post_c], errors="coerce")
        n_eval = int((vpre.notna() & vpost.notna()).sum())
        rows.append({
            "organism": organism,
            "n_evaluable": n_eval,
            "n_pre_positive_total": int((vpre == 1).sum()),
            "n_post_positive_total": int((vpost == 1).sum()),
            "n_pre0_post0": int(((vpre == 0) & (vpost == 0)).sum()),
            "n_pre0_post1_ACQUISITION": int(((vpre == 0) & (vpost == 1)).sum()),
            "n_pre1_post0": int(((vpre == 1) & (vpost == 0)).sum()),
            "n_pre1_post1_persistence": int(((vpre == 1) & (vpost == 1)).sum()),
            "n_pre_negative_evaluable": int((vpre == 0).sum()),
            "meets_10_acquisition_event_bar": int(((vpre == 0) & (vpost == 1)).sum()) >= MIN_ACQUISITION_EVENTS,
        })
    return pd.DataFrame(rows)


def load_genus_table() -> pd.DataFrame:
    """sample_list-indexed genus relative-abundance (%) table for the 5 named
    genera + an Enterobacterales composite (sum of all g__ clades under
    o__Enterobacterales), read-based MetaPhlAn3."""
    gt = pd.read_csv(GENUS_TSV, sep="\t")
    gt = gt.set_index("clade_name")
    named = {}
    for genus in ["Klebsiella", "Escherichia", "Enterococcus", "Staphylococcus", "Pseudomonas"]:
        rows = [c for c in gt.index if f"g__{genus}" in c]
        assert len(rows) == 1, f"expected exactly 1 clade row for g__{genus}, got {rows}"
        named[genus] = gt.loc[rows[0]]
    entero_rows = [c for c in gt.index if "o__Enterobacterales" in c]
    named["__ENTEROBACTERALES_ALL__"] = gt.loc[entero_rows].sum(axis=0)
    out = pd.DataFrame(named)
    out.index.name = "sample_list"
    return out


def _load_table(path: str) -> pd.DataFrame:
    """Load a `feature` x sample RGI per-sample table, transpose to
    sample_list-indexed rows x feature columns, coerce to numeric counts."""
    df = pd.read_csv(path, sep="\t").set_index("feature")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.T
    df.index.name = "sample_list"
    return df


def load_resistome_tables(level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """level: 'contig' (PRIMARY -- whole-assembly resistome via
    rgi_contig_*_per_sample, not limited by MAG binning success) or 'mag'
    (SECONDARY sensitivity check -- rgi_*_per_sample, limited to genes
    recovered onto a dereplicated/binned MAG). Returns (aro_table,
    family_table), both sample_list-indexed."""
    if level == "contig":
        return _load_table(RGI_CONTIG_ARO_TSV), _load_table(RGI_CONTIG_FAMILY_TSV)
    elif level == "mag":
        return _load_table(RGI_MAG_ARO_TSV), _load_table(RGI_MAG_FAMILY_TSV)
    raise ValueError(level)


def build_gene_presence_flags(level: str) -> tuple[pd.DataFrame, dict]:
    """Per-organism gene-presence (0/1) flags, sample_list-indexed, for every
    organism in ORGANISM_RESISTOME_LEVEL. Family-level organisms (CRE/CRKP/
    CRPA) use the CARD-curated AMR-gene-family table directly (no manual
    ARO->family regex needed -- RGI already assigned it). ARO-level organisms
    (thirdGCEC/VRE/VREFM/MRSA) use specific allele names. Returns
    (flags_df, detection_summary) where detection_summary records exactly
    which family members / ARO alleles were found >0 in >=1 sample (for the
    report, so "zero detections" is verifiable, not silently absent)."""
    aro, fam = load_resistome_tables(level)
    flags = pd.DataFrame(index=aro.index)
    detection_summary = {}
    for organism, (kind, genes) in ORGANISM_RESISTOME_LEVEL.items():
        if organism == "thirdGCEC":
            genes = _esbl_aro_names(list(aro.columns))
        table = fam if kind == "family" else aro
        present_genes = [g for g in genes if g in table.columns]
        missing_genes = [g for g in genes if g not in table.columns]
        if present_genes:
            flags[organism] = (table[present_genes].sum(axis=1) > 0).astype(int)
        else:
            flags[organism] = 0
        detection_summary[organism] = {
            "kind": kind, "genes_targeted": genes, "genes_present_in_ontology": present_genes,
            "genes_never_detected_in_cohort": missing_genes,
            "n_positive_samples": int(flags[organism].sum()) if present_genes else 0,
        }
    return flags, detection_summary
