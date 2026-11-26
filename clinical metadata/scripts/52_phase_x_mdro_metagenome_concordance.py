#!/usr/bin/env python3
"""52_phase_x_mdro_metagenome_concordance.py

Task B, STEP 2 SECONDARY analysis (user request, 2026-09-02): among ALL
patients (not just the pre-negative subset used for the acquisition
prediction in script 51), how well does the admission gut metagenome agree
with PRE-EXISTING culture-based MDRO status (`pre.*` flags)? This is a
DETECTION/CONCORDANCE question, not a prediction question -- it uses
CONCURRENT data (the gut sample and the `pre.*` culture status reflect the
same point: same day or before, per the data-owner-confirmed temporal
anchor recorded in script 51/`utils/mdro_data.py`), so it CANNOT support a
prediction claim, but it is a legitimate surveillance question in its own
right: "does shotgun metagenomics agree with culture-based screening, and
does it flag anything screening missed?"

Population: all 430 patients in `metadata/microbiome.clinical.csv` (no
pre-negative restriction -- the reference standard here IS pre-status
itself). VREFM (E.-faecium-specific VRE definition) is EXCLUDED from this
analysis: 0/430 patients are pre.VRE-positive under that definition, so
sensitivity is undefined (0/0) -- reported as a note, not silently dropped.

Two predictor families, exactly as script 51's acquisition analysis, but
using the SIMPLEST/most detection-relevant variant of each (not the full
threshold grid -- the concordance question is "does metagenomics detect
what culture already found", which is naturally a presence/absence
comparison, not a domination-threshold comparison):
  (a) TAXONOMIC: genus presence at the profiler's detection limit (>0%).
  (b) RESISTOME: gene presence (contig-level, PRIMARY; MAG-level reported
      as a sensitivity check), same gene mapping as script 51.

Reads:
  metadata/microbiome.clinical.csv
  merge/output/metaphlan3/metaphlan3_genus_merged.tsv
  merge/output/mag_summary/rgi_contig/*_per_sample.tsv
  merge/output/mag_summary/rgi/*_per_sample.tsv
Writes:
  clinical_ai/reports/phase_x_mdro_metagenome_concordance.csv
  clinical_ai/reports/phase_x_mdro_metagenome_concordance.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, mdro_data as M  # noqa: E402

log = io_utils.setup_logging("52_phase_x_mdro_metagenome_concordance")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_mdro_metagenome_concordance.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_mdro_metagenome_concordance.md")

CONCORDANCE_ORGANISMS = ["CRE", "CRKP", "thirdGCEC", "CRPA", "MRSA", "VRE"]  # VREFM excluded, see docstring


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, (centre - adj) / denom), min(1.0, (centre + adj) / denom))


def sens_spec_ppv_npv(ref, pred):
    """ref, pred: 0/1 arrays (ref = pre.X culture status, pred = metagenome
    flag). Returns a dict with counts + Wilson-CI point estimates."""
    tp = int(((ref == 1) & (pred == 1)).sum())
    fn = int(((ref == 1) & (pred == 0)).sum())
    fp = int(((ref == 0) & (pred == 1)).sum())
    tn = int(((ref == 0) & (pred == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    sens = tp / n_pos if n_pos else np.nan
    spec = tn / n_neg if n_neg else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    sens_ci = wilson_ci(tp, n_pos)
    spec_ci = wilson_ci(tn, n_neg)
    ppv_ci = wilson_ci(tp, tp + fp) if (tp + fp) else (np.nan, np.nan)
    npv_ci = wilson_ci(tn, tn + fn) if (tn + fn) else (np.nan, np.nan)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "n_pre_positive": n_pos, "n_pre_negative": n_neg,
        "sensitivity": round(sens, 4) if not np.isnan(sens) else np.nan,
        "sensitivity_CI_low": round(sens_ci[0], 4), "sensitivity_CI_high": round(sens_ci[1], 4),
        "specificity": round(spec, 4) if not np.isnan(spec) else np.nan,
        "specificity_CI_low": round(spec_ci[0], 4), "specificity_CI_high": round(spec_ci[1], 4),
        "PPV": round(ppv, 4) if not np.isnan(ppv) else np.nan,
        "PPV_CI_low": round(ppv_ci[0], 4), "PPV_CI_high": round(ppv_ci[1], 4),
        "NPV": round(npv, 4) if not np.isnan(npv) else np.nan,
        "NPV_CI_low": round(npv_ci[0], 4), "NPV_CI_high": round(npv_ci[1], 4),
        "n_metagenome_flagged_not_cultured": fp,  # what shotgun metagenomics flags that screening missed
    }


# ---------------------------------------------------------------------------
# 1. load + join predictors, exactly as script 51
# ---------------------------------------------------------------------------
mc = M.load_clinical()
vre_facts = M.reconcile_vre_and_verify(mc, log=log)
genus = M.load_genus_table()
flags_contig, detect_contig = M.build_gene_presence_flags("contig")
flags_mag, detect_mag = M.build_gene_presence_flags("mag")

mc = mc.set_index("sample_list", drop=False)
genus_j = genus.reindex(mc.index)
resist_contig_j = flags_contig.reindex(mc.index)
resist_mag_j = flags_mag.reindex(mc.index)

rows = []
for organism in CONCORDANCE_ORGANISMS:
    pre_c, _, _ = M.ORGANISM_PREPOST[organism]
    ref = pd.to_numeric(mc[pre_c], errors="coerce")
    n_pre_pos_total = int((ref == 1).sum())
    log.info("=== %s: pre-positive (culture-reference) N=%d/%d ===", organism, n_pre_pos_total, ref.notna().sum())

    # --- (a) taxonomic: presence at detection limit ---
    genera = M.ORGANISM_GENUS[organism]
    tax = genus_j[genera[0]]
    tax_pred = (tax > 0).astype(int)
    valid = ref.notna() & tax_pred.notna()
    r = sens_spec_ppv_npv(ref[valid].astype(int).values, tax_pred[valid].astype(int).values)
    r.update({"organism": organism, "predictor_family": "taxonomic", "variant": "presence_gt0pct",
              "N_evaluable": int(valid.sum())})
    rows.append(r)

    # --- (b) resistome: gene presence, contig (primary) + MAG (sensitivity) ---
    for level, resist_j in [("contig", resist_contig_j), ("mag", resist_mag_j)]:
        pred = resist_j[organism]
        valid = ref.notna() & pred.notna()
        r = sens_spec_ppv_npv(ref[valid].astype(int).values, pred[valid].astype(int).values)
        r.update({"organism": organism, "predictor_family": f"resistome_{level}", "variant": "gene_presence",
                  "N_evaluable": int(valid.sum())})
        rows.append(r)

out = pd.DataFrame(rows)
col_order = ["organism", "predictor_family", "variant", "N_evaluable", "n_pre_positive", "n_pre_negative",
             "TP", "FN", "FP", "TN", "sensitivity", "sensitivity_CI_low", "sensitivity_CI_high",
             "specificity", "specificity_CI_low", "specificity_CI_high", "PPV", "PPV_CI_low", "PPV_CI_high",
             "NPV", "NPV_CI_low", "NPV_CI_high", "n_metagenome_flagged_not_cultured"]
out = out[col_order]
io_utils.save_csv(out, OUT_CSV)
log.info("wrote %s", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. markdown report
# ---------------------------------------------------------------------------
n_vrefm_pre_pos = int((pd.to_numeric(mc["pre.VRE"], errors="coerce") == 1).sum())

md = [
    "# MDRO metagenome-vs-culture concordance/detection",
    "",
    "> **DETECTION/CONCORDANCE, NOT PREDICTION.** This analysis uses CONCURRENT data -- the admission gut "
    "metagenome vs. `pre.*` culture-based MDRO status, which per the data-owner-confirmed temporal anchor "
    "(recorded in script 51/`utils/mdro_data.py`) reflects the same day or before the sample. It answers a "
    "different question from `phase_x_mdro_acquisition_prediction.md`: not \"does the gut sample predict a "
    "FUTURE event\" but \"does shotgun metagenomics AGREE with culture-based screening, and does it flag "
    "anything screening missed?\" -- a legitimate surveillance question in its own right, but no temporal claim "
    "should be read into these numbers.",
    "",
    "## Population and exclusion",
    "",
    "All 430 patients in `metadata/microbiome.clinical.csv` (no pre-negative restriction -- `pre.*` culture "
    "status IS the reference standard here). **VREFM (E.-faecium-specific VRE definition) is EXCLUDED**: "
    f"{n_vrefm_pre_pos}/430 patients are `pre.VRE`-positive under that definition (see script 51's VRE "
    "duplicate-column reconciliation), so sensitivity is undefined (0/0 denominator) -- noted explicitly, not "
    "silently dropped from the analysis set.",
    "",
    "## Predictors (simplest/most detection-relevant variant of each family, not the full threshold grid used "
    "for the acquisition-prediction analysis)",
    "",
    "- **Taxonomic**: genus presence at the MetaPhlAn3 detection limit (relative abundance > 0%).",
    "- **Resistome**: gene presence, contig-level RGI/CARD output (PRIMARY) and MAG-level (SECONDARY "
    "sensitivity check) -- identical gene-phenotype mapping as `phase_x_mdro_acquisition_prediction.md` "
    "(carbapenemase KPC/NDM/OXA-48-like -> CRE/CRKP/CRPA[extension, see that report's caveat]; ESBL CTX-M + "
    "SHV/TEM excluding SHV-1/TEM-1 -> thirdGCEC; vanA/vanB -> VRE; mecA[mecC never detected] -> MRSA).",
    "",
    "## Results",
    "",
    df_to_md_table(out[["organism", "predictor_family", "variant", "N_evaluable", "n_pre_positive", "TP", "FN",
                        "FP", "TN"]]),
    "",
    df_to_md_table(out[["organism", "predictor_family", "sensitivity", "sensitivity_CI_low", "sensitivity_CI_high",
                        "specificity", "specificity_CI_low", "specificity_CI_high"]]),
    "",
    df_to_md_table(out[["organism", "predictor_family", "PPV", "PPV_CI_low", "PPV_CI_high", "NPV", "NPV_CI_low",
                        "NPV_CI_high", "n_metagenome_flagged_not_cultured"]]),
    "",
    "## Interpretation",
    "",
    "**Consistent pattern across organisms: taxonomic genus-presence is a high-sensitivity/low-specificity "
    "screen; the resistome gene-presence marker is lower-sensitivity but much higher-specificity/PPV.** Genus "
    "presence at the detection limit picks up common gut commensals regardless of MDRO status (specificity as "
    "low as 10-25% for CRE/thirdGCEC/VRE), so a positive taxonomic flag alone is a weak indicator of culture-"
    "confirmed MDRO status. The resistome gene marker is more conservative but far more specific -- e.g. CRE: "
    "contig-resistome sensitivity 33.3% but specificity 99.7% and PPV 90.9% (10/11 gene-positive patients were "
    "also culture-CRE-positive); CRKP: gene-presence sensitivity 36.4%, specificity 98.3%, PPV 36.4% (4 TP vs "
    "7 FP -- the sensitivity/PPV equality here is coincidental to this specific 2x2 table, not a general "
    "property). This mirrors exactly the pattern in the acquisition-prediction analysis "
    "(`phase_x_mdro_acquisition_prediction.md`): the resistome predictor targets the resistance determinant "
    "directly (rather than genus abundance) and is empirically the stronger signal of the two families. "
    "**Qualification (claim-strength correction, 2026-09-02, task item 5)**: for VRE specifically this should "
    "not be described as mechanistically direct/tight -- vanA/vanB gene detection in the metagenome is not "
    "linked to an Enterococcus MAG or to the organism subsequently cultured (no isolate-level genotype "
    "confirmation).",
    "",
    "**Precision caveat**: several organisms have very small `n_pre_positive` (culture-reference-positive) "
    "denominators (CRPA=5, VRE=4, MRSA=7, thirdGCEC=9) -- sensitivity estimates for these carry wide Wilson "
    "CIs and should be read as hypothesis-generating, not as validated diagnostic performance. CRE (30) and "
    "CRKP (11) have the most stable denominators.",
    "",
    "NPV is uniformly high across organisms/predictors (expected given low pre-prevalence -- a negative "
    "metagenome result is a weak rule-out on its own in a low-prevalence population; this reflects prevalence, "
    "not a strong negative-predictive property of the assay). `n_metagenome_flagged_not_cultured` (FP column) "
    "is the count of patients the gut metagenome flags as gene/genus-positive despite a NEGATIVE culture-based "
    "`pre.*` status -- these are the patients most relevant to the \"does metagenomics catch what screening "
    "missed\" framing, though without a true independent reference standard (e.g. a contemporaneous rectal "
    "swab culture) it cannot be determined whether these are false positives of the metagenomic marker or "
    "colonization that culture-based surveillance itself missed.",
    "",
    "No threshold or gene set was altered after seeing these numbers -- identical definitions to "
    "`phase_x_mdro_acquisition_prediction.md`'s presence/gene-presence variants, reused here unchanged.",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
