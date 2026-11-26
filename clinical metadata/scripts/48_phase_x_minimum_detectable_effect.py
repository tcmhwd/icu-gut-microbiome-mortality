#!/usr/bin/env python3
"""48_phase_x_minimum_detectable_effect.py

Pure arithmetic from already-published paired-bootstrap standard errors
(task item 3, 2026-09-02) -- no new model fitting or resampling. For each
contrast with a published 95% bootstrap CI on delta AUROC, back out an
implied SE (assuming the percentile CI is approximately symmetric/normal:
SE = (CI_high - CI_low) / (2 x 1.96)), then compute the minimum detectable
effect (MDE) at 80% power / alpha=0.05 two-sided:

    MDE = (z_{1-alpha/2} + z_{1-power}) x SE = 2.801585 x SE

Includes BOTH the original (leaky, single-split, script 21) SEs AND -- if
available -- the nested-CV-corrected SEs from script 46
(`phase_x_c1_c2_nested_bootstrap_delta.csv`), per the task's explicit
instruction to report both when the nested-corrected SE exists.

2026-09-06 additive update (script41_nested_rebuild_DESIGN.md section 6):
added a THIRD SOFA-referenced block, from script 61's nested-CV rebuild of
41_phase_x_microbiome_over_sofa.py (`phase_x_microbiome_over_sofa_nestedcv_
paired_bootstrap.csv`), design string exactly "SOFA-referenced nested-CV-
corrected (script 61)" -- deliberately avoiding the substrings "single-split"
(would be picked up by build_tables.py's `mde_single` filter) and "script 41"
(would be picked up by this script's own `power_sofa_m1`/`sofa_m1_single`
selectors below, which are repointed to the new design). The original
single-split SOFA block (script 41) is left in place, unmodified, as the
superseded row.

Reads:
  clinical_ai/reports/phase_x_paired_bootstrap_delta.csv (script 21, C1/C2a/C2b vs C0)
  clinical_ai/reports/phase_x_microbiome_over_sofa_paired_bootstrap.csv (script 41, S1-S3b vs SOFA/S3ref -- superseded, kept for continuity)
  clinical_ai/reports/phase_x_c1_c2_nested_bootstrap_delta.csv (script 46, if present)
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap.csv (script 61, S1-S3b vs SOFA/S3ref, nested-CV-corrected)
Writes:
  clinical_ai/reports/phase_x_minimum_detectable_effect.csv/.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("48_phase_x_minimum_detectable_effect")

ORIG_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_paired_bootstrap_delta.csv")
SOFA_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_paired_bootstrap.csv")
SOFA_NESTED_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap.csv")
NESTED_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_nested_bootstrap_delta.csv")
NESTED_OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities.csv")
OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_minimum_detectable_effect.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_minimum_detectable_effect.md")

ALPHA = 0.05
POWER = 0.80
Z_ALPHA = norm.ppf(1 - ALPHA / 2)
Z_POWER = norm.ppf(POWER)
K = Z_ALPHA + Z_POWER
log.info("z_(1-alpha/2)=%.6f, z_power(%.0f%%)=%.6f, K=z_alpha+z_power=%.6f", Z_ALPHA, POWER * 100, Z_POWER, K)


def implied_se(ci_low: float, ci_high: float) -> float:
    """Back out an implied SE from a (roughly symmetric) 95% percentile bootstrap CI."""
    return (ci_high - ci_low) / (2 * 1.96)


def mde_from_ci(ci_low: float, ci_high: float) -> tuple[float, float]:
    se = implied_se(ci_low, ci_high)
    return se, K * se


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.values) + " |")
    return "\n".join(lines)


rows = []

# ---------------------------------------------------------------------------
# C1/C2a/C2b vs C0 -- original (leaky, single-split) SEs, script 21
# ---------------------------------------------------------------------------
orig = pd.read_csv(ORIG_BOOT_CSV)
LABELS = {"C1_clinical_plus_M1_LiteratureCore": "C1 vs C0",
          "C2a_clinical_plus_M3_LR": "C2a vs C0",
          "C2b_clinical_plus_M3_CatBoost": "C2b vs C0"}
for _, r in orig.iterrows():
    se, mde = mde_from_ci(r["delta_AUROC_CI_low"], r["delta_AUROC_CI_high"])
    rows.append({"contrast": LABELS.get(r["model"], r["model"]), "design": "original (leaky, single-split, script 21)",
                "observed_delta_AUROC": r["delta_AUROC"], "CI_low": r["delta_AUROC_CI_low"],
                "CI_high": r["delta_AUROC_CI_high"], "implied_SE": round(se, 4), "MDE_80pct_power_alpha05": round(mde, 4),
                "observed_reaches_MDE": abs(r["delta_AUROC"]) >= mde})

# ---------------------------------------------------------------------------
# C1/C2a/C2b vs C0 -- nested-CV-corrected SEs, script 46 (if available)
# ---------------------------------------------------------------------------
if os.path.exists(NESTED_BOOT_CSV):
    nested = pd.read_csv(NESTED_BOOT_CSV)
    for _, r in nested.iterrows():
        se, mde = mde_from_ci(r["delta_AUROC_CI_low"], r["delta_AUROC_CI_high"])
        rows.append({"contrast": LABELS.get(r["model"], r["model"]), "design": "nested-CV-corrected (script 46)",
                    "observed_delta_AUROC": r["delta_AUROC"], "CI_low": r["delta_AUROC_CI_low"],
                    "CI_high": r["delta_AUROC_CI_high"], "implied_SE": round(se, 4),
                    "MDE_80pct_power_alpha05": round(mde, 4),
                    "observed_reaches_MDE": abs(r["delta_AUROC"]) >= mde})
    log.info("nested-CV-corrected SEs found (script 46) -- included alongside the original leaky-design SEs.")
else:
    log.warning("%s not found -- reporting original (leaky) SEs only for C1/C2a/C2b. Re-run after script 46 completes.",
               NESTED_BOOT_CSV)

# ---------------------------------------------------------------------------
# SOFA-referenced contrasts, script 41
# ---------------------------------------------------------------------------
sofa = pd.read_csv(SOFA_BOOT_CSV)
SOFA_LABELS = {
    "S1_SOFA_plus_M1": "SOFA+M1 vs SOFA", "S2a_SOFA_plus_M3LR": "SOFA+M3LR vs SOFA",
    "S2b_SOFA_plus_M3CatBoost": "SOFA+M3CatBoost vs SOFA", "S3ref_SOFA_plus_clinicalAI": "SOFA+clinicalAI vs SOFA",
    "S3a_SOFA_plus_clinicalAI_plus_M1": "SOFA+clinicalAI+M1 vs SOFA+clinicalAI",
    "S3b_SOFA_plus_clinicalAI_plus_M3CB": "SOFA+clinicalAI+M3CatBoost vs SOFA+clinicalAI",
}
for _, r in sofa.iterrows():
    se, mde = mde_from_ci(r["delta_AUROC_CI_low"], r["delta_AUROC_CI_high"])
    rows.append({"contrast": SOFA_LABELS.get(r["model"], r["model"]), "design": "single-split (script 41, SOFA-referenced)",
                "observed_delta_AUROC": r["delta_AUROC"], "CI_low": r["delta_AUROC_CI_low"],
                "CI_high": r["delta_AUROC_CI_high"], "implied_SE": round(se, 4), "MDE_80pct_power_alpha05": round(mde, 4),
                "observed_reaches_MDE": abs(r["delta_AUROC"]) >= mde})

# ---------------------------------------------------------------------------
# SOFA-referenced contrasts -- nested-CV-corrected, script 61 (2026-09-06
# additive update, script41_nested_rebuild_DESIGN.md section 6). This is the
# design T2/Fig3 now use for the SOFA-referenced rows; the script-41
# single-split rows above are kept, unmodified, as the superseded design.
# ---------------------------------------------------------------------------
if os.path.exists(SOFA_NESTED_BOOT_CSV):
    sofa_nested = pd.read_csv(SOFA_NESTED_BOOT_CSV)
    for _, r in sofa_nested.iterrows():
        se, mde = mde_from_ci(r["delta_AUROC_CI_low"], r["delta_AUROC_CI_high"])
        rows.append({"contrast": SOFA_LABELS.get(r["model"], r["model"]),
                    "design": "SOFA-referenced nested-CV-corrected (script 61)",
                    "observed_delta_AUROC": r["delta_AUROC"], "CI_low": r["delta_AUROC_CI_low"],
                    "CI_high": r["delta_AUROC_CI_high"], "implied_SE": round(se, 4),
                    "MDE_80pct_power_alpha05": round(mde, 4),
                    "observed_reaches_MDE": abs(r["delta_AUROC"]) >= mde})
    log.info("nested-CV-corrected SOFA-referenced SEs found (script 61) -- included alongside the "
             "original single-split SOFA rows (script 41).")
else:
    log.warning("%s not found -- reporting single-split (leaky) SEs only for SOFA-referenced contrasts. "
               "Re-run after script 61 completes.", SOFA_NESTED_BOOT_CSV)

out = pd.DataFrame(rows)
io_utils.save_csv(out, OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# verification against the reference values already circulating (task item 3)
# ---------------------------------------------------------------------------
ref_check = {"C1 vs C0": (0.0113, 0.032), "C2b vs C0": (0.0144, 0.041), "SOFA+M1 vs SOFA": (0.0128, 0.036)}
verify_rows = []
for contrast, (ref_se, ref_mde) in ref_check.items():
    match = out[(out["contrast"] == contrast) & (out["design"].str.contains("original|script 41"))]
    if len(match):
        row = match.iloc[0]
        verify_rows.append({"contrast": contrast, "reference_SE": ref_se, "computed_SE": row["implied_SE"],
                            "reference_MDE": ref_mde, "computed_MDE": row["MDE_80pct_power_alpha05"],
                            "SE_match": abs(row["implied_SE"] - ref_se) < 0.0005,
                            "MDE_match": abs(row["MDE_80pct_power_alpha05"] - ref_mde) < 0.001})
verify_df = pd.DataFrame(verify_rows)
log.info("Reference-value verification:\n%s", verify_df.to_string(index=False))

# ---------------------------------------------------------------------------
# power inversion (task item 5, 2026-09-02): the MDE above is an algebraic
# restatement of each contrast's own observed CI width, so it cannot serve
# as independent evidence of underpowering -- that rhetorical use was
# already removed from the paragraph below. What IS a defensible, separate
# question: for THIS exact design (this cohort's own N and event count),
# how large would the cohort need to be to detect a given target
# Delta AUROC at 80% power? This inverts the SAME K = z_alpha + z_power
# formula and the SAME CI-width-implied SE used for the MDE above, just
# solved for N instead of for MDE, under the standard assumption that a
# paired bootstrap SE scales as 1/sqrt(N) (holding the design's own event
# fraction and model-performance level fixed) -- i.e. SE(N) = SE_ref x
# sqrt(N_ref / N), so N_required = N_ref x (K x SE_ref / target_effect)^2.
# N_ref/events_ref are read live from script 46's persisted nested-CV OOF
# cohort (the same file used to build the nested SE rows above), not
# hardcoded.
# ---------------------------------------------------------------------------
TARGET_DELTA_AUROC = [0.02, 0.03, 0.05]

if os.path.exists(NESTED_OOF_CSV):
    oof = pd.read_csv(NESTED_OOF_CSV)
    N_REF = len(oof)
    EVENTS_REF = int(oof["hospital_mortality"].sum())
else:
    log.warning("%s not found -- falling back to the design N/events stated in this project's own "
                "cohort documentation (302 patients / 69 events) rather than reading them live.",
                NESTED_OOF_CSV)
    N_REF, EVENTS_REF = 302, 69
EVENT_FRACTION = EVENTS_REF / N_REF
log.info("Power inversion design: N_ref=%d, events_ref=%d, event_fraction=%.4f (%.1f%%)",
          N_REF, EVENTS_REF, EVENT_FRACTION, 100 * EVENT_FRACTION)

power_rows = []
for _, r in out.iterrows():
    se_ref = r["implied_SE"]
    for target in TARGET_DELTA_AUROC:
        n_req = N_REF * (K * se_ref / target) ** 2
        events_req = n_req * EVENT_FRACTION
        power_rows.append({
            "contrast": r["contrast"], "design": r["design"], "implied_SE_at_N_ref": se_ref,
            "N_ref": N_REF, "events_ref": EVENTS_REF, "target_delta_AUROC": target,
            "N_required_80pct_power": int(np.ceil(n_req)),
            "events_required_80pct_power": int(np.ceil(events_req)),
        })
power_df = pd.DataFrame(power_rows)
log.info("\nPower inversion (N/events required for 80%% power at target Delta AUROC):\n%s",
          power_df.to_string(index=False))

# headline anchors for the prose paragraph: the same three design rows the
# MDE paragraph below already quotes (nested C1/C2b, nested-CV-corrected
# SOFA+M1). Repointed 2026-09-06 from the single-split SOFA+M1 row (script
# 41) to the nested-CV-corrected one (script 61), matching the design T2/
# Fig3 now use for the SOFA-referenced rows (build_tables.py).
power_c1_nested = power_df[(power_df.contrast == "C1 vs C0") & (power_df.design.str.contains("nested"))]
power_c2b_nested = power_df[(power_df.contrast == "C2b vs C0") & (power_df.design.str.contains("nested"))]
power_sofa_m1 = power_df[(power_df.contrast == "SOFA+M1 vs SOFA") & (power_df.design.str.contains("script 61"))]

def _fmt_power_row(df_sub, target):
    row = df_sub[df_sub.target_delta_AUROC == target].iloc[0]
    return f"N={row['N_required_80pct_power']} (events={row['events_required_80pct_power']})"

power_paragraph = (
    f"Inverting the same MDE calculation (K = z_(1-alpha/2) + z_power = {K:.4f}; SE assumed to scale as "
    f"1/sqrt(N) from this design's own N_ref={N_REF}, events_ref={EVENTS_REF}, event fraction "
    f"{100*EVENT_FRACTION:.1f}%): to detect Delta AUROC=0.02/0.03/0.05 at 80% power for C1 vs C0 "
    f"(nested-CV-corrected design) would require "
    + ", ".join(_fmt_power_row(power_c1_nested, t) for t in TARGET_DELTA_AUROC)
    + f" respectively; for C2b vs C0 (nested-CV-corrected design), "
    + ", ".join(_fmt_power_row(power_c2b_nested, t) for t in TARGET_DELTA_AUROC)
    + f"; for SOFA+M1 vs SOFA (SOFA-referenced nested-CV-corrected design, script 61), "
    + ", ".join(_fmt_power_row(power_sofa_m1, t) for t in TARGET_DELTA_AUROC)
    + ". These are algebraic sample-size projections from the current design's own precision, not from a "
    "new power calculation or new data collection."
)

# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
largest_observed = out.loc[out["design"].str.contains("original|nested"), "observed_delta_AUROC"].abs().max()
largest_row = out.loc[out["observed_delta_AUROC"].abs().idxmax()]

# Paragraph quotes the SAME design rows actually used downstream (Table T2 /
# Fig3 use the nested-CV-corrected row for C1/C2a/C2b, and -- repointed
# 2026-09-06, script41_nested_rebuild_DESIGN.md section 6 -- the nested-CV-
# corrected (script 61) row for SOFA+X vs SOFA contrasts -- see
# build_tables.py) so this paragraph stays consistent with the "Full table"
# immediately above it and with Paper 1's T2/Fig3, rather than silently
# defaulting to whichever design row happens to appear first in `out`
# (claim-strength/consistency correction, 2026-09-02).
c1_nested = out.loc[(out.contrast == "C1 vs C0") & (out.design.str.contains("nested")), "MDE_80pct_power_alpha05"].iloc[0]
c2b_nested = out.loc[(out.contrast == "C2b vs C0") & (out.design.str.contains("nested")), "MDE_80pct_power_alpha05"].iloc[0]
sofa_m1_nested = out.loc[(out.contrast == "SOFA+M1 vs SOFA") & (out.design.str.contains("script 61")), "MDE_80pct_power_alpha05"].iloc[0]

paragraph = (
    f"At 80% power and alpha=0.05 (two-sided), and under a normal approximation to each contrast's own "
    f"paired-bootstrap CI (the MDE is algebraically derived from that same CI's width -- implied_SE = "
    f"(CI_high - CI_low) / (2 x 1.96) -- so it is not an independent estimate of precision), this study "
    f"design could have detected a minimum true Delta AUROC of approximately {c1_nested:.4f} for C1 vs C0 "
    f"and {c2b_nested:.4f} for C2b vs C0 (both nested-CV-corrected design, script 46), and {sofa_m1_nested:.4f} "
    "for SOFA+M1 vs SOFA (SOFA-referenced nested-CV-corrected design, script 61) -- these are the design rows "
    f"used in Table T2 and Figure P1_Fig3. Every observed incremental-value point estimate in this study (largest: "
    f"{largest_row['contrast']}, |Delta AUROC|={abs(largest_row['observed_delta_AUROC']):.4f}) falls below its "
    "own contrast's MDE. Because the MDE is reconstructed from the same CI that already reports the effect's "
    "uncertainty, an observed effect falling below its MDE is a restatement of the CI width, not independent "
    "evidence that the null result is attributable to low power rather than a true absence of effect. It "
    "should be read only as what this design was capable of detecting at 80% power -- not as a claim that "
    "underpowering, specifically, explains why no significant incremental value was observed. A substantially "
    "larger validation cohort would be needed to distinguish a small-to-moderate true effect from no effect "
    "at all with adequate power."
)

md = [
    "# Minimum detectable effect (MDE), Phase X incremental-value contrasts", "",
    f"80% power, alpha=0.05 two-sided: MDE = (z_(1-alpha/2) + z_power) x SE = {K:.4f} x SE. SE is backed out "
    "from each contrast's published 95% percentile paired-bootstrap CI as (CI_high - CI_low) / (2 x 1.96), "
    "assuming approximate symmetry/normality of the bootstrap distribution (a standard, conservative "
    "approximation for a moderately-sized bootstrap; the actual percentile CI itself remains the primary "
    "inferential quantity elsewhere in this project's reports).", "",
    "## Full table", "", df_to_md_table(out), "",
    "## Reference-value verification (task item 3's already-circulating SE/MDE numbers)", "",
    df_to_md_table(verify_df), "",
    "## Ready-to-use paragraph", "", paragraph, "",
    "---", "",
    "## Power inversion: cohort size required for 80% power at a given target effect (task item 5, 2026-09-02)",
    "",
    f"The MDE table above restates each contrast's own observed CI width -- it is not independent evidence "
    "of underpowering. This section instead answers a genuinely separate question: for this exact design "
    f"(N_ref={N_REF} patients, events_ref={EVENTS_REF} events, event fraction {100*EVENT_FRACTION:.1f}%, "
    "both read live from script 46's persisted nested-CV OOF cohort), how many patients (and events) would "
    f"be required to detect Delta AUROC=0.02, 0.03, and 0.05 at 80% power, two-sided alpha=0.05? Same "
    f"K = z_(1-alpha/2) + z_power = {K:.4f} and same CI-width-implied SE as the MDE above, inverted for N "
    "under the standard assumption that a paired bootstrap SE scales as 1/sqrt(N) (holding this design's "
    "own event fraction and each contrast's own observed performance level fixed): "
    "N_required = N_ref x (K x implied_SE_at_N_ref / target_delta_AUROC)^2, "
    "events_required = N_required x event_fraction.", "",
    df_to_md_table(power_df), "",
    power_paragraph, "",
    "## Sources", "",
    "`phase_x_paired_bootstrap_delta.csv` (script 21, original leaky-design C1/C2a/C2b vs C0); "
    "`phase_x_c1_c2_nested_bootstrap_delta.csv` (script 46, nested-CV-corrected, if present); "
    "`phase_x_microbiome_over_sofa_paired_bootstrap.csv` (script 41, SOFA-referenced contrasts, "
    "single-split -- superseded, kept for continuity); "
    "`phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap.csv` (script 61, SOFA-referenced "
    "contrasts, nested-CV-corrected, if present); "
    "`phase_x_c1_c2_nested_oof_probabilities.csv` (script 46, N_ref/events_ref for the power inversion).",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s / %s", OUT_CSV, OUT_MD)
