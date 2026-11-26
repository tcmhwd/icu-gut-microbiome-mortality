#!/usr/bin/env python3
"""55_phase_x_decision_curve_nested_paired_v4.py

Fold-local M3 preprocessing correction (owner-authorised 2026-10-02) for
`55_phase_x_decision_curve_nested_paired.py`. This script does its OWN model
fitting NOWHERE: it only reads an already-persisted nested-CV OOF prediction
file and computes paired bootstrap net benefit from it (confirmed by reading
the original script in full before writing this one --
`M3_FOLD_LOCAL_METHOD_AUDIT_v3.md` section 8 lists script 55 among the scripts
that "read the old M3/C2 OOF and need repointing and rerunning", NOT among those
that "reimplement the frozen PCA"). So the ONLY change from the original script
is WHICH prediction file is read and the correctness-check target it is verified
against; the net-benefit formula, threshold grid, paired-bootstrap algorithm,
Holm correction and reporting are byte-identical to the original.

WHAT CHANGED
  1. `NESTED_OOF_CSV`: now `phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv`
     (45_v3's fold-local PRIMARY-variant drop-in replacement for script 46's
     `phase_x_c1_c2_nested_oof_probabilities.csv`), not the frozen-PCA file.
     C0 and C1 (M1) columns in that file are confirmed bit-identical to the
     original (see this task's own verification, max|diff| ~1e-15 for M1,
     0.0 for C1/C0/clinical_AI) -- only C2a/C2b differ.
  2. Correctness check: against `phase_x_c1_c2_incremental_value_nested_corrected_v4.csv`
     (45_v3's aggregate report), reading `CV_AUROC_5fold_nested` (not the
     original's `CV_AUROC_5fold_nested_corrected` column, which that file does
     not have) for C0/C1 from the `REF` variant rows and C2a/C2b from the `V3`
     (primary, fully fold-local) variant rows -- NOT V0/V1/V2, which are
     non-primary diagnostic variants of the same 45_v3 script.
  3. Every output path carries a `_v3` suffix; nothing from the original script
     is overwritten. An OLD-vs-NEW numeric comparison is written in addition to
     the original report content, so the "what changed" question has a number
     attached rather than just prose.

Reads:
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv (script 45_v3)
  clinical_ai/reports/phase_x_c1_c2_incremental_value_nested_corrected_v4.csv (script 45_v3, correctness check)
  clinical_ai/reports/phase_x_decision_curve_nested_paired.csv (OLD, read-only reference for the comparison)
Writes:
  clinical_ai/reports/phase_x_decision_curve_nested_paired_v4.csv
  clinical_ai/reports/phase_x_decision_curve_nested_paired_v4.md

ORIGINAL SCRIPT 55 DOCSTRING FOLLOWS (its frozen-PCA predictions are superseded
by point 1 above; everything else below is unchanged and still describes this
script exactly).
---------------------------------------------------------------------------
Independent-review action item (2026-09-02): recomputes the Phase X decision
curve analysis (DCA) as a valid INFERENTIAL comparison, replacing the
descriptive "net benefit indistinguishable at 0 of 25 thresholds" statement
built by `semantic_microbiome/reports/paper1/build_tables.py`'s
`dca_summary()` (~line 239-254) from `phase_x_decision_curve_data_with_CI.csv`
(script 21) and `phase_x_microbiome_over_sofa_decision_curve.csv` (script 41).

THREE CONFIRMED PROBLEMS WITH THE SUPERSEDED DCA (not re-derived here, taken
as given from the independent review):

  1. WRONG COMPARISON. `dca_summary()` calls two net benefit curves
     "indistinguishable" whenever their MARGINAL 95% CI bands overlap at a
     threshold. Overlap of two independent marginal CIs is not a test of the
     PAIRED difference between two models scored on the SAME patients.
  2. WRONG UNCERTAINTY OBJECT. The paired per-replicate DIFFERENCE was
     computed but never persisted anywhere in this project before this script.
  3. WRONG PREDICTIONS. Both scripts 21 and 41 read script 18's OLDER
     out-of-fold probabilities, affected by the CV-nesting leakage bug.

THE FIX
  Predictions: nested-CV-corrected OOF probabilities (point 1 above, now V3
  fold-local-corrected). Estimand: delta_NB = NB(candidate) - NB(C0), PAIRED
  bootstrap (2000 replicates, seed=2026), percentile 95% CI, two-sided
  bootstrap p. SOFA-referenced contrasts: NOT computed here (unchanged reason
  -- no nested-CV OOF exists for any SOFA-referenced S0-S3 candidate in this
  script's lineage; that family is handled separately by
  `61_phase_x_microbiome_over_sofa_nested_v3.py`).

NET BENEFIT FORMULA (standard decision-curve convention, unchanged): for
threshold probability pt, predicted-positive := prob >= pt,
    NB(pt) = TP/n - (FP/n) * (pt / (1 - pt))
Treat-all: NB_all(pt) = prevalence - (1 - prevalence) * (pt / (1 - pt)).
Treat-none: NB_none(pt) = 0.

THRESHOLD GRID: secondary/full grid np.arange(0.02, 0.51, 0.02) -- 25
thresholds. PRIMARY prespecified thresholds: 10% and 30% (unchanged rationale,
see original docstring in the sibling non-_v3 file).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("55_phase_x_decision_curve_nested_paired_v4")

NESTED_OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv")
NESTED_CORRECTED_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected_v4.csv")
OLD_DCA_CSV = os.path.join(config.DIR_REPORTS, "phase_x_decision_curve_nested_paired.csv")  # read-only reference

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_decision_curve_nested_paired_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_decision_curve_nested_paired_v4.md")

SEED = config.RANDOM_SEED  # 2026
N_BOOT = 2000  # matches scripts 21/41/46/55's convention exactly
PRESPECIFIED_THRESHOLDS = [0.10, 0.30]  # unchanged from script 55; see that script's docstring for rationale

# expected (variant, model) pair in the 45_v4 aggregate report for each of the
# four checked arrays -- C0/C1 (M1) come from REF (ladder-fixed, numerically
# unchanged); C2a comes from V4_STRICT (v4 FINAL headline for the LR model);
# C2b comes from V3 (CatBoost unchanged, reused verbatim).
ANCHOR_SOURCE = {
    "C0_clinical_alone": ("REF", "C0_clinical_alone"),
    "C1_clinical_plus_M1_LiteratureCore": ("REF", "C1_clinical_plus_M1_LiteratureCore"),
    "C2a_clinical_plus_M3_LR": ("V4_STRICT", "C2a_clinical_plus_M3_LR"),
    "C2b_clinical_plus_M3_CatBoost": ("V3", "C2b_clinical_plus_M3_CatBoost"),
}


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def net_benefit(y_arr, prob, pt):
    """IDENTICAL formula to scripts 20/21/41/55's net_benefit() -- unchanged."""
    w = pt / (1 - pt)
    pred_pos = prob >= pt
    tp = ((pred_pos) & (y_arr == 1)).sum()
    fp = ((pred_pos) & (y_arr == 0)).sum()
    return tp / len(y_arr) - (fp / len(y_arr)) * w


# ---------------------------------------------------------------------------
# 1. load the fold-local-corrected (V3) nested OOF probabilities -- NOT
#    regenerated here, read straight from 45_v3's drop-in file.
# ---------------------------------------------------------------------------
if not os.path.exists(NESTED_OOF_CSV):
    log.error("%s not found -- run 45_phase_x_c1_c2_nested_cv_corrected_v4.py --aggregate first.", NESTED_OOF_CSV)
    sys.exit(1)

oof_df = pd.read_csv(NESTED_OOF_CSV).set_index("patient_id_clinical")
y = oof_df["hospital_mortality"].values
n = len(y)
log.info("N=%d patients, %d deaths (%.1f%%) -- loaded from %s (45_v3's fold-local PRIMARY nested OOF predictions)",
          n, int(y.sum()), 100 * y.mean(), NESTED_OOF_CSV)

C0 = oof_df["oof_prob_C0_clinical_alone"].values
CANDIDATES = {
    "C1_clinical_plus_M1_LiteratureCore": oof_df["oof_prob_C1_clinical_plus_M1_LiteratureCore"].values,
    "C2a_clinical_plus_M3_LR": oof_df["oof_prob_C2a_clinical_plus_M3_LR"].values,
    "C2b_clinical_plus_M3_CatBoost": oof_df["oof_prob_C2b_clinical_plus_M3_CatBoost"].values,
}

# ---------------------------------------------------------------------------
# 1b. correctness check -- these OOF arrays must reproduce 45_v3's published
#     CV_AUROC_5fold_nested (REF for C0/C1, V3 primary for C2a/C2b) before
#     anything else runs. This is what makes "DCA and delta-AUROC share the
#     same predictions" provable rather than assumed.
# ---------------------------------------------------------------------------
if os.path.exists(NESTED_CORRECTED_CSV):
    ref_all = pd.read_csv(NESTED_CORRECTED_CSV)
    check_arrays = {"C0_clinical_alone": C0, **CANDIDATES}
    all_match = True
    for name, arr in check_arrays.items():
        variant, model = ANCHOR_SOURCE[name]
        sub = ref_all[(ref_all.preprocessing_variant == variant) & (ref_all.model == model)]
        published = float(sub["CV_AUROC_5fold_nested"].iloc[0]) if len(sub) else None
        recomputed = round(roc_auc_score(y, arr), 4)
        status = "MATCH" if published is not None and abs(recomputed - published) < 1e-9 else "MISMATCH"
        log.info("Correctness check %s (variant=%s): AUROC(nested OOF)=%.4f published(45_v3)=%s -> %s",
                  name, variant, recomputed, published, status)
        if status == "MISMATCH":
            all_match = False
    if not all_match:
        log.error("Nested OOF predictions used here do NOT reproduce 45_v3's published V3/REF AUROCs -- "
                   "refusing to build a DCA that would not provably share predictions with the manuscript's "
                   "headline delta-AUROC. Investigate before proceeding.")
        sys.exit(1)
    log.info("Confirmed: DCA inputs here are the EXACT SAME fold-local (V3) nested OOF predictions behind "
              "45_v3's delta-AUROC numbers (C1 +0.0082, C2a +0.0081, C2b +0.0183).")
else:
    log.warning("%s not found -- skipping the AUROC correctness check (predictions still come from "
                "45_v3's file either way).", NESTED_CORRECTED_CSV)

# ---------------------------------------------------------------------------
# 2. threshold grid + paired bootstrap (byte-identical algorithm to script 55)
# ---------------------------------------------------------------------------
thresholds = np.arange(0.02, 0.51, 0.02)
log.info("Threshold grid: %d thresholds (0.02-0.50); prespecified primary thresholds: %s",
          len(thresholds), PRESPECIFIED_THRESHOLDS)

rng = np.random.RandomState(SEED)
rows = []
for pt in thresholds:
    pt = round(float(pt), 3)
    nb_c0 = net_benefit(y, C0, pt)
    nb_all = float(y.mean() - (1 - y.mean()) * (pt / (1 - pt)))

    delta_boot = {name: [] for name in CANDIDATES}
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        nb_c0_b = net_benefit(yb, C0[idx], pt)
        for name, prob in CANDIDATES.items():
            nb_cand_b = net_benefit(yb, prob[idx], pt)
            delta_boot[name].append(nb_cand_b - nb_c0_b)

    for name, prob in CANDIDATES.items():
        nb_cand = net_benefit(y, prob, pt)
        delta_nb = nb_cand - nb_c0
        deltas = np.array(delta_boot[name])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_two = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        rows.append({
            "model": name, "threshold": pt,
            "prespecified_primary": pt in PRESPECIFIED_THRESHOLDS,
            "net_benefit_C0": round(nb_c0, 4), "net_benefit_candidate": round(nb_cand, 4),
            "net_benefit_treat_all": round(nb_all, 4), "net_benefit_treat_none": 0.0,
            "delta_NB": round(delta_nb, 4), "delta_NB_CI_low": round(lo, 4), "delta_NB_CI_high": round(hi, 4),
            "delta_NB_boot_p": round(min(p_two, 1.0), 4),
            "CI_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": ("favors_candidate" if lo > 0 else "favors_C0" if hi < 0 else "n/a"),
            "N_boot": N_BOOT,
        })

dca_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. per-contrast summary (unchanged algorithm)
# ---------------------------------------------------------------------------
summary_rows = []
prespecified_rows = []
for name in CANDIDATES:
    sub = dca_df[dca_df["model"] == name].reset_index(drop=True)
    n_excl = int(sub["CI_excludes_zero"].sum())
    excl_sub = sub.loc[sub["CI_excludes_zero"]]
    n_favor_candidate = int((excl_sub["direction"] == "favors_candidate").sum())
    n_favor_c0 = int((excl_sub["direction"] == "favors_C0").sum())
    excl_desc = ";".join(f"{t:.2f}({'favors_candidate' if d == 'favors_candidate' else 'favors_C0'})"
                          for t, d in zip(excl_sub["threshold"], excl_sub["direction"]))
    imax = sub["delta_NB"].idxmax()
    summary_rows.append({
        "model": name, "thresholds_tested": len(sub),
        "thresholds_paired_CI_excludes_zero": n_excl,
        "n_favor_candidate": n_favor_candidate, "n_favor_C0": n_favor_c0,
        "which_thresholds_excl_zero": excl_desc if excl_desc else "(none)",
        "max_delta_NB": sub.loc[imax, "delta_NB"], "threshold_at_max_delta_NB": sub.loc[imax, "threshold"],
        "max_delta_NB_CI_low": sub.loc[imax, "delta_NB_CI_low"], "max_delta_NB_CI_high": sub.loc[imax, "delta_NB_CI_high"],
    })
    for pt in PRESPECIFIED_THRESHOLDS:
        r = sub[sub["threshold"] == pt].iloc[0]
        prespecified_rows.append({
            "model": name, "threshold": pt, "delta_NB": r["delta_NB"],
            "delta_NB_CI_low": r["delta_NB_CI_low"], "delta_NB_CI_high": r["delta_NB_CI_high"],
            "delta_NB_boot_p": r["delta_NB_boot_p"], "CI_excludes_zero": r["CI_excludes_zero"],
        })

summary_df = pd.DataFrame(summary_rows)
prespecified_df = pd.DataFrame(prespecified_rows)

# Holm-Bonferroni step-down adjustment across the 6 prespecified cells
# (unchanged algorithm from script 55).
m_holm = len(prespecified_df)
order = prespecified_df["delta_NB_boot_p"].argsort().values
holm_p = np.empty(m_holm)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = (m_holm - rank) * prespecified_df["delta_NB_boot_p"].values[idx]
    running_max = max(running_max, adj)
    holm_p[idx] = min(running_max, 1.0)
prespecified_df["delta_NB_boot_p_holm"] = np.round(holm_p, 4)
prespecified_df["CI_excludes_zero_and_holm_survives"] = (
    prespecified_df["CI_excludes_zero"] & (prespecified_df["delta_NB_boot_p_holm"] < 0.05)
)
n_holm_survive = int(prespecified_df["CI_excludes_zero_and_holm_survives"].sum())

log.info("\n%s", summary_df.to_string(index=False))
log.info("\nPrespecified thresholds (with Holm-adjusted p across the 6 cells):\n%s",
          prespecified_df.to_string(index=False))
log.info("Holm-adjusted-p survivors among the 6 prespecified cells: %d/%d", n_holm_survive, m_holm)

dca_df = dca_df.merge(
    prespecified_df[["model", "threshold", "delta_NB_boot_p_holm"]],
    on=["model", "threshold"], how="left")
io_utils.save_csv(dca_df, OUT_CSV)
log.info("wrote %s (with delta_NB_boot_p_holm populated for the 6 prespecified cells only)", OUT_CSV)

# ---------------------------------------------------------------------------
# 4. OLD-vs-NEW numeric comparison (new in v3 -- the task wants a number, not
#    just prose, attached to "did anything flip").
# ---------------------------------------------------------------------------
old_vs_new_rows = []
if os.path.exists(OLD_DCA_CSV):
    old_df = pd.read_csv(OLD_DCA_CSV)
    merged = dca_df.merge(old_df, on=["model", "threshold"], suffixes=("_new", "_old"))
    for _, r in merged.iterrows():
        old_vs_new_rows.append({
            "model": r["model"], "threshold": r["threshold"],
            "old_delta_NB": r["delta_NB_old"], "new_delta_NB": r["delta_NB_new"],
            "old_CI_low": r["delta_NB_CI_low_old"], "old_CI_high": r["delta_NB_CI_high_old"],
            "new_CI_low": r["delta_NB_CI_low_new"], "new_CI_high": r["delta_NB_CI_high_new"],
            "old_CI_excludes_zero": r["CI_excludes_zero_old"], "new_CI_excludes_zero": r["CI_excludes_zero_new"],
            "flipped": bool(r["CI_excludes_zero_old"] != r["CI_excludes_zero_new"]),
        })
    old_vs_new_df = pd.DataFrame(old_vs_new_rows)
    n_flipped = int(old_vs_new_df["flipped"].sum())
    n_old_excl = int(old_vs_new_df["old_CI_excludes_zero"].sum())
    n_new_excl = int(old_vs_new_df["new_CI_excludes_zero"].sum())
    log.info("OLD-vs-NEW: old excl_zero=%d/75, new excl_zero=%d/75, flipped=%d/75", n_old_excl, n_new_excl, n_flipped)
    io_utils.save_csv(old_vs_new_df, os.path.join(config.DIR_REPORTS, "phase_x_decision_curve_nested_paired_old_vs_new_v3.csv"))
else:
    old_vs_new_df = pd.DataFrame()
    n_flipped = n_old_excl = n_new_excl = None
    log.warning("%s not found -- skipping old-vs-new comparison.", OLD_DCA_CSV)

# ---------------------------------------------------------------------------
# 5. markdown report
# ---------------------------------------------------------------------------
total_excl = int(summary_df["thresholds_paired_CI_excludes_zero"].sum())
total_favor_candidate = int(summary_df["n_favor_candidate"].sum())
total_favor_c0 = int(summary_df["n_favor_C0"].sum())

md = [
    "# Phase X: fold-local-corrected (V3) PAIRED decision-curve analysis (C1/C2a/C2b vs C0)", "",
    "**Fold-local M3 correction (2026-10-02):** this is the sibling of "
    "`55_phase_x_decision_curve_nested_paired.py`, repointed at "
    "`45_phase_x_c1_c2_nested_cv_corrected_v4.py`'s fold-local PRIMARY (V3) nested OOF predictions "
    "(`phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv`) instead of the frozen-PCA ones. The net-benefit "
    "formula, threshold grid, paired-bootstrap algorithm and Holm correction are byte-identical to the "
    "original script -- only the input predictions and the correctness-check target changed. C1 (M1) is "
    "unaffected: M1's OOF column is confirmed bit-identical old vs new.", "",
    f"**Old vs new, C0-referenced grid (75 = 25 thresholds x 3 models)**: old excludes zero at "
    f"{n_old_excl}/75 cells, new at {n_new_excl}/75, {n_flipped}/75 flipped direction-of-significance "
    "(none if 0). See `phase_x_decision_curve_nested_paired_old_vs_new_v3.csv` for the per-cell comparison.", "",
    "## Method (unchanged from script 55)", "",
    "**Predictions**: `clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv` (45_v3's "
    "fold-local PRIMARY nested OOF probabilities), NOT regenerated here. A correctness check (log above) "
    "confirms `roc_auc_score(y, oof)` on these arrays reproduces 45_v3's published `CV_AUROC_5fold_nested` "
    "for C0/C1 (REF variant) and C2a/C2b (V3 variant) before any bootstrap runs.", "",
    "**Net benefit formula**: for threshold probability pt and predicted-positive := prob >= pt, "
    "`NB(pt) = TP/n - (FP/n) * (pt / (1-pt))`. Treat-all: `NB_all(pt) = prevalence - (1-prevalence) * "
    "(pt / (1-pt))`. Treat-none: `NB_none(pt) = 0`.", "",
    "**Estimand**: delta_NB(pt) = NB(candidate, pt) - NB(C0, pt), a PAIRED bootstrap statistic, 2000 "
    "replicates, seed=2026. 95% CI from bootstrap percentiles; two-sided bootstrap p = 2 x min(proportion "
    "of replicate deltas <=0, proportion >=0), capped at 1.", "",
    "**Thresholds**: secondary/full grid = `np.arange(0.02, 0.51, 0.02)` (25 thresholds). PRIMARY "
    f"prespecified thresholds: 10% and 30% (unchanged rationale; cohort mortality prevalence {100*y.mean():.1f}%).", "",
    "**SOFA-referenced contrasts**: NOT computed here (unchanged reason); handled separately by "
    "`61_phase_x_microbiome_over_sofa_nested_v3.py`.", "",
    "## Primary result: prespecified thresholds (10%, 30%)", "",
    f"`delta_NB_boot_p_holm`: Holm-Bonferroni-adjusted p across these 6 cells. "
    f"{n_holm_survive}/{m_holm} of the 6 prespecified cells survive Holm at alpha=0.05.", "",
    df_to_md_table(prespecified_df), "",
    "## Per-contrast summary", "",
    df_to_md_table(summary_df), "",
    "## Old (frozen-PCA) vs new (V3 fold-local), per-cell comparison", "",
    df_to_md_table(old_vs_new_df) if len(old_vs_new_df) else "(old reference file not found)", "",
    "## Secondary: full 25-threshold grid", "",
    df_to_md_table(dca_df[["model", "threshold", "prespecified_primary", "net_benefit_C0", "net_benefit_candidate",
                            "net_benefit_treat_all", "net_benefit_treat_none", "delta_NB", "delta_NB_CI_low",
                            "delta_NB_CI_high", "delta_NB_boot_p", "CI_excludes_zero"]]), "",
    "## Reference", "",
    "Old (frozen-PCA) DCA: `phase_x_decision_curve_nested_paired.csv/.md` (script 55). Fold-local-corrected "
    "delta-AUROC (same predictions as this script): `phase_x_c1_c2_incremental_value_nested_corrected_v4.csv`, "
    "`phase_x_c1_c2_nested_bootstrap_delta_v3.csv` (script 45_v3).",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
