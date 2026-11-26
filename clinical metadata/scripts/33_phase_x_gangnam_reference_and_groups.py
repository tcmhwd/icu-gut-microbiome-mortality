#!/usr/bin/env python3
"""33_phase_x_gangnam_reference_and_groups.py

Paper-story restructuring (user request, 2026-08-17). Three additions that
were requested as "discussion-only reference material" alongside the primary
SOFA-only 302-patient comparison (script 29, unchanged):

Part A -- REFERENCE ONLY: conventional severity scores (APACHE II / SOFA /
  qSOFA / SAPS II) computed on ICU.csv's Gangnam site subset (hospital=='G',
  ~23k landmark-eligible ADMISSIONS, NOT the same 302 microbiome patients --
  this is explicitly an assumption/approximation the user asked for "for
  reference only", not a fair same-cohort comparison). Paired against the
  Reduced Clinical AI's own external-on-Gangnam performance (already computed
  in reduced_model_comparison.csv, dev=Sinchon/MIMIC, held out on Gangnam)
  and Microbiome AI's Gangnam-internal-CV performance, to sanity-check
  whether the assumption "large-cohort Gangnam scores ~ what the 302 patients
  would look like" is even in the right ballpark.

Part B -- figure only for script 29's existing SOFA-only 302-patient table
  (phase_x_severity_score_comparison_302.csv) -- that script wrote a table
  but no figure; user asked for "표와 그래프" (table AND graph).

Part C -- A/B/C/D concordance-group baseline characteristics table (age,
  sex, SOFA, CCI comorbidity index, albumin, CRP, WBC, key comorbidity
  flags) for the 4 quadrants already defined in phase_x_concordance_groups.csv
  (script 24/26): A=both agree high, B=clinical-only flags high, C=microbiome-
  only flags high ("rescue" cases), D=both agree low (missed). This is the
  characteristics table version of the existing time-to-death figure
  (phase_x_acuity_by_concordance_group.jpg).

hospital_mortality / mortality_harmonized used for evaluation only in Parts
A and B (no fitting). Part C uses no outcome-driven feature selection --
plain descriptive stats.

Reads:
  ICU.csv (raw, Gangnam subset only)
  clinical_ai/reports/reduced_model_comparison.csv
  clinical_ai/reports/phase_x_severity_score_comparison_302.csv
  clinical_ai/reports/phase_x_concordance_groups.csv
  metadata/microbiome.clinical.csv
  semantic_microbiome/harmonized/discovery_cohort_primary.csv
Writes:
  clinical_ai/reports/phase_x_gangnam_reference_conventional_scores.csv/.md
  clinical_ai/figures/phase_x_gangnam_reference_comparison.svg/.jpg
  clinical_ai/figures/phase_x_severity_score_comparison_302.svg/.jpg
  clinical_ai/reports/phase_x_concordance_group_characteristics.csv/.md
  clinical_ai/figures/phase_x_concordance_group_characteristics.svg/.jpg
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, harmonize, io_utils  # noqa: E402

log = io_utils.setup_logging("33_phase_x_gangnam_reference_and_groups")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEED = config.RANDOM_SEED
N_BOOT = config.N_BOOTSTRAP


def savefig(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.jpg", bbox_inches="tight", dpi=200)
    plt.close(fig)


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def bootstrap_auroc(y, score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    point = roc_auc_score(y, score)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ===========================================================================
# Part A: Gangnam-subset (ICU.csv, hospital=='G') conventional scores --
# REFERENCE ONLY, not the same 302 patients.
# ===========================================================================
log.info("Part A: loading ICU.csv, filtering hospital=='G'...")
icu_raw = io_utils.load_raw("ICU")
gangnam_raw = icu_raw[icu_raw["hospital"] == "G"].copy()
log.info("Gangnam raw rows: %d", len(gangnam_raw))

bundle = harmonize.build_model_ready_frame(gangnam_raw, "ICU", config.ID_COL_ICU)
severity = bundle["severity"]
target = bundle["target"].astype(int)
log.info("Gangnam landmark-eligible: N=%d, deaths=%d", len(target), int(target.sum()))

rows_a = []
for score in config.SEVERITY_SCORE_COMPARATORS:
    s = pd.to_numeric(severity[score], errors="coerce")
    valid = s.notna()
    y = target[valid].values
    sv = s[valid].values
    if len(np.unique(y)) < 2 or valid.sum() == 0:
        continue
    auc, lo, hi = bootstrap_auroc(y, sv)
    rows_a.append({
        "score": config.SEVERITY_SCORE_DISPLAY_NAMES[score], "N": int(valid.sum()),
        "AUROC": round(auc, 4), "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4),
    })
conv_gangnam = pd.DataFrame(rows_a).sort_values("AUROC", ascending=False)

OUT_A_CSV = os.path.join(config.DIR_REPORTS, "phase_x_gangnam_reference_conventional_scores.csv")
OUT_A_MD = os.path.join(config.DIR_REPORTS, "phase_x_gangnam_reference_conventional_scores.md")
io_utils.save_csv(conv_gangnam, OUT_A_CSV)

# Reduced Clinical AI external-on-Gangnam (already computed, script 09/14/15) +
# Microbiome AI internal-CV (already computed, semantic_microbiome lit09/lit10)
reduced_cmp_path = os.path.join(config.DIR_REPORTS, "reduced_model_comparison.csv")
reduced_cmp = pd.read_csv(reduced_cmp_path)
best_ext_mimic = reduced_cmp[(reduced_cmp.model_name == "catboost") & (reduced_cmp.development_dataset == "MIMIC")]
best_ext_sinchon = reduced_cmp[(reduced_cmp.model_name == "catboost") & (reduced_cmp.development_dataset == "Sinchon")]

reference_rows = list(conv_gangnam.itertuples(index=False))
composite = conv_gangnam.rename(columns={"score": "model"}).copy()
composite["model"] = "[REF, ICU.csv Gangnam ~23k] " + composite["model"]
extra = pd.DataFrame([
    {"model": "[REF, ext-on-Gangnam] Reduced Clinical AI (CatBoost, dev=MIMIC)", "N": None,
     "AUROC": round(float(best_ext_mimic["Gangnam_relevant_AUROC"].iloc[0]), 4),
     "AUROC_CI_low": None, "AUROC_CI_high": None},
    {"model": "[REF, ext-on-Gangnam] Reduced Clinical AI (CatBoost, dev=Sinchon)", "N": None,
     "AUROC": round(float(best_ext_sinchon["Gangnam_relevant_AUROC"].iloc[0]), 4),
     "AUROC_CI_low": None, "AUROC_CI_high": None},
    {"model": "[REF, internal CV] Microbiome AI M1 Literature Core", "N": 302,
     "AUROC": 0.6243, "AUROC_CI_low": None, "AUROC_CI_high": None},
    {"model": "[REF, internal CV] Microbiome AI M3 Data-driven CatBoost", "N": 302,
     "AUROC": 0.6490, "AUROC_CI_low": None, "AUROC_CI_high": None},
    {"model": "[TRUE, 302 same-cohort] SOFA (script 29)", "N": 302, "AUROC": 0.7593,
     "AUROC_CI_low": 0.6865, "AUROC_CI_high": 0.8269},
])
composite = pd.concat([composite, extra], ignore_index=True).sort_values("AUROC", ascending=False)
OUT_COMPOSITE_CSV = os.path.join(config.DIR_REPORTS, "phase_x_gangnam_reference_composite.csv")
io_utils.save_csv(composite, OUT_COMPOSITE_CSV)

with open(OUT_A_MD, "w") as fh:
    fh.write("\n".join([
        "# REFERENCE ONLY: conventional severity scores on ICU.csv's Gangnam site subset",
        "",
        "**Not the same 302 patients as the microbiome cohort.** This uses ALL ICU.csv "
        "admissions with hospital=='G' (Gangnam Severance), landmark-eligible, N="
        f"{len(target)} (deaths={int(target.sum())}) -- a much larger and only partially "
        "overlapping population (the microbiome cohort is a 302-patient subset of Gangnam "
        "admissions, not all of them; timeframes/inclusion criteria also differ). This is "
        "supplied as an approximate sanity-check / reference figure only, per user request "
        "2026-08-17 -- it should NOT be presented as a fair same-cohort comparison. The "
        "primary, fair, same-cohort comparison is script 29's SOFA-only-on-302 table "
        "(`phase_x_severity_score_comparison_302.csv`).",
        "",
        df_to_md_table(conv_gangnam),
        "",
        "## Composite reference figure data (conventional scores above + Reduced Clinical AI "
        "external-on-Gangnam + Microbiome AI internal-CV + the TRUE 302-patient SOFA number)",
        "",
        df_to_md_table(composite),
    ]))
log.info("wrote %s / %s / %s", OUT_A_CSV, OUT_A_MD, OUT_COMPOSITE_CSV)

# Figure: composite reference bar chart
fig, ax = plt.subplots(figsize=(9, 5.5))
plot_df = composite.copy()
colors = ["#b0631e" if m.startswith("[REF, ICU.csv") else
          "#33549c" if m.startswith("[REF, ext-on-Gangnam]") else
          "#3d7a56" if m.startswith("[REF, internal CV]") else
          "#7a4f9c" for m in plot_df["model"]]
y_pos = np.arange(len(plot_df))
ax.barh(y_pos, plot_df["AUROC"], color=colors, edgecolor="white")
for i, (auc, lo, hi) in enumerate(zip(plot_df["AUROC"], plot_df["AUROC_CI_low"], plot_df["AUROC_CI_high"])):
    if pd.notna(lo) and pd.notna(hi):
        ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="none", ecolor="black", capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["model"], fontsize=8)
ax.invert_yaxis()
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("AUROC")
ax.set_title("REFERENCE (not a fair same-cohort comparison, see caveats in report)\n"
              "Gangnam-relevant AUROC: conventional scores (orange) vs Reduced Clinical AI\n"
              "external-on-Gangnam (blue) vs Microbiome AI internal-CV (green) vs 302-cohort SOFA (purple)",
              fontsize=10)
fig.tight_layout()
savefig(fig, os.path.join(config.DIR_FIGURES, "phase_x_gangnam_reference_comparison"))
log.info("wrote figure phase_x_gangnam_reference_comparison")

# ===========================================================================
# Part B: figure for script 29's existing SOFA-only-302 table
# ===========================================================================
sev302_path = os.path.join(config.DIR_REPORTS, "phase_x_severity_score_comparison_302.csv")
sev302 = pd.read_csv(sev302_path).sort_values("AUROC")
fig, ax = plt.subplots(figsize=(8, 4.5))
y_pos = np.arange(len(sev302))
colors2 = ["#b0631e" if m.startswith("SOFA") else "#33549c" if "Clinical AI alone" in m else "#7a4f9c" for m in sev302["model"]]
ax.barh(y_pos, sev302["AUROC"], color=colors2, edgecolor="white")
for i, (auc, lo, hi) in enumerate(zip(sev302["AUROC"], sev302["AUROC_CI_low"], sev302["AUROC_CI_high"])):
    ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="none", ecolor="black", capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(sev302["model"], fontsize=9)
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel(f"AUROC (bootstrap 95% CI, N={N_BOOT})")
ax.set_title("Same 302 Gangnam patients: SOFA vs Clinical AI vs Clinical+Microbiome AI\n"
              "(fair, same-cohort comparison -- see phase_x_severity_score_comparison_302.md)", fontsize=10)
fig.tight_layout()
savefig(fig, os.path.join(config.DIR_FIGURES, "phase_x_severity_score_comparison_302"))
log.info("wrote figure phase_x_severity_score_comparison_302")

# ===========================================================================
# Part C: A/B/C/D concordance-group characteristics table
# ===========================================================================
log.info("Part C: building concordance-group characteristics table...")
groups = pd.read_csv(os.path.join(config.DIR_REPORTS, "phase_x_concordance_groups.csv"))
# groups.csv was written with index=False (io_utils.save_csv default), so its patient_id
# index was dropped on write -- but its ROW ORDER is exactly oof.index's row order (script
# 24 built it directly from `oof = pd.read_csv(OOF_CSV, index_col=0)` with no reordering).
# Recover patient_id from that same OOF file, not from discovery_cohort_primary.csv (whose
# row order is not guaranteed to match).
OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")
oof_ids = pd.read_csv(OOF_CSV, index_col=0)
disc = pd.read_csv(os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv"))
raw_clin = pd.read_csv(os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv"))

if len(oof_ids) == len(groups):
    groups = groups.copy()
    groups["patient_id_clinical"] = oof_ids.index.values
else:
    log.error("Could not align concordance_groups.csv to patient IDs (len mismatch: %d vs %d). "
              "Part C characteristics table will be skipped.", len(groups), len(oof_ids))
    groups = None

if groups is not None:
    disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
    merged = groups.merge(disc_ids, on="patient_id_clinical", how="left")
    merged = merged.merge(raw_clin, left_on="kcdc_no", right_on="KCDC_no", how="left")

    for c in ["age", "SOFA", "CCI", "Alb", "CRP", "WBC"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged["sex_female_pct"] = (merged["sex"].astype(str).str.upper().str.startswith("F")).astype(float)

    GROUP_LABELS = {
        "A_both_flag_high": "A: 둘 다 고위험 (both agree, high)",
        "B_clinical_only_flags_high": "B: 임상 AI만 고위험",
        "C_microbiome_only_flags_high": "C: microbiome AI만 고위험 ('rescue')",
        "D_both_flag_low": "D: 둘 다 저위험 (both agree, low)",
    }
    rows_c = []
    for gkey, glabel in GROUP_LABELS.items():
        sub = merged[merged.concordance_group == gkey]
        if len(sub) == 0:
            continue
        n_died = int(sub["hospital_mortality"].sum())
        row = {
            "group": glabel, "N": len(sub), "N_died": n_died,
            "mortality_pct": round(100 * n_died / len(sub), 1),
            "age_mean": round(sub["age"].mean(), 1), "female_pct": round(100 * sub["sex_female_pct"].mean(), 1),
            "SOFA_mean": round(sub["SOFA"].mean(), 2), "CCI_mean": round(sub["CCI"].mean(), 2),
            "Albumin_mean": round(sub["Alb"].mean(), 2), "CRP_mean": round(sub["CRP"].mean(), 1),
            "WBC_mean": round(sub["WBC"].mean(), 2),
        }
        for comorb in ["DM", "HTN", "CVA", "COPD", "cancer"]:
            if comorb in sub.columns:
                flag = sub[comorb].astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true"])
                row[f"{comorb}_pct"] = round(100 * flag.mean(), 1)
        rows_c.append(row)
    char_df = pd.DataFrame(rows_c)
    OUT_C_CSV = os.path.join(config.DIR_REPORTS, "phase_x_concordance_group_characteristics.csv")
    OUT_C_MD = os.path.join(config.DIR_REPORTS, "phase_x_concordance_group_characteristics.md")
    io_utils.save_csv(char_df, OUT_C_CSV)
    with open(OUT_C_MD, "w") as fh:
        fh.write("\n".join([
            "# Baseline characteristics by clinical/microbiome AI concordance group (N=302)",
            "",
            "A=both AIs flag high risk; B=clinical AI only flags high risk; C=microbiome AI "
            "only flags high risk ('rescue' cases -- clinical AI under-calls, microbiome "
            "catches it); D=both flag low risk (of which some are missed deaths). Descriptive "
            "only -- subgroup Ns are modest, no formal hypothesis testing here (see "
            "phase_x_residual_risk_marker_profile.csv / phase_x_fp_correction_marker_profile.csv "
            "for the microbiome-marker-level statistical comparisons). Companion figure to "
            "phase_x_acuity_by_concordance_group.jpg (time-to-death by the same 4 groups).",
            "",
            df_to_md_table(char_df),
        ]))
    log.info("wrote %s / %s", OUT_C_CSV, OUT_C_MD)

    # Figure: small-multiple bar charts for key characteristics across A/B/C/D
    metrics = [("mortality_pct", "Mortality %"), ("SOFA_mean", "Mean SOFA"),
               ("Albumin_mean", "Mean Albumin (g/dL)"), ("age_mean", "Mean age")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors3 = {"A: 둘 다 고위험 (both agree, high)": "#2e7d32", "B: 임상 AI만 고위험": "#c0392b",
               "C: microbiome AI만 고위험 ('rescue')": "#2980b9", "D: 둘 다 저위험 (both agree, low)": "#7f8c8d"}
    for ax, (col, label) in zip(axes, metrics):
        bar_colors = [colors3.get(g, "gray") for g in char_df["group"]]
        ax.bar(range(len(char_df)), char_df[col], color=bar_colors)
        ax.set_xticks(range(len(char_df)))
        ax.set_xticklabels(["A", "B", "C", "D"], fontsize=10)
        ax.set_title(label, fontsize=10)
    fig.suptitle("Concordance-group characteristics (N=302; descriptive)", fontsize=11)
    fig.tight_layout()
    savefig(fig, os.path.join(config.DIR_FIGURES, "phase_x_concordance_group_characteristics"))
    log.info("wrote figure phase_x_concordance_group_characteristics")

# ===========================================================================
# Part D: Full-feature cohort (ICU+MIMIC, 64-feature) -- Full Clinical AI vs
# SAPS II vs APACHE II, clean comparison table + figure (user request:
# "그래프로 AUROC, AUPRC등도 이 3개를 한번에 비교하는 그래프")
# ===========================================================================
log.info("Part D: full-feature-cohort Full Clinical AI vs conventional scores...")
conv_full = pd.read_csv(os.path.join(config.DIR_PERFORMANCE, "conventional_score_performance.csv"))
model_perf = pd.read_csv(os.path.join(config.DIR_PERFORMANCE, "model_performance_comparison.csv"))

# Best Full Clinical AI = catboost, bidirectional external evaluation (Stage1 summary:
# "Best average external AUROC: catboost (0.800)")
cb = model_perf[(model_perf.model_name == "catboost") & (model_perf.pipeline_name == "Universal")]

rows_d = []
for _, r in cb.iterrows():
    rows_d.append({"comparator": f"Full Clinical AI (CatBoost, dev={r.development_dataset})",
                    "evaluated_on": r.external_dataset, "N": int(r.external_N),
                    "AUROC": round(r.external_AUROC, 4), "AUPRC": round(r.external_AUPRC, 4)})
for _, r in conv_full.iterrows():
    rows_d.append({"comparator": r.score, "evaluated_on": r.dataset, "N": int(r.available_N),
                    "AUROC": round(r.AUROC, 4), "AUPRC": round(r.AUPRC, 4)})
full_cmp = pd.DataFrame(rows_d)
OUT_D_CSV = os.path.join(config.DIR_REPORTS, "phase_x_full_cohort_score_comparison.csv")
io_utils.save_csv(full_cmp, OUT_D_CSV)
with open(os.path.join(config.DIR_REPORTS, "phase_x_full_cohort_score_comparison.md"), "w") as fh:
    fh.write("\n".join([
        "# Full-feature cohort (ICU.csv+mimic.csv, 64-feature schema): Full Clinical AI "
        "vs conventional severity scores", "",
        "Full Clinical AI = CatBoost, evaluated on the SAME held-out external dataset it was "
        "never trained on (dev=MIMIC -> evaluated on ICU, or dev=ICU -> evaluated on MIMIC). "
        "Conventional scores (APACHE II, SAPS II, SOFA, qSOFA) evaluated on their own full "
        "dataset (not train/test split -- these are not fitted models). Source: "
        "clinical_ai/performance/model_performance_comparison.csv + conventional_score_performance.csv.",
        "", df_to_md_table(full_cmp),
    ]))
log.info("wrote %s", OUT_D_CSV)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
for ax, dset in zip(axes, ["ICU", "MIMIC"]):
    sub = full_cmp[full_cmp.evaluated_on == dset].sort_values("AUROC")
    colors4 = ["#33549c" if c.startswith("Full Clinical AI") else "#b0631e" for c in sub["comparator"]]
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["AUROC"], color=colors4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["comparator"], fontsize=9)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("AUROC")
    ax.set_title(f"Evaluated on {dset} (N={int(sub['N'].iloc[0]) if len(sub) else 0})", fontsize=10)
fig.suptitle("Full-feature cohort: Full Clinical AI (blue, external-validated) vs conventional scores (orange)", fontsize=11)
fig.tight_layout()
savefig(fig, os.path.join(config.DIR_FIGURES, "phase_x_full_cohort_score_comparison"))
log.info("wrote figure phase_x_full_cohort_score_comparison")

log.info("Done.")
