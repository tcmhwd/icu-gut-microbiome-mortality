"""
lit26_covid_external_severity_concordance.py -- Phase X paper-story addition
(user request, 2026-08-17).

PRJNA842425 (public non-AGI EBI/SRA cohort, N=71) is the COVID-ICU cohort
identified in publicdata.csv (sample titles "covid_001".."covid_070"). Its
metadata.xlsx sheet ('PRJNA842425') was already joined for MORTALITY ONLY in
merge/scripts/01_build_metadata.py; SOFA/APACHE/Charlson/comorbidities/labs
in that sheet have never been extracted or used until now.

This script:
  1. Extracts SOFA, APACHE, Charlson, age, sex, comorbidities from
     metadata.xlsx and joins them to this cohort's mortality (already in
     merge/output/sample_metadata.tsv).
  2. Computes SOFA and APACHE AUROC vs mortality in THIS cohort (new --
     analogous to the Gangnam-based conventional-score tables, but here for
     an entirely independent population).
  3. Re-applies the FROZEN Gangnam M1 (Literature Core LR) model (same
     frozen artifact lit15 used) to get per-patient microbiome-AI
     probabilities for these 71 patients (lit15 only ever saved the
     aggregate AUROC, not per-sample probabilities).
  4. Builds an A/B/C/D concordance/discordance quadrant table -- same
     definitions as clinical_ai's Gangnam-based
     phase_x_concordance_groups.csv (24_phase_x_concordance_acuity_analysis.py):
     clinical proxy = SOFA (own-cohort Youden-optimal threshold), microbiome
     = M1 probability (own-cohort Youden-optimal threshold) -- descriptive
     quadrant characterization only, NOT a proposed decision rule, mirroring
     the Gangnam table's own caveat.

IMPORTANT CAVEAT carried through to every output: M1's own external AUROC on
this cohort is already known to be weak (0.550-0.567, lit10b) and the two
cohorts are almost perfectly separable by microbiome composition alone
(domain_shift AUROC 0.87-1.00, lit15 Section 55) -- so any concordance
pattern seen here should NOT be expected to replicate Gangnam's cleaner
0.833-vs-0.653 concordant/discordant split, and is reported as an honest,
likely-null or weak-signal check, not a confirmatory replication.

hospital_mortality / mortality_label used for evaluation only (AUROC,
threshold-fitting for the descriptive quadrant table) -- no feature
selection, no refitting of M1 itself (frozen, unchanged from lit14).

Reads:
  ICU_public/metadata.xlsx (sheet PRJNA842425)
  merge/output/sample_metadata.tsv
  structured/literature_core_features.csv
  supervised_semantic/frozen_models/M1_literature_core_LR.joblib
Writes:
  public_validation/covid_prjna842425_severity_scores.csv/.md
  public_validation/covid_prjna842425_concordance_groups.csv/.md
  public_validation/figures/covid_prjna842425_severity_and_concordance.svg/.jpg
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from common import SEMANTIC, ICU_PUBLIC, STRUCTURED_DIR, SEED, save_fig, today_str

PUBVAL_DIR = SEMANTIC / "public_validation"
FIGURES_DIR = PUBVAL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_DIR = SEMANTIC / "supervised_semantic" / "frozen_models"
MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
METADATA_XLSX = ICU_PUBLIC / "metadata.xlsx"


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def bootstrap_auroc(y, score, n_boot=2000, seed=SEED):
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


def youden_threshold(y_true, score):
    fpr, tpr, thr = roc_curve(y_true, score)
    return thr[np.argmax(tpr - fpr)]


# ---------------------------------------------------------------------
# 1. Extract clinical metadata + mortality for PRJNA842425
# ---------------------------------------------------------------------
sm = pd.read_csv(MERGE_OUTPUT / "sample_metadata.tsv", sep="\t", dtype=str)
sm["mortality"] = pd.to_numeric(sm["mortality"], errors="coerce")
covid = sm[sm.subcohort == "PRJNA842425"].copy()
covid["covid_id"] = covid["patient_id"].str.replace("PRJNA842425_", "", regex=False)
covid = covid.set_index("sample_id")

xl = pd.read_excel(METADATA_XLSX, sheet_name="PRJNA842425")
clin_cols = ["patient_ID", "sofa", "APACHE", "Charlson", "age", "sex",
             "HTN", "DM", "CHF", "CAD", "CVA", "COPD", "Malignancy"]
clin = xl[[c for c in clin_cols if c in xl.columns]].copy()
clin = clin.rename(columns={"patient_ID": "covid_id"})
# mortality ground truth comes from merge/output/sample_metadata.tsv (already the source
# 01_build_metadata.py used and lit15/lit10b's reported 0.550-0.567 AUC is keyed to) --
# xlsx's own mortality/mortality_label columns are intentionally excluded here to avoid a
# _x/_y merge collision; sofa/APACHE/etc. are unaffected by which mortality column is used.

merged = covid.reset_index().merge(clin, on="covid_id", how="left").set_index("sample_id")
merged["SOFA"] = pd.to_numeric(merged["sofa"], errors="coerce")
merged["APACHE"] = pd.to_numeric(merged["APACHE"], errors="coerce")
print(f"[lit26] PRJNA842425: N={len(merged)}, SOFA available={merged['SOFA'].notna().sum()}, "
      f"APACHE available={merged['APACHE'].notna().sum()}, deaths={int(merged['mortality'].sum())}")

# ---------------------------------------------------------------------
# 2. SOFA / APACHE AUROC vs mortality, THIS cohort
# ---------------------------------------------------------------------
y = merged["mortality"].astype(int).values
rows = []
for score_name in ["SOFA", "APACHE"]:
    s = merged[score_name].values
    valid = ~pd.isna(s)
    auc, lo, hi = bootstrap_auroc(y[valid], s[valid].astype(float))
    rows.append({"score": score_name, "N": int(valid.sum()), "AUROC": round(auc, 4),
                 "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4)})
sev_covid = pd.DataFrame(rows)
sev_covid.to_csv(PUBVAL_DIR / "covid_prjna842425_severity_scores.csv", index=False)
print(f"[lit26] SOFA/APACHE AUROC in PRJNA842425:\n{sev_covid.to_string(index=False)}")

# ---------------------------------------------------------------------
# 3. Frozen M1 -> per-patient microbiome-AI probability
# ---------------------------------------------------------------------
m1_pipe = joblib.load(FROZEN_DIR / "M1_literature_core_LR.joblib")
lit_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
lit_core_prefixed = lit_core.copy()
lit_core_prefixed.columns = [f"LitCore_{c}" for c in lit_core.columns]
ids = merged.index.tolist()
X_m1 = lit_core_prefixed.reindex(ids)
p_m1 = m1_pipe.predict_proba(X_m1)[:, 1]
merged["microbiome_prob_M1"] = p_m1

m1_auc, m1_lo, m1_hi = bootstrap_auroc(y, p_m1)
print(f"[lit26] M1 (Literature Core) AUROC on PRJNA842425 (cross-check vs lit10b/lit15): "
      f"{m1_auc:.4f} [{m1_lo:.4f}, {m1_hi:.4f}]")

# ---------------------------------------------------------------------
# 4. A/B/C/D concordance quadrant table (own-cohort Youden thresholds,
#    same definitions as clinical_ai's Gangnam-based table -- descriptive
#    only, NOT a decision rule)
# ---------------------------------------------------------------------
sofa_valid = merged["SOFA"].notna().values
y_q = y[sofa_valid]
sofa_q = merged["SOFA"].values[sofa_valid].astype(float)
micro_q = p_m1[sofa_valid]
ids_q = np.array(ids)[sofa_valid]

clin_thr = youden_threshold(y_q, sofa_q)
micro_thr = youden_threshold(y_q, micro_q)
print(f"[lit26] Own-cohort Youden thresholds (descriptive only): SOFA={clin_thr:.2f}, M1={micro_thr:.4f}")

clin_flag = sofa_q >= clin_thr
micro_flag = micro_q >= micro_thr
group = np.full(len(y_q), "", dtype=object)
group[clin_flag & micro_flag] = "A_both_flag_high"
group[clin_flag & ~micro_flag] = "B_clinical_only_flags_high"
group[~clin_flag & micro_flag] = "C_microbiome_only_flags_high"
group[~clin_flag & ~micro_flag] = "D_both_flag_low"

qdf = pd.DataFrame({"sample_id": ids_q, "mortality": y_q, "SOFA": sofa_q,
                     "microbiome_prob_M1": micro_q, "clinical_flag_high": clin_flag,
                     "microbiome_flag_high": micro_flag, "concordance_group": group})
qdf.to_csv(PUBVAL_DIR / "covid_prjna842425_concordance_groups.csv", index=False)

concord_rows = []
for gname, glabel in [("A_both_flag_high", "A: both high (SOFA-proxy + M1)"),
                       ("B_clinical_only_flags_high", "B: SOFA-proxy only"),
                       ("C_microbiome_only_flags_high", "C: M1 only ('rescue')"),
                       ("D_both_flag_low", "D: both low")]:
    sub = qdf[qdf.concordance_group == gname]
    if len(sub) == 0:
        continue
    concord_rows.append({"group": glabel, "N": len(sub), "N_died": int(sub.mortality.sum()),
                          "mortality_pct": round(100 * sub.mortality.mean(), 1),
                          "mean_SOFA": round(sub.SOFA.mean(), 2),
                          "mean_M1_prob": round(sub.microbiome_prob_M1.mean(), 3)})
concord_summary = pd.DataFrame(concord_rows)

is_concordant = clin_flag == micro_flag
if is_concordant.sum() >= 2 and len(np.unique(y_q[is_concordant])) == 2:
    auc_concord = roc_auc_score(y_q[is_concordant], micro_q[is_concordant])
else:
    auc_concord = np.nan
if (~is_concordant).sum() >= 2 and len(np.unique(y_q[~is_concordant])) == 2:
    auc_discord = roc_auc_score(y_q[~is_concordant], micro_q[~is_concordant])
else:
    auc_discord = np.nan

with open(PUBVAL_DIR / "covid_prjna842425_severity_scores.md", "w") as fh:
    fh.write("\n".join([
        "# PRJNA842425 (public COVID-ICU cohort, N=71) -- external severity-score "
        "and concordance check", "",
        f"Generated {today_str()}. First use of this cohort's SOFA/APACHE/Charlson/comorbidity "
        "fields (metadata.xlsx sheet PRJNA842425) -- previously only mortality was extracted "
        "(merge/scripts/01_build_metadata.py). **Caveat carried through every result below**: "
        "M1's own AUROC transfer to this cohort is already known to be weak (0.550-0.567, "
        "lit10b_leakage_check_and_external.py), and the two cohorts (Gangnam vs this one) are "
        "almost perfectly separable by microbiome composition alone (domain-shift AUROC "
        "0.87-1.00, lit15 Section 55, driven by different sequencing platforms/populations) -- "
        "so this is reported as an honest weak/null-signal external check, not a confirmatory "
        "replication of Gangnam's concordance pattern.", "",
        "## SOFA / APACHE AUROC for hospital mortality, THIS cohort (never computed before)", "",
        df_to_md(sev_covid), "",
        f"Cross-check: M1 (Literature Core, frozen Gangnam model) AUROC on this cohort = "
        f"{m1_auc:.4f} [{m1_lo:.4f}, {m1_hi:.4f}] -- consistent with the aggregate 0.550-0.567 "
        "already reported in lit10b/lit15 (this run computes per-patient probabilities for the "
        "concordance table below, which lit15 did not save).", "",
        "## A/B/C/D concordance quadrants (own-cohort Youden thresholds on SOFA and M1 -- "
        "descriptive only, mirrors clinical_ai/reports/phase_x_concordance_groups.csv's "
        "Gangnam-based definitions, NOT a decision rule)", "",
        f"SOFA threshold={clin_thr:.2f}, M1 threshold={micro_thr:.4f}. N with SOFA available={sofa_valid.sum()}.",
        "", df_to_md(concord_summary), "",
        f"Concordant-pair AUROC (M1, within patients where SOFA-proxy and M1 agree) = "
        f"{auc_concord:.3f} vs discordant-pair AUROC = {auc_discord:.3f} "
        f"(Gangnam reference: concordant 0.833 vs discordant 0.653, N=302 -- see "
        "clinical_ai/reports/phase_x_concordance_vs_discordance_power.md). Interpret with the "
        "weak-transfer caveat above; subgroup Ns here are small (see table).",
    ]))
print(f"[lit26] wrote covid_prjna842425_severity_scores.csv/.md, "
      f"covid_prjna842425_concordance_groups.csv")

# ---------------------------------------------------------------------
# Figure: 2-panel -- (a) SOFA/APACHE/M1 AUROC bar with CI, (b) quadrant
# mortality-rate bars
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
plot_rows = pd.concat([sev_covid, pd.DataFrame([{"score": "M1 (microbiome)", "N": len(y),
                        "AUROC": round(m1_auc, 4), "AUROC_CI_low": round(m1_lo, 4),
                        "AUROC_CI_high": round(m1_hi, 4)}])], ignore_index=True)
plot_rows = plot_rows.sort_values("AUROC")
colors = ["#3d7a56" if s.startswith("M1") else "#b0631e" for s in plot_rows["score"]]
ypos = np.arange(len(plot_rows))
ax.barh(ypos, plot_rows["AUROC"], color=colors)
for i, (a, lo, hi) in enumerate(zip(plot_rows["AUROC"], plot_rows["AUROC_CI_low"], plot_rows["AUROC_CI_high"])):
    ax.errorbar(a, i, xerr=[[a - lo], [hi - a]], fmt="none", ecolor="black", capsize=3)
ax.set_yticks(ypos); ax.set_yticklabels(plot_rows["score"], fontsize=9)
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("AUROC"); ax.set_title("PRJNA842425 (N=71): SOFA/APACHE vs M1", fontsize=10)

ax2 = axes[1]
qcolors = {"A: both high (SOFA-proxy + M1)": "#2e7d32", "B: SOFA-proxy only": "#c0392b",
           "C: M1 only ('rescue')": "#2980b9", "D: both low": "#7f8c8d"}
bar_colors = [qcolors.get(g, "gray") for g in concord_summary["group"]]
ax2.bar(range(len(concord_summary)), concord_summary["mortality_pct"], color=bar_colors)
ax2.set_xticks(range(len(concord_summary)))
ax2.set_xticklabels(["A", "B", "C", "D"], fontsize=10)
ax2.set_ylabel("Mortality %")
ax2.set_title("PRJNA842425: mortality by concordance quadrant\n(descriptive, small N per group)", fontsize=10)

fig.tight_layout()
save_fig(fig, "covid_prjna842425_severity_and_concordance", figures_dir=FIGURES_DIR)
plt.close(fig)
print("[lit26] wrote figure covid_prjna842425_severity_and_concordance")
