#!/usr/bin/env python3
"""34_table1_baseline_characteristics.py

Paper-story addition (user request, 2026-08-17): a standard "Table 1"
(baseline characteristics, survivors vs deaths, with between-group p-values)
for the three cohorts used in the paper story -- something no existing
report in this project provides. Descriptive only; no model fitting.

Continuous variables: median [IQR], Mann-Whitney U test (non-parametric,
robust to the skew typical of ICU labs/vitals).
Categorical variables: n (%), chi-square test (Fisher's exact if any
expected cell count < 5).

Cohorts:
  (A) Gangnam 302-patient microbiome cohort (metadata/microbiome.clinical.csv)
  (B) ICU.csv full-feature cohort (landmark-eligible, N=64,167)
  (C) mimic.csv full-feature cohort (landmark-eligible, N=63,212)

hospital_mortality used only as the grouping variable for descriptive
comparison -- no fitting, no feature selection.

Reads:
  metadata/microbiome.clinical.csv, semantic_microbiome/harmonized/discovery_cohort_primary.csv
  ICU.csv, mimic.csv (via utils.harmonize)
Writes:
  clinical_ai/reports/table1_gangnam_302.csv/.md
  clinical_ai/reports/table1_icu_full.csv/.md
  clinical_ai/reports/table1_mimic_full.csv/.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, harmonize, io_utils  # noqa: E402

log = io_utils.setup_logging("34_table1_baseline_characteristics")
MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def cont_row(label, values, group):
    values = pd.to_numeric(values, errors="coerce")
    v0, v1 = values[group == 0].dropna(), values[group == 1].dropna()
    if len(v0) < 2 or len(v1) < 2:
        return {"variable": label, "survivors": "-", "deaths": "-", "p": "-", "test": "-"}
    med0, q1_0, q3_0 = v0.median(), v0.quantile(.25), v0.quantile(.75)
    med1, q1_1, q3_1 = v1.median(), v1.quantile(.25), v1.quantile(.75)
    try:
        _, p = stats.mannwhitneyu(v0, v1, alternative="two-sided")
    except ValueError:
        p = np.nan
    return {"variable": label, "survivors": f"{med0:.2f} [{q1_0:.2f}-{q3_0:.2f}] (N={len(v0)})",
            "deaths": f"{med1:.2f} [{q1_1:.2f}-{q3_1:.2f}] (N={len(v1)})",
            "p": fmt_p(p), "test": "Mann-Whitney U"}


def fmt_p(p):
    if pd.isna(p):
        return "-"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def cat_row(label, values, group):
    flag = values.astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true", "male", "m"])
    ct = pd.crosstab(flag, group)
    if ct.shape != (2, 2):
        return {"variable": label, "survivors": "-", "deaths": "-", "p": "-", "test": "-"}
    n0, n1 = (group == 0).sum(), (group == 1).sum()
    pct0 = 100 * ct.get(0, pd.Series([0, 0])).get(True, 0) / n0 if n0 else np.nan
    pct1 = 100 * ct.get(1, pd.Series([0, 0])).get(True, 0) / n1 if n1 else np.nan
    n_pos0, n_pos1 = ct.get(0, pd.Series([0, 0])).get(True, 0), ct.get(1, pd.Series([0, 0])).get(True, 0)
    expected_min = stats.contingency.expected_freq(ct.values).min() if ct.values.sum() else 0
    try:
        if expected_min < 5:
            _, p = stats.fisher_exact(ct.values)
            test = "Fisher exact"
        else:
            _, p, _, _ = stats.chi2_contingency(ct.values)
            test = "chi-square"
    except ValueError:
        p, test = np.nan, "-"
    return {"variable": label, "survivors": f"{n_pos0} ({pct0:.1f}%)", "deaths": f"{n_pos1} ({pct1:.1f}%)",
            "p": fmt_p(p), "test": test}


# ===========================================================================
# (A) Gangnam 302-patient microbiome cohort
# ===========================================================================
log.info("(A) Gangnam 302-patient cohort...")
raw_clin = pd.read_csv(os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv"))
disc = pd.read_csv(os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv"))
disc_ids = disc[["patient_id_clinical", "kcdc_no", "hospital_mortality"]].drop_duplicates()
merged = disc_ids.merge(raw_clin, left_on="kcdc_no", right_on="KCDC_no", how="inner")
log.info("Gangnam Table 1 cohort: N=%d (deaths=%d)", len(merged), int(merged.hospital_mortality.sum()))
g = merged["hospital_mortality"].astype(int)

rows_a = [
    cont_row("Age (years)", merged["age"], g),
    cat_row("Sex, female", merged["sex"].astype(str).str.upper().str.startswith("F"), g),
    cont_row("BMI", merged["BMI"], g),
    cont_row("SOFA score", merged["SOFA"], g),
    cont_row("Charlson Comorbidity Index (CCI)", merged["CCI"], g),
    cat_row("Diabetes mellitus", merged["DM"], g),
    cat_row("Hypertension", merged["HTN"], g),
    cat_row("Cerebrovascular accident (CVA)", merged["CVA"], g),
    cat_row("COPD", merged["COPD"], g),
    cat_row("Cancer (any)", merged["cancer"], g),
    cat_row("Chronic liver disease", merged["liver.ds"], g),
    cat_row("Mechanical ventilation", merged.get("ventilator", pd.Series(dtype=str)), g) if "ventilator" in merged.columns else
        {"variable": "Mechanical ventilation", "survivors": "n/a", "deaths": "n/a", "p": "-", "test": "-"},
    cont_row("Albumin (g/dL)", merged["Alb"], g),
    cont_row("CRP (mg/L)", merged["CRP"], g),
    cont_row("WBC (x10^3/uL)", merged["WBC"], g),
    cont_row("Platelet (x10^3/uL)", merged["PLT"], g),
    cont_row("Creatinine (mg/dL)", merged["Cr"], g),
    cont_row("Total bilirubin (mg/dL)", merged["T.bil"], g),
]
t1a = pd.DataFrame(rows_a)
io_utils.save_csv(t1a, os.path.join(config.DIR_REPORTS, "table1_gangnam_302.csv"))
with open(os.path.join(config.DIR_REPORTS, "table1_gangnam_302.md"), "w") as fh:
    fh.write("\n".join([
        "# Table 1. Baseline characteristics -- Gangnam 302-patient microbiome cohort",
        "",
        f"N=302 total (survivors N={int((g==0).sum())}, deaths N={int((g==1).sum())}). Continuous: "
        "median [IQR], Mann-Whitney U. Categorical: n (%), chi-square (Fisher's exact if expected "
        "cell count <5). Descriptive only -- no feature selection or model fitting.",
        "", df_to_md(t1a),
    ]))
log.info("wrote table1_gangnam_302.csv/.md")

# ===========================================================================
# (B)/(C) ICU.csv / mimic.csv full-feature cohorts
# ===========================================================================
FULL_COMORBIDITIES = [("DM", "Diabetes mellitus"), ("HTN", "Hypertension"), ("COPD", "COPD"),
                       ("cancer", "Cancer (any)"), ("CVD", "Cardiovascular disease"),
                       ("renal ds", "Chronic renal disease"), ("liver ds", "Chronic liver disease")]

for dataset, label, outname in [("ICU", "ICU.csv (Sinchon+Gangnam)", "table1_icu_full"),
                                 ("MIMIC", "mimic.csv", "table1_mimic_full")]:
    log.info("(%s) full-feature cohort...", dataset)
    raw = io_utils.load_raw(dataset)
    bundle = harmonize.build_model_ready_frame(raw, dataset, config.ID_COL_ICU if dataset == "ICU" else config.ID_COL_MIMIC)
    landmark = bundle["landmark"]
    predictors = bundle["predictors"]
    severity = bundle["severity"]
    target = bundle["target"].astype(int)
    log.info("%s Table 1 cohort: N=%d (deaths=%d)", dataset, len(target), int(target.sum()))

    rows = [
        cont_row("Age (years)", landmark["age"], target),
        cat_row("Sex, male", landmark["sex"], target),
        cont_row("SOFA score", severity["SOFA"], target),
        cont_row("APACHE II score", severity["apach2"], target),
        cont_row("SAPS II score", severity["SAPS2"], target),
        cont_row("qSOFA score", severity["qSOFA"], target),
    ]
    for raw_col, disp in FULL_COMORBIDITIES:
        clean = raw_col.strip().replace(" ", "_").replace("/", "_").replace(".", "_")
        if clean in predictors.columns:
            flag = predictors[clean]
            n0, n1 = (target == 0).sum(), (target == 1).sum()
            n_pos0 = int((flag[target == 0] == 1).sum())
            n_pos1 = int((flag[target == 1] == 1).sum())
            ct = np.array([[n0 - n_pos0, n1 - n_pos1], [n_pos0, n_pos1]])
            try:
                _, p, _, _ = stats.chi2_contingency(ct)
            except ValueError:
                p = np.nan
            rows.append({"variable": disp, "survivors": f"{n_pos0} ({100*n_pos0/n0:.1f}%)",
                         "deaths": f"{n_pos1} ({100*n_pos1/n1:.1f}%)",
                         "p": fmt_p(p), "test": "chi-square"})
    for col, disp in [("WBC_max", "WBC max (x10^3/uL)"), ("Cr_max", "Creatinine max (mg/dL)"),
                       ("albumin_min", "Albumin min (g/dL)"), ("bilirubin_max", "Bilirubin max (mg/dL)"),
                       ("PLT_min", "Platelet min (x10^3/uL)")]:
        if col in predictors.columns:
            rows.append(cont_row(disp, predictors[col], target))
    t1 = pd.DataFrame(rows)
    io_utils.save_csv(t1, os.path.join(config.DIR_REPORTS, f"{outname}.csv"))
    with open(os.path.join(config.DIR_REPORTS, f"{outname}.md"), "w") as fh:
        fh.write("\n".join([
            f"# Table 1. Baseline characteristics -- {label} (landmark-eligible, N={len(target)})",
            "",
            f"Survivors N={int((target==0).sum())}, deaths N={int((target==1).sum())}. At this sample "
            "size nearly all comparisons reach p<0.001 -- effect sizes (the median/% values themselves) "
            "matter more than p-values here; reported for completeness/reviewer expectation, not as "
            "evidence of anything beyond 'large-N ICU cohort, sicker patients die more'.",
            "", df_to_md(t1),
        ]))
    log.info("wrote %s.csv/.md", outname)

log.info("Done.")
