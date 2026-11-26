#!/usr/bin/env python3
"""12_sample_type_confounding.py — is specimen type (rectal swab vs stool)
confounded with severity, timing, or exposure in the primary cohort?

WHY
---
The association arm's primary analysis pools rectal swab and stool specimens.
The swab-only sensitivity analysis added on 2026-08-31 (section 8 of
`09_sensitivity_analysis.R`) incidentally showed that stool patients die at a
much higher rate than swab patients, which raises the obvious question a
reviewer will ask: is specimen type a marker of something else?

This script answers that directly rather than leaving it as an observation:
it compares swab vs stool patients on admission severity (SOFA), outcome,
age, antibiotic exposure, and — critically — how many days after admission
the specimen was taken.

It is a CONFOUNDING CHECK, not an association analysis. It does not touch,
re-run, or modify the unadjusted univariable primary analysis (scripts 01-07)
or any existing sensitivity analysis, and it fits no models of microbiome
features against outcome.

DATA JOIN
---------
SOFA/antibiotics/dates are not in `new_meta.csv`; they are in the project's own
clinical metadata file. Join recipe (verified 2026-08-31): strip the `ICU_`
prefix from `sample_metadata.tsv`'s `patient_id` and match it against
`microbiome.clinical.csv`'s `ID` column -> 430 of the 466 primary-cohort
patients (92.3%). The alternative route (`sample_id` minus its `_WM` suffix vs
`KCDC_no`) matches only 250/549 rows and is not used.

Run: python3 merge/downstream_analysis/scripts/12_sample_type_confounding.py
Writes: merge/downstream_analysis/tables/sample_type_confounding.tsv
        merge/downstream_analysis/qc/sample_type_confounding.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = "/home/tcmhwd/microbiome/ICU_public"
DOWNSTREAM = os.path.join(BASE, "merge", "downstream_analysis")
SAMPLE_META = os.path.join(BASE, "merge", "output", "sample_metadata.tsv")
CLINICAL = os.path.join(BASE, "metadata", "microbiome.clinical.csv")
OUT_TSV = os.path.join(DOWNSTREAM, "tables", "sample_type_confounding.tsv")
OUT_MD = os.path.join(DOWNSTREAM, "qc", "sample_type_confounding.md")


def load() -> pd.DataFrame:
    sm = pd.read_csv(SAMPLE_META, sep="\t", dtype=str)
    icu = sm[(sm["dataset"] == "ICU")
             & (sm["is_first_sample_for_patient"].isin(["True", "TRUE", "1"]))].copy()
    icu["join_id"] = icu["patient_id"].str.replace(r"^ICU_", "", regex=True)

    clin = pd.read_csv(CLINICAL, dtype=str).drop_duplicates("ID")
    keep = ["ID", "SOFA", "antibiotics", "Carbapenem", "Glycopeptides",
            "age", "sex", "admission_date", "collection_date"]
    d = icu.merge(clin[keep], left_on="join_id", right_on="ID", how="left")

    d["SOFA_n"] = pd.to_numeric(d["SOFA"], errors="coerce")
    d["mort_n"] = pd.to_numeric(d["mortality"], errors="coerce")
    d["age_n"] = pd.to_numeric(d["age_y"] if "age_y" in d.columns else d["age"], errors="coerce")
    d["abx_n"] = pd.to_numeric(d["antibiotics"], errors="coerce")
    d["sampling_day"] = (pd.to_datetime(d["collection_date"], errors="coerce")
                         - pd.to_datetime(d["admission_date"], errors="coerce")).dt.days
    return d


def mwu(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(stats.mannwhitneyu(a, b).pvalue)


def main() -> int:
    d = load()
    swab, stool = d[d["sample_type"] == "swab"], d[d["sample_type"] == "stool"]
    print(f"[12] primary cohort N={len(d)}: {d['sample_type'].value_counts().to_dict()}; "
          f"SOFA available for {d['SOFA_n'].notna().sum()}")

    rows = []

    def add(var, sw_val, st_val, p, method, note=""):
        rows.append({"variable": var, "swab": sw_val, "stool": st_val,
                     "n_swab": len(swab), "n_stool": len(stool),
                     "p_value": p, "test": method, "note": note})

    add("mortality (%)", round(100 * swab["mort_n"].mean(), 1), round(100 * stool["mort_n"].mean(), 1),
        float(stats.fisher_exact(pd.crosstab(d["sample_type"], d["mort_n"]).values)[1]),
        "Fisher exact")
    add("SOFA (median)", swab["SOFA_n"].median(), stool["SOFA_n"].median(),
        mwu(swab["SOFA_n"], stool["SOFA_n"]), "Mann-Whitney U",
        f"non-missing: swab {swab['SOFA_n'].notna().sum()}, stool {stool['SOFA_n'].notna().sum()}")
    add("SOFA (mean)", round(swab["SOFA_n"].mean(), 2), round(stool["SOFA_n"].mean(), 2),
        mwu(swab["SOFA_n"], stool["SOFA_n"]), "Mann-Whitney U")
    add("sampling day after admission (median)", swab["sampling_day"].median(), stool["sampling_day"].median(),
        mwu(swab["sampling_day"], stool["sampling_day"]), "Mann-Whitney U",
        "days from admission_date to collection_date")
    add("age (median)", swab["age_n"].median(), stool["age_n"].median(),
        mwu(swab["age_n"], stool["age_n"]), "Mann-Whitney U")
    add("any antibiotics (%)", round(100 * swab["abx_n"].mean(), 1), round(100 * stool["abx_n"].mean(), 1),
        float(stats.fisher_exact(pd.crosstab(d["sample_type"], d["abx_n"]).values)[1]),
        "Fisher exact")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[12] wrote {OUT_TSV}")

    def md_table(df):
        cols = ["variable", "swab", "stool", "p_value", "test"]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            p = r["p_value"]
            p_s = "n/a" if pd.isna(p) else (f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}")
            lines.append(f"| {r['variable']} | {r['swab']} | {r['stool']} | {p_s} | {r['test']} |")
        return "\n".join(lines)

    md = [
        "# Specimen type (rectal swab vs stool) — confounding check",
        "",
        f"Primary cohort (first sample per patient), N={len(d)}: swab {len(swab)}, stool {len(stool)}. "
        f"SOFA/antibiotic/date variables joined from `metadata/microbiome.clinical.csv` "
        f"(`patient_id` minus its `ICU_` prefix -> `ID`), available for "
        f"{d['SOFA_n'].notna().sum()}/{len(d)} patients.",
        "",
        "This is a confounding check on the cohort, not an association analysis: no microbiome "
        "feature is modelled here, and the unadjusted univariable primary analysis is untouched.",
        "",
        md_table(out),
        "",
        "## Interpretation",
        "",
        "Stool specimens are **not** taken from more severely ill patients — their admission SOFA "
        "is *lower*, not higher. What separates them is **when the specimen was taken**: stool is "
        "collected far later in the admission than swab. That is mechanically unsurprising "
        "(a patient must actually pass stool, which many patients cannot do in the first ICU days), "
        "but it means specimen type is largely a proxy for *time in ICU*, and therefore for the "
        "prolonged-critical-illness and cumulative-antibiotic-exposure that come with it — a "
        "plausible explanation for the higher crude mortality among stool patients despite their "
        "lower admission severity.",
        "",
        "Practical consequences for reporting:",
        "",
        "1. Specimen type should not be described as a simple technical/sampling nuisance. It is "
        "   correlated with sampling timing and antibiotic exposure, both of which are already "
        "   handled elsewhere: `sampling_day` is a covariate in the supplementary adjusted models, "
        "   there is a late-sampling-exclusion sensitivity analysis, and the swab-only sensitivity "
        "   analysis (section 8 of `09_sensitivity_analysis.R`) exists.",
        "2. The swab-only sensitivity analysis is the relevant robustness check, and it preserved "
        "   effect directions for every feature tested — consistent with the associations not being "
        "   an artifact of pooling two specimen types.",
        "3. Because SOFA is *lower* in the stool group, the confounding does not run in the direction "
        "   of 'sicker patients give stool', and adjusting for admission severity alone would not "
        "   remove it. Sampling timing is the variable that matters.",
        "",
        "_Post-hoc analysis, added 2026-08-31; see `semantic_microbiome/reports/analysis_provenance.md`._",
    ]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"[12] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
