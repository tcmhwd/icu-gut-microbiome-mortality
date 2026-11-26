"""
lit_cc5_a3_severity_antibiotics.py -- Addendum 2, analysis A3 (POST_HOC_EXPLORATORY).
Plan: semantic_microbiome/reports/paper1/CRITICAL_CARE_ADDENDUM2_ANALYSIS_PLAN.md

Question: how strongly do the microbiome score / pathobiont-balance features track SOFA and antibiotic exposure,
and does their unadjusted association with in-hospital death attenuate after adjustment?
Read together with TS4 (clinical variables did not identify the microbiome-only quadrant, AUROC 0.537): the two
findings are not interchangeable, and this script's output does not by itself support "already contained in
the clinical information".

Inputs, all pre-existing: nested-CV OOF M1 probability + SOFA + outcome (phase_x_sofa_nested_oof_probabilities.csv),
literature-core features (structured/), antibiotic exposure from metadata/microbiome.clinical.csv
(binary `antibiotics`; number of antibiotic classes = non-empty tokens of `struc_anti`).
Association model: logistic regression per SD of the z-scored exposure. Spearman for correlations.
Family of tests is stated in the output; no multiplicity correction is applied (exploratory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import SEMANTIC, ICU_PUBLIC

OUT = SEMANTIC / "reports" / "paper1" / "critical_care_external" / "addendum2"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6

o = pd.read_csv(ICU_PUBLIC / "clinical metadata" / "clinical_ai" / "predictions" / "phase_x_sofa_nested_oof_probabilities.csv")
disc = pd.read_csv(SEMANTIC / "harmonized" / "discovery_cohort_primary.csv")[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
clin = pd.read_csv(ICU_PUBLIC / "metadata" / "microbiome.clinical.csv").merge(
    disc, left_on="KCDC_no", right_on="kcdc_no", how="inner")
lit = pd.read_csv(SEMANTIC / "structured" / "literature_core_features.csv", index_col="sample_id")

d = o[["patient_id_clinical", "hospital_mortality", "SOFA", "micro_prob_M1"]].merge(
    clin[["patient_id_clinical", "antibiotics", "struc_anti"]], on="patient_id_clinical", how="left", validate="one_to_one")
d = d.join(lit.reindex(d["patient_id_clinical"]).reset_index(drop=True))
d["n_abx_classes"] = d["struc_anti"].fillna("").apply(lambda s: sum(1 for t in str(s).split(",") if t.strip()))
d["M1_logit"] = np.log(np.clip(d["micro_prob_M1"], EPS, 1 - EPS) / (1 - np.clip(d["micro_prob_M1"], EPS, 1 - EPS)))
assert len(d) == 302 and int(d["hospital_mortality"].sum()) == 69 and d["antibiotics"].notna().all()

exposures = {
    "M1 score (logit of out-of-fold probability)": "M1_logit",
    "Facultative-to-obligate anaerobe balance": "Literature_facultative_to_obligate_anaerobe_balance",
    "Pathobiont burden": "Literature_pathobiont_burden",
    "Klebsiella (genus rel. abundance)": "Literature_Klebsiella_genus_relabund",
    "Enterococcus (genus rel. abundance)": "Literature_Enterococcus_genus_relabund",
    "Escherichia (genus rel. abundance)": "Literature_Escherichia_genus_relabund",
    "Shannon diversity": "Literature_alpha_diversity_shannon",
}

corr_rows, assoc_rows = [], []
for lab, col in exposures.items():
    x = d[col].astype(float)
    for cl, cc in (("SOFA", "SOFA"), ("antibiotic exposure (binary)", "antibiotics"), ("number of antibiotic classes", "n_abx_classes")):
        rho, p = stats.spearmanr(x, d[cc].astype(float), nan_policy="omit")
        corr_rows.append(dict(exposure=lab, versus=cl, spearman_rho=round(rho, 3), p=round(p, 4), N=int(x.notna().sum())))
    z = (x - x.mean()) / x.std(ddof=0)
    y = d["hospital_mortality"].values
    for adj_lab, adj_cols in (("unadjusted", []), ("+ SOFA", ["SOFA"]), ("+ SOFA + antibiotic exposure", ["SOFA", "antibiotics"])):
        X = pd.concat([z.rename("z")] + [d[c].astype(float) for c in adj_cols], axis=1)
        ok = X.notna().all(axis=1)
        fit = sm.Logit(y[ok.values], sm.add_constant(X[ok])).fit(disp=0)
        lo, hi = np.exp(fit.conf_int().loc["z"])
        assoc_rows.append(dict(exposure=lab, model=adj_lab, OR_per_SD=round(float(np.exp(fit.params["z"])), 3),
                               ci_low=round(float(lo), 3), ci_high=round(float(hi), 3),
                               p=round(float(fit.pvalues["z"]), 4), N=int(ok.sum())))

corr = pd.DataFrame(corr_rows); assoc = pd.DataFrame(assoc_rows)
corr.to_csv(OUT / "A3_correlations.csv", index=False)
assoc.to_csv(OUT / "A3_adjusted_association.csv", index=False)
md = ["# A3 — relation of the microbiome signal to severity and antibiotics (POST_HOC_EXPLORATORY)", "",
      f"N=302, 69 deaths. {len(corr)} Spearman correlations and {len(assoc)} logistic models; no multiplicity correction.",
      "Association models use a fixed covariate set (SOFA; SOFA + binary antibiotic exposure). The M1 score is the",
      "out-of-fold probability, so its adjusted OR is anti-conservative in the same way as the S1 row of Table 2.",
      "This output alone does not show that the microbiome signal is 'contained in' the clinical information (see TS4).", "",
      "## Correlations", "", corr.to_markdown(index=False), "", "## Association with in-hospital death (per SD)", "",
      assoc.to_markdown(index=False)]
(OUT / "A3_severity_antibiotics.md").write_text("\n".join(md))
print(corr.to_string(index=False)); print(assoc.to_string(index=False))
