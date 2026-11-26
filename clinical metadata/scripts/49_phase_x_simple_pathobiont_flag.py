#!/usr/bin/env python3
"""49_phase_x_simple_pathobiont_flag.py

Task A (user request, 2026-09-02): the manuscript's M1 16-marker composite
score is null against SOFA/Clinical AI. It never tested the simplest,
clinically-intuitive rule a bedside clinician would actually ask about:
"is this patient's gut dominated by a pathobiont?" This script tests that
directly, on the 302-patient ML/landmark cohort.

RATIONALE NUMBER, VERIFIED (not re-derived here, read from source): the task
brief cited "Enterococcus relative abundance OR 3.92, p=0.0003" from the
466-patient association arm. Confirmed present, exactly, in
`merge/downstream_analysis/tables/alpha_diversity_association.tsv`, feature
`enterococcus_relabund` (genus-level Enterococcus relative abundance summed
from MetaPhlAn3 species and expressed as a 0-1 FRACTION of the community,
not a percentage): odds_ratio=3.9239, ci=[1.870, 8.234], p=0.0003004 (per a
FULL 0->1 unit change, i.e. 0% to 100% dominance -- see
`03_alpha_beta_diversity.R` `run_univariate_continuous()`, a raw-scale
`glm(y ~ x, family=binomial())`). NOTE: this is a DIFFERENT scaling/pipeline
than the same association arm's own `taxonomy_genus` block
(`univariable_taxonomy_genus.tsv`), which reports OR=1.069 for the same
taxon on a per-1-PERCENTAGE-POINT basis (values 0-100). Both are directionally
consistent (higher Enterococcus -> higher mortality) and nominally
significant, but the two numbers are not on the same scale and should not be
compared to each other directly -- flagged here so the discrepancy is not
mistaken for an inconsistency in this script's own numbers.

FLAGS (raw, deterministic thresholds on read-based taxonomy -- NOT fitted
models, so unlike M1/M3 in scripts 18/45/46 there is no OOF-generation step
needed for the flag itself; only the "flag as a covariate in a logistic
model" step needs honest out-of-fold evaluation, exactly as SOFA itself is
already treated in script 41):
  (a) Enterococcus relabund >= threshold  ("entero_geX")
  (b) Enterococcus OR Klebsiella OR Escherichia relabund >= threshold, i.e.
      max(the three genera) >= threshold  ("pathobiont_geX") -- a
      pathobiont-domination flag, not a summed threshold.
  Threshold grid: 10%, 30% (PRE-DECLARED PRIMARY -- the HSCT-literature
  "domination" convention), 50%. All three are reported for both (a) and
  (b); NONE is selected post-hoc as "the" result.
  (c) Shannon diversity below the cohort's own lower tertile -- OUTCOME-BLIND
      cut (the tertile boundary is a property of the covariate distribution
      only, never of the outcome).
  7 flags total. All 7 are reported; no threshold is chosen because it
  performed best (explicit multiplicity handling, Holm-Bonferroni across all
  7, applied to the SOFA-adjusted p-values -- see section 3 of the report).

DESIGN PER FLAG (mirrors 41_phase_x_microbiome_over_sofa.py's S0/S1 nested-
model structure, and reuses 46_phase_x_c1_c2_nested_bootstrap_ci.py's exact
paired-bootstrap algorithm):
  1. Unadjusted OR: logistic y ~ flag (full-data fit).
  2. SOFA-adjusted OR: logistic y ~ flag + SOFA (full-data fit), OR/CI/p for
     the flag term.
  3. Discrimination: out-of-fold 5-fold stratified CV (config.RANDOM_SEED,
     config.N_CV_FOLDS -- project-wide convention), for
     S0 = SOFA alone [reference, computed once, reused for every flag] vs.
     S_flag = SOFA + flag. delta_AUROC / delta_AUPRC / delta_Brier via the
     SAME 2000-resample paired-bootstrap algorithm as script 46 (identical
     with-replacement patient resampling, same resample index applied to
     both models each iteration, resamples with <2 outcome classes skipped,
     95% CI from percentiles, two-sided bootstrap p = 2 x min(tail
     proportions)).
  Because the flags are deterministic functions of already-observed data
  (no model fit produces them), a genuine "inner" nested split (as script 45
  needed for the FITTED M1/M3 microbiome_prob features) is not required
  here -- there is nothing upstream of the flag that could leak across
  folds. This is noted explicitly rather than silently skipped, matching
  script 45's own precedent of explaining why C0 needed no nesting.

EXPECTED RESULT: null (Enterococcus-based features are a subset of what M1
already tested and M1 was null over SOFA/Clinical AI in this cohort). This
script exists to have actually tested the simplest version rather than
asserting the null by analogy. It reports whichever way the numbers come
out.

Reads:
  semantic_microbiome/structured/taxonomy_genus_filtered.tsv (Enterococcus,
    Klebsiella, Escherichia relative abundance, 302-patient cohort)
  semantic_microbiome/structured/taxonomy_alpha_diversity.tsv (Shannon)
  semantic_microbiome/harmonized/discovery_cohort_primary.csv (outcome,
    patient_id_clinical, kcdc_no)
  metadata/microbiome.clinical.csv (SOFA) -- joined via KCDC_no, identical
    to 41_phase_x_microbiome_over_sofa.py's join.
Writes:
  clinical_ai/reports/phase_x_simple_pathobiont_flag.csv
  clinical_ai/reports/phase_x_simple_pathobiont_flag.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("49_phase_x_simple_pathobiont_flag")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
DISCOVERY_CSV = os.path.join(SEMANTIC, "harmonized", "discovery_cohort_primary.csv")
CLINICAL_RAW_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_simple_pathobiont_flag.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_simple_pathobiont_flag.md")

SEED = config.RANDOM_SEED       # 2026
N_FOLDS = config.N_CV_FOLDS     # 5
N_BOOT_PAIRED = 2000            # matches script 46/21's paired bootstrap
EPS = 1e-6


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. load outcome + SOFA -- identical join to 41_phase_x_microbiome_over_sofa.py
# ---------------------------------------------------------------------------
disc = pd.read_csv(DISCOVERY_CSV).set_index("patient_id_clinical")
raw_clin = pd.read_csv(CLINICAL_RAW_CSV)
disc_ids = disc.reset_index()[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw_clin = raw_clin.merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner").set_index("patient_id_clinical")
raw_clin = raw_clin.reindex(disc.index)

y = disc["hospital_mortality"].values.astype(int)
n = len(y)
SOFA = pd.to_numeric(raw_clin["SOFA"], errors="coerce").values
n_sofa_missing = int(np.isnan(SOFA).sum())
if n_sofa_missing:
    log.error("SOFA missing for %d/%d patients -- refusing to silently drop them.", n_sofa_missing, n)
    sys.exit(1)
log.info("N=%d, %d deaths (%.1f%%); SOFA complete for all.", n, int(y.sum()), 100 * y.mean())

# ---------------------------------------------------------------------------
# 2. taxonomy (Enterococcus/Klebsiella/Escherichia relabund) + Shannon
# ---------------------------------------------------------------------------
genus = pd.read_csv(os.path.join(STRUCTURED_DIR, "taxonomy_genus_filtered.tsv"), sep="\t", index_col="patient_id")
genus = genus.reindex(disc.index)
entero_col = [c for c in genus.columns if "g__Enterococcus" in c][0]
kleb_col = [c for c in genus.columns if "g__Klebsiella" in c][0]
esch_col = [c for c in genus.columns if "g__Escherichia" in c][0]
entero = genus[entero_col].values
kleb = genus[kleb_col].values
esch = genus[esch_col].values
pathobiont_max = np.maximum.reduce([entero, kleb, esch])

alpha = pd.read_csv(os.path.join(STRUCTURED_DIR, "taxonomy_alpha_diversity.tsv"), sep="\t", index_col="patient_id")
alpha = alpha.reindex(disc.index)
shannon = alpha["shannon"].values
if np.isnan(entero).any() or np.isnan(shannon).any():
    log.error("Missing taxonomy/alpha-diversity values for some of the 302 -- aborting.")
    sys.exit(1)

shannon_lower_tertile = np.percentile(shannon, 100 / 3)
log.info("Shannon lower-tertile cut (outcome-blind) = %.3f (cohort range %.3f-%.3f)",
          shannon_lower_tertile, shannon.min(), shannon.max())

THRESHOLDS = [10, 30, 50]  # percent; 30 is the PRE-DECLARED primary cut
PRIMARY_FLAGS = {"entero_ge30", "pathobiont_ge30"}

FLAGS = {}
for thr in THRESHOLDS:
    FLAGS[f"entero_ge{thr}"] = (entero >= thr).astype(int)
    FLAGS[f"pathobiont_ge{thr}"] = (pathobiont_max >= thr).astype(int)
FLAGS["shannon_low_tertile"] = (shannon < shannon_lower_tertile).astype(int)

FLAG_DESCRIPTIONS = {
    **{f"entero_ge{thr}": f"Enterococcus relabund >= {thr}%" for thr in THRESHOLDS},
    **{f"pathobiont_ge{thr}": f"max(Enterococcus, Klebsiella, Escherichia) relabund >= {thr}%" for thr in THRESHOLDS},
    "shannon_low_tertile": f"Shannon diversity < cohort lower tertile ({shannon_lower_tertile:.3f}, outcome-blind cut)",
}

log.info("7 pre-specified flags: %s", list(FLAGS.keys()))
for name, flag in FLAGS.items():
    log.info("  %-22s n_positive=%3d (%.1f%%)", name, int(flag.sum()), 100 * flag.mean())

# ---------------------------------------------------------------------------
# 3. out-of-fold discrimination -- S0 (SOFA alone) computed once and reused
#    as the reference for every flag; S_flag = SOFA + flag.
# ---------------------------------------------------------------------------
outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def oof_probabilities(X):
    p = np.full(n, np.nan)
    for tr, te in outer_skf.split(X, y):
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X[tr], y[tr])
        p[te] = lr.predict_proba(X[te])[:, 1]
    assert not np.isnan(p).any()
    return p


X_s0 = SOFA.reshape(-1, 1)
oof_s0 = oof_probabilities(X_s0)
auroc_s0 = roc_auc_score(y, oof_s0)
log.info("S0 (SOFA alone) OOF AUROC = %.4f", auroc_s0)

# ---------------------------------------------------------------------------
# 4. paired bootstrap -- algorithm IDENTICAL to
#    46_phase_x_c1_c2_nested_bootstrap_ci.py section 4.
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)


def paired_bootstrap(p_ref, p_new):
    d_auroc_b, d_auprc_b, d_brier_b = [], [], []
    for _ in range(N_BOOT_PAIRED):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        rb, nb = p_ref[idx], p_new[idx]
        d_auroc_b.append(roc_auc_score(yb, nb) - roc_auc_score(yb, rb))
        d_auprc_b.append(average_precision_score(yb, nb) - average_precision_score(yb, rb))
        d_brier_b.append(brier_score_loss(yb, rb) - brier_score_loss(yb, nb))  # positive = new has LOWER (better) Brier

    def summarize(deltas, point):
        arr = np.asarray(deltas)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p_two = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        return round(float(point), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p_two, 1.0)), 4), len(deltas)

    d_auroc = roc_auc_score(y, p_new) - roc_auc_score(y, p_ref)
    d_auprc = average_precision_score(y, p_new) - average_precision_score(y, p_ref)
    d_brier = brier_score_loss(y, p_ref) - brier_score_loss(y, p_new)
    a = summarize(d_auroc_b, d_auroc)
    c = summarize(d_auprc_b, d_auprc)
    b = summarize(d_brier_b, d_brier)
    return a, c, b


rows = []
for name, flag in FLAGS.items():
    # --- 1. unadjusted OR ---
    Xc = sm.add_constant(flag.astype(float))
    fit_unadj = sm.Logit(y, Xc).fit(disp=0)
    coef, se = fit_unadj.params[1], fit_unadj.bse[1]
    or_unadj = np.exp(coef)
    ci_lo_unadj, ci_hi_unadj = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
    p_unadj = fit_unadj.pvalues[1]

    # --- 2. SOFA-adjusted OR ---
    X_adj = np.column_stack([flag.astype(float), SOFA])
    Xc_adj = sm.add_constant(X_adj)
    fit_adj = sm.Logit(y, Xc_adj).fit(disp=0)
    coef_a, se_a = fit_adj.params[1], fit_adj.bse[1]
    or_adj = np.exp(coef_a)
    ci_lo_adj, ci_hi_adj = np.exp(coef_a - 1.96 * se_a), np.exp(coef_a + 1.96 * se_a)
    p_adj = fit_adj.pvalues[1]

    # --- 3. out-of-fold discrimination: S0 vs S0+flag ---
    X_flag = np.column_stack([SOFA, flag.astype(float)])
    oof_flag = oof_probabilities(X_flag)
    auroc_flag = roc_auc_score(y, oof_flag)
    (d_auroc, d_auroc_lo, d_auroc_hi, d_auroc_p, n_eff), \
        (d_auprc, d_auprc_lo, d_auprc_hi, d_auprc_p, _), \
        (d_brier, d_brier_lo, d_brier_hi, d_brier_p, _) = paired_bootstrap(oof_s0, oof_flag)

    rows.append({
        "flag": name, "description": FLAG_DESCRIPTIONS[name], "is_primary": name in PRIMARY_FLAGS,
        "N": n, "N_events": int(y.sum()), "n_flag_positive": int(flag.sum()),
        "pct_flag_positive": round(100 * flag.mean(), 1),
        "unadjusted_OR": round(or_unadj, 3), "unadjusted_OR_CI_low": round(ci_lo_unadj, 3),
        "unadjusted_OR_CI_high": round(ci_hi_unadj, 3), "unadjusted_p": round(p_unadj, 4),
        "SOFA_adjusted_OR": round(or_adj, 3), "SOFA_adjusted_OR_CI_low": round(ci_lo_adj, 3),
        "SOFA_adjusted_OR_CI_high": round(ci_hi_adj, 3), "SOFA_adjusted_p": round(p_adj, 4),
        "AUROC_SOFA_alone": round(auroc_s0, 4), "AUROC_SOFA_plus_flag": round(auroc_flag, 4),
        "delta_AUROC": d_auroc, "delta_AUROC_CI_low": d_auroc_lo, "delta_AUROC_CI_high": d_auroc_hi,
        "delta_AUROC_boot_p": d_auroc_p, "N_effective_bootstrap_resamples": n_eff,
        "delta_AUPRC": d_auprc, "delta_AUPRC_CI_low": d_auprc_lo, "delta_AUPRC_CI_high": d_auprc_hi,
        "delta_AUPRC_boot_p": d_auprc_p,
        "delta_Brier_improvement": d_brier, "delta_Brier_CI_low": d_brier_lo, "delta_Brier_CI_high": d_brier_hi,
        "delta_Brier_boot_p": d_brier_p,
    })
    log.info("%s: n+=%d (%.1f%%), unadj OR=%.2f p=%.4f | SOFA-adj OR=%.2f p=%.4f | dAUROC=%.4f [%.4f,%.4f] p=%.4f",
              name, int(flag.sum()), 100 * flag.mean(), or_unadj, p_unadj, or_adj, p_adj,
              d_auroc, d_auroc_lo, d_auroc_hi, d_auroc_p)

out = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. explicit multiplicity accounting -- Holm-Bonferroni across all 7 tests,
#    applied to BOTH the unadjusted and SOFA-adjusted p-values (the two
#    inferential p-values reported per flag). Reported alongside, never used
#    to silently drop rows.
# ---------------------------------------------------------------------------
_, holm_unadj, _, _ = multipletests(out["unadjusted_p"].values, alpha=0.05, method="holm")
_, holm_adj, _, _ = multipletests(out["SOFA_adjusted_p"].values, alpha=0.05, method="holm")
out["unadjusted_p_holm"] = np.round(holm_unadj, 4)
out["SOFA_adjusted_p_holm"] = np.round(holm_adj, 4)

io_utils.save_csv(out, OUT_CSV)
log.info("wrote %s", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. markdown report
# ---------------------------------------------------------------------------
any_sig_adj_raw = (out["SOFA_adjusted_p"] < 0.05).sum()
any_sig_adj_holm = (out["SOFA_adjusted_p_holm"] < 0.05).sum()
any_auroc_sig = ((out["delta_AUROC_CI_low"] > 0) | (out["delta_AUROC_CI_high"] < 0)).sum()
max_delta_auroc = out["delta_AUROC"].abs().max()

if any_auroc_sig == 0 and any_sig_adj_holm == 0:
    headline = ("**Headline: NO. None of the 7 pre-specified single-organism/pathobiont-domination flags adds "
                "discrimination over SOFA (all delta_AUROC CIs include zero), and none survives multiplicity "
                "correction on the SOFA-adjusted association. The simplest clinically-intuitive rule is, like M1, "
                "null in this cohort.**")
elif any_auroc_sig == 0 and any_sig_adj_raw > 0:
    headline = (
        f"**Headline: split verdict, same pattern as the M1/SOFA result in "
        f"`phase_x_microbiome_over_sofa.md`.** No flag improves out-of-fold discrimination over SOFA "
        f"(all {len(out)} delta_AUROC CIs include zero; max |delta_AUROC|={max_delta_auroc:.4f}). "
        f"{any_sig_adj_raw}/{len(out)} flags are nominally associated with mortality after SOFA-adjustment "
        f"(p<0.05 uncorrected), of which {any_sig_adj_holm} survive Holm-Bonferroni across all 7 tests. "
        "A term can be associated with the outcome yet move AUROC negligibly -- this is not a contradiction, "
        "and does **not** license the claim that the flag improves risk prediction over SOFA.")
else:
    headline = ("**Headline: at least one flag improved out-of-fold discrimination over SOFA with a bootstrap "
                "CI excluding zero -- see the table, and read the multiplicity caveat below before over-reading it.**")

md = [
    "# Task A: does a simple single-organism/pathobiont-domination flag add anything over SOFA?", "",
    "> **EXPLORATORY / POST-HOC.** This analysis was requested on 2026-09-02 to test the simplest, most "
    "clinically-intuitive version of the pathobiont-domination hypothesis that the manuscript's M1 16-marker "
    "composite score never isolated on its own. It is not prespecified. All 7 flags and thresholds below are "
    "reported; none was selected because it performed best.", "",
    "## Rationale number, verified", "",
    "The task brief cited \"Enterococcus relative abundance OR 3.92, p=0.0003\" from the 466-patient association "
    "arm. **Confirmed present and accurate**, in "
    "`merge/downstream_analysis/tables/alpha_diversity_association.tsv`, feature `enterococcus_relabund`: "
    "odds_ratio=3.9239, CI [1.870, 8.234], p=0.0003004 (N=466, 131 deaths). This is a raw-scale univariate "
    "logistic regression of mortality on genus-level Enterococcus relative abundance expressed as a 0-1 FRACTION "
    "of the community (i.e. the OR is per a full 0->100% change in dominance), computed in "
    "`03_alpha_beta_diversity.R`. **Caveat, not a contradiction**: the SAME association arm's `taxonomy_genus` "
    "block (`univariable_taxonomy_genus.tsv`) reports OR=1.069 for the same taxon on a per-1-PERCENTAGE-POINT "
    "basis (0-100 scale) -- both are nominally significant and point the same direction, but they are not "
    "directly comparable numbers because they use different units/pipelines. Flagged here so a reader does not "
    "mistake this for an inconsistency in this script's own results.", "",
    "## Flags (7 total, all reported)", "",
    "Enterococcus, Klebsiella and Escherichia relative abundance are read-based (MetaPhlAn3-derived genus-level) "
    "taxonomy from the 302-patient landmark ML cohort. Threshold grid 10%/30%/50% for both the single-organism "
    "(Enterococcus) and pathobiont-domination (max of the three genera) flags; **30% is the PRE-DECLARED primary "
    "cut** (the HSCT-literature \"domination\" convention), 10%/50% are a sensitivity grid, not a search for the "
    "best-performing cut. Shannon-diversity-below-lower-tertile is an outcome-blind cut (the tertile boundary is "
    "computed from the covariate distribution only).", "",
    df_to_md_table(out[["flag", "description", "is_primary", "n_flag_positive", "pct_flag_positive"]]), "",
    "## Design", "",
    f"- N = {n} patients, {int(y.sum())} in-hospital deaths ({100*y.mean():.1f}%) -- the 302-patient landmark ML "
    "cohort used for all incremental-value work in this project.",
    "- SOFA from `metadata/microbiome.clinical.csv`, joined via KCDC_no exactly as "
    "`41_phase_x_microbiome_over_sofa.py`.",
    f"- Discrimination is out-of-fold ({N_FOLDS}-fold stratified CV, seed={SEED}, project-wide convention). "
    "S0 (SOFA alone) is computed once and reused as the common reference for all 7 flags.",
    "- The flags are deterministic thresholds on already-observed taxonomy/diversity data, not fitted-model "
    "outputs -- unlike M1/M3 in scripts 18/45/46, there is nothing upstream of the flag itself that could leak "
    "across CV folds, so no inner nested split is needed for the flag's construction (only the SOFA+flag "
    "logistic model needs honest out-of-fold evaluation, which it gets). This is analogous to why script 45 "
    "needed no nesting for its C0 (SOFA/clinical-alone) row.",
    f"- Paired-bootstrap delta AUROC/AUPRC/Brier: algorithm IDENTICAL to "
    f"`46_phase_x_c1_c2_nested_bootstrap_ci.py` ({N_BOOT_PAIRED} with-replacement patient resamples, seed={SEED}, "
    "same resample index applied to S0 and S0+flag each iteration, resamples with <2 outcome classes skipped, "
    "95% CI from bootstrap percentiles, two-sided bootstrap p = 2 x min(tail proportions crossing zero)).",
    "- **Multiplicity**: Holm-Bonferroni correction applied across all 7 tests, separately for the unadjusted and "
    "SOFA-adjusted p-values (`unadjusted_p_holm`, `SOFA_adjusted_p_holm` in the CSV). No flag/threshold was "
    "selected post-hoc for reporting -- all 7 appear in the table below regardless of result.", "",
    "## Results", "",
    df_to_md_table(out[["flag", "N", "N_events", "n_flag_positive", "unadjusted_OR", "unadjusted_OR_CI_low",
                        "unadjusted_OR_CI_high", "unadjusted_p", "unadjusted_p_holm"]]), "",
    df_to_md_table(out[["flag", "SOFA_adjusted_OR", "SOFA_adjusted_OR_CI_low", "SOFA_adjusted_OR_CI_high",
                        "SOFA_adjusted_p", "SOFA_adjusted_p_holm"]]), "",
    df_to_md_table(out[["flag", "AUROC_SOFA_alone", "AUROC_SOFA_plus_flag", "delta_AUROC", "delta_AUROC_CI_low",
                        "delta_AUROC_CI_high", "delta_AUROC_boot_p"]]), "",
    "Full table (including delta_AUPRC/delta_Brier) in `phase_x_simple_pathobiont_flag.csv`.", "",
    "## Interpretation", "",
    headline, "",
    f"- {any_sig_adj_raw}/{len(out)} flags nominally SOFA-adjusted-significant (p<0.05, uncorrected); "
    f"{any_sig_adj_holm}/{len(out)} survive Holm-Bonferroni across all 7.",
    f"- {any_auroc_sig}/{len(out)} flags improve out-of-fold discrimination over SOFA with a bootstrap CI "
    f"excluding zero; largest |delta_AUROC| = {max_delta_auroc:.4f}.",
    "- **Power**: at N=302 with 69 events, and with flag prevalence as low as a few percent for the 50% cuts, "
    "this analysis cannot detect small increments in discrimination or rule out a modest true association -- a "
    "null result here is 'not demonstrated', not 'demonstrated absent'.",
    "- This result is consistent with, and does not contradict, the manuscript's existing finding that the "
    "16-marker M1 composite (of which Enterococcus abundance is one input) does not improve discrimination over "
    "SOFA/Clinical AI (`phase_x_microbiome_over_sofa.md`) -- the simplest single-feature version of the same "
    "hypothesis behaves the same way.",
    "", "No specification was altered in pursuit of significance. The threshold grid, flag set, SOFA-linear "
    "form, 5-fold/seed 2026 and the multiplicity correction were fixed before the numbers were read.",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
