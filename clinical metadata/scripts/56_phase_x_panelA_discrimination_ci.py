#!/usr/bin/env python3
"""56_phase_x_panelA_discrimination_ci.py

Audit action items D2 + D3 (fable_audit_20260904.md, 2026-09-04): P1_Fig2
Panel A is the manuscript's head-to-head discrimination panel. As shipped it
carried a bootstrap CI for SOFA only, no CI for the reduced clinical AI model
(C0) and no CI at all for the microbiome-alone score (M1) -- and it drew SOFA
from `phase_x_severity_score_comparison_302.csv` while drawing C0 from
`phase_x_c1_c2_incremental_value.csv`, i.e. two different C0 estimands in one
panel. This script emits one self-consistent three-bar table with a CI on
every bar.

WHICH C0 (the D3 decision, recorded here so the figure does not have to
re-litigate it):
  C0 has two documented, both-correct values on these same 302 patients:
    0.7560 -- `clinical_AI_probability_PRIMARY` scored directly (the frozen
              catboost_Gangnamdev bundle applied as-is; script 16's PRIMARY
              row, and the "Reduced Clinical AI alone (C0)" row of script 29).
    0.7528 -- `oof_prob_C0_clinical_alone`, the same frozen probability passed
              through the 5-fold stacking baseline (per-fold logistic refit on
              logit(clinical), seed 2026) used by scripts 18/45/46.
  Panel A uses 0.7528. Criterion: the panel's C0 bar must be the baseline that
  the paper's headline ΔAUROC is measured FROM, so that bar differences and
  reported deltas are arithmetically reconcilable by a reader. The nested
  deltas in `phase_x_c1_c2_incremental_value_nested_corrected.csv` /
  `phase_x_c1_c2_nested_bootstrap_delta.csv` are all computed against 0.7528.
  0.7560 is not a stacking baseline and no reported Δ is measured from it.

WHY M1 CANNOT BE A DIRECT-APPLICATION QUANTITY (the D2 decision):
  `microbiome_prob_M1` is NOT the frozen M1 model applied to the 302. It is
  produced by `18_phase_x_incremental_value.py:213`,
  `df["microbiome_prob_M1"] = oof_scores_lr(X_m1)` -- a fresh out-of-fold
  refit (elastic-net LogisticRegressionCV, StratifiedKFold(5, shuffle=True,
  random_state=2026)) of the 16 frozen literature-core FEATURES. The marker
  list is literature-derived and outcome-blind; the coefficients are fitted
  inside each training fold. Script 18's own section-2/3 comment records why:
  applying the frozen all-302 M1 bundle directly was found to be in-sample /
  overfit. So there is no honest direct-application M1 AUROC to put in a
  direct-application panel -- M1 can only enter Panel A as an out-of-fold
  quantity, which is the second reason Panel A is built on the OOF layer.

  M1-alone is a STAGE-1 quantity and is untouched by the nested-CV correction
  (`45_phase_x_c1_c2_nested_cv_corrected.py`), which restructures only the
  stacked C1/C2 meta-model layer. Script 45 recomputes the identical stage-1
  column at line 241 (`single_split_oof(X_m1, fit_predict_lr)`, same folds,
  same estimator, same seed) and keeps it unchanged; it simply is not
  persisted into `phase_x_c1_c2_nested_oof_probabilities.csv`. Verified here:
  `oof_prob_C0_clinical_alone` is bit-identical (max|diff| ~2e-16) between the
  pre-nested and nested prediction files, confirming the two files share the
  same stage-1/C0 layer and differ only in the stacked rows.

BOOTSTRAP CONVENTION -- matched to `29_phase_x_severity_score_comparison.py`
  `bootstrap_auroc()` so all three bars use one resampling protocol:
    np.random.RandomState(2026); per resample idx = rng.randint(0, n, n)
    (with-replacement patient resampling); resamples containing a single
    outcome class are skipped; 95% CI = np.percentile(boots, [2.5, 97.5]).
  N_BOOT = 1000. NOTE: script 29's .md prose says "bootstrap 95% CI, N=2000",
  but its code used `config.N_BOOTSTRAP`, which is 1000 -- and 1000 is what
  reproduces its published SOFA [0.6865, 0.8269] and C0-direct
  [0.6918, 0.8132] exactly (2000 does not: [0.6886, 0.8255] / [0.6928,
  0.8173]). The prose is stale; the code and the published numbers agree at
  1000. This script asserts that reproduction at import time.
  The bootstrap resamples patients while holding the out-of-fold probability
  vector fixed; it does not re-run the cross-validation inside each resample.
  That is exactly what script 29 did for its own OOF rows, so the three bars
  remain mutually comparable, but the interval is conditional on the realised
  CV and is not a full nested-bootstrap interval.

Reads (read-only):
  clinical_ai/predictions/phase_x_c1_c2_oof_probabilities.csv       (script 18)
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities.csv (script 45)
  semantic_microbiome/harmonized/discovery_cohort_primary.csv
  ICU_public/metadata/microbiome.clinical.csv                       (SOFA)
Writes:
  clinical_ai/reports/phase_x_panelA_discrimination_ci.csv
  clinical_ai/reports/phase_x_panelA_discrimination_ci.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("56_phase_x_panelA_discrimination_ci")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")
NESTED_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities.csv")
CLINICAL_RAW_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")
DISCOVERY_CSV = os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv")
OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_panelA_discrimination_ci.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_panelA_discrimination_ci.md")

SEED = config.RANDOM_SEED          # 2026
N_BOOT = config.N_BOOTSTRAP        # 1000 -- see docstring
EPS = 1e-6

# published values this script must reproduce before it is allowed to emit
# anything, so the convention match is enforced rather than asserted in prose
REPRO_TARGETS = {
    "SOFA": (0.7593, 0.6865, 0.8269),          # phase_x_severity_score_comparison_302.csv
    "C0_direct": (0.7560, 0.6918, 0.8132),     # ditto; also script 16's PRIMARY row
}


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def bootstrap_auroc(y, score, n_boot=N_BOOT, seed=SEED):
    """Byte-for-byte the resampling protocol of
    29_phase_x_severity_score_comparison.py:bootstrap_auroc()."""
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
    return point, lo, hi, len(boots)


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------
for path in (OOF_CSV, NESTED_CSV, CLINICAL_RAW_CSV, DISCOVERY_CSV):
    if not os.path.exists(path):
        log.error("missing input: %s", path)
        sys.exit(1)

oof = pd.read_csv(OOF_CSV, index_col=0)
nested = pd.read_csv(NESTED_CSV, index_col=0).reindex(oof.index)
disc = pd.read_csv(DISCOVERY_CSV)
raw_clin = pd.read_csv(CLINICAL_RAW_CSV)
disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw_clin = (raw_clin.merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner")
            .set_index("patient_id_clinical").reindex(oof.index))

y = oof["hospital_mortality"].values
SOFA = pd.to_numeric(raw_clin["SOFA"], errors="coerce").values
log.info("N=%d, events=%d (%.1f%%); SOFA non-missing=%d",
         len(y), int(y.sum()), 100 * y.mean(), int((~pd.isna(SOFA)).sum()))

# ---------------------------------------------------------------------------
# 2. provenance assertion: the pre-nested and nested files share the same
#    stage-1 / C0 layer, so C0 = 0.7528 is NOT superseded by the nested
#    correction and M1 (stage 1) is likewise unaffected by it.
# ---------------------------------------------------------------------------
c0_delta = float((nested["oof_prob_C0_clinical_alone"] - oof["oof_prob_C0_clinical_alone"]).abs().max())
clin_delta = float((nested["clinical_AI_probability_PRIMARY"] - oof["clinical_AI_probability_PRIMARY"]).abs().max())
log.info("pre-nested vs nested: max|C0 OOF diff|=%.3e, max|clinical PRIMARY diff|=%.3e", c0_delta, clin_delta)
assert c0_delta < 1e-12 and clin_delta < 1e-12, "C0/clinical layer differs between the two prediction files"

# ---------------------------------------------------------------------------
# 3. reproduce script 29's published intervals -> proves the convention match
# ---------------------------------------------------------------------------
valid_sofa = ~pd.isna(SOFA)
repro = {
    "SOFA": bootstrap_auroc(y[valid_sofa], SOFA[valid_sofa])[:3],
    "C0_direct": bootstrap_auroc(y, oof["clinical_AI_probability_PRIMARY"].values)[:3],
}
for k, (want_pt, want_lo, want_hi) in REPRO_TARGETS.items():
    got = tuple(round(float(v), 4) for v in repro[k])
    log.info("reproduction check %-9s got=%s want=%s", k, got, (want_pt, want_lo, want_hi))
    assert got == (want_pt, want_lo, want_hi), f"{k} does not reproduce script 29 -- convention drift"

# ---------------------------------------------------------------------------
# 4. the three Panel A bars
# ---------------------------------------------------------------------------
BARS = [
    ("SOFA", "SOFA (conventional severity score)", SOFA,
     "raw score, no cohort fitting of any kind (AUROC is rank-based, so no "
     "recalibration is possible or needed)"),
    ("C0", "Reduced Clinical AI alone (C0)", oof["oof_prob_C0_clinical_alone"].values,
     "frozen external clinical model (catboost, Gangnam-dev, trained on "
     "ICU.csv/mimic.csv -- disjoint from these 302), passed through the 5-fold "
     "stacking baseline (per-fold logistic refit on logit(clinical), seed 2026); "
     "this is the baseline the reported dAUROC values are measured from"),
    ("M1", "Microbiome literature-core score alone (M1)", oof["microbiome_prob_M1"].values,
     "16 frozen literature-derived markers (feature set outcome-blind); "
     "elastic-net logistic coefficients refit out-of-fold, StratifiedKFold(5, "
     "shuffle=True, random_state=2026) -- the SAME fold partition as the C0 bar"),
]

rows = []
for key, label, score, design in BARS:
    valid = ~pd.isna(score)
    pt, lo, hi, n_eff = bootstrap_auroc(y[valid], score[valid])
    rows.append({"bar": key, "model": label, "N": int(valid.sum()), "N_events": int(y[valid].sum()),
                 "AUROC": round(pt, 4), "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4),
                 "N_boot": N_BOOT, "N_effective_resamples": n_eff, "seed": SEED,
                 "estimand": "out-of-fold (patient's own outcome never used to produce that patient's score)",
                 "design": design})
result = pd.DataFrame(rows)
io_utils.save_csv(result, OUT_CSV)
log.info("\n%s", result[["bar", "N", "AUROC", "AUROC_CI_low", "AUROC_CI_high"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. sensitivity: how much would SOFA move if it were pushed through the same
#    per-fold logistic recalibration that produces C0's 0.7528? (Reported so
#    the panel's one un-recalibrated bar is quantified, not just asserted.)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sofa_v = SOFA[valid_sofa].astype(float).reshape(-1, 1)
y_v = y[valid_sofa]
skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=SEED)
sofa_oof = np.full(len(y_v), np.nan)
for tr, te in skf.split(sofa_v, y_v):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(sofa_v[tr], y_v[tr])
    sofa_oof[te] = lr.predict_proba(sofa_v[te])[:, 1]
sofa_recal_pt, sofa_recal_lo, sofa_recal_hi, _ = bootstrap_auroc(y_v, sofa_oof)
log.info("SOFA sensitivity: per-fold recalibrated OOF AUROC=%.4f [%.4f, %.4f] vs raw %.4f",
         sofa_recal_pt, sofa_recal_lo, sofa_recal_hi, result.loc[0, "AUROC"])

# C0 direct-application reference (NOT a panel bar) for the caption
c0_direct_pt, c0_direct_lo, c0_direct_hi = repro["C0_direct"]

# ---------------------------------------------------------------------------
# 6. markdown
# ---------------------------------------------------------------------------
md = [
    "# P1_Fig2 Panel A: head-to-head discrimination with 95% CI on every bar", "",
    f"All three bars evaluated on the SAME {len(y)} Gangnam patients ({int(y.sum())} deaths, "
    f"{100 * y.mean():.1f}%), one bootstrap protocol (percentile 95% CI, {N_BOOT} with-replacement "
    f"patient resamples, `np.random.RandomState({SEED})`, `rng.randint(0, n, n)`, single-class "
    "resamples skipped, percentiles 2.5/97.5) -- identical to "
    "`29_phase_x_severity_score_comparison.py:bootstrap_auroc()`, whose published SOFA and "
    "C0-direct intervals this script reproduces exactly as a precondition for writing this file.", "",
    df_to_md_table(result[["bar", "model", "N", "N_events", "AUROC", "AUROC_CI_low", "AUROC_CI_high"]]), "",
    "## Which C0, and why", "",
    f"C0 has two documented, both-correct values on these patients: **{result.loc[1, 'AUROC']:.4f} "
    f"[{result.loc[1, 'AUROC_CI_low']:.4f}, {result.loc[1, 'AUROC_CI_high']:.4f}]** (out-of-fold "
    f"stacking baseline) and **{c0_direct_pt:.4f} [{c0_direct_lo:.4f}, {c0_direct_hi:.4f}]** "
    "(`clinical_AI_probability_PRIMARY` scored directly). Panel A uses the out-of-fold value "
    "because it is the baseline the paper's reported dAUROC values are measured from "
    "(`phase_x_c1_c2_incremental_value_nested_corrected.csv`, "
    "`phase_x_c1_c2_nested_bootstrap_delta.csv`), so bar differences in the figure and deltas in "
    "the text reconcile arithmetically. The direct-application value is not a stacking baseline "
    "and no reported delta is measured from it. The two differ by "
    f"{abs(c0_direct_pt - result.loc[1, 'AUROC']):.4f} AUROC.", "",
    "## What M1 alone is", "",
    "`microbiome_prob_M1` is **not** the frozen M1 bundle applied to these patients. It is "
    "`18_phase_x_incremental_value.py:213`, `oof_scores_lr(X_m1)`: a fresh out-of-fold refit "
    "(elastic-net `LogisticRegressionCV`) of the **16 frozen literature-core features**, using "
    f"`StratifiedKFold(5, shuffle=True, random_state={SEED})` -- the same fold partition as the C0 "
    "bar. The marker list is literature-derived and outcome-blind; the coefficients are fitted "
    "inside each training fold. Script 18's own code comment records why the frozen bundle is not "
    "applied directly: doing so was in-sample/overfit. There is therefore no honest "
    "direct-application M1 AUROC, which is why Panel A is built on the out-of-fold layer rather "
    "than on the direct-application layer of "
    "`phase_x_severity_score_comparison_302.csv`.", "",
    "M1 alone is a **stage-1** quantity. The nested-CV correction "
    "(`45_phase_x_c1_c2_nested_cv_corrected.py`) restructures only the stacked C1/C2 meta-model "
    "layer; script 45 recomputes the identical stage-1 column at line 241 "
    "(`single_split_oof(X_m1, fit_predict_lr)`, same folds, same estimator, same seed) and leaves "
    "it unchanged -- it is simply not persisted into "
    "`phase_x_c1_c2_nested_oof_probabilities.csv`. Verified here: `oof_prob_C0_clinical_alone` is "
    f"bit-identical between the two prediction files (max|diff| = {c0_delta:.1e}), confirming they "
    "share one stage-1/C0 layer and differ only in the stacked rows. "
    "Separately, the semantic arm reports M1 = 0.624 +/- 0.066 under a 25-fold nested-CV design; "
    "both are correct and are distinguished in `manuscript_numbers.json`. Panel A uses 0.592 "
    "because it is on the identical fold partition as the C0 bar and is therefore the only M1 "
    "estimate that is paired with it.", "",
    "## Sensitivity: the one un-recalibrated bar", "",
    "SOFA enters raw, with no cohort fitting. If it were pushed through the same per-fold logistic "
    f"recalibration that produces C0's out-of-fold value, its AUROC would be {sofa_recal_pt:.4f} "
    f"[{sofa_recal_lo:.4f}, {sofa_recal_hi:.4f}] versus {result.loc[0, 'AUROC']:.4f} "
    f"[{result.loc[0, 'AUROC_CI_low']:.4f}, {result.loc[0, 'AUROC_CI_high']:.4f}] raw "
    f"(difference {abs(sofa_recal_pt - result.loc[0, 'AUROC']):.4f}). AUROC is invariant to "
    "monotone recalibration within a fold; the residual movement is only the artefact of pooling "
    "folds with different fitted intercepts. SOFA is reported raw because it is a fixed published "
    "score, not a model being developed here.", "",
    "## Limitation, stated for the figure caption", "",
    "The bootstrap resamples patients while holding the out-of-fold probability vector fixed; it "
    "does not re-run the cross-validation inside each resample. Each interval is therefore "
    "conditional on the realised CV and is expected to be slightly narrower than a full "
    "nested-bootstrap interval. The same is true of every interval in "
    "`phase_x_severity_score_comparison_302.csv`, so the bars remain mutually comparable.", "",
    "## Note on a stale string elsewhere", "",
    "`phase_x_severity_score_comparison_302.md` states \"bootstrap 95% CI, N=2000\". Its code used "
    f"`config.N_BOOTSTRAP` = {N_BOOT}, and {N_BOOT} is what reproduces its published intervals "
    "exactly (2000 gives SOFA [0.6886, 0.8255] and C0-direct [0.6928, 0.8173], neither of which "
    "matches the file). The published numbers are correct; the prose is stale and should read "
    f"N={N_BOOT}.",
]
io_utils.save_text("\n".join(md), OUT_MD)
log.info("wrote %s / %s", OUT_CSV, OUT_MD)
