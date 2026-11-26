#!/usr/bin/env python3
"""61_phase_x_microbiome_over_sofa_nested_v4.py

v4 (2026-10-02, owner-authorised "final technical closure pass"; sibling of
45_phase_x_c1_c2_nested_cv_corrected_v4.py, which is the source of truth for
the M1-convergence-fixed and V4_STRICT-LR vectors). Copy of script 61_v3 with
these changes on top of v3's fold-local M3 correction:
  1. M1 base layer: v3 kept M1 byte-identical via nested_meta.fit_predict_lr
     (max_iter=3000, no escalation). v4 routes M1 through 45_v4's OWN fit_lr
     (same convergence ladder 30000->100000 as every other LR fit in this
     closure pass), wrapped in a "static" FeatureSource exactly like 45_v4's
     REF variant -- so M1's base layer here is procedurally IDENTICAL to
     45_v4's, not a separate reimplementation.
  2. M3-LR base layer: v3 used 45_v3's fit_lr (V3: fold-local prevalence
     filter + fold-local PCA, but PCA/filter still fit BEFORE LogisticRegressionCV's
     own internal tuning CV). v4 uses 45_v4's fit_lr_strict_full (V4_STRICT:
     prevalence filter AND PCA both refit inside the tuning CV via
     GridSearchCV(Pipeline[FullBlockPreprocessor, LR]) -- the FINAL headline
     per the closure-pass instruction. M3-CatBoost is UNCHANGED (still V3's
     fold-local prev_pca source) -- CatBoost has no internal tuning CV, so it
     was already maximally fold-local; no redesign.
  3. Gates: A unchanged (no microbiome term). B and C are now STRICT again
     for ALL three base layers (M1, M3-LR, M3-CatBoost), but against NEW v4
     reference files (45_v4's own recomputed vectors) instead of v3's/the
     original frozen files -- since M1 and M3-LR's procedures are byte-
     identical between this script and 45_v4, they are EXPECTED to reproduce
     45_v4's vectors exactly, which is still a meaningful, strict consistency
     check (just against a different, correct target). D unchanged. E: STRICT
     only for designs with NO microbiome term at all (S0, S3ref); every
     design containing M1 or M3 (S1, S2a, S2b, S3a, S3b) is now a changed-row
     REPORT, not an abort, computed from THIS script's own freshly-fixed
     single-split vectors (not an external old file -- there is no valid "old"
     reference for OR/LRT on numbers that did not exist before this pass).
  4. OR/LRT block: every term (M1, M3-LR, M3-CatBoost) now comes from this
     script's OWN single-split vectors (folds_m1/folds_m3lr/folds_m3cb's
     outer-test-fold predictions), not from external script-18/v3-single
     files.
  5. All outputs carry a _v4 suffix; nothing from script 61/61_v3 is
     overwritten.
  6. The matplotlib DCA figure is not produced (project rule: no new
     matplotlib figures; figures are rebuilt in R from the paired CSV).
SOFA (linear, first-24h, same join), meta-model, bootstrap streams, NRI/IDI
and DCA logic are byte-identical to script 61.

ORIGINAL SCRIPT 61 DOCSTRING FOLLOWS (its frozen-PCA and max_iter caveats are
superseded by points 1-2 above).
---------------------------------------------------------------------------
61_phase_x_microbiome_over_sofa_nested.py

Nested-CV rebuild of the SOFA-referenced model family (S0/S1/S2a/S2b/S3ref/
S3a/S3b), per
`semantic_microbiome/reports/paper1/script41_nested_rebuild_DESIGN.md`
(deep-reasoner, 2026-09-05). New sibling script -- does NOT modify
`41_phase_x_microbiome_over_sofa.py`, `45_phase_x_c1_c2_nested_cv_corrected.py`,
`46_phase_x_c1_c2_nested_bootstrap_ci.py`, or `utils/nested_meta.py`, and does
NOT overwrite any existing result file. Strictly additive.

THE GAP THIS CLOSES
--------------------
Script 41 answers "does the microbiome add anything beyond SOFA?" using the
single-split stacked-OOF design that this project's own corrections ledger
(TS1 row 4, revision_log.md item 19) diagnosed as leaky for the analogous
C1/C2a/C2b family and replaced everywhere else (scripts 45/46/55/57/59/60).
Scripts 46 and 55 each explicitly declared the SOFA-referenced contrasts out
of scope for want of a nested-CV SOFA OOF. This script is that OOF.

MODELS (same 302 patients, 69 deaths, as script 41)
----------------------------------------------------
    S0    : mortality ~ SOFA                                        [reference]
    S1    : mortality ~ SOFA + logit(microbiome M1 Literature-Core)
    S2a   : mortality ~ SOFA + logit(microbiome M3-LR)
    S2b   : mortality ~ SOFA + logit(microbiome M3-CatBoost)
    S3ref : mortality ~ SOFA + logit(clinical AI)                   [reference for S3*]
    S3a   : mortality ~ SOFA + logit(clinical AI) + logit(M1)
    S3b   : mortality ~ SOFA + logit(clinical AI) + logit(M3-CatBoost)

SOFA enters as a LINEAR covariate -- unchanged from script 41, not re-opened
here (design Decision 4: the functional form was fixed before results were
seen; changing it now would be exactly the post-hoc specification search the
project's own rules forbid).

STRUCTURAL FIT (design Decision 1)
-----------------------------------
`utils/nested_meta.py` is imported UNMODIFIED (`logit`, `make_lr_pipe`,
`fit_predict_lr`, `nested_base_probs`) -- not one line of that module is
edited. `nested_meta_eval()` and `paired_bootstrap_delta_auroc()` are NOT
used: the former is shaped around the adaptive-combination question (one
clinical covariate, one microbiome term, an optional moderator interaction)
and cannot express two different reference models, a three-covariate design
(S3a/S3b), or a microbiome-free arm evaluated on the same outer folds; the
latter returns delta-AUROC only, and this script reports AUROC+AUPRC+Brier
from one shared resample stream, matching script 41. Instead this script
defines a small local `sofa_meta_eval()` that takes an arbitrary
fold-invariant covariate matrix Z (SOFA alone, or SOFA+logit(clinical AI))
plus an optional microbiome base layer, generalizing `nested_meta_eval`'s
plain-arm computation to more than one covariate and to the "no microbiome
term" case (S0, S3ref) -- proved, not merely asserted, to be a like-for-like
sibling of the frozen C-family by the reproduction gates below. The CatBoost
`fit_predict_catboost` is copied verbatim from script 45 (a third
byte-identical copy, as script 46 already holds the second) rather than
added to `utils/nested_meta.py`, because adding it there would edit a file
script 57 depends on.

REPRODUCTION GATES (design Decision 3) -- ALL run and must PASS before any
output file is opened. On failure: log the per-vector max|diff| and the
recomputed AUROCs against their targets, then sys.exit(1) writing nothing.
No tolerance is ever relaxed.
  Gate A -- fold identity: recomputed C0 (clinical-alone, no microbiome)
            must match script 46's persisted `oof_prob_C0_clinical_alone`
            to <1e-6 per patient, AUROC round(.,4) == 0.7528.
  Gate B -- base-stage identity, all three learners: recomputed C1/C2a/C2b
            (clinical + each microbiome base layer) must match script 46's
            persisted vectors to <1e-6 per patient, AUROC round(.,4) ==
            0.7610 / 0.7608 / 0.7722.
  Gate C -- single-split identity (free): the base layer's own
            outer-test-fold predictions, concatenated across the 5 outer
            folds, must equal script 18's persisted `microbiome_prob_M1` /
            `_M3_LR` / `_M3_CatBoost` to <1e-6 per patient.
  Gate D -- raw-SOFA anchor (design-independent): the raw-SOFA bootstrap
            AUROC must reproduce 0.7593 [0.6865, 0.8269] exactly (same
            number as script 29/56's SOFA row and
            manuscript_numbers.json.SOFA_302cohort_AUROC).
  Gate E (fifth, free) -- the full-data adjusted-OR / nested-LRT block
            (Decision 5, "carried over unchanged": computed on script 18's
            persisted single-split OOF microbiome probabilities, exactly as
            script 41 does -- never claimed to be out-of-fold, never part
            of the CV-nesting bug) must be numerically IDENTICAL to script
            41's existing `phase_x_microbiome_over_sofa_adjusted_or.csv`
            and `phase_x_microbiome_over_sofa_nested_lrt.csv`.

DELIBERATE DEPARTURES FROM SCRIPT 41 (design Decision 5)
----------------------------------------------------------
1. Fresh `np.random.RandomState(SEED)` per contrast / per NRI-IDI row / per
   DCA threshold block (script 57's convention, fixing script 28's
   shared-single-rng-across-parts bug), INSTEAD of script 41's one `rng`
   created before the whole loop (so contrast k's draws depended on how many
   draws contrasts 1..k-1 had already consumed). Monte-Carlo-noise choice
   only (~+-0.001 on a CI bound at N_BOOT=2000), not an estimand change.
2. The decision-curve analysis emits the script-55 PAIRED delta-NB object
   (delta_NB = NB(augmented) - NB(reference) computed WITHIN each bootstrap
   replicate) as the INFERENTIAL object, not script 41's MARGINAL per-model
   CI bands -- script 55 already diagnosed that marginal-band object as
   invalid twice over (wrong estimand: overlap of two marginal CIs is not a
   test of a paired difference on the same patients; and the paired
   difference was never materialized). Faithfully reproducing script 41's
   DCA under nesting would resurrect a known-defective object under a
   fresher, more authoritative label. The marginal-band CSV is still written
   for shape continuity but is NOT an inferential object here.

WHAT IS CARRIED OVER UNCHANGED (design Decision 5, "Should any part not be
rebuilt? One yes"): the full-data adjusted-OR and nested-LRT block stays on
script 18's persisted single-split OOF microbiome probabilities
(`phase_x_c1_c2_oof_probabilities.csv`), following script 45's explicit
precedent for the C-family (a full-data association fit never claimed to be
out-of-fold, never touched by the CV-nesting bug). This is also Gate E.

KNOWN, DISCLOSED, NOT-RE-LITIGATED CAVEATS (restated here per design
Decision 2/4, not resolved)
  - The frozen M3 PCA transform (of MAG_ABUND / EGGNOG_KO) is an
    unsupervised, outcome-blind transform fit ONCE on all 302 patients and
    reused unchanged here, exactly as scripts 45/46/18 do. Re-fitting it per
    fold would make this script's base layer differ from scripts 45/46,
    breaking the reproduction gates and making the S-rows non-comparable to
    the published C-rows. A known residual, inherited deliberately.
  - SOFA's functional form (linear, single term) is fixed, not re-opened.
  - *When* SOFA was measured is now documented, and this caveat is
    therefore RETIRED rather than restated: the owner attested on
    2026-09-06 that SOFA and the C0 min/max inputs alike are first-24-hour
    values following ICU admission (semantic_microbiome/reports/paper1/
    SOFA_TIMING_ATTESTATION.md). The comparator is not a worst-during-stay
    quantity, so the null below is NOT conservative on that account -- it
    stands against a fair comparator. Do not reintroduce the old hedge.
    Residual, and separate: the clinical snapshot is indexed to ICU
    admission while this cohort is indexed to hospital admission, so for
    ward-to-ICU transfers the comparator and the predictor start different
    clocks. That is a Methods statement, not a defect in these numbers.
  - Saga's `ConvergenceWarning: max_iter was reached` on the M3 elastic-net
    base model is pre-existing (inherited unchanged from scripts 18/45) and
    is NOT "fixed" here (changing max_iter would change the frozen C1/C2
    numbers).

COMPARABLE TO (same nested 5x5 design, same outer folds, same 302 patients,
same seed 2026): script 45's CV_AUROC_5fold_nested_corrected, script 46's
phase_x_c1_c2_nested_bootstrap_delta.csv, script 55's
phase_x_decision_curve_nested_paired.csv, script 57's nested plain/C0
vectors.
NOT COMPARABLE TO: script 41's own existing SOFA-referenced numbers
(different CV design -- that is the entire point); script 18/21's
single-split C-rows; `phase_x_severity_score_comparison_302.csv`'s
C1/C2a/C2b rows (already-flagged superseded leaky-CV values).

Reads (read-only):
  clinical_ai/predictions/reduced_model_microbiome_cohort_predictions.csv
  clinical_ai/reports/selected_reduced_model.json
  clinical_ai/predictions/phase_x_c1_c2_oof_probabilities.csv       (script 18 -- Gate C, Gate E/OR-LRT)
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities.csv (script 46 -- Gates A/B)
  ICU_public/semantic_microbiome/harmonized/discovery_cohort_primary.csv
  ICU_public/metadata/microbiome.clinical.csv                       (SOFA)
  ICU_public/semantic_microbiome/structured/*.tsv, literature_core_features.csv
  ICU_public/semantic_microbiome/supervised_semantic/frozen_models/M3_pca_loadings/*
  clinical_ai/reports/phase_x_microbiome_over_sofa_adjusted_or.csv   (script 41 -- Gate E comparison only)
  clinical_ai/reports/phase_x_microbiome_over_sofa_nested_lrt.csv    (script 41 -- Gate E comparison only)
Writes (all new, additive):
  clinical_ai/predictions/phase_x_sofa_nested_oof_probabilities.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_auroc.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_adjusted_or.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_model_lrt.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_nri_idi.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_decision_curve_paired.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv_decision_curve_marginal.csv
  clinical_ai/reports/phase_x_microbiome_over_sofa_nestedcv.md
  clinical_ai/figures/phase_x_microbiome_over_sofa_nestedcv_decision_curve.svg/.jpg
"""
from __future__ import annotations

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler  # noqa: F401  (imported for parity w/ script45 feature block)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, nested_meta  # noqa: E402  (logit() only, as of v4 -- fit_predict_lr no longer used)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "v4fl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "45_phase_x_c1_c2_nested_cv_corrected_v4.py"))
v4fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4fl)  # defs/constants only (main guarded) -- FeatureSource, fit_lr, fit_lr_strict_full, fit_catboost, MAX_ITER_LADDER, ...

log = io_utils.setup_logging("61_phase_x_microbiome_over_sofa_nested_v4")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
SEMANTIC = os.path.join(MICROBIOME_BASE, "semantic_microbiome")
STRUCTURED_DIR = os.path.join(SEMANTIC, "structured")
FROZEN_DIR = os.path.join(SEMANTIC, "supervised_semantic", "frozen_models")
DISCOVERY_CSV = os.path.join(SEMANTIC, "harmonized", "discovery_cohort_primary.csv")
CLINICAL_PRED_CSV = os.path.join(config.DIR_PREDICTIONS, "reduced_model_microbiome_cohort_predictions.csv")
CLINICAL_RAW_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")

OOF18_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")          # script 18 (historical reference only, v4 no longer gates against it)
NESTED46_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities.csv")  # script 46 (Gate A only, as of v4)
OLD_OR_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_adjusted_or.csv")     # script 41 (historical reference only, v4 no longer gates against it)
OLD_LRT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nested_lrt.csv")     # script 41 (historical reference only, v4 no longer gates against it)

OUT_OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_sofa_nested_oof_probabilities_v4.csv")
OUT_AUROC = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_auroc_v4.csv")
OUT_BOOT = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap_v4.csv")
OUT_OR = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_adjusted_or_v4.csv")
OUT_LRT = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_model_lrt_v4.csv")
OUT_NRI = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_nri_idi_v4.csv")
OUT_DCA_PAIRED = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_decision_curve_paired_v4.csv")
OUT_DCA_MARGINAL = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_decision_curve_marginal_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_v4.md")
OUT_GATE_E_CHANGED = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_gateE_changed_rows_v4.csv")
OUT_CONV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_m3_convergence_v4.csv")
OUT_SOFA_COMPARE = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_v3_vs_v4.csv")
OLD61_BOOT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap_v3.csv")
OLD61_AUROC_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_auroc_v3.csv")
OLD61_OR_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_adjusted_or_v3.csv")
OLD61_LRT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_model_lrt_v3.csv")
OLD61_NRI_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_nri_idi_v3.csv")
OLD61_DCA_CSV = os.path.join(config.DIR_REPORTS, "phase_x_microbiome_over_sofa_nestedcv_decision_curve_paired_v3.csv")
# v4 reference files: 45_v4's own recomputed vectors -- M1 (ladder-fixed) and M3-LR (V4_STRICT) are EXPECTED to
# reproduce these exactly (same procedure, same script), so Gates B/C stay strict against these, not v3's files.
V4_SINGLE_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_m3_single_level_oof_probabilities_v4_V4_STRICT.csv")  # 45_v4
V4_NESTED_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv")     # 45_v4

SEED = config.RANDOM_SEED               # 2026
N_FOLDS = config.N_CV_FOLDS             # 5, outer AND inner
N_BOOT_MARGINAL = config.N_BOOTSTRAP    # 1000, matching script 29/41's marginal AUROC CIs
N_BOOT_PAIRED = 2000                    # matching scripts 21/41/46/55/57
EPS = 1e-6

DESIGN_LABEL = "SOFA-referenced nested-CV-corrected (script 61 v3, fold-local M3)"  # must contain neither "single-split" nor
                                                                   # "script 41" -- three selectors key off those
                                                                   # substrings downstream (script 48, build_tables.py)

LOW_DIM_FILES = ["taxonomy_species_filtered.tsv", "taxonomy_genus_filtered.tsv", "humann_pathway_filtered.tsv",
                  "contig_rgi_drugclass_filtered.tsv", "contig_rgi_rpkm_drugclass_filtered.tsv",
                  "mag_rgi_drugclass_filtered.tsv", "vfdb_category_filtered.tsv",
                  "antismash_bgc_types_filtered.tsv", "eggnog_cog_filtered.tsv"]
HIGH_DIM_FILES = {"MAG_ABUND": "mag_abundance_filtered.tsv", "EGGNOG_KO": "eggnog_kegg_ko_filtered.tsv"}


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


logit = nested_meta.logit  # identical to script 45's logit() -- imported, not redefined


def fit_predict_catboost(X_tr, y_tr, X_te):
    """Copied VERBATIM from 45_phase_x_c1_c2_nested_cv_corrected.py (lines
    212-219) -- a third byte-identical copy, matching how script 46 already
    holds the second. NOT added to utils/nested_meta.py (that would edit a
    file script 57 depends on -- design Decision 2)."""
    imp = SimpleImputer(strategy="median").fit(X_tr)
    cb = CatBoostClassifier(verbose=False, random_state=SEED, thread_count=config.get_n_jobs(),
                             iterations=200, depth=4, learning_rate=0.05)
    cb.fit(imp.transform(X_tr), y_tr)
    return cb.predict_proba(imp.transform(X_te))[:, 1]


# ---------------------------------------------------------------------------
# 1. load outcome + clinical AI probability -- identical join to scripts 45/46
#    (verified read-only, 2026-09-05: this join's index is IDENTICAL, in both
#    content and order, to script 18's phase_x_c1_c2_oof_probabilities.csv
#    index and script 46's persisted nested OOF index -- so no separate
#    "script-41-style" index is needed anywhere in this script).
# ---------------------------------------------------------------------------
outcome = pd.read_csv(DISCOVERY_CSV)[["patient_id_clinical", "hospital_mortality"]].set_index("patient_id_clinical")
clin_pred = pd.read_csv(CLINICAL_PRED_CSV).set_index("patient_id_clinical")
if "clinical_AI_probability_PRIMARY" not in clin_pred.columns:
    log.error("No PRIMARY column in %s.", CLINICAL_PRED_CSV)
    sys.exit(1)

df = outcome.join(clin_pred[["clinical_AI_probability_PRIMARY"]], how="inner")
n = len(df)
log.info("N=%d patients with both outcome and clinical AI probability", n)

y = df["hospital_mortality"].values.astype(int)
logit_clin = logit(df["clinical_AI_probability_PRIMARY"].values)

# ---------------------------------------------------------------------------
# 2. SOFA join -- IDENTICAL to script 41's join (verified: same index, same
#    order, same values as script 41's own SOFA array). Keep script 41's hard
#    abort on missing SOFA VERBATIM (design Decision 4): currently a no-op
#    (0/302 missing), and that is exactly why it must stay.
# ---------------------------------------------------------------------------
disc = pd.read_csv(DISCOVERY_CSV)
raw_clin = pd.read_csv(CLINICAL_RAW_CSV)
disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw_clin = raw_clin.merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner").set_index("patient_id_clinical")
raw_clin = raw_clin.reindex(df.index)

SOFA = pd.to_numeric(raw_clin["SOFA"], errors="coerce").values
n_sofa_missing = int(np.isnan(SOFA).sum())
if n_sofa_missing:
    log.error("SOFA missing for %d/%d patients -- refusing to silently drop them.", n_sofa_missing, n)
    sys.exit(1)
log.info("N=%d, %d deaths (%.1f%%); SOFA complete for all (median=%.1f, IQR %.1f-%.1f, range %.0f-%.0f)",
         n, int(y.sum()), 100 * y.mean(), np.median(SOFA),
         np.percentile(SOFA, 25), np.percentile(SOFA, 75), SOFA.min(), SOFA.max())

# ---------------------------------------------------------------------------
# 3. microbiome feature matrices (M1 Literature Core; M3 low-dim + frozen,
#    outcome-blind PCA of the high-dim views) -- copied verbatim from script
#    45 section 2 / script 46 section 2. Do NOT re-fit the M3 PCA (design
#    Decision 2) -- it is an unsupervised, outcome-blind transform fit once
#    on all 302 patients; reusing it unchanged matches scripts 18/45/46.
# ---------------------------------------------------------------------------
lit_core = pd.read_csv(os.path.join(STRUCTURED_DIR, "literature_core_features.csv"), index_col="sample_id")
lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
X_m1 = lit_core.reindex(df.index)

# v4: M1 now goes through v4fl.fit_lr (ladder) via a "static" FeatureSource --
# procedurally IDENTICAL to 45_v4's own REF variant, not a separate M1 path.
m1_source = v4fl.FeatureSource("static", df.index.values, X_static=X_m1)
# M3-CatBoost: UNCHANGED, still V3's fold-local prevalence filter + PCA (no
# internal tuning CV to fix -- already maximally fold-local).
raw_views = v4fl.load_raw_views(df.index.values)
prev_ok, prev_chk = v4fl.verify_raw_reproduces_filtered(raw_views, log)
if not prev_ok:
    log.error("Raw views do not reproduce structured/*_filtered.tsv with all 302 -- STOP (nothing written).")
    sys.exit(1)
m3_source_v3 = v4fl.FeatureSource("prev_pca", df.index.values, raw_views=raw_views)
# M3-LR: v4 FINAL headline, V4_STRICT -- prevalence filter AND PCA both tuning-local.
m3_source_v4strict = v4fl.FeatureSource("raw_concat_full", df.index.values, raw_views=raw_views)

# ---------------------------------------------------------------------------
# 4. base layers -- run ONCE each, reused by every downstream S-row and by
#    Gates A/B/C. 90 base fits total (3 layers x 30), exactly script 45's
#    count -- design's mandatory verification anchor. As of v4, ALL THREE
#    layers go through the same fold-local machinery (nested_base_probs_foldlocal
#    + a v4fl.FeatureSource), M1 included -- nested_meta.nested_base_probs /
#    fit_predict_lr are no longer used anywhere in this script.
# ---------------------------------------------------------------------------
log.info("Computing nested base-model probabilities (3 layers: M1-LR, M3-LR, M3-CatBoost) ...")


def nested_base_probs_foldlocal(source, y_, learner, max_iter, code, n_folds=N_FOLDS, seed=SEED):
    """Same partition and same returned structure as nested_meta.nested_base_probs
    (p[tr_outer] = inner-OOF, p[te_outer] = outer-full fit), but every fit rebuilds
    the feature matrix from its own training rows (45_v4 make_runner/FeatureSource)."""
    from sklearn.model_selection import StratifiedKFold
    run = v4fl.make_runner(source, learner, y_, max_iter, "V4_script61", code)
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_, n_fit = [], 0
    for k, (tr_o, te_o) in enumerate(outer.split(np.zeros(len(y_)), y_)):
        inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        p = np.full(len(y_), np.nan)
        for j, (tr_i, te_i) in enumerate(inner.split(np.zeros(len(tr_o)), y_[tr_o])):
            p[tr_o[te_i]] = run(tr_o[tr_i], tr_o[te_i], "nested_inner", k, j)   # (a) inner-train-only transform
            n_fit += 1
        p[te_o] = run(tr_o, te_o, "nested_outer_full", k, -1)                   # (b) full tr_outer transform
        n_fit += 1
        folds_.append({"tr_outer": tr_o, "te_outer": te_o, "p": p})
    return folds_, n_fit


LR_LEARNERS = (v4fl.fit_lr, v4fl.fit_lr_strict_full)


def _timed_foldlocal(label, learner, code, source):
    import time
    for max_iter in (v4fl.MAX_ITER_LADDER if learner in LR_LEARNERS else [None]):
        n0 = len(v4fl.CONV_LOG)
        t0 = time.time()
        folds_, n_fit = nested_base_probs_foldlocal(source, y, learner, max_iter, code)
        new = v4fl.CONV_LOG[n0:]
        n_bad = sum(1 for r in new if not r["converged"])
        log.info("%s fold-local base layer done in %.1f s (%d fits; max_iter=%s; non-converged LR fits=%d)",
                 label, time.time() - t0, n_fit, max_iter, n_bad)
        if n_bad == 0:
            return folds_, n_fit, max_iter
    log.error("%s: %d non-converged LR fits even at max_iter=%s -- STOP (blocker, nothing written).", label, n_bad, max_iter)
    sys.exit(1)


folds_m1, n_fits_m1, M1_MAX_ITER = _timed_foldlocal("M1-LR", v4fl.fit_lr, "M1", m1_source)
folds_m3lr, n_fits_m3lr, M3LR_MAX_ITER = _timed_foldlocal("M3-LR", v4fl.fit_lr_strict_full, "M3_LR", m3_source_v4strict)
folds_m3cb, n_fits_m3cb, _ = _timed_foldlocal("M3-CatBoost", v4fl.fit_catboost, "M3_CatBoost", m3_source_v3)

for label, n_fits in [("M1-LR", n_fits_m1), ("M3-LR", n_fits_m3lr), ("M3-CatBoost", n_fits_m3cb)]:
    log.info("%s base layer: %d base-model fits (expected 30 = 5 outer x (5 inner + 1 outer-full))", label, n_fits)
    if n_fits != N_FOLDS * (N_FOLDS + 1):
        log.error("%s base layer fit count mismatch: got %d, expected %d. STOP.", label, n_fits, N_FOLDS * (N_FOLDS + 1))
        sys.exit(1)
total_base_fits = n_fits_m1 + n_fits_m3lr + n_fits_m3cb
log.info("Total base-model fits across all 3 layers: %d (expected 90 = 3 x 30, matching script 45's count)",
          total_base_fits)
if total_base_fits != 90:
    log.error("Total base-fit count != 90 (got %d). STOP.", total_base_fits)
    sys.exit(1)

# Sanity check (not a formal gate -- mathematically guaranteed): the outer
# fold partition depends only on (y, n_folds, seed), not on X_micro, so it
# must be IDENTICAL across the three base layers.
for k in range(N_FOLDS):
    if not (np.array_equal(folds_m1[k]["tr_outer"], folds_m3lr[k]["tr_outer"])
            and np.array_equal(folds_m1[k]["tr_outer"], folds_m3cb[k]["tr_outer"])):
        log.error("Outer fold partition differs across base layers at fold %d -- should be impossible "
                  "(StratifiedKFold.split depends only on y/n/seed). STOP.", k)
        sys.exit(1)
log.info("Sanity check PASSED: outer fold partition identical across all 3 base layers.")


# ---------------------------------------------------------------------------
# 5. sofa_meta_eval() -- the script-61-local meta stage (design Decision 1).
#    A fold-invariant covariate matrix Z (SOFA alone, or SOFA+logit_clin),
#    plus an OPTIONAL microbiome base layer (folds, with its cached full-
#    length p array: p[tr_outer]=inner-OOF, p[te_outer]=outer-fit -- see
#    nested_meta.nested_base_probs docstring). One unpenalized
#    LogisticRegression(penalty=None) per outer fold, exactly the meta-model
#    spec used everywhere else in this project (scripts 41/45/46/57).
# ---------------------------------------------------------------------------
def sofa_meta_eval(folds, y, Z, use_micro):
    """Returns the outer-fold OOF probability vector (n,) for the design
    [Z, optionally logit(micro_prob)]. `folds` supplies the outer/inner
    partition (identical across all 3 base layers, asserted above) and, if
    use_micro, the cached base-model probabilities. Z must already be
    positionally aligned with y (same order as df.index everywhere in this
    script)."""
    n_ = len(y)
    y_ = np.asarray(y)
    Z_ = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z_.shape[0] != n_:
        Z_ = Z_.T
    oof = np.full(n_, np.nan)
    for f in folds:
        tr, te = f["tr_outer"], f["te_outer"]
        if use_micro:
            lm = logit(f["p"])
            X_tr = np.column_stack([Z_[tr], lm[tr]])
            X_te = np.column_stack([Z_[te], lm[te]])
        else:
            X_tr, X_te = Z_[tr], Z_[te]
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_tr, y_[tr])
        oof[te] = model.predict_proba(X_te)[:, 1]
    assert not np.isnan(oof).any()
    return oof


def bootstrap_auroc(yv, score, n_boot=N_BOOT_MARGINAL, seed=SEED):
    """Identical to script 41's bootstrap_auroc() -- a FRESH RandomState per
    call, so it is stream-position-independent (design Gate D relies on
    this)."""
    rng = np.random.RandomState(seed)
    point = roc_auc_score(yv, score)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        boots.append(roc_auc_score(yv[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ===========================================================================
# 6. REPRODUCTION GATES A-D -- must ALL pass before any output file opens.
# ===========================================================================
log.info("=== Gate A: fold identity (recomputed C0 vs script 46's persisted vector) ===")
ref46 = pd.read_csv(NESTED46_CSV, index_col=0).reindex(df.index)

recomputed_c0 = sofa_meta_eval(folds_m1, y, logit_clin.reshape(-1, 1), use_micro=False)
max_diff_c0 = float(np.max(np.abs(recomputed_c0 - ref46["oof_prob_C0_clinical_alone"].values)))
auroc_c0 = round(roc_auc_score(y, recomputed_c0), 4)
log.info("Gate A: max|diff| vs script 46 C0 = %.3e (want <1e-6); AUROC=%.4f (want 0.7528)", max_diff_c0, auroc_c0)
gate_a_ok = max_diff_c0 < 1e-6 and abs(auroc_c0 - 0.7528) < 1e-9

log.info("=== Gate B: base-stage identity, all three learners (v4: against 45_v4's OWN vectors -- "
         "M1 and M3-LR are EXPECTED to differ from v3/the original, but are expected to reproduce "
         "45_v4's independently-computed numbers exactly, since both scripts call the identical "
         "v4fl.fit_lr / v4fl.fit_lr_strict_full) ===")
recomputed_c1 = sofa_meta_eval(folds_m1, y, logit_clin.reshape(-1, 1), use_micro=True)
recomputed_c2a = sofa_meta_eval(folds_m3lr, y, logit_clin.reshape(-1, 1), use_micro=True)
recomputed_c2b = sofa_meta_eval(folds_m3cb, y, logit_clin.reshape(-1, 1), use_micro=True)

refv4 = pd.read_csv(V4_NESTED_CSV, index_col=0).reindex(df.index)
gate_b_checks = [   # (label, vector, reference frame, column) -- target AUROC read from 45_v4 itself, not hardcoded
    ("C1", recomputed_c1, refv4, "oof_prob_C1_clinical_plus_M1_LiteratureCore"),    # 45_v4 REF (M1 ladder-fixed)
    ("C2a", recomputed_c2a, refv4, "oof_prob_C2a_clinical_plus_M3_LR"),             # 45_v4 V4_STRICT
    ("C2b", recomputed_c2b, refv4, "oof_prob_C2b_clinical_plus_M3_CatBoost"),       # 45_v4 (= V3, reused verbatim)
]
gate_b_ok = True
for label, vec, refdf, refcol in gate_b_checks:
    target_auroc = round(roc_auc_score(y, refdf[refcol].values), 4)
    max_diff = float(np.max(np.abs(vec - refdf[refcol].values)))
    auroc_v = round(roc_auc_score(y, vec), 4)
    log.info("Gate B %s: max|diff| vs 45_v4 = %.3e (want <1e-6); AUROC=%.4f (45_v4's own AUROC=%.4f)",
              label, max_diff, auroc_v, target_auroc)
    if max_diff >= 1e-6 or abs(auroc_v - target_auroc) >= 1e-9:
        gate_b_ok = False

log.info("=== Gate C: single-split identity (v4: against 45_v4's OWN vectors, all three terms) ===")
v4single = pd.read_csv(V4_SINGLE_CSV, index_col=0).reindex(df.index)  # 45_v4: M1 (REF) + M3_LR (V4_STRICT) + M3_CatBoost (V3, reused)

single_split = {}
for label, folds_ in [("M1", folds_m1), ("M3_LR", folds_m3lr), ("M3_CatBoost", folds_m3cb)]:
    vec = np.full(n, np.nan)
    for f in folds_:
        vec[f["te_outer"]] = f["p"][f["te_outer"]]
    assert not np.isnan(vec).any()
    single_split[label] = vec

gate_c_ok = True
for label in ["M1", "M3_LR", "M3_CatBoost"]:
    ref_col = f"microbiome_prob_{label}"
    max_diff = float(np.max(np.abs(single_split[label] - v4single[ref_col].values)))
    log.info("Gate C %s: max|diff| vs 45_v4 %s = %.3e (want <1e-6)", label, ref_col, max_diff)
    if max_diff >= 1e-6:
        gate_c_ok = False

log.info("=== Gate D: raw-SOFA anchor (design-independent) ===")
sofa_point, sofa_lo, sofa_hi = bootstrap_auroc(y, SOFA)
log.info("Gate D: raw SOFA AUROC = %.4f [%.4f, %.4f] (want 0.7593 [0.6865, 0.8269])",
          sofa_point, sofa_lo, sofa_hi)
gate_d_ok = (abs(round(sofa_point, 4) - 0.7593) < 1e-9 and abs(round(sofa_lo, 4) - 0.6865) < 1e-9
             and abs(round(sofa_hi, 4) - 0.8269) < 1e-9)

# ---------------------------------------------------------------------------
# 6b. Gate E (fifth, free) -- v4: the OR/LRT block now uses THIS SCRIPT'S OWN
#     freshly-fixed single-split vectors (single_split, computed just above
#     for Gate C), not an external file. There is no valid "old" OR/LRT
#     reference for numbers that did not exist before this closure pass --
#     designs with NO microbiome term (S0, S3ref) are still checked strictly
#     against script 41 (completely unaffected by M1/M3 changes); every
#     microbiome-containing design (S1, S2a, S2b, S3a, S3b) is now a
#     changed-row REPORT, not an abort (owner-approved 2026-10-02: an
#     intended change must not abort the run).
# ---------------------------------------------------------------------------
log.info("=== Gate E: adjusted-OR / nested-LRT -- strict for no-microbiome designs (S0,S3ref) vs script 41; "
         "all microbiome-containing designs (S1,S2a,S2b,S3a,S3b) reported old-vs-new, not gated ===")
lg_m1_or = logit(single_split["M1"])
lg_m3lr_or = logit(single_split["M3_LR"])
lg_m3cb_or = logit(single_split["M3_CatBoost"])

DESIGNS = {
    "S0_SOFA_alone":                       (["SOFA"], np.column_stack([SOFA])),
    "S1_SOFA_plus_M1":                     (["SOFA", "logit_M1"], np.column_stack([SOFA, lg_m1_or])),
    "S2a_SOFA_plus_M3LR":                  (["SOFA", "logit_M3_LR"], np.column_stack([SOFA, lg_m3lr_or])),
    "S2b_SOFA_plus_M3CatBoost":            (["SOFA", "logit_M3_CatBoost"], np.column_stack([SOFA, lg_m3cb_or])),
    "S3ref_SOFA_plus_clinicalAI":          (["SOFA", "logit_clinicalAI"], np.column_stack([SOFA, logit_clin])),
    "S3a_SOFA_plus_clinicalAI_plus_M1":    (["SOFA", "logit_clinicalAI", "logit_M1"],
                                            np.column_stack([SOFA, logit_clin, lg_m1_or])),
    "S3b_SOFA_plus_clinicalAI_plus_M3CB":  (["SOFA", "logit_clinicalAI", "logit_M3_CatBoost"],
                                            np.column_stack([SOFA, logit_clin, lg_m3cb_or])),
}
REFERENCE_OF = {
    "S1_SOFA_plus_M1": "S0_SOFA_alone",
    "S2a_SOFA_plus_M3LR": "S0_SOFA_alone",
    "S2b_SOFA_plus_M3CatBoost": "S0_SOFA_alone",
    "S3ref_SOFA_plus_clinicalAI": "S0_SOFA_alone",
    "S3a_SOFA_plus_clinicalAI_plus_M1": "S3ref_SOFA_plus_clinicalAI",
    "S3b_SOFA_plus_clinicalAI_plus_M3CB": "S3ref_SOFA_plus_clinicalAI",
}

or_rows, lrt_rows = [], []
for name, (terms, X) in DESIGNS.items():
    Xc = sm.add_constant(X)
    try:
        fit = sm.Logit(y, Xc).fit(disp=0)
    except Exception as e:  # pragma: no cover
        log.warning("%s: logistic regression failed (%s)", name, e)
        or_rows.append({"model": name, "term": "ALL", "note": f"FAILED: {e}"})
        continue
    for j, term in enumerate(terms, start=1):
        coef, se = fit.params[j], fit.bse[j]
        sd = float(np.std(X[:, j - 1], ddof=1))
        or_rows.append({
            "model": name, "term": term,
            "OR_per_1_unit": round(float(np.exp(coef)), 3),
            "OR_CI_low": round(float(np.exp(coef - 1.96 * se)), 3),
            "OR_CI_high": round(float(np.exp(coef + 1.96 * se)), 3),
            "p_value": round(float(fit.pvalues[j]), 4),
            "OR_per_1_SD": round(float(np.exp(coef * sd)), 3),
            "OR_per_1_SD_CI_low": round(float(np.exp((coef - 1.96 * se) * sd)), 3),
            "OR_per_1_SD_CI_high": round(float(np.exp((coef + 1.96 * se) * sd)), 3),
            "term_SD_in_cohort": round(sd, 3),
            "model_LL": round(float(fit.llf), 3),
            "model_pseudo_R2": round(float(fit.prsquared), 4),
        })
    if name in REFERENCE_OF:
        ref = REFERENCE_OF[name]
        fit_r = sm.Logit(y, sm.add_constant(DESIGNS[ref][1])).fit(disp=0)
        lr_stat = 2 * (fit.llf - fit_r.llf)
        ddof = Xc.shape[1] - sm.add_constant(DESIGNS[ref][1]).shape[1]
        lrt_rows.append({"model": name, "reference": ref,
                         "LRT_chi2": round(float(lr_stat), 3), "LRT_df": int(ddof),
                         "LRT_p": round(float(stats.chi2.sf(lr_stat, ddof)), 4),
                         "logLik_reference": round(float(fit_r.llf), 3),
                         "logLik_model": round(float(fit.llf), 3),
                         "pseudo_R2_reference": round(float(fit_r.prsquared), 4),
                         "pseudo_R2_model": round(float(fit.prsquared), 4)})
or_df = pd.DataFrame(or_rows)
lrt_df = pd.DataFrame(lrt_rows)

gate_e_ok = True
gate_e_detail = []
# v4: CHANGED_MODELS = every design containing ANY microbiome term (M1 or M3) --
# M1 is now ladder-fixed and M3-LR is now V4_STRICT, so S1/S3a (M1) join S2a/S2b/S3b
# (M3) as expected-change REPORT rows. Only S0 (SOFA alone) and S3ref (SOFA+clinical,
# no microbiome) have zero microbiome content and stay STRICT vs script 41.
CHANGED_MODELS = {"S1_SOFA_plus_M1", "S2a_SOFA_plus_M3LR", "S2b_SOFA_plus_M3CatBoost",
                   "S3a_SOFA_plus_clinicalAI_plus_M1", "S3b_SOFA_plus_clinicalAI_plus_M3CB"}
if os.path.exists(OLD_OR_CSV) and os.path.exists(OLD_LRT_CSV):
    old_or = pd.read_csv(OLD_OR_CSV)
    old_lrt = pd.read_csv(OLD_LRT_CSV)
    or_check_cols = ["OR_per_1_unit", "OR_CI_low", "OR_CI_high", "p_value", "OR_per_1_SD",
                      "OR_per_1_SD_CI_low", "OR_per_1_SD_CI_high", "term_SD_in_cohort", "model_LL", "model_pseudo_R2"]
    merged_or_all = or_df.merge(old_or, on=["model", "term"], suffixes=("_new", "_old"))
    merged_lrt_all = lrt_df.merge(old_lrt, on=["model", "reference"], suffixes=("_new", "_old"))
    # Microbiome-containing rows: intended change -> old-vs-new REPORT, not an abort (owner-approved 2026-10-02)
    gate_e_changed = pd.concat([
        merged_or_all[merged_or_all.model.isin(CHANGED_MODELS)].assign(block="adjusted_OR"),
        merged_lrt_all[merged_lrt_all.model.isin(CHANGED_MODELS)].assign(block="LRT")], ignore_index=True)
    for _, r in merged_or_all[merged_or_all.model.isin(CHANGED_MODELS)].iterrows():
        log.info("Gate E (REPORT, changed row) %s %s: OR %s -> %s, p %s -> %s", r.model, r.term,
                 r.OR_per_1_unit_old, r.OR_per_1_unit_new, r.p_value_old, r.p_value_new)
    for _, r in merged_lrt_all[merged_lrt_all.model.isin(CHANGED_MODELS)].iterrows():
        log.info("Gate E (REPORT, changed row) %s LRT: chi2 %s -> %s, p %s -> %s", r.model,
                 r.LRT_chi2_old, r.LRT_chi2_new, r.LRT_p_old, r.LRT_p_new)
    # STRICT rows: zero microbiome term anywhere in the design (S0, S3ref only, as of v4)
    merged_or = merged_or_all[~merged_or_all.model.isin(CHANGED_MODELS)]
    n_strict_or = int((~or_df.model.isin(CHANGED_MODELS)).sum())
    gate_e_detail.append(f"STRICT OR rows checked: {len(merged_or)} (expected {n_strict_or}); changed rows reported: "
                         f"{int(merged_or_all.model.isin(CHANGED_MODELS).sum())}")
    if len(merged_or) != n_strict_or:
        gate_e_ok = False
        gate_e_detail.append(f"OR merge row count mismatch: new={len(or_df)} merged={len(merged_or)}")
    for c in or_check_cols:
        diff = (merged_or[f"{c}_new"] - merged_or[f"{c}_old"]).abs().max()
        gate_e_detail.append(f"OR column {c}: max|diff|={diff:.3e}")
        if diff > 1e-9:
            gate_e_ok = False

    lrt_check_cols = ["LRT_chi2", "LRT_df", "LRT_p", "logLik_reference", "logLik_model",
                       "pseudo_R2_reference", "pseudo_R2_model"]
    merged_lrt = merged_lrt_all[~merged_lrt_all.model.isin(CHANGED_MODELS)]
    n_strict_lrt = int((~lrt_df.model.isin(CHANGED_MODELS)).sum())
    if len(merged_lrt) != n_strict_lrt:
        gate_e_ok = False
        gate_e_detail.append(f"LRT merge row count mismatch: new={len(lrt_df)} merged={len(merged_lrt)}")
    for c in lrt_check_cols:
        diff = (merged_lrt[f"{c}_new"] - merged_lrt[f"{c}_old"]).abs().max()
        gate_e_detail.append(f"LRT column {c}: max|diff|={diff:.3e}")
        if diff > 1e-9:
            gate_e_ok = False
    for line in gate_e_detail:
        log.info("Gate E: %s", line)
else:
    log.warning("Gate E: %s or %s not found -- skipping OR/LRT equality check (this should not happen; "
                "script 41 is expected to have already run).", OLD_OR_CSV, OLD_LRT_CSV)
    gate_e_ok = False

# ---------------------------------------------------------------------------
# 6c. Abort on any gate failure -- WRITE NOTHING. Never relax a tolerance.
# ---------------------------------------------------------------------------
all_gates_ok = gate_a_ok and gate_b_ok and gate_c_ok and gate_d_ok and gate_e_ok
log.info("=== Gate summary: A=%s B=%s C=%s D=%s E=%s ===", gate_a_ok, gate_b_ok, gate_c_ok, gate_d_ok, gate_e_ok)
if not all_gates_ok:
    log.error("ONE OR MORE REPRODUCTION GATES FAILED. Aborting BEFORE opening any output file. "
              "Do not relax a tolerance; if Gate B failed only on C2b (CatBoost), the one permitted retry is "
              "resubmitting at --cpus-per-task=4 with OMP/OPENBLAS/MKL_NUM_THREADS=4 (already this job's "
              "configuration); if it still fails, report the magnitude to the owner.")
    sys.exit(1)
log.info("ALL GATES PASSED (E strict on S0/S3ref only; every microbiome-containing row reported). "
         "Proceeding to compute S0-S3b nested OOF and write outputs.")
io_utils.save_csv(gate_e_changed, OUT_GATE_E_CHANGED)
io_utils.save_csv(pd.DataFrame(v4fl.CONV_LOG), OUT_CONV)
log.info("wrote %s and %s (M1 max_iter: %s; M3-LR max_iter: %s)", OUT_GATE_E_CHANGED, OUT_CONV, M1_MAX_ITER, M3LR_MAX_ITER)

# ===========================================================================
# 7. S0-S3b nested outer-fold OOF probabilities. Persisted IMMEDIATELY
#    (design checklist step 5) so a later-stage failure does not lose the
#    90 base fits.
# ===========================================================================
oof_probs = {
    "S0_SOFA_alone": sofa_meta_eval(folds_m1, y, SOFA.reshape(-1, 1), use_micro=False),
    "S1_SOFA_plus_M1": sofa_meta_eval(folds_m1, y, SOFA.reshape(-1, 1), use_micro=True),
    "S2a_SOFA_plus_M3LR": sofa_meta_eval(folds_m3lr, y, SOFA.reshape(-1, 1), use_micro=True),
    "S2b_SOFA_plus_M3CatBoost": sofa_meta_eval(folds_m3cb, y, SOFA.reshape(-1, 1), use_micro=True),
    "S3ref_SOFA_plus_clinicalAI": sofa_meta_eval(folds_m1, y, np.column_stack([SOFA, logit_clin]), use_micro=False),
    "S3a_SOFA_plus_clinicalAI_plus_M1": sofa_meta_eval(folds_m1, y, np.column_stack([SOFA, logit_clin]), use_micro=True),
    "S3b_SOFA_plus_clinicalAI_plus_M3CB": sofa_meta_eval(folds_m3cb, y, np.column_stack([SOFA, logit_clin]), use_micro=True),
}

oof_out = pd.DataFrame({"hospital_mortality": y, "SOFA": SOFA, "logit_clin": logit_clin}, index=df.index)
oof_out.index.name = "patient_id_clinical"
for name, vec in oof_probs.items():
    oof_out[f"oof_prob_{name}"] = vec
for label in ["M1", "M3_LR", "M3_CatBoost"]:
    oof_out[f"micro_prob_{label}"] = single_split[label]
io_utils.save_csv(oof_out, OUT_OOF_CSV, index=True)
log.info("wrote %s (90 base fits now safely persisted)", OUT_OOF_CSV)

# ===========================================================================
# 8. marginal AUROC + bootstrap 95% CI (raw SOFA anchor + 7 S-designs)
# ===========================================================================
auroc_rows = [{"model": "raw SOFA score (no model, anchor = script 29/41's SOFA row)",
               "terms": "SOFA", "OOF": "n/a (raw score)", "design": DESIGN_LABEL,
               "AUROC": round(sofa_point, 4), "AUROC_CI_low": round(sofa_lo, 4), "AUROC_CI_high": round(sofa_hi, 4)}]
for name, (terms, _) in DESIGNS.items():
    p_, lo_, hi_ = bootstrap_auroc(y, oof_probs[name])
    auroc_rows.append({"model": name, "terms": " + ".join(terms),
                       "OOF": f"nested {N_FOLDS}x{N_FOLDS} (outer x inner), seed={SEED}", "design": DESIGN_LABEL,
                       "AUROC": round(p_, 4), "AUROC_CI_low": round(lo_, 4), "AUROC_CI_high": round(hi_, 4)})
auroc_df = pd.DataFrame(auroc_rows)
io_utils.save_csv(auroc_df, OUT_AUROC)
log.info("\n%s", auroc_df.to_string(index=False))

# ===========================================================================
# 9. paired bootstrap delta AUROC / AUPRC / Brier vs. each model's nested
#    reference -- FRESH RandomState(SEED) PER CONTRAST (deliberate departure
#    from script 41's single shared rng; design Decision 5).
# ===========================================================================
boot_rows = []
for name, ref in REFERENCE_OF.items():
    rng = np.random.RandomState(SEED)  # fresh per contrast
    p_new, p_ref = oof_probs[name], oof_probs[ref]
    d_auroc_b, d_auprc_b, d_brier_b = [], [], []
    for _ in range(N_BOOT_PAIRED):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        nb, rb = p_new[idx], p_ref[idx]
        d_auroc_b.append(roc_auc_score(yb, nb) - roc_auc_score(yb, rb))
        d_auprc_b.append(average_precision_score(yb, nb) - average_precision_score(yb, rb))
        d_brier_b.append(brier_score_loss(yb, rb) - brier_score_loss(yb, nb))  # positive = augmented has LOWER Brier

    def summarize(deltas, point):
        arr = np.asarray(deltas)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p_two = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        return round(point, 4), round(lo, 4), round(hi, 4), round(min(p_two, 1.0), 4)

    a_pt, a_lo, a_hi, a_p = summarize(d_auroc_b, roc_auc_score(y, p_new) - roc_auc_score(y, p_ref))
    c_pt, c_lo, c_hi, c_p = summarize(d_auprc_b, average_precision_score(y, p_new) - average_precision_score(y, p_ref))
    b_pt, b_lo, b_hi, b_p = summarize(d_brier_b, brier_score_loss(y, p_ref) - brier_score_loss(y, p_new))
    boot_rows.append({
        "model": name, "reference": ref, "design": DESIGN_LABEL, "N_boot": N_BOOT_PAIRED,
        "delta_AUROC": a_pt, "delta_AUROC_CI_low": a_lo, "delta_AUROC_CI_high": a_hi, "delta_AUROC_boot_p": a_p,
        "delta_AUPRC": c_pt, "delta_AUPRC_CI_low": c_lo, "delta_AUPRC_CI_high": c_hi, "delta_AUPRC_boot_p": c_p,
        "delta_Brier_improvement": b_pt, "delta_Brier_CI_low": b_lo, "delta_Brier_CI_high": b_hi,
        "delta_Brier_boot_p": b_p,
    })
boot_df = pd.DataFrame(boot_rows)
io_utils.save_csv(boot_df, OUT_BOOT)
log.info("\n%s", boot_df.to_string(index=False))

# ===========================================================================
# 10. adjusted OR / nested LRT -- already computed above for Gate E; persist.
# ===========================================================================
io_utils.save_csv(lrt_df, OUT_LRT)
io_utils.save_csv(or_df, OUT_OR)
log.info("wrote %s / %s (Gate E confirmed these equal script 41's existing files)", OUT_OR, OUT_LRT)

# ===========================================================================
# 11. NRI (continuous + categorical) and IDI, paired-bootstrap CIs.
#     FRESH RandomState(SEED) PER ROW (design Decision 5).
# ===========================================================================
THR_CAT = float(y.mean())


def idi(yv, p_old, p_new):
    ev, ne = yv == 1, yv == 0
    return (p_new[ev].mean() - p_old[ev].mean()) - (p_new[ne].mean() - p_old[ne].mean())


def nri_continuous(yv, p_old, p_new):
    d = p_new - p_old
    ev, ne = yv == 1, yv == 0
    return (np.mean(d[ev] > 0) - np.mean(d[ev] < 0)) + (np.mean(d[ne] < 0) - np.mean(d[ne] > 0))


def nri_categorical(yv, p_old, p_new, thr):
    hi_o, hi_n = p_old >= thr, p_new >= thr
    ev, ne = yv == 1, yv == 0
    ev_part = np.mean(hi_n[ev] & ~hi_o[ev]) - np.mean(~hi_n[ev] & hi_o[ev])
    ne_part = np.mean(~hi_n[ne] & hi_o[ne]) - np.mean(hi_n[ne] & ~hi_o[ne])
    return ev_part + ne_part, ev_part, ne_part


nri_rows = []
for name, ref in REFERENCE_OF.items():
    rng = np.random.RandomState(SEED)  # fresh per row
    p_new, p_old = oof_probs[name], oof_probs[ref]
    b_idi, b_nric, b_nrik = [], [], []
    for _ in range(N_BOOT_PAIRED):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        b_idi.append(idi(yb, p_old[idx], p_new[idx]))
        b_nric.append(nri_continuous(yb, p_old[idx], p_new[idx]))
        b_nrik.append(nri_categorical(yb, p_old[idx], p_new[idx], THR_CAT)[0])

    def summ(bs, point):
        arr = np.asarray(bs)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p_two = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        return round(point, 4), round(lo, 4), round(hi, 4), round(min(p_two, 1.0), 4)

    nri_c = nri_continuous(y, p_old, p_new)
    nri_k, nri_k_ev, nri_k_ne = nri_categorical(y, p_old, p_new, THR_CAT)
    i_pt, i_lo, i_hi, i_p = summ(b_idi, idi(y, p_old, p_new))
    c_pt, c_lo, c_hi, c_p = summ(b_nric, nri_c)
    k_pt, k_lo, k_hi, k_p = summ(b_nrik, nri_k)
    nri_rows.append({
        "model": name, "reference": ref, "design": DESIGN_LABEL,
        "IDI": i_pt, "IDI_CI_low": i_lo, "IDI_CI_high": i_hi, "IDI_boot_p": i_p,
        "NRI_continuous": c_pt, "NRI_continuous_CI_low": c_lo, "NRI_continuous_CI_high": c_hi,
        "NRI_continuous_boot_p": c_p,
        f"NRI_categorical_thr{THR_CAT:.3f}": k_pt, "NRI_cat_CI_low": k_lo, "NRI_cat_CI_high": k_hi,
        "NRI_cat_boot_p": k_p,
        "NRI_cat_event_component": round(nri_k_ev, 4), "NRI_cat_nonevent_component": round(nri_k_ne, 4),
    })
nri_df = pd.DataFrame(nri_rows)
io_utils.save_csv(nri_df, OUT_NRI)
log.info("\n%s", nri_df.to_string(index=False))

# ===========================================================================
# 12. decision curve analysis. net_benefit() identical to scripts 20/21/41/55.
#     PAIRED delta_NB (script-55 estimand) is the INFERENTIAL object (design
#     Decision 5); the marginal per-model bands are written for shape
#     continuity ONLY and are NOT cited as inferential. FRESH
#     RandomState(SEED) PER THRESHOLD BLOCK (design Decision 5) -- within one
#     threshold's block, the SAME resample index scores every model (paired
#     resampling, script 41/55's convention), so both the marginal bands and
#     the paired deltas at that threshold come from the identical draws.
# ===========================================================================
def net_benefit(y_arr, prob, pt):
    w = pt / (1 - pt)
    pred_pos = prob >= pt
    tp = ((pred_pos) & (y_arr == 1)).sum()
    fp = ((pred_pos) & (y_arr == 0)).sum()
    return tp / len(y_arr) - (fp / len(y_arr)) * w


ALL_MODELS = list(DESIGNS.keys())  # S0 .. S3b, 7 models
CONTRASTS = list(REFERENCE_OF.items())  # 6 (candidate, reference) pairs
thresholds = np.arange(0.02, 0.51, 0.02)
PRESPECIFIED_THRESHOLDS = [0.10, 0.30]  # same two thresholds as script 55, same POST_HOC_BUT_METHOD_DRIVEN status

marginal_rows = []
paired_rows = []
for pt in thresholds:
    pt = round(float(pt), 3)
    rng = np.random.RandomState(SEED)  # fresh per threshold block
    nb_boot = {m: [] for m in ALL_MODELS}
    for _ in range(N_BOOT_PAIRED):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        for m in ALL_MODELS:
            nb_boot[m].append(net_benefit(yb, oof_probs[m][idx], pt))

    nb_all = float(y.mean() - (1 - y.mean()) * (pt / (1 - pt)))
    m_row = {"threshold": pt, "design": DESIGN_LABEL,
             "net_benefit_treat_all": round(nb_all, 4), "net_benefit_treat_none": 0.0}
    for m in ALL_MODELS:
        nb_point = net_benefit(y, oof_probs[m], pt)
        lo_m, hi_m = np.percentile(nb_boot[m], [2.5, 97.5])
        m_row[f"net_benefit_{m}"] = round(nb_point, 4)
        m_row[f"net_benefit_{m}_CI_low"] = round(float(lo_m), 4)
        m_row[f"net_benefit_{m}_CI_high"] = round(float(hi_m), 4)
    marginal_rows.append(m_row)

    for candidate, reference in CONTRASTS:
        nb_cand = net_benefit(y, oof_probs[candidate], pt)
        nb_ref = net_benefit(y, oof_probs[reference], pt)
        deltas = np.array(nb_boot[candidate]) - np.array(nb_boot[reference])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_two = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        delta_nb = nb_cand - nb_ref
        paired_rows.append({
            "model": candidate, "reference": reference, "design": DESIGN_LABEL, "threshold": pt,
            "prespecified_primary": pt in PRESPECIFIED_THRESHOLDS,
            "net_benefit_reference": round(nb_ref, 4), "net_benefit_candidate": round(nb_cand, 4),
            "net_benefit_treat_all": round(nb_all, 4), "net_benefit_treat_none": 0.0,
            "delta_NB": round(delta_nb, 4), "delta_NB_CI_low": round(lo, 4), "delta_NB_CI_high": round(hi, 4),
            "delta_NB_boot_p": round(min(p_two, 1.0), 4),
            "CI_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": ("favors_candidate" if lo > 0 else "favors_reference" if hi < 0 else "n/a"),
            "N_boot": N_BOOT_PAIRED,
        })

marginal_df = pd.DataFrame(marginal_rows)
paired_df = pd.DataFrame(paired_rows)

# Holm-Bonferroni across the 12 prespecified cells (6 contrasts x 2
# thresholds) WITHIN this script's own family -- script 55's per-script
# convention; the eventual combined-T2b family size (6+12) is an owner
# decision out of scope here (design Decision 6).
prespecified_mask = paired_df["prespecified_primary"]
pre_df = paired_df.loc[prespecified_mask].copy()
m_holm = len(pre_df)
order = pre_df["delta_NB_boot_p"].argsort().values
holm_p = np.empty(m_holm)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = (m_holm - rank) * pre_df["delta_NB_boot_p"].values[idx]
    running_max = max(running_max, adj)
    holm_p[idx] = min(running_max, 1.0)
pre_df["delta_NB_boot_p_holm"] = np.round(holm_p, 4)
paired_df = paired_df.merge(
    pre_df[["model", "reference", "threshold", "delta_NB_boot_p_holm"]],
    on=["model", "reference", "threshold"], how="left")

io_utils.save_csv(marginal_df, OUT_DCA_MARGINAL)
io_utils.save_csv(paired_df, OUT_DCA_PAIRED)
log.info("wrote %s (continuity only, NOT cited as inferential)", OUT_DCA_MARGINAL)
log.info("wrote %s (the INFERENTIAL DCA object)", OUT_DCA_PAIRED)
log.info("\n%s", pre_df[["model", "reference", "threshold", "delta_NB", "delta_NB_CI_low", "delta_NB_CI_high",
                          "delta_NB_boot_p", "delta_NB_boot_p_holm", "CI_excludes_zero"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 12b. figure -- same 2-panel shape as script 41 (S0 vs S2b; S3ref vs S3b),
#      drawn from the marginal bands (a net-benefit CURVE is inherently a
#      per-model, not a paired-delta, quantity). Caption states explicitly
#      that the inferential comparison is the paired table above, not these
#      marginal bands (design Decision 5's "cite the paired columns only").
# ---------------------------------------------------------------------------
log.info("v3: DCA figure not drawn here (no new matplotlib figures); rebuild in R from %s", OUT_DCA_PAIRED)

# ===========================================================================
# 13. markdown report
# ===========================================================================
def verdict(row):
    lo, hi = row["delta_AUROC_CI_low"], row["delta_AUROC_CI_high"]
    if lo > 0:
        return "adds discrimination (CI excludes 0)"
    if hi < 0:
        return "degrades discrimination (CI excludes 0)"
    return "no evidence of added discrimination (CI includes 0)"


boot_show = boot_df[["model", "reference", "delta_AUROC", "delta_AUROC_CI_low", "delta_AUROC_CI_high",
                     "delta_AUROC_boot_p", "delta_AUPRC", "delta_AUPRC_boot_p",
                     "delta_Brier_improvement", "delta_Brier_boot_p"]].copy()
boot_show["verdict"] = boot_df.apply(verdict, axis=1)

micro_or = or_df[or_df["term"].astype(str).str.startswith("logit_M")][
    ["model", "term", "OR_per_1_unit", "OR_CI_low", "OR_CI_high", "p_value",
     "OR_per_1_SD", "OR_per_1_SD_CI_low", "OR_per_1_SD_CI_high"]]
lrt_show = lrt_df[["model", "reference", "LRT_chi2", "LRT_df", "LRT_p", "pseudo_R2_reference", "pseudo_R2_model"]]

# Only the five models that actually CARRY a microbiome term may license a
# statement about the microbiome. S3ref (SOFA + clinical AI) does not: it is
# the reference the S3 arms are measured against. Filtering on it is not
# cosmetic -- S3ref is the ONE prespecified DCA cell whose CI excludes zero,
# so counting it made the headline assert that "at least one microbiome
# model improved ... net benefit" on the strength of a contrast containing
# no microbiome feature at all.
MICROBIOME_MODELS = frozenset({
    "S1_SOFA_plus_M1", "S2a_SOFA_plus_M3LR", "S2b_SOFA_plus_M3CatBoost",
    "S3a_SOFA_plus_clinicalAI_plus_M1", "S3b_SOFA_plus_clinicalAI_plus_M3CB",
})
assert MICROBIOME_MODELS < set(DESIGNS), "MICROBIOME_MODELS names must exist in DESIGNS"
assert "S3ref_SOFA_plus_clinicalAI" not in MICROBIOME_MODELS
assert "S0_SOFA_alone" not in MICROBIOME_MODELS

boot_micro = boot_df[boot_df.model.isin(MICROBIOME_MODELS)]
pre_micro = pre_df[pre_df.model.isin(MICROBIOME_MODELS)]
auroc_sig = boot_micro[(boot_micro.delta_AUROC_CI_low > 0) | (boot_micro.delta_AUROC_CI_high < 0)]
micro_sig = micro_or[(micro_or.OR_CI_low > 1) | (micro_or.OR_CI_high < 1)]
# The numerator and the denominator must be filtered together. Filtering only
# the numerator produced "4 of 6", mixing microbiome-only rejections against a
# denominator that included S3ref -- the clinical-AI contrast, which carries no
# microbiome term. The honest statement is 4 of the 5 microbiome-term tests.
lrt_micro = lrt_df[lrt_df.model.isin(MICROBIOME_MODELS)]
lrt_sig = lrt_micro[lrt_micro.LRT_p < 0.05]
assert len(lrt_micro) == len(MICROBIOME_MODELS), (
    f"expected one LRT per microbiome model, got {len(lrt_micro)}")
dca_sig = pre_micro[pre_micro["CI_excludes_zero"]]
dca_sig_holm = pre_micro[pre_micro["CI_excludes_zero"] & (pre_micro["delta_NB_boot_p_holm"] < 0.05)]

# Counted over the FULL threshold grid, not the prespecified subset, so the
# split-verdict text below can state what was actually examined. The earlier
# wording claimed no cell excluded zero across "25 thresholds x 6 contrasts"
# while the object tested was the 12 prespecified cells -- and several
# non-prespecified cells (0.04, 0.08) do exclude zero.
grid_micro = paired_df[paired_df.model.isin(MICROBIOME_MODELS)]
grid_micro_sig = grid_micro[grid_micro["CI_excludes_zero"]]
grid_sig_thresholds = ", ".join(f"{t:g}" for t in sorted(grid_micro_sig["threshold"].unique())) or "none"
# If a full-grid cell that excludes zero ever lands ON a prespecified
# threshold, the sentence built from these counts would be self-contradictory
# (0 prespecified cells excluding zero, yet a grid cell at a prespecified
# threshold doing so). Fail loudly rather than print it.
assert not (set(grid_micro_sig["threshold"]) & set(PRESPECIFIED_THRESHOLDS) and dca_sig.empty), (
    "grid/prespecified DCA counts disagree -- do not emit the split-verdict text")

if auroc_sig.empty and micro_sig.empty and dca_sig.empty:
    headline = ("**Headline: NO. Under genuine nested cross-validation, the microbiome score adds no demonstrable "
                "prognostic value over SOFA in this cohort, on any criterion examined (discrimination, adjusted "
                "association, or net benefit).**")
elif auroc_sig.empty and dca_sig.empty and not micro_sig.empty:
    headline = (
        "**Headline: split verdict, and the split is the finding -- SURVIVES nested CV.** "
        f"No microbiome model improves out-of-fold *discrimination* over its nested reference -- all "
        f"{len(boot_micro)} paired-bootstrap delta-AUROC confidence intervals include zero. On net "
        f"benefit, 0 of the {len(pre_micro)} evaluated microbiome decision-curve cells has a paired "
        # "evaluated", not "prespecified": F2's 6 cells were fixed 2026-09-05 before this
        # script ran, but F3's 4 were added by a 2026-09-06 addendum after these results
        # were seen (wording rule 3, extended 2026-09-08). Corrected here so a future run
        # emits it correctly; the script was NOT re-run for a wording change.
        f"delta-NB CI excluding zero; across the full {len(grid_micro)}-cell threshold grid "
        f"{len(grid_micro_sig)} cells do, at thresholds {grid_sig_thresholds}. Those lie outside the "
        "prespecified pair and so are not members of the Holm-corrected family; they are descriptive "
        "only and no Holm-adjusted p exists for them. But "
        f"{len(micro_sig)} of {len(micro_or)} microbiome terms are nominally significantly associated with "
        f"mortality after adjustment for SOFA (full-data fit, unaffected by the CV-nesting design), and "
        f"{len(lrt_sig)} of {len(lrt_micro)} nested likelihood-ratio tests on microbiome terms "
        "reject at 0.05. "
        "**The defensible claim from these data remains association, not added discriminative or "
        "decision-analytic performance.**")
else:
    headline = ("**Headline: at least one microbiome model improved out-of-fold discrimination or net benefit "
                "over its nested reference with a CI excluding zero under genuine nested CV -- see sections 2 and "
                "5, and read the multiplicity caveat.**")

md = [
    "# Incremental value of the microbiome score OVER SOFA -- nested cross-validation rebuild",
    "",
    "> Nested-CV rebuild of `41_phase_x_microbiome_over_sofa.py`'s single-split stacked-OOF design, per "
    "`semantic_microbiome/reports/paper1/script41_nested_rebuild_DESIGN.md` (deep-reasoner, 2026-09-05), "
    "answering reviewer critique R4 / remaining-work item 9 (1B-4): script 41's SOFA-referenced contrasts were "
    "the last model family sitting on the leaky design this project's corrections ledger diagnosed and replaced "
    "everywhere else (TS1 row 4). Strictly additive: no existing script or result file was modified. Original "
    "(superseded, non-nested) result: `phase_x_microbiome_over_sofa.md` -- NOT modified by this script.",
    "",
    "## Design",
    "",
    f"- N = {n} patients, {int(y.sum())} in-hospital deaths ({100 * y.mean():.1f}%) -- identical cohort to the "
    "C0/C1/C2a/C2b family (scripts 45/46/55/57) and to script 41.",
    "- SOFA enters as a **linear** covariate; no spline, categorisation or alternative specification -- fixed "
    "before any nested-CV number here was seen, not re-opened by this rebuild.",
    f"- Genuine nested 5x5 cross-validation: one OUTER {N_FOLDS}-fold split for honest evaluation; a fresh INNER "
    f"{N_FOLDS}-fold split, run ONLY within each outer training fold, generates the meta-training microbiome "
    "features. Base model specs (elastic-net LR / CatBoost) and the meta-model spec "
    "(`LogisticRegression(penalty=None)`) are UNCHANGED from scripts 18/41/45/46 -- only the CV nesting differs.",
    "- Three base layers (M1 elastic-net LR, M3 elastic-net LR, M3 CatBoost), 30 fits each = 90 total, run ONCE "
    "and reused by every S-design and by the reproduction gates -- exactly script 45's fit count.",
    "- **Five reproduction gates, all run and PASSED before any output file was opened** (see run log for exact "
    "values): (A) fold identity vs script 46's C0 (unaffected by this pass -- no microbiome term); (B) base-stage "
    "identity, all three learners, against 45_v4's own recomputed C1/C2a/C2b (M1 ladder-fixed, M3-LR V4_STRICT, "
    "M3-CatBoost unchanged); (C) single-split identity, all three terms, against 45_v4's own vectors; "
    "(D) raw-SOFA anchor, design-independent; (E) the full-data adjusted-OR/nested-LRT block reproduces script "
    "41's existing files exactly only for designs with zero microbiome content (S0, S3ref) -- every microbiome-"
    "containing design (S1, S2a, S2b, S3a, S3b) is reported old-vs-new instead, since both M1 and M3-LR are "
    "intended changes in this pass. No tolerance was relaxed on the rows that remained strict.",
    "- **Deliberate departures from script 41** (documented, not silent): (1) a FRESH `RandomState(SEED)` per "
    "contrast / per NRI-IDI row / per DCA threshold block, instead of script 41's one shared `rng` (Monte-Carlo "
    "noise only, ~+-0.001 on a CI bound, not an estimand change -- script 57's convention, fixing script 28's "
    "shared-rng bug). (2) The decision-curve INFERENTIAL object is the script-55 PAIRED delta_NB (computed "
    "within each bootstrap replicate against each model's own nested reference), not script 41's marginal-CI-band "
    "object -- script 55 already diagnosed that object as invalid twice over. The marginal-band CSV is retained "
    "for shape continuity only and is not cited as inferential here.",
    "- **What changed from the original script 41 design**: the full-data adjusted-OR and nested-LRT block used "
    "to read script 18's persisted single-split OOF microbiome probabilities (v1-v3); as of v4 it uses THIS "
    "script's own freshly-fixed single-split vectors (M1 ladder-fixed, M3-LR V4_STRICT, M3-CatBoost unchanged), "
    "since there is no valid external reference for OR/LRT values on numbers that did not exist before this "
    "closure pass. This is also Gate E.",
    "- **v3 (2026-10-02): fold-local M3.** The frozen M3 PCA and the global prevalence filter were replaced by "
    "45_v3's fold-local construction (variant V3): every M3 base fit rebuilds its prevalence mask, imputer, "
    "scaler and PCA(20) from its own training rows. M1 stayed byte-identical to the original in v3.",
    "- **v4 (2026-10-02): M1 convergence fix + strict tuning-local M3-LR (FINAL technical closure pass).** "
    "M1 now runs through the same convergence ladder (max_iter=" + str(M1_MAX_ITER) + ", zero non-converged fits) "
    "as every other LR fit in this pass, via 45_v4's own fit_lr, instead of staying pinned at the old max_iter=3000. "
    "M3-LR moves from V3 to V4_STRICT: the prevalence filter AND the PCA are now BOTH refit inside "
    "LogisticRegressionCV's own hyperparameter-selection CV (not just inside the outer/stacking-inner folds), via "
    "GridSearchCV(Pipeline[FullBlockPreprocessor, LR]) -- this is the FINAL headline estimate for the data-driven "
    "LR model, superseding V3. M3-LR runs at max_iter=" + str(M3LR_MAX_ITER) + " with zero non-converged fits "
    "(per-fit log: `" + os.path.basename(OUT_CONV) + "`). M3-CatBoost is unchanged (still V3's fold-local source; "
    "no internal tuning CV to fix). Gates B/C are strict for ALL three base layers, but against 45_v4's own "
    "vectors (not v3's or the original script 46/18 files) since M1 and M3-LR are now expected to differ from "
    "those. Gate E is strict only for designs with zero microbiome content (S0, S3ref); every microbiome-"
    "containing design (S1, S2a, S2b, S3a, S3b) is an old-vs-new report, not an abort "
    "(`" + os.path.basename(OUT_GATE_E_CHANGED) + "`).",
    "",
    "- **SOFA timing (previously a caveat here, now resolved).** The owner attested on 2026-09-06 that SOFA and "
    "the C0 `min`/`max` inputs are first-24-hour values following ICU admission, `min`/`max` meaning extremes "
    "*within* that window (`semantic_microbiome/reports/paper1/SOFA_TIMING_ATTESTATION.md`). Earlier versions of "
    "this report hedged that the comparator might be a worst-during-stay quantity and the null therefore "
    "conservative. That hedge is retired: the comparator is fair, and the null below stands on its own terms. "
    "One genuine residual remains, and it is a Methods statement rather than a defect in these numbers -- the "
    "clinical snapshot is indexed to ICU admission while this cohort is indexed to hospital admission, so for "
    "patients transferred to the ICU from a ward the comparator and the predictor start different clocks.",
    "",
    "## 1. Out-of-fold AUROC (nested)",
    "",
    df_to_md_table(auroc_df),
    "",
    "## 2. Paired-bootstrap change vs. the nested reference model",
    "",
    "Each augmented model is tested against its own nested reference (S1/S2a/S2b/S3ref against S0; S3a/S3b "
    "against S3ref), fresh `RandomState(2026)` per contrast, same resample index scoring both models each "
    "replicate. `delta_Brier_improvement` > 0 means the augmented model has LOWER (better) Brier score.",
    "",
    df_to_md_table(boot_show),
    "",
    "## 3. Adjusted odds ratios and nested likelihood-ratio tests (Gate E: S0/S3ref identical to script 41; every microbiome-containing design uses this script's own v4 single-split probabilities)",
    "",
    "Same design as script 41 (full-data logistic regression on a single-split microbiome_prob, controlling for "
    "SOFA and, in S3a/S3b, the clinical AI), but as of v4 the microbiome_prob itself comes from this script's own "
    "freshly-fixed base layer (M1 ladder-fixed, M3-LR V4_STRICT), not script 18's persisted file: this is a "
    "full-data association fit, never claimed to be out-of-fold, and was never affected by the CV-nesting bug "
    "that motivated this rebuild -- see script 45's identical treatment of the C-family's OR/LRT block.",
    "",
    df_to_md_table(micro_or),
    "",
    df_to_md_table(lrt_show),
    "",
    "## 4. Reclassification: NRI and IDI (nested)",
    "",
    df_to_md_table(nri_df[["model", "reference", "IDI", "IDI_CI_low", "IDI_CI_high", "IDI_boot_p",
                            "NRI_continuous", "NRI_continuous_CI_low", "NRI_continuous_CI_high",
                            "NRI_continuous_boot_p"]]),
    "",
    df_to_md_table(nri_df[["model", "reference", f"NRI_categorical_thr{THR_CAT:.3f}", "NRI_cat_CI_low",
                            "NRI_cat_CI_high", "NRI_cat_boot_p", "NRI_cat_event_component",
                            "NRI_cat_nonevent_component"]]),
    "",
    "## 5. Decision curve analysis -- PAIRED delta_NB is the inferential object",
    "",
    "At the two prespecified thresholds (10%, 30%; same POST_HOC_BUT_METHOD_DRIVEN status as script 55), with "
    "Holm-Bonferroni adjustment across these 12 cells (6 contrasts x 2 thresholds), WITHIN this script's own "
    "family (the eventual combined-with-T2b family size is an owner decision, out of scope here):",
    "",
    df_to_md_table(pre_df[["model", "reference", "threshold", "delta_NB", "delta_NB_CI_low", "delta_NB_CI_high",
                            "delta_NB_boot_p", "delta_NB_boot_p_holm", "CI_excludes_zero", "direction"]]),
    "",
    f"{len(dca_sig)}/{len(pre_df)} prespecified cells have a paired delta_NB CI excluding zero on the nominal p; "
    f"{len(dca_sig_holm)}/{len(pre_df)} survive Holm at alpha=0.05. Full 25-threshold grid in "
    f"`{os.path.basename(OUT_DCA_PAIRED)}`. Marginal per-model bands (continuity only, NOT inferential) in "
    f"`{os.path.basename(OUT_DCA_MARGINAL)}` and the figure.",
    "",
    "## 6. Interpretation",
    "",
    headline,
    "",
    "### Comparability",
    "",
    "**Comparable to** (same nested 5x5 design, same outer folds, same 302 patients, same seed 2026): script 45's "
    "`CV_AUROC_5fold_nested_corrected`, script 46's `phase_x_c1_c2_nested_bootstrap_delta.csv`, script 55's "
    "`phase_x_decision_curve_nested_paired.csv`, script 57's nested plain/C0 vectors.",
    "",
    "**NOT comparable to**: script 41's own existing SOFA-referenced numbers (different CV design -- that is "
    "the entire point of this rebuild); script 18/21's single-split C-rows; "
    "`phase_x_severity_score_comparison_302.csv`'s C1/C2a/C2b rows (already-flagged superseded leaky-CV values).",
    "",
    "### Caveats (identical in substance to script 41's, restated for this design)",
    "",
    f"- **Multiplicity.** {len(boot_df)} augmented-vs-reference AUROC/AUPRC/Brier comparisons, {len(nri_df)} "
    f"NRI/IDI comparisons and {len(pre_df)} prespecified DCA cells are reported. No cross-family multiplicity "
    "correction is applied here (consistent with the rest of this project's Phase X work); the DCA block applies "
    "Holm only within its own 12-cell family (see section 5).",
    f"- **Power.** At N={n} with {int(y.sum())} events, this study cannot detect small increments in "
    "discrimination. A null delta AUROC is 'not demonstrated', not 'demonstrated absent'.",
    "- Standard OOF-stacking optimism is still present at the meta-model layer for every design here (a "
    "patient's microbiome/meta prediction is out-of-fold for that patient, but the meta model is trained on "
    "other patients whose base-model features came from models that, for other outer folds, did see that "
    "patient) -- identical caveat to scripts 41/45/46/57, and it favours the microbiome if anything, so it does "
    "not undermine a null result.",
    "",
    "No specification was altered in pursuit of significance. SOFA-linear, nested 5x5/seed 2026 and the model "
    "set above were fixed before the numbers were read, and are reported whatever they show.",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
