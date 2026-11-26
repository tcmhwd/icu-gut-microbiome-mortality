#!/usr/bin/env python3
"""lit_LOCO_AGI_M1_refit.py

Follow-up to lit_LOCO_validation_sensitivity.py: the AGI LOCO pass found
Literature_Prevotella_genus_relabund loses Tier A/B eligibility once PMID
40013797 (the AGI cohort's own publication) is masked -- ALL of Prevotella's
"direct human ICU mortality evidence" rows in evidence_flat.csv came from
that single paper; its other 6 supporting PMIDs are non-ICU/indirect
evidence only. This is a real (not fast-path) feature-panel change, so per
task Section 10-13 this builds the actual LOCO-AGI M1 model:

  - drop ONLY Literature_Prevotella_genus_relabund (16 -> 15 features);
    Literature_obligate_anaerobe_preservation's FROZEN FORMULA
    (sum[Faecalibacterium, Prevotella], lit08/lit14) is left unmodified --
    per the task's explicit "do not alter biological score formulas" rule,
    this composite's definition is not retroactively rewritten. It is
    flagged as an indirectly-affected feature in the report, not silently
    fixed.
  - refit using the IDENTICAL architecture/hyperparameters/seed as
    lit14_freeze_pipeline.py's M1 fit (Pipeline: median-impute, scale,
    LogisticRegressionCV(Cs=8, cv=3, penalty=elasticnet, solver=saga,
    l1_ratios=[0.1,0.5,0.9], max_iter=3000, scoring=roc_auc,
    random_state=SEED=2026, n_jobs=4)), fit ONLY on the 302 Gangnam
    discovery-cohort patients (hospital_mortality) -- no new feature
    selection, no new hyperparameter search, no outcome-based tuning.
  - apply ONLY to AGI_BGISEQ (frozen application, no refit/threshold-tune/
    calibration/PCA on AGI), using the exact same external-application code
    path as lit15_public_external_validation.py (bootstrap_auc_ci,
    calibration_intercept_slope, n_boot=2000, seed=2026).
  - compare paired predictions (original frozen M1 vs LOCO-AGI M1) on the
    same 210 AGI patients: Pearson r, Spearman rho, mean/median/max abs diff.

PRJNA842425 needs no equivalent script: its exclusion set is empty (its own
paper is not in the corpus and no follow-up was found), so its LOCO panel is
identical to the original M1 by construction (verified separately).

Writes (overwrites nothing outside loco_validation/):
  literature_prior/loco_validation/LOCO_AGI_M1_refit_performance.csv/.md
  literature_prior/loco_validation/LOCO_AGI_M1_paired_predictions.csv
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEMANTIC, HARMONIZED_DIR, STRUCTURED_DIR, ICU_PUBLIC, SEED  # noqa: E402

FROZEN_DIR = SEMANTIC / "supervised_semantic" / "frozen_models"
OUT_DIR = SEMANTIC / "literature_prior" / "loco_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_FEATURE = "Literature_Prevotella_genus_relabund"

# ---------------------------------------------------------------------
# Gangnam training data (identical to lit14)
# ---------------------------------------------------------------------
cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(
    columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
y_gangnam = cohort["hospital_mortality"].astype(int)

lit_core_full = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
assert DROP_FEATURE in lit_core_full.columns, f"{DROP_FEATURE} not found -- check literature_core_features.csv schema"
lit_core_loco = lit_core_full.drop(columns=[DROP_FEATURE])
lit_core_loco.columns = [f"LitCore_{c}" for c in lit_core_loco.columns]
lit_core_orig = lit_core_full.copy()
lit_core_orig.columns = [f"LitCore_{c}" for c in lit_core_orig.columns]

X_m1_loco = lit_core_loco.reindex(y_gangnam.index)
print(f"[LOCO-AGI-refit] Gangnam training matrix: {X_m1_loco.shape} (dropped {DROP_FEATURE}; "
      f"{lit_core_full.shape[1]} -> {lit_core_loco.shape[1]} features)")

# ---------------------------------------------------------------------
# refit, IDENTICAL hyperparameters/seed to lit14_freeze_pipeline.py's M1 fit
# ---------------------------------------------------------------------
loco_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                  l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                  random_state=SEED, n_jobs=4)),
]).fit(X_m1_loco, y_gangnam)
joblib.dump(loco_pipe, OUT_DIR / "M1_literature_core_LR_LOCO_AGI.joblib")
print(f"[LOCO-AGI-refit] refit on all {len(y_gangnam)} Gangnam patients, seed={SEED}, saved.")

orig_pipe = joblib.load(FROZEN_DIR / "M1_literature_core_LR.joblib")

# ---------------------------------------------------------------------
# external application to AGI_BGISEQ ONLY -- identical code path to lit15
# ---------------------------------------------------------------------
sm = pd.read_csv(ICU_PUBLIC / "merge" / "output" / "sample_metadata.tsv", sep="\t", dtype=str)
sm["mortality"] = pd.to_numeric(sm["mortality"], errors="coerce")
public = sm[sm.dataset != "ICU"].set_index("sample_id")
ids = public[public.subcohort == "AGI_BGISEQ"].index.tolist()
y_ext = public.loc[ids, "mortality"].astype(int).values
print(f"[LOCO-AGI-refit] AGI_BGISEQ external cohort: N={len(ids)}, deaths={int(y_ext.sum())}")

X_ext_loco = lit_core_loco.reindex(ids)
X_ext_orig = lit_core_orig.reindex(ids)
p_loco = loco_pipe.predict_proba(X_ext_loco)[:, 1]


def manual_predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    """Bypass sklearn's LogisticRegressionCV.predict_proba (broken across the
    sklearn 1.8.0 [frozen pickle] -> 1.4.2 [this env] version gap --
    AttributeError: 'LogisticRegressionCV' object has no attribute
    'multi_class') by reading the fitted imputer/scaler/coef_/intercept_
    arrays directly and computing the sigmoid by hand -- numerically
    identical to what predict_proba would do, avoids the incompatible
    method entirely. Same technique already used/verified in script 37
    (phase_x_rescue_m1_marker_contributions.py) this session."""
    imputer, scaler, clf = pipe.named_steps["impute"], pipe.named_steps["scale"], pipe.named_steps["clf"]
    Xi = imputer.transform(X.values)
    Xs = (Xi - scaler.mean_) / scaler.scale_
    logit = Xs @ clf.coef_[0] + clf.intercept_[0]
    return 1.0 / (1.0 + np.exp(-logit))


p_orig = manual_predict_proba(orig_pipe, X_ext_orig)
# sanity check: manual computation must match sklearn's own predict_proba on
# the freshly-fit LOCO pipe (same sklearn version, so predict_proba works
# natively there) -- confirms the bypass is numerically correct before
# trusting it for the (otherwise-unloadable) original frozen pipe
p_loco_manual = manual_predict_proba(loco_pipe, X_ext_loco)
max_manual_diff = float(np.max(np.abs(p_loco_manual - p_loco)))
print(f"[LOCO-AGI-refit] manual-vs-sklearn predict_proba sanity check (LOCO pipe, same sklearn version): "
      f"max|diff|={max_manual_diff:.8f}")
assert max_manual_diff < 1e-6, "manual predict_proba bypass does not match sklearn's own output -- do not trust p_orig"


def bootstrap_auc_ci(y_true, p, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        bidx = rng.choice(idx, len(idx), replace=True)
        if len(set(y_true[bidx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[bidx], p[bidx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def calibration_intercept_slope(y_true, p):
    eps = 1e-6
    logit_p = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression().fit(logit_p, y_true)
    return lr.intercept_[0], lr.coef_[0][0]


auc_loco = roc_auc_score(y_ext, p_loco)
auc_orig = roc_auc_score(y_ext, p_orig)
ci_loco = bootstrap_auc_ci(y_ext, p_loco)
ci_orig = bootstrap_auc_ci(y_ext, p_orig)
brier_loco = brier_score_loss(y_ext, p_loco)
brier_orig = brier_score_loss(y_ext, p_orig)
cal_loco = calibration_intercept_slope(y_ext, p_loco)
cal_orig = calibration_intercept_slope(y_ext, p_orig)

pear_r, pear_p = pearsonr(p_orig, p_loco)
spear_r, spear_p = spearmanr(p_orig, p_loco)
abs_diff = np.abs(p_orig - p_loco)

perf = pd.DataFrame([{
    "cohort": "AGI_BGISEQ", "N": len(ids), "deaths": int(y_ext.sum()),
    "original_M1_AUROC": round(auc_orig, 4), "original_AUROC_CI_low": round(ci_orig[0], 4), "original_AUROC_CI_high": round(ci_orig[1], 4),
    "LOCO_AGI_M1_AUROC": round(auc_loco, 4), "LOCO_AUROC_CI_low": round(ci_loco[0], 4), "LOCO_AUROC_CI_high": round(ci_loco[1], 4),
    "delta_AUROC": round(auc_loco - auc_orig, 4),
    "original_Brier": round(brier_orig, 4), "LOCO_Brier": round(brier_loco, 4), "delta_Brier": round(brier_loco - brier_orig, 4),
    "original_calib_intercept": round(cal_orig[0], 4), "original_calib_slope": round(cal_orig[1], 4),
    "LOCO_calib_intercept": round(cal_loco[0], 4), "LOCO_calib_slope": round(cal_loco[1], 4),
    "feature_count_original": lit_core_full.shape[1], "feature_count_LOCO": lit_core_loco.shape[1],
    "features_removed": DROP_FEATURE,
    "prediction_Pearson_r": round(pear_r, 4), "prediction_Spearman_r": round(spear_r, 4),
    "mean_abs_pred_diff": round(float(abs_diff.mean()), 4), "median_abs_pred_diff": round(float(np.median(abs_diff)), 4),
    "max_abs_pred_diff": round(float(abs_diff.max()), 4),
    "abs_delta_AUROC_lt_0.02": bool(abs(auc_loco - auc_orig) < 0.02),
    "spearman_rho_ge_0.90": bool(spear_r >= 0.90),
    "feature_panel_overlap_pct": round(100 * (lit_core_loco.shape[1] - 1) / lit_core_full.shape[1], 1),
}])
perf.to_csv(OUT_DIR / "LOCO_AGI_M1_refit_performance.csv", index=False)

pd.DataFrame({"sample_id": ids, "y_true": y_ext, "p_original_M1": p_orig, "p_LOCO_AGI_M1": p_loco,
              "abs_diff": abs_diff}).to_csv(OUT_DIR / "LOCO_AGI_M1_paired_predictions.csv", index=False)

with open(OUT_DIR / "LOCO_AGI_M1_refit_performance.md", "w") as fh:
    fh.write("\n".join([
        "# LOCO-AGI M1 refit: external performance on AGI_BGISEQ (own-cohort-publication masked)",
        "",
        f"Original M1 (16 features, includes {DROP_FEATURE}) AUROC on AGI_BGISEQ (N={len(ids)}, "
        f"deaths={int(y_ext.sum())}) = **{auc_orig:.4f}** [{ci_orig[0]:.4f}, {ci_orig[1]:.4f}].",
        "",
        f"LOCO-AGI M1 (15 features, {DROP_FEATURE} removed -- its only direct-human-ICU-mortality-evidence "
        f"support was the AGI cohort's own publication, PMID 40013797) AUROC on the SAME AGI_BGISEQ patients "
        f"= **{auc_loco:.4f}** [{ci_loco[0]:.4f}, {ci_loco[1]:.4f}].",
        "",
        f"**delta AUROC = {auc_loco - auc_orig:+.4f}** (descriptive robustness indicator, not a formal "
        f"equivalence test: |delta| < 0.02 -> {'YES' if abs(auc_loco - auc_orig) < 0.02 else 'NO'}).",
        f"Paired-prediction Spearman rho = {spear_r:.4f} (>=0.90 -> {'YES' if spear_r >= 0.90 else 'NO'}), "
        f"Pearson r = {pear_r:.4f}. Mean|diff|={abs_diff.mean():.4f}, max|diff|={abs_diff.max():.4f}.",
        "",
        "Caveat: `Literature_obligate_anaerobe_preservation` (sum[Faecalibacterium, Prevotella]) still uses "
        "Prevotella internally -- its frozen formula was left unmodified per the task's explicit "
        "'do not alter biological score formulas' instruction, even though Prevotella itself lost standalone "
        "Tier A eligibility under AGI-cohort masking. This composite is flagged, not silently corrected -- see "
        "knowledge_leakage_addendum.md.",
    ]))

print(f"[LOCO-AGI-refit] original AUROC={auc_orig:.4f}, LOCO AUROC={auc_loco:.4f}, delta={auc_loco-auc_orig:+.4f}, "
      f"Spearman rho={spear_r:.4f}")
print(f"[LOCO-AGI-refit] wrote outputs to {OUT_DIR}")
