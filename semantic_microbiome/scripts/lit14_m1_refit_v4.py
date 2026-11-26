"""
lit14_m1_refit_v4.py -- M1 convergence fix, final frozen refit (owner-authorised
2026-10-02 "final technical closure pass", Task 1).

Scope: ONLY the M1 Literature Core LR refit from lit14_freeze_pipeline.py
(section 1, lines 70-82 of that file). lit14's M3 refit (section 2, LR +
CatBoost with Optuna hyperparameter search) and clustering freeze (section 3)
are completely out of scope for this fix and are NOT reproduced here -- they
are untouched, frozen artifacts exactly as lit14 left them. This script does
not import or touch lit14_freeze_pipeline.py, and does not write to
supervised_semantic/frozen_models/ (the original, still-live frozen dir) --
it writes to a separate frozen_models_v4/ directory so the original
M1_literature_core_LR.joblib (used by any script not yet repointed to v4)
is never disturbed.

WHAT CHANGED vs lit14's M1 refit: ONLY max_iter, via the same escalating
ladder (30000 -> 100000) already used by
clinical metadata/scripts/45_phase_x_c1_c2_nested_cv_corrected_v4.py for M1's
internal nested-CV fits. 16-marker feature set, feature construction
(LitCore_ prefix), penalty (elasticnet/saga), C grid (Cs=8), l1_ratio grid
([0.1, 0.5, 0.9]), inner cv=3, scoring (roc_auc), seed, and cohort (all 302
Gangnam discovery patients) are byte-identical to lit14. Convergence is
checked exactly like 45_v4: ConvergenceWarning captured (never suppressed)
with warnings.catch_warnings(record=True) + simplefilter("always"), plus a
check that no LogisticRegressionCV path has n_iter_ >= max_iter.

Output:
  semantic_microbiome/supervised_semantic/frozen_models_v4/M1_literature_core_LR.joblib
  clinical metadata/clinical_ai/reports/lit14_m1_refit_convergence_v4.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEMANTIC, HARMONIZED_DIR, STRUCTURED_DIR, SEED

SUPERVISED_DIR = SEMANTIC / "supervised_semantic"
FROZEN_DIR_ORIG = SUPERVISED_DIR / "frozen_models"  # read-only reference, never written here
FROZEN_DIR_V4 = SUPERVISED_DIR / "frozen_models_v4"
FROZEN_DIR_V4.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("/home/tcmhwd/microbiome/ICU_public/clinical metadata/clinical_ai/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LADDER = [30000, 100000]

cohort = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(
    columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
y = cohort["hospital_mortality"].astype(int)

lit_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
lit_core.columns = [f"LitCore_{c}" for c in lit_core.columns]
X_m1 = lit_core.reindex(y.index)

rows = []
chosen_pipe, chosen_max_iter = None, None
for max_iter in LADDER:
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                    l1_ratios=[0.1, 0.5, 0.9], max_iter=max_iter, scoring="roc_auc",
                                                    random_state=SEED, n_jobs=4))])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X_m1, y)
    conv_w = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    clf = pipe.named_steps["clf"]
    n_iter = np.asarray(clf.n_iter_)
    n_at_max = int((n_iter >= max_iter).sum())
    converged = bool(n_at_max == 0 and len(conv_w) == 0)
    rows.append({"attempt_max_iter": max_iter, "n_cv_paths": int(n_iter.size),
                 "n_iter_max_over_cv_paths": int(n_iter.max()), "n_iter_median_over_cv_paths": float(np.median(n_iter)),
                 "n_cv_paths_reaching_max_iter": n_at_max, "n_convergence_warnings_incl_refit": len(conv_w),
                 "selected_C": float(clf.C_[0]), "selected_l1_ratio": float(clf.l1_ratio_[0]), "converged": converged})
    print(f"[lit14_m1_refit_v4] max_iter={max_iter}: n_iter_max={int(n_iter.max())}, "
          f"n_at_max={n_at_max}, ConvergenceWarnings={len(conv_w)}, converged={converged}")
    chosen_pipe, chosen_max_iter = pipe, max_iter
    if converged:
        break

conv_df = pd.DataFrame(rows)
conv_df.to_csv(REPORTS_DIR / "lit14_m1_refit_convergence_v4.csv", index=False)
final_converged = bool(conv_df.iloc[-1]["converged"])
print(f"[lit14_m1_refit_v4] FINAL: max_iter={chosen_max_iter}, converged={final_converged}")
if not final_converged:
    raise SystemExit("BLOCKER: M1 frozen refit did not converge even at max_iter=100000 without changing model design.")

joblib.dump(chosen_pipe, FROZEN_DIR_V4 / "M1_literature_core_LR.joblib")
print(f"[lit14_m1_refit_v4] M1 Literature Core LR refit on all {len(y)} Gangnam, "
      f"saved to {FROZEN_DIR_V4}/M1_literature_core_LR.joblib")

# Sanity: coefficients should be close to (not necessarily identical to) the
# old 3000-cap fit, since the problem is convex -- a large change would be a
# red flag, not expected here given 45_v4's internal-CV M1 result also barely
# moved (standalone AUROC 0.5920, unchanged to 4dp).
old_pipe = joblib.load(FROZEN_DIR_ORIG / "M1_literature_core_LR.joblib")
old_coef = old_pipe.named_steps["clf"].coef_[0]
new_coef = chosen_pipe.named_steps["clf"].coef_[0]
max_abs_coef_diff = float(np.max(np.abs(old_coef - new_coef)))
print(f"[lit14_m1_refit_v4] max |old_coef - new_coef| = {max_abs_coef_diff:.6f} "
      f"(old selected_C={old_pipe.named_steps['clf'].C_[0]:.4g}, new selected_C={chosen_pipe.named_steps['clf'].C_[0]:.4g})")
