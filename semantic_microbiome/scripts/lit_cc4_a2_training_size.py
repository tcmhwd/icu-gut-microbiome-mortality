"""
lit_cc4_a2_training_size.py -- Addendum 2, analysis A2 (POST_HOC_EXPLORATORY).
Plan: semantic_microbiome/reports/paper1/CRITICAL_CARE_ADDENDUM2_ANALYSIS_PLAN.md

A2a  learning curve of the M1 arm: the 16-feature literature-core panel with the project's own base-model spec
     (elastic-net LogisticRegressionCV, median imputation, standardisation; identical to utils.nested_meta.make_lr_pipe)
     trained on 20/40/60/80/100 % of each outer training partition, evaluated on the SAME held-out outer folds.
A2b  size-matched clinical reference: the same 29 reduced clinical inputs that the frozen clinical AI (C0) uses
     (built by script 10's build_mapped_X), same elastic-net LR spec, same folds and training fractions.
     C0 itself (CatBoost, trained on >100k patients elsewhere) is a frozen reference, not refit here.
Repeated stratified 5-fold CV (R repeats, seed 2026 + repeat). AUROC is pooled over the 5 outer folds per repeat;
the reported band is the 2.5-97.5 percentile across repeats (resampling variability of the CV procedure,
NOT patient sampling uncertainty). No hyperparameter search beyond the frozen specification.
If AUROC is still rising at 100 % that is reported as such; nothing is extrapolated here.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, SEMANTIC, ICU_PUBLIC

warnings.filterwarnings("ignore")
R = int(os.environ.get("A2_REPEATS", 20))
FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)
N_JOBS = max(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), 1)
OUT = SEMANTIC / "reports" / "paper1" / "critical_care_external" / "addendum2"
OUT.mkdir(parents=True, exist_ok=True)

# clinical-AI input builder from script 10 (imported, not modified)
S10 = ICU_PUBLIC / "clinical metadata" / "scripts" / "10_apply_reduced_model_to_microbiome_cohort.py"
sys.path.insert(0, str(S10.parent))
spec = importlib.util.spec_from_file_location("s10", S10)
s10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s10)

disc = pd.read_csv(SEMANTIC / "harmonized" / "discovery_cohort_primary.csv")
clin = pd.read_csv(ICU_PUBLIC / "metadata" / "microbiome.clinical.csv")
ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
clin = clin.merge(ids, left_on="KCDC_no", right_on="kcdc_no", how="inner")
assert len(clin) == 302
Xc = s10.build_mapped_X(clin)
Xc.index = clin["patient_id_clinical"].values
Xc = Xc.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
outc = disc.drop_duplicates("patient_id_clinical").set_index("patient_id_clinical")["hospital_mortality"]
lit = pd.read_csv(SEMANTIC / "structured" / "literature_core_features.csv", index_col="sample_id")

pid = list(Xc.index)
y = outc.reindex(pid).values.astype(int)
Xm = lit.reindex(pid)
Xc = Xc.reindex(pid)
assert len(y) == 302 and int(y.sum()) == 69, (len(y), int(y.sum()))
assert Xm.shape[1] == 16, Xm.shape


def pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                  l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                  random_state=SEED, n_jobs=1))])


def one_repeat(r):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
    oof = {(arm, f): np.full(len(y), np.nan) for arm in ("M1", "clinical") for f in FRACS}
    ntr = {f: [] for f in FRACS}
    for k, (tr, te) in enumerate(skf.split(Xm, y)):
        for f in FRACS:
            if f < 1.0:
                sub, _ = train_test_split(tr, train_size=f, stratify=y[tr], random_state=SEED + 1000 * r + k)
            else:
                sub = tr
            ntr[f].append(len(sub))
            for arm, X in (("M1", Xm), ("clinical", Xc)):
                p = pipe().fit(X.iloc[sub], y[sub])
                oof[(arm, f)][te] = p.predict_proba(X.iloc[te])[:, 1]
    rows = []
    for (arm, f), v in oof.items():
        rows.append(dict(repeat=r, arm=arm, fraction=f, mean_train_N=float(np.mean(ntr[f])),
                         AUROC=roc_auc_score(y, v)))
    return rows


res = Parallel(n_jobs=N_JOBS)(delayed(one_repeat)(r) for r in range(R))
df = pd.DataFrame([x for rr in res for x in rr])
df.to_csv(OUT / "A2_learning_curve_raw.csv", index=False)

summ = (df.groupby(["arm", "fraction"])
          .agg(mean_train_N=("mean_train_N", "mean"), AUROC_mean=("AUROC", "mean"),
               AUROC_p2_5=("AUROC", lambda s: np.percentile(s, 2.5)),
               AUROC_p97_5=("AUROC", lambda s: np.percentile(s, 97.5)), repeats=("AUROC", "size"))
          .reset_index().round(4))
summ.to_csv(OUT / "A2_learning_curve_summary.csv", index=False)

# per-repeat gain from 40 % -> 100 % of the training data (is the curve still rising?)
gain = []
for arm in ("M1", "clinical"):
    w = df[df.arm == arm].pivot(index="repeat", columns="fraction", values="AUROC")
    g = w[1.0] - w[0.4]
    gain.append(dict(arm=arm, gain_0p4_to_1p0_mean=round(g.mean(), 4),
                     p2_5=round(np.percentile(g, 2.5), 4), p97_5=round(np.percentile(g, 97.5), 4),
                     gain_0p8_to_1p0_mean=round((w[1.0] - w[0.8]).mean(), 4)))
gain = pd.DataFrame(gain)
gain.to_csv(OUT / "A2_learning_curve_gain.csv", index=False)

md = ["# A2 — training-size asymmetry (POST_HOC_EXPLORATORY)", "",
      f"Repeated 5-fold CV, {R} repeats, seed 2026+repeat, N=302 (69 deaths). Band = 2.5–97.5 percentile ACROSS REPEATS",
      "(CV resampling variability, not patient sampling uncertainty). Nothing is extrapolated.",
      "Reference (already reported): M1 OOF AUROC 0.592; frozen C0 (externally trained CatBoost) 0.7528.", "",
      summ.to_markdown(index=False), "", "## Gain from 40% to 100% of the training partition", "",
      gain.to_markdown(index=False)]
(OUT / "A2_learning_curve.md").write_text("\n".join(md))
print(summ.to_string(index=False)); print(gain.to_string(index=False))
