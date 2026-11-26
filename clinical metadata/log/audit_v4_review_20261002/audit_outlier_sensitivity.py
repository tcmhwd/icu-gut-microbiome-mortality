"""Audit-only sensitivity check (review pass, 2026-10-02): how much of the v4
V4_STRICT M3-LR 'nominally significant' pattern depends on the single patient
whose OOF probability is exactly 1.0 (logit clipped to +13.82)?

Recomputes, with the SAME estimators as scripts 45_v4 / 61_v4 / 42_v4 (statsmodels
Logit, PHReg efron, Fine-Gray reduction with G(t)=1), the point estimates and
asymptotic / sandwich p-values under: (A) as published; (B) the influential patient dropped;
(C) logit winsorised to [-5, 5]; (D) clip EPS = 1e-3 instead of 1e-6.
No bootstrap. Nothing in the project is modified; outputs go to the scratchpad.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = "/home/tcmhwd/microbiome/ICU_public"
PRED = os.path.join(BASE, "clinical metadata", "clinical_ai", "predictions")
DISC = os.path.join(BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv")
CLIN = os.path.join(BASE, "metadata", "microbiome.clinical.csv")
STRUCT = os.path.join(BASE, "semantic_microbiome", "structured")
OUT = os.path.dirname(os.path.abspath(__file__))
OUTLIER = os.environ.get("INFLUENTIAL_PATIENT_ID", "").strip()  # [REDACTED for public release] patient whose data-driven LR OOF probability is 1.0
if not OUTLIER:  # [PUBLIC RELEASE] required; validated before any data is read
    raise SystemExit("INFLUENTIAL_PATIENT_ID must be set to the pseudonymised ID of the influential patient (see README).")

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

oof4 = pd.read_csv(os.path.join(PRED, "phase_x_m3_single_level_oof_probabilities_v4_V4_STRICT.csv"), index_col=0)
if OUTLIER not in oof4.index:  # [PUBLIC RELEASE] a wrong ID would otherwise make analysis B identical to A
    raise SystemExit(f"INFLUENTIAL_PATIENT_ID={OUTLIER!r} is not in the 302-patient OOF index.")
oof3 = pd.read_csv(os.path.join(PRED, "phase_x_m3_single_level_oof_probabilities_v3_V3.csv"), index_col=0).reindex(oof4.index)
nest4 = pd.read_csv(os.path.join(PRED, "phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv"), index_col=0).reindex(oof4.index)
disc = pd.read_csv(DISC); raw = pd.read_csv(CLIN)
ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw = raw.merge(ids, left_on="KCDC_no", right_on="kcdc_no", how="inner").set_index("patient_id_clinical").reindex(oof4.index)
assert len(raw) == 302
y = oof4["hospital_mortality"].astype(int).values
SOFA = pd.to_numeric(raw["SOFA"], errors="coerce").values
assert not np.isnan(SOFA).any()
lclin = logit(oof4["clinical_AI_probability_PRIMARY"].values)
d = pd.DataFrame(index=oof4.index)
for s, c in [("collection", "collection_date"), ("discharge", "discharge_date"), ("death", "death.date")]:
    d[s] = pd.to_datetime(raw[c], errors="coerce")
d["t_death"] = (d.death - d.collection).dt.days
d["discharge_eff"] = d.death.where(d.death.notna(), d.discharge)
d["t_dis"] = (d.discharge_eff - d.collection).dt.days
B_time = np.where(y == 1, d.t_death, d.t_dis).astype(float)
CS_status = (y == 1).astype(int)
TMAX = float(B_time.max())
FG_time = np.where(y == 1, B_time, TMAX); FG_status = CS_status.copy()

def cox(time, status, X, groups=None):
    m = sm.PHReg(np.asarray(time, float), np.asarray(X, float), status=np.asarray(status, int), ties="efron")
    return m.fit(groups=groups) if groups is not None else m.fit()

def hr_row(r, j, sd, label):
    b, se = float(r.params[j]), float(r.bse[j])
    return {"HR_per_SD": round(np.exp(b * sd), 3), "lo": round(np.exp((b - 1.96 * se) * sd), 3),
            "hi": round(np.exp((b + 1.96 * se) * sd), 3), "p": round(2 * stats.norm.sf(abs(b / se)), 4)}

def run(label, lm3, keep):
    """lm3: logit vector (full 302); keep: boolean mask of rows retained."""
    yk, lc, so, lm = y[keep], lclin[keep], SOFA[keep], lm3[keep]
    sd = float(np.std(lm, ddof=1))
    out = {"scenario": label, "N": int(keep.sum()), "events": int(yk.sum()), "SD_logit": round(sd, 3),
           "max_logit": round(float(lm.max()), 2)}
    # C2a adjusted OR (45_v4 adjusted_or): Logit(y ~ logit_clin + logit_M3)
    f = sm.Logit(yk, sm.add_constant(np.column_stack([lc, lm]))).fit(disp=0)
    out.update({"C2a_OR_unit": round(np.exp(f.params[2]), 3), "C2a_p": round(f.pvalues[2], 4),
                "C2a_OR_perSD": round(np.exp(f.params[2] * sd), 3)})
    # S2a adjusted OR + LRT vs SOFA alone (61_v4)
    f1 = sm.Logit(yk, sm.add_constant(np.column_stack([so, lm]))).fit(disp=0)
    f0 = sm.Logit(yk, sm.add_constant(so)).fit(disp=0)
    chi2 = 2 * (f1.llf - f0.llf)
    out.update({"S2a_OR_unit": round(np.exp(f1.params[2]), 3), "S2a_OR_perSD": round(np.exp(f1.params[2] * sd), 3),
                "S2a_perSD_lo": round(np.exp((f1.params[2] - 1.96 * f1.bse[2]) * sd), 3),
                "S2a_perSD_hi": round(np.exp((f1.params[2] + 1.96 * f1.bse[2]) * sd), 3),
                "S2a_Wald_p": round(f1.pvalues[2], 4), "S2a_LRT_chi2": round(chi2, 3),
                "S2a_LRT_p": round(stats.chi2.sf(chi2, 1), 4)})
    # cause-specific Cox alone / + SOFA
    r = cox(B_time[keep], CS_status[keep], lm.reshape(-1, 1)); h = hr_row(r, 0, sd, "cox_alone")
    out.update({f"Cox_alone_{k}": v for k, v in h.items()})
    r = cox(B_time[keep], CS_status[keep], np.column_stack([lm, so])); h = hr_row(r, 0, sd, "cox_sofa")
    out.update({f"Cox_SOFA_{k}": v for k, v in h.items()})
    # Fine-Gray reduction, sandwich SE (42_v4 fallback)
    n = int(keep.sum())
    for nm, X in [("FG_alone", lm.reshape(-1, 1)), ("FG_SOFA", np.column_stack([lm, so]))]:
        r = cox(FG_time[keep], FG_status[keep], X); rr = cox(FG_time[keep], FG_status[keep], X, groups=np.arange(n))
        b, se = float(r.params[0]), float(rr.bse[0])
        out.update({f"{nm}_HR_per_SD": round(np.exp(b * sd), 3), f"{nm}_lo": round(np.exp((b - 1.96 * se) * sd), 3),
                    f"{nm}_hi": round(np.exp((b + 1.96 * se) * sd), 3), f"{nm}_p": round(2 * stats.norm.sf(abs(b / se)), 4)})
    # standalone AUROC of M3-LR and nested C2a vs C0 (point only)
    out["M3LR_AUROC"] = round(roc_auc_score(yk, lm), 4)
    c0, c2a = nest4["oof_prob_C0_clinical_alone"].values[keep], nest4["oof_prob_C2a_clinical_plus_M3_LR"].values[keep]
    out["C2a_minus_C0_AUROC"] = round(roc_auc_score(yk, c2a) - roc_auc_score(yk, c0), 4)
    return out

rows = []
keep_all = np.ones(302, bool); keep_drop = (oof4.index != OUTLIER)
for ver, oof in [("v4_STRICT", oof4), ("v3_V3", oof3)]:
    p = oof["microbiome_prob_M3_LR"].values
    lm = logit(p)
    rows.append(run(f"{ver} A: as published (clip 1e-6)", lm, keep_all))
    rows.append(run(f"{ver} B: drop {OUTLIER}", lm, keep_drop))
    rows.append(run(f"{ver} C: winsorise logit to [-5,5]", np.clip(lm, -5, 5), keep_all))
    rows.append(run(f"{ver} D: clip EPS=1e-3 (logit +-6.9)", logit(p, 1e-3), keep_all))
# CatBoost for reference (v4 file == v3 file)
lmcb = logit(oof4["microbiome_prob_M3_CatBoost"].values)
rows.append(run("CatBoost (unchanged) A: as published", lmcb, keep_all))
rows.append(run(f"CatBoost (unchanged) B: drop {OUTLIER}", lmcb, keep_drop))
res = pd.DataFrame(rows)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60)
print(res.to_string(index=False))
res.to_csv(os.path.join(OUT, "audit_outlier_sensitivity.csv"), index=False)

# ---- where is the influential patient extreme in feature space? (filtered views, z vs others)
print("\n=== influential patient feature-space extremity (filtered structured views; z-score vs the other 301) ===")
zrows = []
for f in sorted(os.listdir(STRUCT)):
    if not f.endswith("_filtered.tsv"): continue
    m = pd.read_csv(os.path.join(STRUCT, f), sep="\t", index_col="patient_id")
    if OUTLIER not in m.index: print(f, ": patient absent"); continue
    others = m.drop(index=OUTLIER)
    mu, sdv = others.mean(), others.std(ddof=1).replace(0, np.nan)
    z = ((m.loc[OUTLIER] - mu) / sdv).dropna()
    top = z.abs().sort_values(ascending=False).head(3)
    for c in top.index:
        zrows.append({"view": f, "feature": c, "z": round(float(z[c]), 1), "value": float(m.loc[OUTLIER, c]),
                      "others_max": float(others[c].max()), "others_median": float(others[c].median())})
    print(f"{f}: n_feat={m.shape[1]}, n|z|>10: {(z.abs()>10).sum()}, n|z|>5: {(z.abs()>5).sum()}, max|z|={z.abs().max():.1f}")
pd.DataFrame(zrows).to_csv(os.path.join(OUT, "audit_outlier_features.csv"), index=False)
print(pd.DataFrame(zrows).to_string(index=False))
# a few clinical facts about the patient
print("\ninfluential patient:", {k: raw.loc[OUTLIER, k] for k in ["SOFA", "collection_date", "death.date", "discharge_date"] if k in raw.columns},
      "y=", int(oof4.loc[OUTLIER, "hospital_mortality"]), "clinAI=", round(float(oof4.loc[OUTLIER, "clinical_AI_probability_PRIMARY"]), 3),
      "M1=", round(float(oof4.loc[OUTLIER, "microbiome_prob_M1"]), 3), "CB=", round(float(oof4.loc[OUTLIER, "microbiome_prob_M3_CatBoost"]), 3))
print("B_time influential patient =", B_time[list(oof4.index).index(OUTLIER)], "TMAX =", TMAX)
