"""
lit_cc2_cross_cohort_biology.py -- Critical Care strengthening task,
Priorities 4/5 (user request, 2026-09-14), prespecified in
CRITICAL_CARE_EXTERNAL_ANALYSIS_PLAN.md Section 7.

Two analyses, both restricted to the 16 FROZEN M1 markers (never redesigned,
never re-selected) and reusing the same pooled, already-existing
literature_core_features.csv (verified to cover Gangnam + AGI_BGISEQ +
PRJNA842425 on one consistent 0-100 relative-abundance scale -- confirmed by
direct comparison against structured/taxonomy_genus_filtered.tsv, the file
the internal cohort's own frozen Enterococcus >=30% threshold was defined
against; Gangnam-subset means match to within rounding: 8.451 vs 8.451).

A. Marker x cohort effect-size table: Spearman rho (as lit24 already
   computed, recomputed here restricted to the 16-marker subset for a
   self-contained table) AND OR per SD from univariate logistic regression
   (NEW -- lit24 only has rho). Then the pairwise Spearman correlation of
   the 16-marker rho vector between cohort pairs (NEW).

B. Enterococcus >=30% domination replication: the internal cohort's frozen,
   never-re-optimized threshold, applied unchanged to AGI_BGISEQ and
   PRJNA842425. Unadjusted OR (Fisher exact) in both; severity-adjusted OR +
   LRT + delta-AUROC only in PRJNA842425 (AGI has no severity score, see
   AGI_METADATA_AUDIT.md).

Reads:
  structured/literature_core_features.csv
  harmonized/discovery_cohort_primary.csv (Gangnam mortality)
  merge/output/sample_metadata.tsv (external-cohort mortality + subcohort)
  public_validation/covid_prjna842425_concordance_groups.csv (SOFA, for the
    severity-adjusted Enterococcus arm)
Writes:
  reports/paper1/critical_care_external/M1_CROSS_COHORT_MARKER_RESULTS.csv
  reports/paper1/critical_care_external/cross_cohort_effect_size_correlation.csv
  reports/paper1/critical_care_external/ENTEROCOCCUS_REPLICATION_RESULTS.csv
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import SEMANTIC, ICU_PUBLIC, HARMONIZED_DIR, STRUCTURED_DIR, today_str

OUT_DIR = SEMANTIC / "reports" / "paper1" / "critical_care_external"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
PUBVAL_DIR = SEMANTIC / "public_validation"
FROZEN_DIR = SEMANTIC / "supervised_semantic" / "frozen_models"
EPS = 1e-6

# ---------------------------------------------------------------------------
# 0. the 16 frozen M1 markers, exact names, read from the model artifact
#    itself (not retyped) so this can never silently drift from M1's own
#    feature list.
# ---------------------------------------------------------------------------
m1_pipe = joblib.load(FROZEN_DIR / "M1_literature_core_LR.joblib")
M1_FEATURES = [f.replace("LitCore_", "") for f in m1_pipe.feature_names_in_]
assert len(M1_FEATURES) == 16, f"expected 16 M1 features, found {len(M1_FEATURES)}"

lit_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
missing = [m for m in M1_FEATURES if m not in lit_core.columns]
assert not missing, f"M1 features missing from literature_core_features.csv: {missing}"

gangnam_ids = pd.read_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv").rename(
    columns={"patient_id_clinical": "patient_id"}).set_index("patient_id")
y_gangnam = gangnam_ids["hospital_mortality"].astype(int)

sm = pd.read_csv(MERGE_OUTPUT / "sample_metadata.tsv", sep="\t", dtype=str)
sm["mortality"] = pd.to_numeric(sm["mortality"], errors="coerce")
public = sm[sm.dataset != "ICU"].set_index("sample_id")

COHORTS = {
    "Gangnam": (y_gangnam.index.tolist(), y_gangnam),
    "AGI_BGISEQ": None,
    "PRJNA842425": None,
}
for cname in ["AGI_BGISEQ", "PRJNA842425"]:
    ids = public[public.subcohort == cname].index.tolist()
    COHORTS[cname] = (ids, public.loc[ids, "mortality"].astype(int))

# ENDPOINT FIX (fable audit finding B1, applied here too for consistency with
# lit_cc1_prjna842425_incremental.py): PRJNA842425's pooled `mortality` column
# is vital status at end of follow-up, not in-hospital death (5/32 events were
# discharged alive and died later). Overridden with in-hospital death
# (DischargeLocation=="Death") to match AGI_BGISEQ's genuinely-in-hospital
# endpoint and this manuscript's "in-hospital mortality" label everywhere else.
xl_covid = pd.read_excel(ICU_PUBLIC / "metadata.xlsx", sheet_name="PRJNA842425")
xl_covid = xl_covid.rename(columns={"patient_ID": "covid_id"})
id_map_covid = sm[sm.subcohort == "PRJNA842425"][["sample_id", "patient_id"]].copy()
id_map_covid["covid_id"] = id_map_covid["patient_id"].str.replace("PRJNA842425_", "", regex=False)
# validate="one_to_one" (Codex review finding #14): guard against a silent
# drop/duplicate if the upstream id map or xlsx sheet ever changes shape.
inhosp = id_map_covid.merge(xl_covid[["covid_id", "DischargeLocation"]], on="covid_id", how="left", validate="one_to_one")
assert inhosp["DischargeLocation"].notna().all(), (
    "DischargeLocation missing for some PRJNA842425 patients -- a missing value must not "
    "silently become a non-death (Codex review finding #14).")
inhosp["inhosp_death"] = (inhosp["DischargeLocation"] == "Death").astype(int)
inhosp = inhosp.set_index("sample_id")["inhosp_death"]
ids842425, _ = COHORTS["PRJNA842425"]
COHORTS["PRJNA842425"] = (ids842425, inhosp.reindex(ids842425))
n_flip = int((inhosp.reindex(ids842425) != public.loc[ids842425, "mortality"].astype(int)).sum())
print(f"[cc2] PRJNA842425 endpoint switched to in-hospital death: "
      f"{int(inhosp.reindex(ids842425).sum())} events (was {int(public.loc[ids842425,'mortality'].sum())}); "
      f"{n_flip} patients' label changed")


def or_per_sd(x, y):
    """Univariate OR per 1-SD increase, Wald CI. Firth fallback on complete
    separation (statsmodels raises/warns on perfect separation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    xz = (x - x.mean()) / x.std(ddof=0)
    import statsmodels.api as sm_
    X = sm_.add_constant(xz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm_.Logit(y, X).fit(disp=0, maxiter=200)
            beta, se = fit.params[1], fit.bse[1]
            or_, lo, hi, p = np.exp(beta), np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se), fit.pvalues[1]
            # separation/instability check: not just a raw SE cutoff (a
            # moderate SE on a large beta can still explode once
            # exponentiated) -- check the OR and its CI ratio directly.
            if not np.isfinite(se) or or_ > 50 or or_ < 0.02 or (hi / max(lo, 1e-12)) > 1000:
                raise np.linalg.LinAlgError("unstable OR/CI (likely separation)")
            method = "Wald"
        except Exception:
            # Firth via R's logistf, matching this project's small-sample convention elsewhere
            # (utils/run_firth_logistf.R). If the R call itself fails, this returns NaN + a
            # "FAILED:<reason>" method string rather than any kind of approximate fallback
            # (Codex review finding #15 caught a stale comment here promising a "coarse
            # penalized" fallback that was never implemented -- corrected to describe what
            # the code actually does).
            import subprocess, tempfile, json as json_
            with tempfile.TemporaryDirectory() as td:
                pd.DataFrame({"y": y, "x": xz}).to_csv(f"{td}/d.csv", index=False)
                rscript = ("d<-read.csv('%s/d.csv'); library(logistf); "
                           "f<-logistf(y~x, data=d); "
                           "cat(jsonlite::toJSON(list(or=exp(coef(f))[['x']], "
                           "lo=exp(confint(f))[2,1], hi=exp(confint(f))[2,2], p=f$prob[['x']])))" % td)
                try:
                    out = subprocess.run(["Rscript", "-e", rscript], capture_output=True, text=True, timeout=60)
                    j = json_.loads(out.stdout.strip())
                    unwrap = lambda v: v[0] if isinstance(v, list) else v
                    or_, lo, hi, p = unwrap(j["or"]), unwrap(j["lo"]), unwrap(j["hi"]), unwrap(j["p"])
                    method = "Firth"
                except Exception as e2:
                    or_, lo, hi, p, method = np.nan, np.nan, np.nan, np.nan, f"FAILED:{e2}"
    return or_, lo, hi, p, method


# ---------------------------------------------------------------------------
# A. marker x cohort table
# ---------------------------------------------------------------------------
rows = []
for cname, (ids, y) in COHORTS.items():
    X = lit_core.reindex(ids)
    for marker in M1_FEATURES:
        vals = X[marker]
        valid = vals.notna()
        yy = y.reindex(vals.index)[valid].values
        xx = vals[valid].values
        if len(xx) < 10 or len(np.unique(yy)) < 2:
            rows.append(dict(cohort=cname, marker=marker, N=int(valid.sum()), spearman_rho=np.nan,
                              spearman_p=np.nan, OR_per_SD=np.nan, OR_lo=np.nan, OR_hi=np.nan,
                              OR_p=np.nan, OR_method="insufficient N/events"))
            continue
        rho, p = stats.spearmanr(xx, yy)
        or_, lo, hi, orp, method = or_per_sd(xx, yy)
        rows.append(dict(cohort=cname, marker=marker, N=len(xx), spearman_rho=round(rho, 3),
                          spearman_p=round(p, 4), OR_per_SD=round(or_, 3) if np.isfinite(or_) else np.nan,
                          OR_lo=round(lo, 3) if np.isfinite(lo) else np.nan,
                          OR_hi=round(hi, 3) if np.isfinite(hi) else np.nan,
                          OR_p=round(orp, 4) if np.isfinite(orp) else np.nan, OR_method=method))

marker_tbl = pd.DataFrame(rows)
marker_tbl.to_csv(OUT_DIR / "M1_CROSS_COHORT_MARKER_RESULTS.csv", index=False)
print(marker_tbl.to_string(index=False))

# directional concordance vs Gangnam, restricted to the 16 M1 markers
piv_rho = marker_tbl.pivot(index="marker", columns="cohort", values="spearman_rho")
for ext in ["AGI_BGISEQ", "PRJNA842425"]:
    same_sign = np.sign(piv_rho["Gangnam"]) == np.sign(piv_rho[ext])
    n_ok = same_sign.sum()
    print(f"[cc2] 16-marker directional concordance, {ext} vs Gangnam: {n_ok}/16 ({100*n_ok/16:.0f}%)")

# ---------------------------------------------------------------------------
# cross-cohort effect-size (rho) correlation
# ---------------------------------------------------------------------------
corr_rows = []
pairs = [("Gangnam", "AGI_BGISEQ"), ("Gangnam", "PRJNA842425"), ("AGI_BGISEQ", "PRJNA842425")]
for a, b in pairs:
    sub = piv_rho[[a, b]].dropna()
    rho_corr, p_corr = stats.spearmanr(sub[a], sub[b])
    corr_rows.append(dict(cohort_a=a, cohort_b=b, n_markers=len(sub),
                           spearman_of_effect_sizes=round(rho_corr, 3), p=round(p_corr, 4)))
corr_tbl = pd.DataFrame(corr_rows)
corr_tbl.to_csv(OUT_DIR / "cross_cohort_effect_size_correlation.csv", index=False)
print(corr_tbl.to_string(index=False))

# ---------------------------------------------------------------------------
# B. Enterococcus >=30% domination replication
# ---------------------------------------------------------------------------
ent_rows = []
sofa_df = pd.read_csv(PUBVAL_DIR / "covid_prjna842425_concordance_groups.csv").set_index("sample_id")

for cname, (ids, y) in COHORTS.items():
    if cname == "Gangnam":
        continue  # already the established, frozen internal result; not rerun here
    ent = lit_core.reindex(ids)["Literature_Enterococcus_genus_relabund"]
    valid = ent.notna()
    yy = y.reindex(ent.index)[valid]
    flag = (ent[valid] >= 30).astype(int)
    n_exp, n_unexp = int(flag.sum()), int((1 - flag).sum())
    d_exp, d_unexp = int(yy[flag == 1].sum()), int(yy[flag == 0].sum())
    table = [[d_exp, n_exp - d_exp], [d_unexp, n_unexp - d_unexp]]
    odds_ratio, fisher_p = stats.fisher_exact(table)
    ent_rows.append(dict(cohort=cname, N_exposed=n_exp, deaths_exposed=d_exp,
                          N_unexposed=n_unexp, deaths_unexposed=d_unexp,
                          unadjusted_OR=round(odds_ratio, 3), fisher_p=round(fisher_p, 4),
                          adjusted_OR=np.nan, adjusted_OR_lo=np.nan, adjusted_OR_hi=np.nan,
                          LRT_p=np.nan, delta_AUROC=np.nan, delta_AUROC_lo=np.nan, delta_AUROC_hi=np.nan))

    if cname == "PRJNA842425":
        sofa = sofa_df.reindex(ent[valid].index)["SOFA"].astype(float).values
        yv = yy.values
        flagv = flag.values.astype(float)
        sofa_z = (sofa - sofa.mean()) / sofa.std(ddof=0)
        X_full = np.column_stack([sofa_z, flagv])
        import statsmodels.api as sm_
        fit_full = sm_.Logit(yv, sm_.add_constant(X_full)).fit(disp=0)
        fit_red = sm_.Logit(yv, sm_.add_constant(sofa_z)).fit(disp=0)
        lrt = 2 * (fit_full.llf - fit_red.llf)
        lrt_p = 1 - stats.chi2.cdf(lrt, df=1)
        adj_or = np.exp(fit_full.params[2])
        ci = fit_full.conf_int()
        adj_lo, adj_hi = np.exp(ci[2][0]), np.exp(ci[2][1])
        p_full = fit_full.predict(sm_.add_constant(X_full))
        p_red = fit_red.predict(sm_.add_constant(sofa_z))
        auc_full, auc_red = roc_auc_score(yv, p_full), roc_auc_score(yv, p_red)
        rng = np.random.RandomState(2026)
        deltas = []
        for _ in range(2000):
            idx = rng.randint(0, len(yv), len(yv))
            if len(np.unique(yv[idx])) < 2:
                continue
            deltas.append(roc_auc_score(yv[idx], p_full[idx]) - roc_auc_score(yv[idx], p_red[idx]))
        d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        ent_rows[-1].update(adjusted_OR=round(adj_or, 3), adjusted_OR_lo=round(adj_lo, 3),
                             adjusted_OR_hi=round(adj_hi, 3), LRT_p=round(lrt_p, 4),
                             delta_AUROC=round(auc_full - auc_red, 4),
                             delta_AUROC_lo=round(d_lo, 4), delta_AUROC_hi=round(d_hi, 4))

ent_tbl = pd.DataFrame(ent_rows)
ent_tbl.to_csv(OUT_DIR / "ENTEROCOCCUS_REPLICATION_RESULTS.csv", index=False)
print(ent_tbl.to_string(index=False))
print("[cc2] wrote M1_CROSS_COHORT_MARKER_RESULTS.csv, cross_cohort_effect_size_correlation.csv, "
      "ENTEROCOCCUS_REPLICATION_RESULTS.csv")
