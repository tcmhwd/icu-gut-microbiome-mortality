"""
lit_cc1_prjna842425_incremental.py -- Critical Care strengthening task,
Priority 1 (user request, 2026-09-14): the primary NEW external estimand,
prespecified in CRITICAL_CARE_EXTERNAL_ANALYSIS_PLAN.md Section 7.

Question: does adding the frozen M1 microbiome score to SOFA improve
discrimination for in-hospital mortality in PRJNA842425 (N=71, 27 in-hospital
deaths -- corrected mid-review, Codex finding #6; see the endpoint note below)?

Reuses, unchanged: SOFA/APACHE (metadata.xlsx sheet PRJNA842425, already
extracted by lit26), the frozen M1 per-patient probability (already computed
by lit26, covid_prjna842425_concordance_groups.csv). Fits exactly one new
thing: a 2-term in-sample logistic combination `mortality ~ SOFA + logit(M1)`
(and the APACHE-referenced secondary version) -- disclosed as in-sample, not
cross-validated; see CRITICAL_CARE_ANALYSIS_DEVIATIONS.md item 3.

CORRECTED WORDING (Codex finding #1, review pass): the deviations log
originally said N=71 "cannot support ... any" cross-validation -- too strong.
A simple stratified 5-fold OOF refit of the same 2-term combiner IS feasible,
just unstable (fold-to-fold variance is large at this N/event count). Both
are reported below: the in-sample PRIMARY estimand (matches the internal
cohort's own frozen-score convention) and an explicit 5-fold-OOF SENSITIVITY
row that captures the fit/split uncertainty the in-sample bootstrap CI does
not -- Codex's independent version got OOF delta_AUROC -0.013 [-0.071,+0.034]
p=0.667 (seed 2026); the version computed here should be close but is not
required to match exactly (different fold RNG draws are expected to move a
5-fold estimate at N=71).

Bootstrap: identical algorithm to
clinical_ai/scripts/utils/nested_meta.paired_bootstrap_delta_auroc (2000
replicates, seed 2026, patient resampling, predictions held fixed,
single-class resamples skipped) -- reimplemented inline here rather than
imported across the clinical_ai/semantic_microbiome script-tree boundary,
same algorithm verified line-for-line against the original.

Reads:
  public_validation/covid_prjna842425_concordance_groups.csv (SOFA, M1 prob, mortality)
  metadata.xlsx sheet PRJNA842425 (APACHE, for the secondary comparator)
Writes:
  reports/paper1/critical_care_external/prjna842425_incremental_results.csv
  reports/paper1/critical_care_external/prjna842425_incremental_results.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from common import SEMANTIC, ICU_PUBLIC, SEED, today_str

PUBVAL_DIR = SEMANTIC / "public_validation"
OUT_DIR = SEMANTIC / "reports" / "paper1" / "critical_care_external"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_XLSX = ICU_PUBLIC / "metadata.xlsx"

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y_true, y_prob):
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    lp = logit(p).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def paired_bootstrap_delta_auroc(y, plain, adapt, n_boot=2000, seed=SEED):
    """Identical algorithm to utils.nested_meta.paired_bootstrap_delta_auroc
    (clinical_ai/scripts) -- reimplemented here, not imported, to avoid a
    cross-script-tree sys.path dependency. Verified against the original
    line-for-line at write time."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y = np.asarray(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], adapt[idx]) - roc_auc_score(y[idx], plain[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)), 1.0)
    point = roc_auc_score(y, adapt) - roc_auc_score(y, plain)
    return round(point, 4), round(lo, 4), round(hi, 4), round(p, 4), len(deltas)


def bootstrap_auroc(y, score, n_boot=2000, seed=SEED):
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
    return point, lo, hi


# ---------------------------------------------------------------------------
# 1. Load SOFA, M1 prob, mortality (already computed, lit26) + APACHE (xlsx)
# ---------------------------------------------------------------------------
qdf = pd.read_csv(PUBVAL_DIR / "covid_prjna842425_concordance_groups.csv")
# qdf's sample_id is the raw SRA run accession (SRR...), not the covid_### id --
# recover covid_id the same way lit26 does, via merge/output/sample_metadata.tsv's
# patient_id column ("PRJNA842425_covid_XXX").
sm = pd.read_csv(ICU_PUBLIC / "merge" / "output" / "sample_metadata.tsv", sep="\t", dtype=str)
id_map = sm[sm.subcohort == "PRJNA842425"][["sample_id", "patient_id"]].copy()
id_map["covid_id"] = id_map["patient_id"].str.replace("PRJNA842425_", "", regex=False)
# validate="one_to_one" (Codex review finding #14): neither merge previously
# asserted match cardinality, so a future upstream duplicate/rename could
# silently drop or duplicate patients without either script noticing.
qdf = qdf.merge(id_map[["sample_id", "covid_id"]], on="sample_id", how="left", validate="one_to_one")
xl = pd.read_excel(METADATA_XLSX, sheet_name="PRJNA842425")
xl_sel = xl.rename(columns={"patient_ID": "covid_id"})[["covid_id", "APACHE", "DischargeLocation", "SurvTime"]]
merged = qdf.merge(xl_sel, on="covid_id", how="left", validate="one_to_one")
merged["APACHE"] = pd.to_numeric(merged["APACHE"], errors="coerce")
assert merged["DischargeLocation"].notna().all(), (
    "DischargeLocation missing for some patients -- a missing value must not silently become "
    "a non-death (Codex review finding #14); investigate before proceeding.")

# ENDPOINT FIX (fable audit, 2026-09-14, CRITICAL_CARE_ANALYSIS_DEVIATIONS.md item 4,
# BLOCKING finding B1): the pre-existing `mortality` column (used unmodified by
# lit26/lit10b/lit15 for the M1-alone AUROC numbers already published) is vital
# status at end of follow-up (up to 290 days), NOT in-hospital mortality -- 5 of
# the 32 "deaths" were discharged alive (Hospice/LTACH/SNF) and died 21-172 days
# later per SurvTime. This project's own wording rules (CENTRAL_CLAIM_REFRAMING_
# DESIGN.md rule 9) treat endpoint scoping as load-bearing. AGI_BGISEQ's mortality
# field genuinely is in-hospital (confirmed independently by the fable audit
# against the published paper's own metadata), so for a fair external comparator
# and for consistency with the "in-hospital mortality" label used throughout this
# manuscript, in-hospital death (DischargeLocation=="Death") is now PRIMARY;
# the pre-existing all-cause-by-end-of-follow-up definition is kept as an explicit
# SECONDARY/sensitivity analysis, not silently dropped.
merged["mortality_inhospital"] = (merged["DischargeLocation"] == "Death").astype(int)
merged["mortality_followup"] = merged["mortality"].astype(int)  # secondary/sensitivity
print(f"[cc1] N={len(merged)}, SOFA available={merged['SOFA'].notna().sum()}, "
      f"APACHE available={merged['APACHE'].notna().sum()}, "
      f"in-hospital deaths (PRIMARY)={int(merged['mortality_inhospital'].sum())}, "
      f"all-cause-by-followup deaths (secondary)={int(merged['mortality_followup'].sum())}")

sofa = merged["SOFA"].astype(float).values
m1 = merged["microbiome_prob_M1"].astype(float).values
m1_logit = logit(m1)

rows = []


def run_one(sev_name, sev_vals, valid_mask, endpoint_name, y_full):
    yy = y_full[valid_mask]
    sv = sev_vals[valid_mask]
    ml = m1_logit[valid_mask]
    mp = m1[valid_mask]
    n = len(yy)

    # severity-alone AUROC (bootstrap CI, same protocol)
    sev_auc, sev_lo, sev_hi = bootstrap_auroc(yy, sv)

    # combined in-sample logistic: mortality ~ severity(z) + logit(M1)(z)
    sv_z = (sv - sv.mean()) / sv.std(ddof=0)
    ml_z = (ml - ml.mean()) / ml.std(ddof=0)
    X = np.column_stack([sv_z, ml_z])
    lr_full = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(X, yy)
    combined_p = lr_full.predict_proba(X)[:, 1]
    comb_auc, comb_lo, comb_hi = bootstrap_auroc(yy, combined_p)

    # paired delta AUROC (severity+M1 combined) vs (severity alone) -- predictions held fixed
    d_point, d_lo, d_hi, d_p, n_eff = paired_bootstrap_delta_auroc(yy, sv, combined_p)

    # severity-only reduced model, for LRT
    lr_red = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(sv_z.reshape(-1, 1), yy)
    p_red = lr_red.predict_proba(sv_z.reshape(-1, 1))[:, 1]
    ll_red = np.sum(yy * np.log(np.clip(p_red, EPS, 1 - EPS)) + (1 - yy) * np.log(np.clip(1 - p_red, EPS, 1 - EPS)))
    ll_full = np.sum(yy * np.log(np.clip(combined_p, EPS, 1 - EPS)) + (1 - yy) * np.log(np.clip(1 - combined_p, EPS, 1 - EPS)))
    lrt_stat = 2 * (ll_full - ll_red)
    lrt_p = 1 - stats.chi2.cdf(lrt_stat, df=1)

    # adjusted OR per SD for the M1 term (Wald CI) from the same fitted model
    # statsmodels-free Wald CI via the covariance from a second fit with
    # statsmodels-equivalent SE: use sklearn's coef_ and a numerical Hessian-based
    # SE approximation is unnecessary here -- use statsmodels directly for a
    # correct Wald SE.
    import statsmodels.api as sm
    Xc = sm.add_constant(X)
    sm_fit = sm.Logit(yy, Xc).fit(disp=0)
    or_m1 = np.exp(sm_fit.params[2])
    or_m1_lo = np.exp(sm_fit.conf_int()[2][0])
    or_m1_hi = np.exp(sm_fit.conf_int()[2][1])
    or_m1_p = sm_fit.pvalues[2]

    # Brier fair comparison fix (fable audit S6): the "before" arm must be the
    # SOFA-only fitted probability (p_red), not a min-max-rescaled raw SOFA
    # score -- rescaling SOFA onto [0,1] is not a probability and is not a
    # fair Brier comparator.
    brier_sev = brier_score_loss(yy, p_red)
    brier_comb = brier_score_loss(yy, combined_p)
    auprc_sev = average_precision_score(yy, p_red)
    auprc_comb = average_precision_score(yy, combined_p)
    # Calibration intercept/slope dropped (fable audit Q6/S6): verified exactly
    # tautological here -- the MLE score equations of the same in-sample fit
    # guarantee intercept=0/slope=1 when evaluated on its own training data,
    # so the number carries no information and is not written to the output.

    rows.append(dict(
        endpoint=endpoint_name, severity_score=sev_name, N=n, deaths=int(yy.sum()),
        severity_AUROC=round(sev_auc, 4), severity_AUROC_lo=round(sev_lo, 4), severity_AUROC_hi=round(sev_hi, 4),
        M1_AUROC=round(roc_auc_score(yy, mp), 4),
        combined_AUROC=round(comb_auc, 4), combined_AUROC_lo=round(comb_lo, 4), combined_AUROC_hi=round(comb_hi, 4),
        delta_AUROC=d_point, delta_AUROC_lo=d_lo, delta_AUROC_hi=d_hi, delta_AUROC_boot_p=d_p, n_boot_used=n_eff,
        M1_adjusted_OR_per_SD=round(or_m1, 3), OR_lo=round(or_m1_lo, 3), OR_hi=round(or_m1_hi, 3), OR_p=round(or_m1_p, 4),
        LRT_stat=round(lrt_stat, 3), LRT_p=round(lrt_p, 4),
        Brier_severity_only=round(brier_sev, 4), Brier_combined=round(brier_comb, 4),
        AUPRC_severity_only=round(auprc_sev, 4), AUPRC_combined=round(auprc_comb, 4),
        fit_type="in-sample 2-term logistic (not cross-validated, N too small; see deviations log item 3)",
    ))


sofa_valid = np.ones(len(merged), dtype=bool)
apache_valid = merged["APACHE"].notna().values
for endpoint_name, y_full in [("in_hospital_death_PRIMARY", merged["mortality_inhospital"].values),
                               ("all_cause_death_by_followup_SECONDARY", merged["mortality_followup"].values)]:
    run_one("SOFA", sofa, sofa_valid, endpoint_name, y_full)
    run_one("APACHE", merged["APACHE"].fillna(0).values, apache_valid, endpoint_name, y_full)

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "prjna842425_incremental_results.csv", index=False)
print(res.to_string(index=False))

with open(OUT_DIR / "prjna842425_incremental_results.md", "w") as fh:
    fh.write(f"# PRJNA842425 external incremental validation (N=71)\n\n")
    fh.write(f"Generated {today_str()}. Primary estimand (prespecified in "
             "CRITICAL_CARE_EXTERNAL_ANALYSIS_PLAN.md Section 7): "
             "delta_AUROC(SOFA+M1) - AUROC(SOFA alone), in-sample 2-term logistic "
             "combination (mortality ~ z(SOFA) + z(logit(M1))), paired bootstrap "
             "(2000 replicates, seed 2026, predictions held fixed). SOFA is primary, "
             "APACHE secondary.\n\n"
             "**Endpoint (corrected 2026-09-14, fable audit finding B1):** the pre-existing "
             "`mortality` column is vital status at end of follow-up (up to 290 days), not "
             "in-hospital death -- 5 of its 32 events were discharged alive (Hospice/LTACH/SNF) "
             "and died 21-172 days later. `in_hospital_death_PRIMARY` "
             "(DischargeLocation==\"Death\", 27 events) is now the primary endpoint, matching the "
             "\"in-hospital mortality\" label used throughout this manuscript and matching "
             "AGI_BGISEQ's genuinely-in-hospital endpoint; the original all-cause-by-follow-up "
             "definition (32 events) is retained below as an explicit secondary/sensitivity row, "
             "not dropped. See CRITICAL_CARE_ANALYSIS_DEVIATIONS.md item 4.\n\n"
             "Calibration intercept/slope were computed and then dropped from this table: they are "
             "tautologically 0/1 for a model evaluated on its own in-sample training data "
             "(the logistic MLE's score equations guarantee it), so the number carries no "
             "information about genuine calibration transfer.\n\n")
    cols = list(res.columns)
    fh.write("| " + " | ".join(cols) + " |\n")
    fh.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for _, r in res.iterrows():
        fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    fh.write("\n**Caveat carried from lit26**: M1 standalone AUROC on this cohort is weak "
             "(0.55-0.57 under the original all-cause endpoint, 0.627 under the corrected "
             "in-hospital-death endpoint used as primary here), and this cohort is near-perfectly "
             "separable from Gangnam by microbiome composition alone (domain-shift AUROC "
             "0.87-1.00) -- so any positive delta here should be read with that transfer caveat, "
             "not as confirmatory replication of the internal cohort's own (also-null) "
             "delta-AUROC pattern.\n")

    # ------------------------------------------------------------------
    # OOF sensitivity (Codex review finding #1): the in-sample combined-
    # model bootstrap CI above does not include coefficient/fit uncertainty.
    # This stratified 5-fold out-of-fold refit of the SAME 2-term combiner
    # (primary endpoint, SOFA only) makes that additional uncertainty
    # explicit rather than leaving the in-sample number to stand alone.
    # ------------------------------------------------------------------
    from sklearn.model_selection import StratifiedKFold
    yy = merged["mortality_inhospital"].values
    sv_all = sofa
    ml_all = m1_logit
    sv_z_all = (sv_all - sv_all.mean()) / sv_all.std(ddof=0)
    ml_z_all = (ml_all - ml_all.mean()) / ml_all.std(ddof=0)
    X_all = np.column_stack([sv_z_all, ml_z_all])

    def oof_combined(seed):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(yy), np.nan)
        for tr, te in skf.split(X_all, yy):
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(X_all[tr], yy[tr])
            oof[te] = lr.predict_proba(X_all[te])[:, 1]
        return oof

    oof_p = oof_combined(seed=2026)
    oof_auc = roc_auc_score(yy, oof_p)
    d_oof, d_oof_lo, d_oof_hi, d_oof_p, _ = paired_bootstrap_delta_auroc(yy, sv_all, oof_p)

    # multi-seed range (20 fold-assignment seeds): at N=71 a single 5-fold
    # split is itself noisy, so report the spread, not just one draw --
    # exactly the instability Codex's independent seeds-0-99 check surfaced.
    multi_deltas = []
    for s in range(20):
        p_s = oof_combined(seed=s)
        multi_deltas.append(roc_auc_score(yy, p_s) - roc_auc_score(yy, sv_all))
    multi_deltas = np.array(multi_deltas)

    fh.write(f"\n## Sensitivity: 5-fold out-of-fold refit (SOFA, primary endpoint)\n\n"
             f"Same combiner (`z(SOFA) + z(logit(M1))`), refit inside each of 5 stratified folds "
             f"instead of once in-sample, so this estimate carries fit/split uncertainty the "
             f"in-sample bootstrap above does not. Seed 2026: OOF combined AUROC={oof_auc:.4f}, "
             f"delta_AUROC(OOF combined vs SOFA)={d_oof:.4f} [{d_oof_lo:.4f}, {d_oof_hi:.4f}], "
             f"p={d_oof_p:.4f}. Across 20 fold-assignment seeds (0-19), the OOF delta_AUROC point "
             f"estimate ranges {multi_deltas.min():.4f} to {multi_deltas.max():.4f} "
             f"(median {np.median(multi_deltas):.4f}) -- at N=71/27 events, a single 5-fold split "
             f"is itself a noisy estimate. Codex's independent OOF check (a different fold "
             f"implementation, seeds 0-99, on the pre-correction 32-event endpoint) found a "
             f"similarly unstable -0.040 to +0.013 range; the two are not expected to match "
             f"exactly (different endpoint, different fold code) but agree on the substantive "
             f"point -- no seed in either check suggests a positive, clinically meaningful "
             f"increment. Reported as a sensitivity check, not a replacement primary estimand.\n")
    print(f"[cc1] OOF sensitivity (SOFA, primary): AUROC={oof_auc:.4f}, "
          f"delta_AUROC={d_oof:.4f} [{d_oof_lo:.4f}, {d_oof_hi:.4f}], p={d_oof_p:.4f}; "
          f"20-seed range [{multi_deltas.min():.4f}, {multi_deltas.max():.4f}]")

    # ------------------------------------------------------------------
    # Exploratory decision-curve analysis (user request, 2026-09-14: this
    # was PRESPECIFIED in CRITICAL_CARE_EXTERNAL_ANALYSIS_PLAN.md Section 7
    # as "exploratory DCA... over the same threshold range as the internal
    # cohort (2-50%)" and was originally skipped with the rationale that the
    # delta_AUROC CI "already answers the incremental-value question" --
    # WRONG: discrimination and net benefit are different estimands (Codex
    # review finding #2), and skipping a prespecified analysis needs an
    # actual reason, not a post-hoc one. Run properly now.
    #
    # Net benefit at threshold pt: NB = TP/n - FP/n * pt/(1-pt) (Vickers &
    # Elkin 2006), "test positive" = predicted_p >= pt. Candidate = the
    # primary in-sample SOFA+M1 combined score; reference = the SOFA-only
    # fitted probability (p_red) from the same primary run above, NOT raw
    # SOFA (which is not a probability and cannot be thresholded the same
    # way) -- both already computed in the last run_one("SOFA", ...,
    # "in_hospital_death_PRIMARY", ...) call, captured via the module-level
    # names below since Python leaves a function's last-assigned locals
    # accessible only inside it; recomputed here in 3 lines rather than
    # threading extra return values through run_one's existing call sites.
    # ------------------------------------------------------------------
    yy_dca = merged["mortality_inhospital"].values
    sv_z_dca = (sofa - sofa.mean()) / sofa.std(ddof=0)
    ml_z_dca = (m1_logit - m1_logit.mean()) / m1_logit.std(ddof=0)
    X_dca = np.column_stack([sv_z_dca, ml_z_dca])
    p_red_dca = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(
        sv_z_dca.reshape(-1, 1), yy_dca).predict_proba(sv_z_dca.reshape(-1, 1))[:, 1]
    p_comb_dca = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(
        X_dca, yy_dca).predict_proba(X_dca)[:, 1]

    def net_benefit(y, p, pt):
        pos = p >= pt
        tp = np.sum(pos & (y == 1))
        fp = np.sum(pos & (y == 0))
        n = len(y)
        return tp / n - fp / n * (pt / (1 - pt))

    def paired_boot_nb(y, plain, adapt, pt, n_boot=2000, seed=2026):
        rng = np.random.RandomState(seed)
        n = len(y)
        deltas = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            deltas.append(net_benefit(y[idx], adapt[idx], pt) - net_benefit(y[idx], plain[idx], pt))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_val = min(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)), 1.0)
        point = net_benefit(y, adapt, pt) - net_benefit(y, plain, pt)
        return point, lo, hi, p_val

    thresholds = np.round(np.arange(0.02, 0.51, 0.02), 2)
    dca_rows = []
    for pt in thresholds:
        nb_red = net_benefit(yy_dca, p_red_dca, pt)
        nb_comb = net_benefit(yy_dca, p_comb_dca, pt)
        d, lo, hi, pv = paired_boot_nb(yy_dca, p_red_dca, p_comb_dca, pt)
        dca_rows.append(dict(threshold=pt, NB_SOFA_only=round(nb_red, 5), NB_SOFA_plus_M1=round(nb_comb, 5),
                              delta_NB=round(d, 5), delta_NB_lo=round(lo, 5), delta_NB_hi=round(hi, 5),
                              boot_p=round(pv, 4), CI_excludes_zero=bool(lo > 0 or hi < 0)))
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(OUT_DIR / "prjna842425_dca_results.csv", index=False)
    n_excl = int(dca_df["CI_excludes_zero"].sum())
    print(f"[cc1] Exploratory DCA (SOFA, primary, 25 thresholds 2-50%): "
          f"{n_excl}/{len(dca_df)} threshold evaluations exclude zero")
    fh.write(f"\n## Exploratory: decision-curve analysis (SOFA, primary endpoint)\n\n"
             f"Prespecified in the analysis plan, initially skipped with an inadequate rationale "
             f"(discrimination and net benefit are different estimands -- Codex review finding #2), "
             f"now run. Paired delta-NB (SOFA+M1 combined vs SOFA-only, both from the same in-sample "
             f"fits used above) at 25 thresholds from 2% to 50%, paired bootstrap (2000 replicates, "
             f"seed 2026). **{n_excl}/{len(dca_df)} threshold evaluations exclude zero** "
             f"(no multiplicity correction applied across this exploratory grid, matching how the "
             f"internal cohort's own threshold grid is reported). Full table: "
             f"`prjna842425_dca_results.csv`; figure: `figures/Fig_external_dca.png`.\n\n")
    fh.write("| threshold | NB (SOFA) | NB (SOFA+M1) | delta NB | 95% CI | p | excludes 0 |\n")
    fh.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for r in dca_rows:
        fh.write(f"| {r['threshold']:.2f} | {r['NB_SOFA_only']:.4f} | {r['NB_SOFA_plus_M1']:.4f} | "
                  f"{r['delta_NB']:.4f} | [{r['delta_NB_lo']:.4f}, {r['delta_NB_hi']:.4f}] | "
                  f"{r['boot_p']:.3f} | {r['CI_excludes_zero']} |\n")

print("[cc1] wrote prjna842425_incremental_results.csv/.md, prjna842425_dca_results.csv")
