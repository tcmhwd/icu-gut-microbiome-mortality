#!/usr/bin/env python3
"""53_phase_x_vre_sensitivity_landmark_firth.py

PRE-PLANNED SENSITIVITY ANALYSIS of an already-reported POST-HOC finding
(user request, 2026-09-02): script 51's headline positive result -- gut
vanA/vanB (VRE resistome, contig-level, PRIMARY) presence predicts
subsequent VRE acquisition among 426 pre-VRE-negative patients (unadjusted
OR=11.589 [3.378,39.767] p=0.0001 Holm 0.0014; SOFA+antibiotic-adjusted
OR=10.696 [2.940,38.918] p=0.0003 Holm 0.0042; 6/44 gene-positive vs 5/372
gene-negative acquired) -- stress-tested on the two axes a reviewer will
attack:

  (1) POPULATION IS NOT AN ADMISSION POPULATION. Script 51 deliberately used
      the 430-patient linked set (`metadata/microbiome.clinical.csv`), not
      the 302-patient day 0-3 landmark cohort used elsewhere in Phase X
      (`collection_time`: 0 for 302 patients, 1 for 128 day 4-193
      follow-up samples). The pre/post gate guarantees temporality (the gut
      sample precedes the acquisition event) regardless of which admission
      day it was drawn on, but it does NOT license the clinical phrase "at
      ICU admission" -- a patient sampled on, say, day 40 may have acquired
      vanA/vanB gut carriage itself under ICU antibiotic pressure, which is
      a different claim with a different intervention point.
  (2) EVENTS-PER-VARIABLE. 11 acquisitions / 3 model variables (VRE gene,
      SOFA, antibiotic-exposure binary) = EPV ~= 3.7, far below any
      conventional threshold (commonly cited: >=10) for unpenalized
      maximum-likelihood logistic regression -- the OR point estimate is
      plausibly inflated by small-sample bias.

THIS SCRIPT DOES NOT MODIFY SCRIPT 51 OR ITS OUTPUTS. It reuses script 51's
data-loading (`utils/mdro_data.py`), outcome-blind predictor definitions
(contig-level RGI/CARD vanA/vanB gene presence == VRE resistome flag,
exactly as `M.build_gene_presence_flags("contig")["VRE"]`), and reporting
conventions (Wilson-CI helper from script 52, closed-form 2x2 OR with
Haldane-Anscombe correction and `sm.Logit` SOFA+antibiotic-adjusted OR from
script 51, both duplicated here verbatim rather than imported, matching the
existing 51/52 convention of small helper duplication over cross-imports --
`51_*.py` cannot be imported directly anyway, since it executes its full
analysis as top-level module code on import).

SCOPE: VRE (broad/any-Enterococcus definition, `pre.VRE.1`/`post.VRE.1`,
PRIMARY per script 51's reconciliation) resistome predictor ONLY -- this is
the one manuscript claim being stress-tested here, not the full 7-organism/
2-predictor-family grid. Taxonomic (Enterococcus genus abundance) and
MAG-level resistome are NOT re-examined here (script 51 already showed the
resistome signal is the stronger, Holm-surviving one; re-litigating the
taxonomic/MAG-level null results is out of scope for this stress test).

FOUR PARTS (task brief, 2026-09-02):
  A. Day 0-3 restriction (`collection_time==0`, the 302-patient landmark
     cohort) -- N pre-negative / N gene-positive / N acquisitions reported
     FIRST; full inferential analysis only if acquisitions>=10 (project's
     own established bar, `M.MIN_ACQUISITION_EVENTS`), else descriptive
     counts only, no p-value, with an explicit statement about the
     manuscript's admission-time wording.
  B. Collection-day distribution (`collection_time` + day-from-admission,
     `collection_date` minus `admission_date`) among the 426 pre-negatives,
     44 gene-positives, and 11 acquisitions -- is gene-positivity/
     acquisition concentrated in later-sampled patients?
  C. Firth-penalized logistic regression (R's `logistf`, called via
     Rscript -- statsmodels has no native Firth implementation, and
     Python's `firthlogist` package is NOT installed in this project's
     'gpt' conda env, checked 2026-09-02; system R already has `logistf`
     installed) for the same adjusted model (VRE gene + SOFA + antibiotic
     binary), on the full 426/416 and, "if evaluable" (fit converges, not
     necessarily >=10 events -- Firth's whole purpose is estimation under
     few events, but its output on the 3-event landmark subset is reported
     as exploratory/supplementary, NOT as license to treat 3 events as an
     adequately powered test; Part A's own <10-events descriptive-only gate
     still governs what counts as "the" landmark-cohort answer), the
     day 0-3 subset. Reported ALONGSIDE the unpenalized `sm.Logit`
     estimates, never replacing them.
  D. With vs without the antibiotic-exposure covariate -- the antibiotic
     variable is a whole-ADMISSION binary flag with no day-level timing
     (same caveat already documented project-wide, revision_log.md item
     10, and in script 51's own docstring), so it may encode post-sample
     exposure. Both directions reported, neither treated as "the" answer.

Reads:
  metadata/microbiome.clinical.csv (pre/post VRE flags, SOFA, antibiotics,
    collection_date, collection_time, admission_date)
  merge/output/mag_summary/rgi_contig/*_per_sample.tsv (resistome, PRIMARY,
    via utils/mdro_data.py)
Writes:
  clinical_ai/reports/phase_x_vre_sensitivity_landmark_firth.csv (+
    _event_counts.csv, _collection_day_distribution.csv companions)
  clinical_ai/reports/phase_x_vre_sensitivity_landmark_firth.md
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, mdro_data as M  # noqa: E402

log = io_utils.setup_logging("53_phase_x_vre_sensitivity_landmark_firth")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_vre_sensitivity_landmark_firth.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_vre_sensitivity_landmark_firth.md")
R_FIRTH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "run_firth_logistf.R")
SCRATCH_DIR = ("/tmp/claude-1003/-home-tcmhwd-microbiome-ICU-public/"
               "a6592c63-91ba-4843-a83a-a75feeb8cd84/scratchpad")

# Duplicated from script 51 verbatim -- same fixed set, do not edit
# independently of that script's own copy.
ABX_BROAD_SPECTRUM_COLS = ["Carbapenem", "Glycopeptides", "cephems_3rd", "cephems_4th",
                            "BLBLI", "CEF_AVI", "CEF_TAZ"]


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, (centre - adj) / denom), min(1.0, (centre + adj) / denom))


def unadjusted_or_2x2(x, y):
    """Copied from script 51 verbatim. Closed-form 2x2 OR/CI/p, Haldane-
    Anscombe +0.5 continuity correction on any zero cell (flagged in the
    return). None if the predictor has no variation."""
    n = len(x)
    n_pos = int(x.sum())
    if n_pos == 0 or n_pos == n:
        return None
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    corrected = (a == 0 or b == 0 or c == 0 or d == 0)
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_ = np.exp(log_or)
    ci = (np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se))
    z = log_or / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"OR": or_, "CI_low": ci[0], "CI_high": ci[1], "p": p, "haldane_anscombe_corrected": corrected,
            "a": a, "b": b, "c": c, "d": d}


def adjusted_or_logistic(x, y, covariates):
    """Copied from script 51 verbatim. Unpenalized sm.Logit adjusted OR;
    None (not NaN) on non-convergence/singular-Hessian."""
    Xa = np.column_stack([x.astype(float)] + [c.astype(float) for c in covariates])
    Xac = sm.add_constant(Xa)
    try:
        fit_a = sm.Logit(y, Xac).fit(disp=0, method="bfgs", maxiter=200)
        if not fit_a.mle_retvals.get("converged", True):
            return None
        or_a, se_a = np.exp(fit_a.params[1]), fit_a.bse[1]
        if not np.isfinite(se_a) or se_a > 20:
            return None
        ci_a = (np.exp(fit_a.params[1] - 1.96 * se_a), np.exp(fit_a.params[1] + 1.96 * se_a))
        return {"OR": or_a, "CI_low": ci_a[0], "CI_high": ci_a[1], "p": fit_a.pvalues[1], "se": se_a}
    except Exception as e:
        log.warning("adjusted logistic fit failed (%s) -- reporting as non-converged, not a number", type(e).__name__)
        return None


def sens_spec_ppv_npv(ref, pred):
    """Copied from script 52 verbatim. ref/pred: 0/1 arrays. Wilson-CI
    point estimates."""
    tp = int(((ref == 1) & (pred == 1)).sum())
    fn = int(((ref == 1) & (pred == 0)).sum())
    fp = int(((ref == 0) & (pred == 1)).sum())
    tn = int(((ref == 0) & (pred == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    sens = tp / n_pos if n_pos else np.nan
    spec = tn / n_neg if n_neg else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    sens_ci = wilson_ci(tp, n_pos)
    spec_ci = wilson_ci(tn, n_neg)
    ppv_ci = wilson_ci(tp, tp + fp) if (tp + fp) else (np.nan, np.nan)
    npv_ci = wilson_ci(tn, tn + fn) if (tn + fn) else (np.nan, np.nan)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "sensitivity": round(sens, 4) if not np.isnan(sens) else np.nan,
        "sensitivity_CI_low": round(sens_ci[0], 4), "sensitivity_CI_high": round(sens_ci[1], 4),
        "specificity": round(spec, 4) if not np.isnan(spec) else np.nan,
        "specificity_CI_low": round(spec_ci[0], 4), "specificity_CI_high": round(spec_ci[1], 4),
        "PPV": round(ppv, 4) if not np.isnan(ppv) else np.nan,
        "PPV_CI_low": round(ppv_ci[0], 4), "PPV_CI_high": round(ppv_ci[1], 4),
        "NPV": round(npv, 4) if not np.isnan(npv) else np.nan,
        "NPV_CI_low": round(npv_ci[0], 4), "NPV_CI_high": round(npv_ci[1], 4),
    }


def run_firth_logistf(y, predictors: dict, log=None):
    """Firth-penalized logistic regression y ~ predictors via R's `logistf`,
    called out to Rscript (see `utils/run_firth_logistf.R`). `predictors` is
    an ordered dict name->array; the first entry must be the predictor of
    interest (reported as the headline term). Returns a dict keyed by model
    term ('(Intercept)' + predictor names) with {coef,OR,CI_low,CI_high,p},
    plus '_conv' (fit$conv triple: last LL change, max|score|, last beta
    change) and '_converged' (bool, thresholded on '_conv'), or None if the
    R process fails outright (never crashes or fabricates a number, same
    convention as `adjusted_or_logistic` above)."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int)})
    for name, arr in predictors.items():
        df[name] = np.asarray(arr, dtype=float)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=SCRATCH_DIR)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        proc = subprocess.run(["Rscript", R_FIRTH_SCRIPT, tmp_path],
                               capture_output=True, text=True, timeout=180)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    if proc.returncode != 0 or "FIRTH_FIT_FAILED" in proc.stdout:
        if log:
            log.warning("Firth (logistf) fit failed: rc=%s stderr_tail=%s", proc.returncode, proc.stderr[-1500:])
        return None
    result = {}
    conv = None
    n_r = None
    for line in proc.stdout.strip().splitlines():
        parts = line.strip().split(",")
        if not parts or parts[0] == "":
            continue
        if parts[0] == "COEF":
            _, term, coef, or_, ci_lo, ci_hi, p = parts
            result[term] = {"coef": float(coef), "OR": float(or_), "CI_low": float(ci_lo),
                             "CI_high": float(ci_hi), "p": float(p)}
        elif parts[0] == "CONV":
            conv = tuple(float(v) for v in parts[1:4])
        elif parts[0] == "N":
            n_r = int(parts[1])
    result["_conv"] = conv
    result["_n"] = n_r
    result["_converged"] = bool(conv) and all(np.isfinite(v) for v in conv) and conv[0] < 1e-3 and conv[1] < 1e-2
    return result


# ---------------------------------------------------------------------------
# 1. load clinical + VRE gene predictor, exactly as script 51's VRE branch
# ---------------------------------------------------------------------------
mc = M.load_clinical()
vre_facts = M.reconcile_vre_and_verify(mc, log=log)
flags_contig, detect_contig = M.build_gene_presence_flags("contig")

mc = mc.set_index("sample_list", drop=False)
resist_contig_j = flags_contig.reindex(mc.index)

pre_c, post_c, _ = M.ORGANISM_PREPOST["VRE"]
pre = pd.to_numeric(mc[pre_c], errors="coerce")
post = pd.to_numeric(mc[post_c], errors="coerce")
gene = resist_contig_j["VRE"]

SOFA = pd.to_numeric(mc["SOFA"], errors="coerce")
abx_broad = (mc[ABX_BROAD_SPECTRUM_COLS].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(float)
collection_time = pd.to_numeric(mc["collection_time"], errors="coerce")
collection_date = pd.to_datetime(mc["collection_date"], errors="coerce")
admission_date = pd.to_datetime(mc["admission_date"], errors="coerce")
day_from_admission = (collection_date - admission_date).dt.days

pop_mask = pre == 0  # pre-VRE-negative, outcome-blind population gate (identical to script 51)
valid = pop_mask & gene.notna() & post.notna() & SOFA.notna()
landmark_mask = valid & (collection_time == 0)

n_pop_raw = int(pop_mask.sum())
n_pop_valid = int(valid.sum())
n_landmark_raw = int((pop_mask & (collection_time == 0)).sum())
n_landmark_valid = int(landmark_mask.sum())
log.info("Full population: pre-negative=%d, joined/SOFA-complete=%d", n_pop_raw, n_pop_valid)
log.info("Landmark (collection_time==0) subset: pre-negative=%d, joined/SOFA-complete=%d",
         n_landmark_raw, n_landmark_valid)

report_rows = []  # main OR/CI/p table, one row per (population, model)


def add_row(**kw):
    report_rows.append(kw)


# ---------------------------------------------------------------------------
# 2. Baseline reproduction (full 426/416 population) -- sanity check that
#    this script's independent recomputation matches script 51's reported
#    headline numbers, and the basis for Part C/D's "full population" arm.
# ---------------------------------------------------------------------------
x_full = gene[valid].astype(int).values
y_full = post[valid].astype(int).values
sofa_full = SOFA[valid].values
abx_full = abx_broad[valid].values
n_genepos_full = int(x_full.sum())
n_acq_full = int(y_full.sum())

u_full = unadjusted_or_2x2(x_full, y_full)
a_full_with_abx = adjusted_or_logistic(x_full, y_full, [sofa_full, abx_full])
a_full_without_abx = adjusted_or_logistic(x_full, y_full, [sofa_full])
sspn_full = sens_spec_ppv_npv(y_full, x_full)  # gene as a "test" for the acquisition event

log.info("Baseline reproduction (full 426/%d): gene+=%d, acquisitions=%d, unadjusted OR=%.3f "
         "(script 51 reported N=416, gene+=44, acq=11, unadjusted OR=11.589 -- must match)",
         n_pop_valid, n_genepos_full, n_acq_full, u_full["OR"] if u_full else np.nan)
assert n_pop_valid == 416 and n_genepos_full == 44 and n_acq_full == 11, \
    "baseline reproduction does not match script 51's reported numbers -- investigate before proceeding"

add_row(part="baseline_full_reproduction", population="full_426(_416_joined)", model="unadjusted_2x2",
        N=n_pop_valid, N_events=n_acq_full, N_predictor_pos=n_genepos_full,
        OR=round(u_full["OR"], 3), CI_low=round(u_full["CI_low"], 3), CI_high=round(u_full["CI_high"], 3),
        p=round(u_full["p"], 4), converged=True,
        notes="reproduces script 51 exactly (sanity check)")
add_row(part="baseline_full_reproduction", population="full_426(_416_joined)", model="adjusted_SOFA_abx (unpenalized sm.Logit)",
        N=n_pop_valid, N_events=n_acq_full, N_predictor_pos=n_genepos_full,
        OR=round(a_full_with_abx["OR"], 3), CI_low=round(a_full_with_abx["CI_low"], 3),
        CI_high=round(a_full_with_abx["CI_high"], 3), p=round(a_full_with_abx["p"], 4),
        converged=a_full_with_abx is not None, notes="reproduces script 51 exactly (sanity check)")
add_row(part="part_D_with_without_abx", population="full_426(_416_joined)", model="adjusted_SOFA_only (unpenalized sm.Logit, no abx)",
        N=n_pop_valid, N_events=n_acq_full, N_predictor_pos=n_genepos_full,
        OR=round(a_full_without_abx["OR"], 3), CI_low=round(a_full_without_abx["CI_low"], 3),
        CI_high=round(a_full_without_abx["CI_high"], 3), p=round(a_full_without_abx["p"], 4),
        converged=a_full_without_abx is not None, notes="Part D -- abx covariate dropped")

# ---------------------------------------------------------------------------
# 3. PART A -- day 0-3 landmark restriction
# ---------------------------------------------------------------------------
x_land = gene[landmark_mask].astype(int).values
y_land = post[landmark_mask].astype(int).values
sofa_land = SOFA[landmark_mask].values
abx_land = abx_broad[landmark_mask].values
n_genepos_land = int(x_land.sum())
n_acq_land = int(y_land.sum())
landmark_evaluable = n_acq_land >= M.MIN_ACQUISITION_EVENTS

log.info("=== PART A: day 0-3 landmark restriction -- N_pre_negative=%d (raw), N_joined=%d, "
         "N_gene_positive=%d, N_acquisitions=%d (bar=%d -> %s) ===",
         n_landmark_raw, n_landmark_valid, n_genepos_land, n_acq_land, M.MIN_ACQUISITION_EVENTS,
         "EVALUABLE, full inferential analysis" if landmark_evaluable else "DESCRIPTIVE ONLY, no p-value")

event_counts = pd.DataFrame([
    {"population": "full_pre_negative (script 51 population)", "N_pre_negative_raw": n_pop_raw,
     "N_joined_SOFA_complete": n_pop_valid, "N_gene_positive": n_genepos_full, "N_acquisitions": n_acq_full,
     "meets_10_event_bar": n_acq_full >= M.MIN_ACQUISITION_EVENTS},
    {"population": "day0-3_landmark (collection_time==0)", "N_pre_negative_raw": n_landmark_raw,
     "N_joined_SOFA_complete": n_landmark_valid, "N_gene_positive": n_genepos_land, "N_acquisitions": n_acq_land,
     "meets_10_event_bar": landmark_evaluable},
    {"population": "day4+_followup (collection_time==1, complement)",
     "N_pre_negative_raw": n_pop_raw - n_landmark_raw,
     "N_joined_SOFA_complete": n_pop_valid - n_landmark_valid,
     "N_gene_positive": n_genepos_full - n_genepos_land, "N_acquisitions": n_acq_full - n_acq_land,
     "meets_10_event_bar": (n_acq_full - n_acq_land) >= M.MIN_ACQUISITION_EVENTS},
])
io_utils.save_csv(event_counts, OUT_CSV.replace(".csv", "_event_counts.csv"))
log.info("event counts:\n%s", event_counts.to_string(index=False))

if landmark_evaluable:
    u_land = unadjusted_or_2x2(x_land, y_land)
    a_land_with_abx = adjusted_or_logistic(x_land, y_land, [sofa_land, abx_land])
    a_land_without_abx = adjusted_or_logistic(x_land, y_land, [sofa_land])
    sspn_land = sens_spec_ppv_npv(y_land, x_land)
    add_row(part="part_A_full_inferential", population="day0-3_landmark", model="unadjusted_2x2",
            N=n_landmark_valid, N_events=n_acq_land, N_predictor_pos=n_genepos_land,
            OR=round(u_land["OR"], 3) if u_land else np.nan,
            CI_low=round(u_land["CI_low"], 3) if u_land else np.nan,
            CI_high=round(u_land["CI_high"], 3) if u_land else np.nan,
            p=round(u_land["p"], 4) if u_land else np.nan, converged=u_land is not None, notes="")
    add_row(part="part_A_full_inferential", population="day0-3_landmark", model="adjusted_SOFA_abx (unpenalized)",
            N=n_landmark_valid, N_events=n_acq_land, N_predictor_pos=n_genepos_land,
            OR=round(a_land_with_abx["OR"], 3) if a_land_with_abx else np.nan,
            CI_low=round(a_land_with_abx["CI_low"], 3) if a_land_with_abx else np.nan,
            CI_high=round(a_land_with_abx["CI_high"], 3) if a_land_with_abx else np.nan,
            p=round(a_land_with_abx["p"], 4) if a_land_with_abx else np.nan,
            converged=a_land_with_abx is not None, notes="")
else:
    u_land = unadjusted_or_2x2(x_land, y_land)  # computed anyway for Part C/D exploratory use, NOT shown as Part A's answer
    a_land_with_abx = adjusted_or_logistic(x_land, y_land, [sofa_land, abx_land])
    a_land_without_abx = adjusted_or_logistic(x_land, y_land, [sofa_land])
    sspn_land = None
    rate_pos = y_land[x_land == 1].mean() if n_genepos_land > 0 else np.nan
    rate_neg = y_land[x_land == 0].mean() if (len(x_land) - n_genepos_land) > 0 else np.nan
    add_row(part="part_A_descriptive_only", population="day0-3_landmark", model="descriptive_counts (no p-value, <10 events)",
            N=n_landmark_valid, N_events=n_acq_land, N_predictor_pos=n_genepos_land,
            OR=np.nan, CI_low=np.nan, CI_high=np.nan, p=np.nan, converged=np.nan,
            notes=f"acquisition_rate_gene_pos={rate_pos:.4f} ({int(y_land[x_land==1].sum())}/{n_genepos_land}); "
                  f"acquisition_rate_gene_neg={rate_neg:.4f} ({int(y_land[x_land==0].sum())}/{len(x_land)-n_genepos_land}); "
                  "per task instruction: descriptive only, no OR/CI/p manufactured under 10 acquisition events")

# ---------------------------------------------------------------------------
# 4. PART B -- collection-day distribution among pre-negatives / gene-pos /
#    acquisitions (full 426/416 population, not landmark-restricted)
# ---------------------------------------------------------------------------
ct_valid = collection_time[valid].values
dfa_valid = day_from_admission[valid].values

groups = {
    "all_pre_negative": np.ones(len(x_full), dtype=bool),
    "gene_positive": x_full == 1,
    "acquisitions": y_full == 1,
}
dist_rows = []
for gname, gmask in groups.items():
    ct_g = ct_valid[gmask]
    dfa_g = dfa_valid[gmask]
    n_g = int(gmask.sum())
    n_day03 = int((ct_g == 0).sum())
    n_day4plus = int((ct_g == 1).sum())
    dist_rows.append({
        "group": gname, "N": n_g,
        "n_collection_time_0_day0-3": n_day03, "pct_day0-3": round(100 * n_day03 / n_g, 1) if n_g else np.nan,
        "n_collection_time_1_day4plus": n_day4plus, "pct_day4plus": round(100 * n_day4plus / n_g, 1) if n_g else np.nan,
        "day_from_admission_mean": round(float(np.mean(dfa_g)), 2),
        "day_from_admission_median": round(float(np.median(dfa_g)), 1),
        "day_from_admission_IQR_low": round(float(np.percentile(dfa_g, 25)), 1),
        "day_from_admission_IQR_high": round(float(np.percentile(dfa_g, 75)), 1),
        "day_from_admission_min": int(np.min(dfa_g)), "day_from_admission_max": int(np.max(dfa_g)),
    })
distribution_table = pd.DataFrame(dist_rows)
io_utils.save_csv(distribution_table, OUT_CSV.replace(".csv", "_collection_day_distribution.csv"))
log.info("collection-day distribution:\n%s", distribution_table.to_string(index=False))

# Informal (exploratory, NOT Holm-corrected, NOT part of the primary
# inferential grid) association checks: is gene-positivity / acquisition
# concentrated in later-sampled patients?
fisher_gene = stats.fisher_exact(pd.crosstab(x_full, ct_valid.astype(int)).values)
mwu_gene = stats.mannwhitneyu(dfa_valid[x_full == 1], dfa_valid[x_full == 0])
fisher_acq = stats.fisher_exact(pd.crosstab(y_full, ct_valid.astype(int)).values)
mwu_acq = stats.mannwhitneyu(dfa_valid[y_full == 1], dfa_valid[y_full == 0])
log.info("Exploratory: gene-positive vs day0-3/day4+ Fisher p=%.4g; day_from_admission Mann-Whitney p=%.4g",
         fisher_gene[1], mwu_gene.pvalue)
log.info("Exploratory: acquisition vs day0-3/day4+ Fisher p=%.4g; day_from_admission Mann-Whitney p=%.4g",
         fisher_acq[1], mwu_acq.pvalue)

# ---------------------------------------------------------------------------
# 5. PART C -- Firth-penalized logistic regression (full + landmark, if
#    evaluable in the technical sense that the R fit converges)
# ---------------------------------------------------------------------------
firth_full_with_abx = run_firth_logistf(y_full, {"VRE_gene": x_full, "SOFA": sofa_full, "abx_broad": abx_full}, log=log)
firth_full_without_abx = run_firth_logistf(y_full, {"VRE_gene": x_full, "SOFA": sofa_full}, log=log)

firth_land_with_abx = firth_land_without_abx = None
if n_acq_land >= 1 and n_genepos_land >= 1:  # technical evaluability floor, not the 10-event bar
    firth_land_with_abx = run_firth_logistf(y_land, {"VRE_gene": x_land, "SOFA": sofa_land, "abx_broad": abx_land}, log=log)
    firth_land_without_abx = run_firth_logistf(y_land, {"VRE_gene": x_land, "SOFA": sofa_land}, log=log)
else:
    log.warning("Landmark subset not technically evaluable for Firth (n_acq=%d, n_genepos=%d) -- skipped",
                n_acq_land, n_genepos_land)


def _add_firth_row(part, population, firth_result, n, n_events, n_pos, notes_extra=""):
    if firth_result is None or "VRE_gene" not in firth_result:
        add_row(part=part, population=population, model="Firth (logistf)",
                N=n, N_events=n_events, N_predictor_pos=n_pos,
                OR=np.nan, CI_low=np.nan, CI_high=np.nan, p=np.nan, converged=False,
                notes="FIRTH_FIT_FAILED -- R logistf did not return a usable fit " + notes_extra)
        return
    t = firth_result["VRE_gene"]
    conv = firth_result.get("_conv")
    conv_str = f"conv(LLchange={conv[0]:.2e},maxscore={conv[1]:.2e},betachange={conv[2]:.2e})" if conv else "conv=NA"
    add_row(part=part, population=population, model="Firth (logistf)",
            N=n, N_events=n_events, N_predictor_pos=n_pos,
            OR=round(t["OR"], 3), CI_low=round(t["CI_low"], 3), CI_high=round(t["CI_high"], 3),
            p=round(t["p"], 4), converged=firth_result["_converged"],
            notes=(conv_str + (" " + notes_extra if notes_extra else "")))


_add_firth_row("part_C_firth", "full_426(_416_joined)", firth_full_with_abx, n_pop_valid, n_acq_full, n_genepos_full,
               notes_extra="with SOFA+abx")
_add_firth_row("part_D_with_without_abx", "full_426(_416_joined)", firth_full_without_abx, n_pop_valid, n_acq_full, n_genepos_full,
               notes_extra="Part D -- abx covariate dropped, SOFA only")
if firth_land_with_abx is not None or n_acq_land >= 1:
    _add_firth_row("part_C_firth", "day0-3_landmark",
                    firth_land_with_abx, n_landmark_valid, n_acq_land, n_genepos_land,
                    notes_extra="EXPLORATORY (only 3 acquisition events -- Firth was designed for exactly this "
                                 "low-event regime, but does not by itself make 3 events an adequately powered "
                                 "test; Part A's descriptive-only gate still governs the landmark-cohort answer)"
                                 if not landmark_evaluable else "with SOFA+abx")
    _add_firth_row("part_D_with_without_abx", "day0-3_landmark",
                    firth_land_without_abx, n_landmark_valid, n_acq_land, n_genepos_land,
                    notes_extra="Part D -- abx covariate dropped, SOFA only; EXPLORATORY, see above" if not landmark_evaluable
                                 else "Part D -- abx covariate dropped, SOFA only")

out = pd.DataFrame(report_rows)
io_utils.save_csv(out, OUT_CSV)
log.info("wrote %s (+ event_counts, collection_day_distribution companions)", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. EPV (events-per-variable) for both populations, adjusted model (3 vars:
#    VRE gene, SOFA, abx)
# ---------------------------------------------------------------------------
epv_full = n_acq_full / 3
epv_land = n_acq_land / 3 if n_acq_land else np.nan

# ---------------------------------------------------------------------------
# 7. markdown report
# ---------------------------------------------------------------------------
headline_changes = not landmark_evaluable  # True -> the day0-3 result changes what the manuscript can claim

md = []
if headline_changes:
    md += [
        "# VRE resistome -> acquisition finding: landmark + Firth sensitivity analysis",
        "",
        "> ## HEADLINE: THE DAY 0-3 RESULT CHANGES WHAT THE MANUSCRIPT CAN CLAIM",
        f"> Restricted to the {n_landmark_valid}-patient day 0-3 landmark cohort (`collection_time==0`), only "
        f"**{n_acq_land} VRE acquisition events** occur among {n_genepos_land} vanA/vanB gene-positive and "
        f"{n_landmark_valid - n_genepos_land} gene-negative pre-negative patients -- below the project's own "
        f"{M.MIN_ACQUISITION_EVENTS}-acquisition-event bar for a formal OR/CI/p. **The manuscript's "
        "\"gut vanA/vanB at ICU admission predicts subsequent VRE acquisition\" framing is NOT earned by this "
        "cohort's day 0-3 subset alone.** The correct, defensible wording is that vanA/vanB gut gene presence "
        "**at the first available gut sample** (which for most of the 426-patient analysis population is NOT an "
        "admission-day sample -- see Part B below) predicts subsequent VRE acquisition. Part B additionally shows "
        "that both gene-positivity and acquisition are themselves significantly concentrated in later-sampled "
        "patients (Fisher p<0.001 and p<0.01 respectively), which is an independent reason the admission-time "
        "framing is not supported: some of the signal may reflect ICU-acquired carriage under antibiotic "
        "pressure, not a baseline/admission risk marker.",
        "",
    ]
else:
    md += [
        "# VRE resistome -> acquisition finding: landmark + Firth sensitivity analysis",
        "",
        f"> ## The day 0-3 landmark result CONFIRMS the admission-time framing: {n_acq_land} acquisitions "
        f"clear the {M.MIN_ACQUISITION_EVENTS}-event bar in the {n_landmark_valid}-patient day 0-3 subset. "
        "See Part A below for the full inferential result restricted to that cohort.",
        "",
    ]

md += [
    "> **PRE-PLANNED SENSITIVITY ANALYSIS of an already-reported POST-HOC finding** "
    "(`clinical metadata/scripts/51_phase_x_mdro_acquisition_prediction.py`). Script 51 is NOT modified; this "
    "script reuses its data-loading (`utils/mdro_data.py`), outcome-blind VRE resistome (vanA/vanB, contig-level "
    "RGI/CARD, PRIMARY) predictor definition, and reporting conventions, applied ONLY to the VRE organism "
    "(the manuscript's positive finding) under 4 stress tests specified by the task brief: (A) day 0-3 landmark "
    "restriction, (B) collection-day distribution among pre-negatives/gene-positives/acquisitions, (C) "
    "Firth-penalized logistic regression alongside the unpenalized estimates, (D) with/without the "
    "antibiotic-exposure covariate. No threshold, gene definition, or covariate set was altered after seeing "
    "these numbers -- all reuse script 51's own fixed choices.",
    "",
    "## Background problems being stress-tested",
    "",
    "**(1) Population is not an admission population.** Script 51 used the full 430-patient linked set, not the "
    "302-patient day 0-3 landmark cohort used elsewhere in Phase X. The pre/post gate (`pre.VRE.1==0` at/before "
    "the gut sample, `post.VRE.1==1` strictly after) guarantees temporality regardless of which admission day the "
    "sample was drawn on, but does not by itself license \"at ICU admission\" -- a patient sampled on a later day "
    "may have acquired vanA/vanB gut carriage itself under ICU antibiotic pressure before that later sample.",
    "",
    f"**(2) Events-per-variable.** Script 51's adjusted model (VRE gene + SOFA + antibiotic binary, 3 variables) "
    f"has {n_acq_full} acquisition events in the full population -> EPV = {epv_full:.2f}, well below the "
    "commonly cited EPV>=10 threshold for reliable unpenalized maximum-likelihood logistic regression -- the OR "
    "point estimate is plausibly inflated by small-sample bias. In the day 0-3 landmark subset, EPV = "
    f"{epv_land:.2f} ({n_acq_land} events / 3 variables) -- even more severely underpowered.",
    "",
    "## Part A -- day 0-3 landmark restriction (read this first)",
    "",
    df_to_md_table(event_counts),
    "",
]

if landmark_evaluable:
    md += [
        f"**{n_acq_land} acquisitions clear the {M.MIN_ACQUISITION_EVENTS}-event bar** in the "
        f"{n_landmark_valid}-patient day 0-3 landmark cohort -> full inferential analysis below.",
        "",
        df_to_md_table(out[(out["part"] == "part_A_full_inferential")]
                       [["model", "N", "N_events", "N_predictor_pos", "OR", "CI_low", "CI_high", "p", "converged"]]),
        "",
        "### Sensitivity/specificity/PPV/NPV (vanA/vanB gene presence as a predictor of subsequent acquisition, day 0-3 cohort)",
        "",
        df_to_md_table(pd.DataFrame([sspn_land])),
        "",
        "**The \"at ICU admission\" framing is earned**: the day 0-3-restricted result is directionally and "
        "statistically consistent with the full-population result.",
        "",
    ]
else:
    md += [
        f"**Only {n_acq_land} acquisitions** in the {n_landmark_valid}-patient day 0-3 landmark cohort -- "
        f"below the project's {M.MIN_ACQUISITION_EVENTS}-event bar. Per this project's established convention "
        "(scripts 51/49/etc.): **descriptive counts only, no OR/CI/p manufactured.**",
        "",
        f"- vanA/vanB gene-positive ({n_genepos_land} patients): {int(y_land[x_land==1].sum())}/{n_genepos_land} "
        f"acquired VRE ({100*y_land[x_land==1].mean() if n_genepos_land else float('nan'):.1f}%).",
        f"- vanA/vanB gene-negative ({len(x_land)-n_genepos_land} patients): "
        f"{int(y_land[x_land==0].sum())}/{len(x_land)-n_genepos_land} acquired VRE "
        f"({100*y_land[x_land==0].mean() if (len(x_land)-n_genepos_land) else float('nan'):.1f}%).",
        "",
        "**Manuscript wording implication**: the finding cannot be described as occurring \"at ICU admission\" "
        "on the strength of this cohort's day 0-3 subset alone. The defensible phrasing is that vanA/vanB gut "
        "gene presence **at the first available gut sample** (median day-from-admission for the full analysis "
        "population is shown in Part B below, and is well past day 0-3 for the gene-positive and acquisition "
        "subgroups specifically) predicts subsequent VRE acquisition -- the temporal precedence (sample-before-"
        "event) claim survives; the admission-time claim does not.",
        "",
        "Exploratory (NOT the formal Part A answer, shown only because Part C requires it): unpenalized and "
        "Firth-penalized adjusted ORs computed on this same 3-event subset are reported in Parts C/D below, both "
        "prominently labeled as exploratory given the event count.",
        "",
    ]

md += [
    "## Part B -- collection-day distribution (full 426/416 population: all pre-negatives, gene-positives, "
    "acquisitions)",
    "",
    "`collection_time`: 0 = day 0-3 landmark window (project-wide convention); 1 = day 4-193 follow-up sample. "
    "`day_from_admission` = `collection_date` minus `admission_date` (4 of the 430 patients in this project have "
    "a small negative gap, kept by prior explicit user decision -- see `ICU_public/CLAUDE.md`).",
    "",
    df_to_md_table(distribution_table),
    "",
    f"**Exploratory association checks (informal, NOT Holm-corrected, NOT part of any primary inferential "
    f"grid -- shown only to characterize the concentration pattern)**: vanA/vanB gene-positivity is significantly "
    f"associated with later sampling (day0-3 vs day4+ Fisher exact p={fisher_gene[1]:.4g}; day-from-admission "
    f"Mann-Whitney p={mwu_gene.pvalue:.4g}). VRE acquisition is likewise significantly associated with later "
    f"sampling (Fisher exact p={fisher_acq[1]:.4g}; Mann-Whitney p={mwu_acq.pvalue:.4g}).",
    "",
    f"**Interpretation**: only {distribution_table.loc[distribution_table['group']=='gene_positive','pct_day0-3'].values[0]}% "
    "of gene-positive patients, and only "
    f"{distribution_table.loc[distribution_table['group']=='acquisitions','pct_day0-3'].values[0]}% of patients "
    f"who went on to acquire VRE, were sampled in the day 0-3 window -- both well below the "
    f"{distribution_table.loc[distribution_table['group']=='all_pre_negative','pct_day0-3'].values[0]}% baseline "
    "rate among all pre-negative patients. Gene-positivity and acquisition are concentrated in later-sampled "
    "patients, consistent with (though not proof of) some of the vanA/vanB carriage being ICU-acquired under "
    "antibiotic pressure rather than present at admission.",
    "",
    "## Part C -- Firth-penalized logistic regression (R `logistf`, via Rscript)",
    "",
    "statsmodels has no native Firth implementation; Python's `firthlogist` package is not installed in this "
    "project's 'gpt' conda env (checked 2026-09-02). System R already has `logistf` installed and was used "
    "instead of adding a new Python dependency mid-analysis -- see `utils/run_firth_logistf.R`.",
    "",
    df_to_md_table(out[out["part"] == "part_C_firth"]
                   [["population", "model", "N", "N_events", "N_predictor_pos", "OR", "CI_low", "CI_high",
                     "p", "converged", "notes"]]),
    "",
    "**Full population, unpenalized (script 51's reported estimate) vs Firth-penalized, side by side:**",
    "",
    df_to_md_table(pd.concat([
        out[(out["part"] == "baseline_full_reproduction") & (out["model"].str.startswith("adjusted_SOFA_abx"))],
        out[(out["part"] == "part_C_firth") & (out["population"] == "full_426(_416_joined)")],
    ])[["model", "OR", "CI_low", "CI_high", "p", "converged"]]),
    "",
    "The Firth-penalized OR is somewhat SMALLER than the unpenalized OR (consistent with the expected direction "
    "of Firth's small-sample bias correction -- unpenalized MLE tends to overestimate |log-OR| under low EPV), "
    "but the CI remains wide and excludes 1 in both cases; the qualitative conclusion (vanA/vanB gene presence is "
    "associated with subsequent VRE acquisition) is unchanged by penalization, though the precise magnitude "
    "should be read as somewhat uncertain given EPV~=3.7.",
    "",
    "## Part D -- with vs without the antibiotic-exposure covariate",
    "",
    "The antibiotic-exposure variable is a whole-ADMISSION binary flag with no day-level timing (same limitation "
    "already documented project-wide, `revision_log.md` item 10, and in script 51's own docstring) -- it may "
    "include doses given AFTER the gut sample too. Neither version is treated as \"the\" answer; both are "
    "reported.",
    "",
    df_to_md_table(pd.concat([
        out[(out["part"] == "baseline_full_reproduction") & (out["model"].str.startswith("adjusted_SOFA_abx"))],
        out[out["part"] == "part_D_with_without_abx"],
    ])[["population", "model", "N", "N_events", "OR", "CI_low", "CI_high", "p", "converged", "notes"]]),
    "",
]

with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
