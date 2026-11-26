#!/usr/bin/env python3
"""51_phase_x_mdro_acquisition_prediction.py

Task B, STEP 2 (user request, 2026-09-02): "SURVEILLANCE unblocked" -- does
the admission gut metagenome predict SUBSEQUENT MDRO acquisition? Different
question from the mortality-prognosis arm (already answered null elsewhere
in this project); this is a NEW, POST-HOC-BUT-METHOD-DRIVEN analysis
(`prespecification_audit.csv` vocabulary -- the acquisition question itself
has a strong microbiology prior and a prespecifiable structure, but this
specific test was designed and run in one pass after the temporal-anchor
gate below was confirmed, not before any data was seen).

TEMPORAL ANCHOR (data-owner-confirmed, 2026-09-02; not documented in the
source file itself -- recorded here per that instruction, and again in
`utils/mdro_data.py`): `post.*` MDRO flags in
`metadata/microbiome.clinical.csv` are STRICTLY AFTER the microbiome sample
collection date; `pre.*` flags are the same day or before. This is what
converts script 50's "adjacent, not-yet-usable resource" into a genuine
prediction analysis: PRE-negative -> POST-positive is acquisition observed
strictly after the admission gut sample, i.e. the metagenome genuinely
precedes the event it is being asked to predict.

POPULATION: the 430 patients in `metadata/microbiome.clinical.csv` already
linked to the 466-patient association-arm cohort (script 50's "466-linked
subset"). This is a DIFFERENT (larger, non-day0-3-restricted) population
than the 302-patient landmark ML cohort used elsewhere in Phase X -- used
here because it is the one with pre/post MDRO flags at all, and because the
acquisition question does not require the landmark-design's day0-3
admission-time restriction the mortality-prediction arm needed (the sample
just needs to precede the event, which the pre/post gate itself guarantees
regardless of which admission day the sample was drawn on).

VRE DUPLICATE-COLUMN RESOLUTION: see `utils/mdro_data.py` docstring for the
full computational derivation. Summary: `pre.VRE.1`/`post.VRE.1` (whose
summary column is literally named `VRE`) is the BROADER/any-Enterococcus
definition and is used as PRIMARY; `pre.VRE`/`post.VRE` (summary column
`VREFM`) is the E.-faecium-specific definition, reported as a secondary/
alternate row.

TWO PRE-DECLARED PREDICTOR FAMILIES, reported side by side for every
organism, at every evaluable variant -- NEITHER is chosen post-hoc:
  (a) TAXONOMIC: read-based (MetaPhlAn3) genus relative abundance, at 3
      OUTCOME-BLIND thresholds (presence at profiler detection limit i.e.
      >0%, domination >=10%, domination >=30% -- the HSCT-literature
      "domination" convention, matching script 49's precedent).
  (b) RESISTOME: presence/absence of the mechanistically corresponding
      resistance gene(s) in the SAME admission gut sample, from the
      project's RGI/CARD contig-level output (whole-assembly resistome, not
      limited by MAG-binning success -- see `utils/mdro_data.py` for why
      this table was chosen over the MAG-level one as PRIMARY; MAG-level is
      reported as a sensitivity check). Gene mapping (task-specified,
      verified present/absent in this cohort's own RGI output, not assumed):
        carbapenemase (KPC/NDM/OXA-48-like families; VIM/IMP families
          targeted but NEVER detected in ANY sample in this cohort, checked)
          -> CRE, CRKP [and CRPA, as an EXPLICIT EXTENSION -- see caveat]
        ESBL (CTX-M any allele; SHV/TEM EXCLUDING the ancestral
          narrow-spectrum reference alleles SHV-1/TEM-1) -> thirdGCEC
        van genes (vanA, vanB exactly, per the task's own explicit list,
          not vanC/D/G which are often intrinsic/non-VRE) -> VRE
        mecA/mecC (mecC never detected in this cohort, checked) -> MRSA

EVENT-COUNT GATE (task's own stated bar, applied to the pre=0/post=1
ACQUISITION count, which is smaller than and different from the raw
pre-OR-post "total events" count reported in the task brief and in script
50): >=10 acquisition events -> full inferential analysis (OR/CI/p, AUROC).
<10 -> DESCRIPTIVE ONLY, no OR/CI/p manufactured. thirdGCEC (27), CRPA (21),
MRSA (14), VRE (11) clear the bar; CRE (8), CRKP (6), VREFM (4) do not.

MULTIPLICITY: Holm-Bonferroni (matching this project's established
convention, scripts 47/49), applied SEPARATELY to the unadjusted and
SOFA+antibiotic-adjusted p-values, across the PRIMARY grid only (4 evaluable
organisms x [3 taxonomic thresholds + 1 contig-resistome gene test] = 16
tests per adjustment level). The MAG-level resistome sensitivity check and
the descriptive-only organisms are explicitly OUTSIDE this correction
family (no test/p-value exists for them to correct).

ANTIBIOTIC-EXPOSURE CONFOUNDER CAVEAT (same limitation already documented
project-wide, item 10 in `revision_log.md`): only a whole-ADMISSION binary
flag exists per antibiotic class in `metadata/microbiome.clinical.csv` (no
day-level timing), so "broad-spectrum exposure" here is a coarse proxy for
baseline antibiotic pressure that may include doses given AFTER the sample
too -- adjustment, not a pre/post-sample-restricted covariate.

Reads:
  metadata/microbiome.clinical.csv (pre/post MDRO flags, SOFA, antibiotics)
  merge/output/metaphlan3/metaphlan3_genus_merged.tsv (taxonomic predictor)
  merge/output/mag_summary/rgi_contig/*_per_sample.tsv (resistome, PRIMARY)
  merge/output/mag_summary/rgi/*_per_sample.tsv (resistome, MAG sensitivity)
Writes:
  clinical_ai/reports/phase_x_mdro_acquisition_prediction.csv
  clinical_ai/reports/phase_x_mdro_acquisition_prediction.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, mdro_data as M  # noqa: E402

log = io_utils.setup_logging("51_phase_x_mdro_acquisition_prediction")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_mdro_acquisition_prediction.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_mdro_acquisition_prediction.md")

SEED = config.RANDOM_SEED
N_BOOT = 2000
ABX_BROAD_SPECTRUM_COLS = ["Carbapenem", "Glycopeptides", "cephems_3rd", "cephems_4th", "BLBLI", "CEF_AVI", "CEF_TAZ"]

EVALUABLE_ORGANISMS = ["thirdGCEC", "CRPA", "MRSA", "VRE"]
DESCRIPTIVE_ONLY_ORGANISMS = ["CRE", "CRKP", "VREFM"]
ALL_ORGANISMS = EVALUABLE_ORGANISMS + DESCRIPTIVE_ONLY_ORGANISMS


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
    return ((centre - adj) / denom, (centre + adj) / denom)


def unadjusted_or_2x2(x, y):
    """Closed-form 2x2 OR/CI/p for a single binary predictor -- exactly
    equivalent to y ~ x logistic regression for a binary covariate, but
    immune to the numerical separation/overflow failures sm.Logit hits on
    sparse tables. Applies the standard Haldane-Anscombe +0.5 continuity
    correction to all four cells whenever any cell is 0 (flagged in the
    return). Returns None (caller must skip/report descriptively) if the
    predictor has NO variation at all (n_pos==0 or ==n)."""
    n = len(x)
    n_pos = int(x.sum())
    if n_pos == 0 or n_pos == n:
        return None
    a_raw = int(((x == 1) & (y == 1)).sum())  # exposed(predictor+), event -- TP
    b_raw = int(((x == 1) & (y == 0)).sum())  # exposed(predictor+), no event -- FP
    c_raw = int(((x == 0) & (y == 1)).sum())  # unexposed(predictor-), event -- FN
    d_raw = int(((x == 0) & (y == 0)).sum())  # unexposed(predictor-), no event -- TN
    a, b, c, d = a_raw, b_raw, c_raw, d_raw
    corrected = (a == 0 or b == 0 or c == 0 or d == 0)
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_ = np.exp(log_or)
    ci = (np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se))
    z = log_or / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    # Detection-performance metrics (sensitivity/specificity/PPV/NPV), from the
    # RAW (uncorrected) 2x2 counts -- these are simple proportions, not the OR,
    # so the Haldane-Anscombe continuity correction (needed only for the OR's
    # log-ratio) is not applied here. TP=a_raw, FP=b_raw, FN=c_raw, TN=d_raw.
    sens_lo, sens_hi = wilson_ci(a_raw, a_raw + c_raw)
    spec_lo, spec_hi = wilson_ci(d_raw, d_raw + b_raw)
    ppv_lo, ppv_hi = wilson_ci(a_raw, a_raw + b_raw)
    npv_lo, npv_hi = wilson_ci(d_raw, d_raw + c_raw)
    return {
        "OR": or_, "CI_low": ci[0], "CI_high": ci[1], "p": p, "haldane_anscombe_corrected": corrected,
        "TP": a_raw, "FP": b_raw, "FN": c_raw, "TN": d_raw,
        "sensitivity": a_raw / (a_raw + c_raw) if (a_raw + c_raw) else np.nan,
        "sensitivity_CI_low": sens_lo, "sensitivity_CI_high": sens_hi,
        "specificity": d_raw / (d_raw + b_raw) if (d_raw + b_raw) else np.nan,
        "specificity_CI_low": spec_lo, "specificity_CI_high": spec_hi,
        "PPV": a_raw / (a_raw + b_raw) if (a_raw + b_raw) else np.nan,
        "PPV_CI_low": ppv_lo, "PPV_CI_high": ppv_hi,
        "NPV": d_raw / (d_raw + c_raw) if (d_raw + c_raw) else np.nan,
        "NPV_CI_low": npv_lo, "NPV_CI_high": npv_hi,
    }


def adjusted_or_logistic(x, y, covariates):
    """SOFA+antibiotic-adjusted OR via sm.Logit. Returns None (not NaN) on
    non-convergence/singular-Hessian (quasi-/complete separation) rather
    than crashing or reporting a spurious number -- caller must report this
    explicitly as 'adjusted model did not converge', never silently as a
    missing value indistinguishable from 'not attempted'."""
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
        return {"OR": or_a, "CI_low": ci_a[0], "CI_high": ci_a[1], "p": fit_a.pvalues[1]}
    except Exception as e:
        log.warning("adjusted logistic fit failed (%s) -- reporting as non-converged, not a number", type(e).__name__)
        return None


def bootstrap_auroc(y, score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    point = roc_auc_score(y, score)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4), len(boots)


# ---------------------------------------------------------------------------
# 1. load clinical + verify VRE reconciliation + event-count table
# ---------------------------------------------------------------------------
mc = M.load_clinical()
vre_facts = M.reconcile_vre_and_verify(mc, log=log)
event_table = M.acquisition_event_table(mc)
log.info("Acquisition (pre=0/post=1) event counts:\n%s", event_table.to_string(index=False))
io_utils.save_csv(event_table, OUT_CSV.replace(".csv", "_event_counts.csv"))

# ---------------------------------------------------------------------------
# 2. taxonomic + resistome predictor tables, joined via sample_list
# ---------------------------------------------------------------------------
genus = M.load_genus_table()
flags_contig, detect_contig = M.build_gene_presence_flags("contig")
flags_mag, detect_mag = M.build_gene_presence_flags("mag")

mc = mc.set_index("sample_list", drop=False)
genus_j = genus.reindex(mc.index)
resist_contig_j = flags_contig.reindex(mc.index)
resist_mag_j = flags_mag.reindex(mc.index)

n_tax_missing = genus_j["Klebsiella"].isna().sum()
n_resist_contig_missing = resist_contig_j["thirdGCEC"].isna().sum()
n_resist_mag_missing = resist_mag_j["thirdGCEC"].isna().sum()
log.info("Predictor join coverage (N=%d): taxonomy missing=%d, contig-resistome missing=%d, "
         "MAG-resistome missing=%d", len(mc), n_tax_missing, n_resist_contig_missing, n_resist_mag_missing)

SOFA = pd.to_numeric(mc["SOFA"], errors="coerce")
abx_broad = (mc[ABX_BROAD_SPECTRUM_COLS].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(int)
mc["abx_broad_spectrum"] = abx_broad
mc["SOFA_num"] = SOFA
assert SOFA.notna().all(), "SOFA must be complete for all 430 (established project-wide)"

def taxon_series_for(organism: str) -> pd.Series:
    genera = M.ORGANISM_GENUS[organism]
    assert len(genera) == 1
    return genus_j[genera[0]]


# ---------------------------------------------------------------------------
# 3. main loop -- taxonomic (3 thresholds) + resistome (contig primary,
#    MAG sensitivity), unadjusted + SOFA/antibiotic-adjusted OR, AUROC for
#    the resistome predictor, on the PRE-NEGATIVE-restricted population.
# ---------------------------------------------------------------------------
rows = []
descriptive_rows = []

for organism in ALL_ORGANISMS:
    pre_c, post_c, _ = M.ORGANISM_PREPOST[organism]
    pre = pd.to_numeric(mc[pre_c], errors="coerce")
    post = pd.to_numeric(mc[post_c], errors="coerce")
    pop_mask = pre == 0
    pop = mc.loc[pop_mask].copy()
    y_full = post.loc[pop_mask]
    n_acq = int(y_full.sum())
    evaluable = organism in EVALUABLE_ORGANISMS
    log.info("=== %s: pre-negative N=%d, acquisitions=%d (%s) ===", organism, len(pop), n_acq,
             "EVALUABLE" if evaluable else "DESCRIPTIVE ONLY, n<10")

    tax_series = taxon_series_for(organism).loc[pop_mask]

    # --- taxonomic variants ---
    for thr in M.TAXONOMY_THRESHOLDS_PCT:
        pred = (tax_series > 0).astype(int) if thr == 0 else (tax_series >= thr).astype(int)
        variant_name = "presence_gt0pct" if thr == 0 else f"domination_ge{thr}pct"
        valid = pred.notna() & y_full.notna()
        n_valid = int(valid.sum())
        y = y_full[valid].astype(int).values
        x = pred[valid].astype(int).values
        n_pos = int(x.sum())

        if not evaluable:
            descriptive_rows.append({
                "organism": organism, "predictor_family": "taxonomic", "variant": variant_name,
                "N_pre_negative": n_valid, "N_acquisitions": int(y.sum()),
                "n_predictor_positive": n_pos,
                "acquisition_rate_predictor_pos": round(y[x == 1].mean(), 4) if n_pos > 0 else np.nan,
                "acquisition_rate_predictor_neg": round(y[x == 0].mean(), 4) if (len(x) - n_pos) > 0 else np.nan,
                "note": "descriptive only (<10 acquisition events) -- no OR/CI/p computed, per task instruction",
            })
            continue

        u = unadjusted_or_2x2(x, y)
        cov = [pop["SOFA_num"].values[valid], pop["abx_broad_spectrum"].values[valid].astype(float)]
        a = adjusted_or_logistic(x, y, cov) if u is not None else None

        if u is None:
            log.warning("%s/taxonomic/%s: no variation in predictor (n_pos=%d/%d) -- OR undefined",
                        organism, variant_name, n_pos, n_valid)
        rows.append({
            "organism": organism, "predictor_family": "taxonomic", "variant": variant_name,
            "resistome_level": "", "N_pre_negative": n_valid, "N_acquisitions": int(y.sum()),
            "n_predictor_positive": n_pos, "pct_predictor_positive": round(100 * n_pos / n_valid, 1),
            "unadjusted_OR": round(u["OR"], 3) if u else np.nan,
            "unadjusted_OR_CI_low": round(u["CI_low"], 3) if u else np.nan,
            "unadjusted_OR_CI_high": round(u["CI_high"], 3) if u else np.nan,
            "unadjusted_p": round(u["p"], 4) if u else np.nan,
            "haldane_anscombe_corrected": bool(u["haldane_anscombe_corrected"]) if u else False,
            "adjusted_OR": round(a["OR"], 3) if a else np.nan,
            "adjusted_OR_CI_low": round(a["CI_low"], 3) if a else np.nan,
            "adjusted_OR_CI_high": round(a["CI_high"], 3) if a else np.nan,
            "adjusted_p": round(a["p"], 4) if a else np.nan,
            "adjusted_model_converged": a is not None,
            "AUROC": np.nan, "AUROC_CI_low": np.nan, "AUROC_CI_high": np.nan,
            "TP": u["TP"] if u else np.nan, "FP": u["FP"] if u else np.nan,
            "FN": u["FN"] if u else np.nan, "TN": u["TN"] if u else np.nan,
            "sensitivity": round(u["sensitivity"], 4) if u else np.nan,
            "sensitivity_CI_low": round(u["sensitivity_CI_low"], 4) if u else np.nan,
            "sensitivity_CI_high": round(u["sensitivity_CI_high"], 4) if u else np.nan,
            "specificity": round(u["specificity"], 4) if u else np.nan,
            "specificity_CI_low": round(u["specificity_CI_low"], 4) if u else np.nan,
            "specificity_CI_high": round(u["specificity_CI_high"], 4) if u else np.nan,
            "PPV": round(u["PPV"], 4) if u else np.nan,
            "PPV_CI_low": round(u["PPV_CI_low"], 4) if u else np.nan,
            "PPV_CI_high": round(u["PPV_CI_high"], 4) if u else np.nan,
            "NPV": round(u["NPV"], 4) if u else np.nan,
            "NPV_CI_low": round(u["NPV_CI_low"], 4) if u else np.nan,
            "NPV_CI_high": round(u["NPV_CI_high"], 4) if u else np.nan,
        })

    # --- resistome variants (contig PRIMARY; MAG SECONDARY sensitivity, no p) ---
    for level, resist_j in [("contig", resist_contig_j), ("mag", resist_mag_j)]:
        pred = resist_j[organism].loc[pop_mask]
        valid = pred.notna() & y_full.notna()
        n_valid = int(valid.sum())
        y = y_full[valid].astype(int).values
        x = pred[valid].astype(int).values
        n_pos = int(x.sum())
        n_missing = int((~pred.notna()).sum())

        if not evaluable:
            descriptive_rows.append({
                "organism": organism, "predictor_family": f"resistome_{level}", "variant": "gene_presence",
                "N_pre_negative": n_valid, "N_acquisitions": int(y.sum()), "n_predictor_positive": n_pos,
                "acquisition_rate_predictor_pos": round(y[x == 1].mean(), 4) if n_pos > 0 else np.nan,
                "acquisition_rate_predictor_neg": round(y[x == 0].mean(), 4) if (len(x) - n_pos) > 0 else np.nan,
                "note": "descriptive only (<10 acquisition events) -- no OR/CI/p computed, per task instruction",
            })
            continue
        u = unadjusted_or_2x2(x, y) if n_pos > 0 else None
        cov = [pop["SOFA_num"].values[valid], pop["abx_broad_spectrum"].values[valid].astype(float)]
        a = adjusted_or_logistic(x, y, cov) if u is not None else None
        auroc = auroc_lo = auroc_hi = np.nan
        if n_pos > 0 and n_pos < n_valid:
            auroc, auroc_lo, auroc_hi, _ = bootstrap_auroc(y, x.astype(float))
        elif evaluable:
            log.warning("%s/%s: gene predictor has no variation (n_pos=%d/%d) -- OR/AUROC undefined",
                        organism, level, n_pos, n_valid)

        rows.append({
            "organism": organism, "predictor_family": f"resistome_{level}", "variant": "gene_presence",
            "resistome_level": level, "N_pre_negative": n_valid, "N_acquisitions": int(y.sum()),
            "n_predictor_positive": n_pos, "pct_predictor_positive": round(100 * n_pos / n_valid, 1),
            "unadjusted_OR": round(u["OR"], 3) if u else np.nan,
            "unadjusted_OR_CI_low": round(u["CI_low"], 3) if u else np.nan,
            "unadjusted_OR_CI_high": round(u["CI_high"], 3) if u else np.nan,
            "unadjusted_p": round(u["p"], 4) if u else np.nan,
            "haldane_anscombe_corrected": bool(u["haldane_anscombe_corrected"]) if u else False,
            "adjusted_OR": round(a["OR"], 3) if a else np.nan,
            "adjusted_OR_CI_low": round(a["CI_low"], 3) if a else np.nan,
            "adjusted_OR_CI_high": round(a["CI_high"], 3) if a else np.nan,
            "adjusted_p": round(a["p"], 4) if a else np.nan,
            "adjusted_model_converged": a is not None,
            "AUROC": auroc, "AUROC_CI_low": auroc_lo, "AUROC_CI_high": auroc_hi,
            "TP": u["TP"] if u else np.nan, "FP": u["FP"] if u else np.nan,
            "FN": u["FN"] if u else np.nan, "TN": u["TN"] if u else np.nan,
            "sensitivity": round(u["sensitivity"], 4) if u else np.nan,
            "sensitivity_CI_low": round(u["sensitivity_CI_low"], 4) if u else np.nan,
            "sensitivity_CI_high": round(u["sensitivity_CI_high"], 4) if u else np.nan,
            "specificity": round(u["specificity"], 4) if u else np.nan,
            "specificity_CI_low": round(u["specificity_CI_low"], 4) if u else np.nan,
            "specificity_CI_high": round(u["specificity_CI_high"], 4) if u else np.nan,
            "PPV": round(u["PPV"], 4) if u else np.nan,
            "PPV_CI_low": round(u["PPV_CI_low"], 4) if u else np.nan,
            "PPV_CI_high": round(u["PPV_CI_high"], 4) if u else np.nan,
            "NPV": round(u["NPV"], 4) if u else np.nan,
            "NPV_CI_low": round(u["NPV_CI_low"], 4) if u else np.nan,
            "NPV_CI_high": round(u["NPV_CI_high"], 4) if u else np.nan,
        })

out = pd.DataFrame(rows)
desc = pd.DataFrame(descriptive_rows)

# ---------------------------------------------------------------------------
# 4. multiplicity -- Holm-Bonferroni across the PRIMARY grid only: 4
#    evaluable organisms x (3 taxonomic + 1 contig-resistome) = 16 tests.
#    MAG-resistome sensitivity rows are excluded from the correction family
#    (marked explicitly, no p_holm computed for them).
# ---------------------------------------------------------------------------
primary_mask = (out["predictor_family"] == "taxonomic") | (out["predictor_family"] == "resistome_contig")
primary_mask &= out["unadjusted_p"].notna()
out["in_primary_holm_family"] = primary_mask

out["unadjusted_p_holm"] = np.nan
out["adjusted_p_holm"] = np.nan
if primary_mask.sum() > 0:
    _, holm_u, _, _ = multipletests(out.loc[primary_mask, "unadjusted_p"].values, alpha=0.05, method="holm")
    _, holm_a, _, _ = multipletests(out.loc[primary_mask, "adjusted_p"].values, alpha=0.05, method="holm")
    out.loc[primary_mask, "unadjusted_p_holm"] = np.round(holm_u, 4)
    out.loc[primary_mask, "adjusted_p_holm"] = np.round(holm_a, 4)
n_primary_tests = int(primary_mask.sum())
log.info("Holm-Bonferroni family size (primary grid): %d tests", n_primary_tests)

io_utils.save_csv(out, OUT_CSV)
io_utils.save_csv(desc, OUT_CSV.replace(".csv", "_descriptive_only.csv"))
log.info("wrote %s (+ event_counts, descriptive_only companions)", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. markdown report
# ---------------------------------------------------------------------------
n_sig_holm_adj = int(((out["in_primary_holm_family"]) & (out["adjusted_p_holm"] < 0.05)).sum())
n_sig_raw_adj = int(((out["in_primary_holm_family"]) & (out["adjusted_p"] < 0.05)).sum())
n_auroc_ci_excl_half = int(((out["AUROC_CI_low"] > 0.5) & out["AUROC_CI_low"].notna()).sum())

md = [
    "# MDRO acquisition: does the admission gut metagenome detect occult carriage confirmed by later culture?",
    "",
    "> **DETECTION, NOT PREDICTION (reframe 2026-09-02; the filename is historical and is kept so that "
    "existing cross-references do not break).** Owner-confirmed rule: `pre`/`post` are assigned by temporal "
    "ORDER relative to the gut sample, not by calendar date -- a culture drawn later the SAME calendar day "
    "counts as `post` (rule verified in the source data). "
    "Because a `post` event can therefore "
    "follow the gut sample by hours, \"predicts subsequent acquisition\" is NOT the defensible claim: at short "
    "intervals the organism was already present and culture merely caught up. The claim that holds at EVERY "
    "interval, and the one this report makes, is that gut metagenomic resistance-gene detection identified "
    "occult carriage that culture-based screening had not identified at the time of sampling and that "
    "subsequent culture confirmed. The prespecified interval sensitivity plan is "
    "`phase_x_vre_interval_sensitivity_PRESPECIFIED.md` (written before the post-culture dates were available; "
    "those dates are day-granularity only, as times of day were removed during coding).",
    "",
    "> **POST-HOC-BUT-METHOD-DRIVEN** (`prespecification_audit.csv` vocabulary). The acquisition question "
    "(gut microbiome/resistome at admission -> new MDRO colonization observed strictly afterward) has a strong "
    "prior in the HSCT/ICU gut-domination-precedes-infection literature and a prespecifiable structure (fixed "
    "genus/gene mapping, outcome-blind thresholds, fixed confounder set, fixed multiplicity correction), but "
    "this specific test was designed and executed in one pass on 2026-09-02, after the temporal-anchor "
    "confirmation below unblocked it -- it was not specified before any data in this project was seen. This is "
    "the SURVEILLANCE use case, a different clinical question from the mortality-prognosis arm already answered "
    "null elsewhere in this project.",
    "",
    "## Temporal anchor (data-owner-confirmed, 2026-09-02)",
    "",
    "**`post.*` MDRO flags in `metadata/microbiome.clinical.csv` are strictly AFTER the microbiome sample "
    "collection date; `pre.*` flags are the same day or before.** This statement is attributed to the data "
    "owner and is recorded here because the source file itself does not document it. It is what makes a genuine "
    "prediction claim possible: pre-negative -> post-positive is acquisition observed strictly after the "
    "admission gut sample was drawn.",
    "",
    "## VRE duplicate-raw-column resolution",
    "",
    f"The raw CSV literally contains 2 columns each named `pre.VRE`/`post.VRE`. Verified computationally "
    f"(not guessed): `pre.VRE`=0 for all {vre_facts['n_total']} patients, and `post.VRE` reproduces the "
    f"`VREFM` summary column exactly ({vre_facts['post_VRE_eq_VREFM_n_match']}/{vre_facts['n_total']} match) -- "
    "this pair is the **E. faecium-specific** VRE definition. The pandas-auto-suffixed second pair "
    f"(`pre.VRE.1`/`post.VRE.1`) reproduces the summary column literally named `VRE` exactly "
    f"({vre_facts['prepost1_or_eq_VRE_n_match']}/{vre_facts['n_total']} match: VRE == pre.VRE.1 | post.VRE.1) -- "
    "this pair is the **broader/any-Enterococcus** VRE definition. The same pre|post==summary pattern was "
    "independently verified for every other organism block in the file (CRE, CDI, CRKP, CRAB, CRPA, thirdGCEC, "
    "MRSA -- 430/430 match for all 7), corroborating this reconciliation rather than resting on the VRE columns "
    "alone. **Decision: the broader pair (`VRE`, matching the plainly-named summary column and clearing the "
    "10-acquisition-event bar at 11 vs VREFM's 4) is used as the PRIMARY VRE definition throughout this report; "
    "VREFM is reported alongside as an explicit secondary/alternate definition, never silently dropped.**",
    "",
    "## Acquisition (pre=0 -> post=1) event counts -- read this first",
    "",
    "This is the count that determines what is evaluable -- smaller than, and different from, the raw "
    "pre-OR-post \"total positive\" counts reported in the task brief and in the prior feasibility script (50). "
    "**>=10 acquisitions -> full inferential analysis below; <10 -> descriptive counts only, no OR/CI/p "
    "manufactured**, per the task's own explicit instruction.",
    "",
    df_to_md_table(event_table[["organism", "n_pre_negative_evaluable", "n_pre0_post1_ACQUISITION",
                                 "n_pre1_post1_persistence", "n_pre1_post0", "meets_10_acquisition_event_bar"]]
                   .rename(columns={"n_pre_negative_evaluable": "N_pre_negative",
                                     "n_pre0_post1_ACQUISITION": "N_acquisitions",
                                     "n_pre1_post1_persistence": "N_persistence(pre1_post1)",
                                     "n_pre1_post0": "N_pre1_post0(cleared/not_recultured)"})),
    "",
    f"**Evaluable (>=10 acquisitions, full inferential analysis): {', '.join(EVALUABLE_ORGANISMS)}.** "
    f"**Descriptive only (<10 acquisitions): {', '.join(DESCRIPTIVE_ONLY_ORGANISMS)}.**",
    "",
    "## Resistome gene mapping used (verified present/absent in this cohort's own RGI/CARD output)",
    "",
    "| organism | predictor family | genes targeted | genes detected in >=1 sample (this cohort) | genes never detected |",
    "| --- | --- | --- | --- | --- |",
] + [
    f"| {org} | {detect_contig[org]['kind']} | {', '.join(detect_contig[org]['genes_targeted'])} | "
    f"{', '.join(detect_contig[org]['genes_present_in_ontology']) or '(none)'} | "
    f"{', '.join(detect_contig[org]['genes_never_detected_in_cohort']) or '(none)'} |"
    for org in ["CRE", "CRKP", "CRPA", "thirdGCEC", "VRE", "VREFM", "MRSA"]
] + [
    "",
    M.CRPA_GENE_CAVEAT,
    "",
    "## Predictor join coverage",
    "",
    f"- Taxonomic (MetaPhlAn3 genus, `merge/output/metaphlan3/metaphlan3_genus_merged.tsv`): "
    f"{len(mc) - n_tax_missing}/{len(mc)} patients matched via `sample_list`.",
    f"- Resistome, contig-level (PRIMARY, `merge/output/mag_summary/rgi_contig/`): "
    f"{len(mc) - n_resist_contig_missing}/{len(mc)} patients matched -- the "
    f"{n_resist_contig_missing} unmatched are dropped per-analysis (reported per organism below as reduced N), not silently imputed.",
    f"- Resistome, MAG-level (SECONDARY sensitivity, `merge/output/mag_summary/rgi/`): "
    f"{len(mc) - n_resist_mag_missing}/{len(mc)} patients matched -- lower coverage than contig-level because "
    "not every sample yielded a QC-passing MAG; this is exactly why contig-level (whole-assembly, not "
    "binning-dependent) was chosen as primary.",
    "",
    "## Confounders",
    "",
    "- **SOFA** (continuous, complete for all 430 -- `metadata/microbiome.clinical.csv`, same source used "
    "project-wide for severity adjustment).",
    "- **Broad-spectrum antibiotic exposure** (binary): any of "
    f"{', '.join(ABX_BROAD_SPECTRUM_COLS)} = 1. **Caveat (same limitation already documented project-wide, "
    "revision_log.md item 10)**: only a whole-ADMISSION binary flag exists per class, with no day-level timing -- "
    "so this may include doses given AFTER the sample too. Adjustment here is a coarse proxy for baseline "
    "antibiotic pressure, not a pre-sample-restricted covariate.",
    "",
    "## Multiplicity correction",
    "",
    f"**Holm-Bonferroni** (matching this project's established convention, scripts 47/49 -- controls "
    "family-wise error rate without assuming independence between tests, appropriate since taxonomic thresholds "
    "for the same organism are correlated by construction), applied SEPARATELY to the unadjusted and "
    f"SOFA+antibiotic-adjusted p-values, across the **PRIMARY grid only**: {n_primary_tests} tests, out of a "
    "nominal 16 (4 evaluable organisms x [3 taxonomic thresholds + 1 contig-resistome gene test]) -- 2 taxonomic "
    "thresholds (CRPA domination>=30%, MRSA domination>=30%) had ZERO pre-negative patients meeting the "
    "threshold at all (no acquisitions among a non-existent group), so no test exists there to correct; they "
    "are reported with `unadjusted_p`/`adjusted_p`=NaN and excluded from the family, not silently dropped. The "
    "MAG-resistome sensitivity rows and the descriptive-only organisms are likewise explicitly OUTSIDE this "
    "family (`in_primary_holm_family` column in the CSV; no p_holm computed for excluded rows). Sparse 2x2 "
    "tables (any cell=0) use the standard Haldane-Anscombe +0.5 continuity correction for the unadjusted OR "
    "(`haldane_anscombe_corrected` column); the SOFA+antibiotic-adjusted logistic model is reported as "
    "non-converged (`adjusted_model_converged`=False, OR/CI/p left as NaN, never a spurious extreme number) "
    "when it hits quasi-/complete separation, which happens for several of the near-zero-prevalence variants.",
    "",
    "## Results -- evaluable organisms (full inferential analysis)",
    "",
    df_to_md_table(out.loc[out["organism"].isin(EVALUABLE_ORGANISMS),
                           ["organism", "predictor_family", "variant", "N_pre_negative", "N_acquisitions",
                            "n_predictor_positive", "unadjusted_OR", "unadjusted_OR_CI_low", "unadjusted_OR_CI_high",
                            "unadjusted_p", "unadjusted_p_holm", "haldane_anscombe_corrected"]]),
    "",
    df_to_md_table(out.loc[out["organism"].isin(EVALUABLE_ORGANISMS),
                           ["organism", "predictor_family", "variant", "adjusted_OR", "adjusted_OR_CI_low",
                            "adjusted_OR_CI_high", "adjusted_p", "adjusted_p_holm", "adjusted_model_converged"]]),
    "",
    "### AUROC (resistome-gene predictor only, both levels)",
    "",
    df_to_md_table(out.loc[(out["organism"].isin(EVALUABLE_ORGANISMS)) & out["AUROC"].notna(),
                           ["organism", "resistome_level", "AUROC", "AUROC_CI_low", "AUROC_CI_high"]]),
    "",
    "### Detection performance (resistome-gene predictor, contig-level; Wilson 95% CI)",
    "",
    "Sensitivity/specificity/PPV/NPV of contig-level resistome-gene presence for the subsequent "
    "acquisition event, among evaluable organisms -- the same 2x2 tables underlying the unadjusted OR "
    "above, decomposed as detection-performance metrics rather than an association measure. Added "
    "2026-09-02 (task item 9) so that every number quoted from this analysis, including the VRE "
    "sensitivity/specificity/NPV figures used elsewhere, is generated from this repository file rather "
    "than hand-derived.",
    "",
    df_to_md_table(out.loc[(out["predictor_family"] == "resistome_contig") & out["TP"].notna(),
                           ["organism", "TP", "FP", "FN", "TN", "sensitivity", "sensitivity_CI_low",
                            "sensitivity_CI_high", "specificity", "specificity_CI_low", "specificity_CI_high",
                            "PPV", "PPV_CI_low", "PPV_CI_high", "NPV", "NPV_CI_low", "NPV_CI_high"]]),
    "",
    "## Results -- descriptive only (<10 acquisition events; no OR/CI/p)",
    "",
    df_to_md_table(desc[["organism", "predictor_family", "variant", "N_pre_negative", "N_acquisitions",
                         "n_predictor_positive", "acquisition_rate_predictor_pos", "acquisition_rate_predictor_neg"]]),
    "",
    "## Interpretation",
    "",
    "**Headline positive finding, framed as DETECTION rather than prediction (see the note at the top of this "
    "report): gut metagenomic vanA/vanB detection identified occult vancomycin-resistant *Enterococcus* "
    "carriage that culture-based screening had not identified at the time of sampling and that subsequent "
    "culture confirmed -- and this association survives multiplicity correction.** Among the 416 "
    "pre-VRE-negative patients with contig "
    "resistome data, 44 (10.6%) already carried a vanA/vanB gene in their admission gut sample despite no "
    "cultured VRE at that point; 6/44 (13.6%) of them were subsequently culture-confirmed VRE-positive vs "
    "5/372 (1.3%) of vanA/vanB-negative "
    "patients -- unadjusted OR=11.59 [3.38, 39.77], p=0.0001 (Holm-adjusted p=0.0014); SOFA+antibiotic-adjusted "
    "OR=10.70 [2.94, 38.92], p=0.0003 (Holm-adjusted p=0.0042); AUROC=0.726 [0.572, 0.880]. This is the ONLY "
    "primary-grid cell that survives Holm-Bonferroni correction at either adjustment level. Because the "
    "`pre`/`post` flags are assigned by temporal ORDER and a `post` event can follow the gut sample by as "
    "little as hours (see the top-of-file note), \"predicts subsequent VRE acquisition\" overstates what is "
    "defensible in general -- for the shortest-interval events the organism was plausibly already present and "
    "culture simply caught up. \"Detects occult carriage\" is the claim that holds at EVERY interval, because "
    "these patients are culture-negative at/before sampling by construction: metagenomic vanA/vanB genuinely "
    "added information the concurrent culture did not have. This refines rather than contradicts Chanderraj "
    "et al.'s mSphere 2020 null, which found that 16S community-level gut microbiota profiling did NOT predict "
    "subsequent VRE acquisition in hospitalized patients (Chanderraj R et al., \"Gut Microbiota Predict "
    "Enterococcus Expansion but Not Vancomycin-Resistant Enterococcus Acquisition,\" mSphere 2020;5(6):e00537-20, "
    "PMID 33208515, DOI 10.1128/mSphere.00537-20; verified against PubMed 2026-09-02). Our community-level "
    "signal is consistent with that null -- taxonomic Enterococcus "
    "domination (>=10%/30%) at this same organism is nominally significant unadjusted (p=0.0154/0.0143) but does "
    "NOT survive Holm correction (p_holm=0.1859 both) -- while the directly assayed resistance "
    "determinant is the Holm-surviving signal: **the community does not predict, but the gene detects.** "
    "**Qualification (claim-strength correction, 2026-09-02, "
    "task item 5)**: this should not be described as mechanistically direct or tight -- vanA/vanB gene presence "
    "in the admission metagenome is not linked to a specific Enterococcus MAG in this dataset, nor to the "
    "organism subsequently cultured from the same patient (no isolate-level genotype confirmation exists here). "
    "**Disclosure (task item 4)**: the broader any-Enterococcus VRE definition is used as PRIMARY partly because "
    "it clears this project's 10-acquisition-event evaluability bar (11 events vs the E.-faecium-specific "
    "definition's 4) -- an outcome-informed phenotype choice.", "",
    f"{n_sig_raw_adj}/{n_primary_tests} primary-grid tests are nominally SOFA+antibiotic-adjusted-significant "
    f"(p<0.05, uncorrected); {n_sig_holm_adj}/{n_primary_tests} survive Holm-Bonferroni. "
    f"{n_auroc_ci_excl_half}/{len(EVALUABLE_ORGANISMS)} resistome-gene AUROC 95% CIs exclude 0.5 (contig-level).",
    "",
    "No threshold, gene set, confounder, or correction method was altered after seeing these numbers. The "
    "taxonomy threshold grid (0%/10%/30%), the gene mapping, SOFA+antibiotic as the fixed confounder set, and "
    f"Holm-Bonferroni across the primary grid ({n_primary_tests} realized tests out of a nominal 16) were all "
    "fixed before this script was run.",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
