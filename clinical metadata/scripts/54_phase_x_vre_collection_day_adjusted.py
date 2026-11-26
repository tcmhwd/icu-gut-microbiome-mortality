#!/usr/bin/env python3
"""54_phase_x_vre_collection_day_adjusted.py

ITEM 0 of the Paper-1 figure/table build (user request, 2026-09-02): script
53's Part B found that BOTH vanA/vanB gene-positivity (47.7% day 0-3 vs
71.6% baseline, Fisher p=0.0006) and VRE acquisition (27.3% vs 71.6%
baseline, Fisher p=0.0026) are significantly concentrated in later-sampled
patients. Sample-collection day (day-from-admission) is therefore an
UNCONTROLLED COMMON CAUSE of both the predictor and the outcome: longer ICU
stay -> more antibiotic pressure -> both more gene carriage and more
acquisition opportunity. Script 51/53's whole-admission binary antibiotic
covariate cannot control for this within-admission timing gradient.

THIS SCRIPT DOES NOT MODIFY SCRIPTS 51 OR 53. It reuses their data-loading
(`utils/mdro_data.py`), outcome-blind VRE resistome predictor definition
(contig-level RGI/CARD vanA/vanB gene presence, `M.build_gene_presence_flags
("contig")["VRE"]`), and helper functions (Wilson CI, closed-form 2x2 OR,
unpenalized `sm.Logit` adjusted OR, Firth-via-Rscript runner) duplicated
here verbatim, matching the established 51/52/53 convention of small helper
duplication over cross-imports.

WHAT'S NEW: `day_from_admission` (continuous, `collection_date` minus
`admission_date`, exactly as computed in script 53 Part B) is added as a
THIRD covariate alongside SOFA (+/- antibiotic), on the full 416-patient
(SOFA-complete) joined population -- NOT restricted to the day 0-3 landmark
subset, since collection day is now itself a covariate rather than a
restriction criterion (restricting AND adjusting for the same axis would be
redundant/circular).

Two models reported, matching script 53 Part D's with/without-antibiotic
structure:
  M1: VRE_gene + SOFA + antibiotic_broad + collection_day  (4 variables)
  M2: VRE_gene + SOFA + collection_day                     (3 variables, no abx)

Both Firth-penalized (primary, appropriate estimator: EPV = 11/4 = 2.75 for
M1, 11/3 = 3.67 for M2 -- far below the EPV>=10 convention) AND unpenalized
`sm.Logit` (reported alongside, explicitly labelled UNRELIABLE given EPV,
never treated as the primary estimate). Reported side-by-side with script
53's existing full-population estimates (no collection-day covariate) so
the shift attributable to adding this covariate is directly visible.

HONEST FRAMING: this tests whether the vanA/vanB -> VRE-acquisition
association survives controlling for the sampling-day confound identified
in script 53 Part B. It is reported whichever way it comes out -- this
script does not tune the model or covariate set after seeing the result.

Reads:
  metadata/microbiome.clinical.csv (pre/post VRE flags, SOFA, antibiotics,
    collection_date, admission_date) via utils/mdro_data.py
  merge/output/mag_summary/rgi_contig/*_per_sample.tsv (resistome, PRIMARY)
Writes:
  clinical_ai/reports/phase_x_vre_collection_day_adjusted.csv
  clinical_ai/reports/phase_x_vre_collection_day_adjusted.md
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, mdro_data as M  # noqa: E402

log = io_utils.setup_logging("54_phase_x_vre_collection_day_adjusted")

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_vre_collection_day_adjusted.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_vre_collection_day_adjusted.md")
R_FIRTH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "run_firth_logistf.R")
SCRATCH_DIR = ("/tmp/claude-1003/-home-tcmhwd-microbiome-ICU-public/"
               "a6592c63-91ba-4843-a83a-a75feeb8cd84/scratchpad")
SENSITIVITY_CSV = os.path.join(config.DIR_REPORTS, "phase_x_vre_sensitivity_landmark_firth.csv")

# Duplicated from scripts 51/53 verbatim -- same fixed set, do not edit
# independently of those scripts' own copies.
ABX_BROAD_SPECTRUM_COLS = ["Carbapenem", "Glycopeptides", "cephems_3rd", "cephems_4th",
                            "BLBLI", "CEF_AVI", "CEF_TAZ"]


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def unadjusted_or_2x2(x, y):
    """Copied from scripts 51/53 verbatim. Closed-form 2x2 OR/CI/p,
    Haldane-Anscombe +0.5 continuity correction on any zero cell."""
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
    return {"OR": or_, "CI_low": ci[0], "CI_high": ci[1], "p": p, "haldane_anscombe_corrected": corrected}


def adjusted_or_logistic(x, y, covariates):
    """Copied from scripts 51/53 verbatim. Unpenalized sm.Logit adjusted OR;
    None on non-convergence/singular-Hessian. LABEL AS UNRELIABLE under low
    EPV by the caller -- this function itself does not gate on EPV."""
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


def run_firth_logistf(y, predictors: dict, log=None):
    """Copied from script 53 verbatim (see that script's docstring for the
    full contract). `predictors`: ordered dict name->array, first entry is
    the headline term."""
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
# 1. load clinical + VRE gene predictor + collection day, exactly as script
#    51/53's VRE branch, on the full (non-landmark-restricted) population
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
collection_date = pd.to_datetime(mc["collection_date"], errors="coerce")
admission_date = pd.to_datetime(mc["admission_date"], errors="coerce")
day_from_admission = (collection_date - admission_date).dt.days.astype(float)

pop_mask = pre == 0  # pre-VRE-negative, outcome-blind population gate (identical to scripts 51/53)
valid = pop_mask & gene.notna() & post.notna() & SOFA.notna() & day_from_admission.notna()

n_pop_raw = int(pop_mask.sum())
n_pop_valid = int(valid.sum())
log.info("Full population: pre-negative=%d, joined/SOFA/collection-day-complete=%d", n_pop_raw, n_pop_valid)

x = gene[valid].astype(int).values
y = post[valid].astype(int).values
sofa = SOFA[valid].values
abx = abx_broad[valid].values
day = day_from_admission[valid].values
n_genepos = int(x.sum())
n_acq = int(y.sum())

log.info("Sanity check vs script 53 baseline (N=416, gene+=44, acq=11 expected; this script requires "
         "collection_date/admission_date non-missing in addition, so N may be <=416): N=%d, gene+=%d, acq=%d",
         n_pop_valid, n_genepos, n_acq)
if n_pop_valid != 416:
    log.warning("N=%d differs from script 53's reported N=416 -- likely a small number of patients missing "
                "collection_date/admission_date not otherwise excluded by script 53's SOFA-complete gate. "
                "Reporting the number this script actually fits on, not manufacturing 416.", n_pop_valid)

epv_M1 = n_acq / 4  # VRE_gene, SOFA, abx, collection_day
epv_M2 = n_acq / 3  # VRE_gene, SOFA, collection_day

# ---------------------------------------------------------------------------
# 2. Fit M1 (with abx) and M2 (without abx), Firth (primary) + unpenalized
#    (secondary, labelled unreliable)
# ---------------------------------------------------------------------------
report_rows = []


def add_row(**kw):
    report_rows.append(kw)


firth_M1 = run_firth_logistf(y, {"VRE_gene": x, "SOFA": sofa, "abx_broad": abx, "collection_day": day}, log=log)
firth_M2 = run_firth_logistf(y, {"VRE_gene": x, "SOFA": sofa, "collection_day": day}, log=log)
unpen_M1 = adjusted_or_logistic(x, y, [sofa, abx, day])
unpen_M2 = adjusted_or_logistic(x, y, [sofa, day])


def _firth_row(model_label, firth_result, epv, notes_extra=""):
    if firth_result is None or "VRE_gene" not in firth_result:
        add_row(model=model_label, estimator="Firth (logistf)", N=n_pop_valid, N_events=n_acq,
                N_predictor_pos=n_genepos, EPV=round(epv, 3),
                OR=np.nan, CI_low=np.nan, CI_high=np.nan, p=np.nan, converged=False,
                notes="FIRTH_FIT_FAILED -- R logistf did not return a usable fit " + notes_extra)
        return
    t = firth_result["VRE_gene"]
    conv = firth_result.get("_conv")
    conv_str = f"conv(LLchange={conv[0]:.2e},maxscore={conv[1]:.2e},betachange={conv[2]:.2e})" if conv else "conv=NA"
    add_row(model=model_label, estimator="Firth (logistf)", N=n_pop_valid, N_events=n_acq,
            N_predictor_pos=n_genepos, EPV=round(epv, 3),
            OR=round(t["OR"], 3), CI_low=round(t["CI_low"], 3), CI_high=round(t["CI_high"], 3),
            p=round(t["p"], 4), converged=firth_result["_converged"],
            notes=(conv_str + (" " + notes_extra if notes_extra else "")))


def _unpen_row(model_label, unpen_result, epv, notes_extra=""):
    add_row(model=model_label, estimator="unpenalized (sm.Logit) -- UNRELIABLE at this EPV, secondary only",
            N=n_pop_valid, N_events=n_acq, N_predictor_pos=n_genepos, EPV=round(epv, 3),
            OR=round(unpen_result["OR"], 3) if unpen_result else np.nan,
            CI_low=round(unpen_result["CI_low"], 3) if unpen_result else np.nan,
            CI_high=round(unpen_result["CI_high"], 3) if unpen_result else np.nan,
            p=round(unpen_result["p"], 4) if unpen_result else np.nan,
            converged=unpen_result is not None, notes=notes_extra)


_firth_row("M1_gene+SOFA+abx+collection_day", firth_M1, epv_M1, notes_extra="4 variables, primary estimator")
_unpen_row("M1_gene+SOFA+abx+collection_day", unpen_M1, epv_M1, notes_extra="4 variables, secondary/UNRELIABLE")
_firth_row("M2_gene+SOFA+collection_day (no abx)", firth_M2, epv_M2, notes_extra="3 variables, primary estimator")
_unpen_row("M2_gene+SOFA+collection_day (no abx)", unpen_M2, epv_M2, notes_extra="3 variables, secondary/UNRELIABLE")

out = pd.DataFrame(report_rows)
io_utils.save_csv(out, OUT_CSV)
log.info("wrote %s", OUT_CSV)
log.info("\n%s", out.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Pull script 53's existing full-population estimates (no collection-day
#    covariate) for side-by-side comparison -- read from file, not hardcoded.
# ---------------------------------------------------------------------------
s53 = pd.read_csv(SENSITIVITY_CSV)
s53_full_with_abx_unpen = s53[(s53["part"] == "baseline_full_reproduction") &
                               (s53["model"].astype(str).str.startswith("adjusted_SOFA_abx"))]
s53_full_without_abx_unpen = s53[(s53["part"] == "part_D_with_without_abx") &
                                  (s53["population"].astype(str).str.startswith("full_426")) &
                                  (s53["model"].astype(str).str.contains("SOFA_only"))]
s53_full_with_abx_firth = s53[(s53["part"] == "part_C_firth") &
                               (s53["population"].astype(str) == "full_426(_416_joined)")]
s53_full_without_abx_firth = s53[(s53["part"] == "part_D_with_without_abx") &
                                  (s53["population"].astype(str) == "full_426(_416_joined)") &
                                  (s53["model"].astype(str) == "Firth (logistf)")]

comparison = pd.concat([
    s53_full_with_abx_unpen.assign(source="script53_no_collection_day", model_group="with_abx", estimator="unpenalized"),
    s53_full_with_abx_firth.assign(source="script53_no_collection_day", model_group="with_abx", estimator="Firth"),
    s53_full_without_abx_unpen.assign(source="script53_no_collection_day", model_group="without_abx", estimator="unpenalized"),
    s53_full_without_abx_firth.assign(source="script53_no_collection_day", model_group="without_abx", estimator="Firth"),
], ignore_index=True)[["source", "model_group", "estimator", "N", "N_events", "OR", "CI_low", "CI_high", "p", "converged"]]

this_script = pd.DataFrame([
    {"source": "script54_WITH_collection_day", "model_group": "with_abx", "estimator": "Firth",
     "N": n_pop_valid, "N_events": n_acq,
     "OR": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth")), "OR"].values[0],
     "CI_low": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth")), "CI_low"].values[0],
     "CI_high": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth")), "CI_high"].values[0],
     "p": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth")), "p"].values[0],
     "converged": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth")), "converged"].values[0]},
    {"source": "script54_WITH_collection_day", "model_group": "with_abx", "estimator": "unpenalized",
     "N": n_pop_valid, "N_events": n_acq,
     "OR": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("unpenalized")), "OR"].values[0],
     "CI_low": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("unpenalized")), "CI_low"].values[0],
     "CI_high": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("unpenalized")), "CI_high"].values[0],
     "p": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("unpenalized")), "p"].values[0],
     "converged": out.loc[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("unpenalized")), "converged"].values[0]},
    {"source": "script54_WITH_collection_day", "model_group": "without_abx", "estimator": "Firth",
     "N": n_pop_valid, "N_events": n_acq,
     "OR": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("Firth")), "OR"].values[0],
     "CI_low": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("Firth")), "CI_low"].values[0],
     "CI_high": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("Firth")), "CI_high"].values[0],
     "p": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("Firth")), "p"].values[0],
     "converged": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("Firth")), "converged"].values[0]},
    {"source": "script54_WITH_collection_day", "model_group": "without_abx", "estimator": "unpenalized",
     "N": n_pop_valid, "N_events": n_acq,
     "OR": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("unpenalized")), "OR"].values[0],
     "CI_low": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("unpenalized")), "CI_low"].values[0],
     "CI_high": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("unpenalized")), "CI_high"].values[0],
     "p": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("unpenalized")), "p"].values[0],
     "converged": out.loc[(out["model"].str.startswith("M2")) & (out["estimator"].str.startswith("unpenalized")), "converged"].values[0]},
])

comparison_full = pd.concat([comparison, this_script], ignore_index=True)
comparison_csv = OUT_CSV.replace(".csv", "_vs_script53.csv")
io_utils.save_csv(comparison_full, comparison_csv)
log.info("wrote %s", comparison_csv)

# ---------------------------------------------------------------------------
# 4. Verdict -- does the CI (Firth, primary estimator, with-abx model,
#    matching script 51/53's headline configuration) exclude 1?
# ---------------------------------------------------------------------------
firth_M1_row = out[(out["model"].str.startswith("M1")) & (out["estimator"].str.startswith("Firth"))].iloc[0]
survives = bool(firth_M1_row["converged"]) and firth_M1_row["CI_low"] > 1.0 if pd.notna(firth_M1_row["CI_low"]) else False
verdict = ("SURVIVES: the CI excludes 1 after adding collection-day as a continuous covariate to the "
           "SOFA+antibiotic-adjusted Firth model." if survives else
           "DOES NOT SURVIVE (as tested here): the CI no longer excludes 1 (or the fit did not converge) once "
           "collection-day is added as a continuous covariate to the SOFA+antibiotic-adjusted Firth model.")
log.info("VERDICT: %s", verdict)

# ---------------------------------------------------------------------------
# 5. markdown report
# ---------------------------------------------------------------------------
md = [
    "# VRE resistome -> acquisition finding: collection-day-adjusted sensitivity (Item 0)",
    "",
    f"> ## VERDICT: {verdict}",
    "",
    "> **Item 0 of the Paper-1 figure/table build** (user request, 2026-09-02). Script 53 Part B found that both "
    "vanA/vanB gene-positivity and VRE acquisition are significantly concentrated in later-sampled patients "
    "(`clinical_ai/reports/phase_x_vre_sensitivity_landmark_firth.md`), making sample-collection day an "
    "uncontrolled common cause: longer ICU stay -> more antibiotic pressure -> both more gene carriage and more "
    "acquisition opportunity. The existing whole-admission binary antibiotic covariate (scripts 51/53) cannot "
    "control for this within-admission timing gradient. This script adds `day_from_admission` (continuous, "
    "`collection_date` minus `admission_date`, identical definition to script 53 Part B) as an explicit covariate "
    "and reports whether the association survives. Neither scripts 51 nor 53 are modified.",
    "",
    "## Population",
    "",
    f"Full (non-landmark-restricted) 426-patient pre-VRE-negative population, joined with SOFA and "
    f"collection-day complete: **N={n_pop_valid}**, vanA/vanB gene-positive={n_genepos}, "
    f"acquisitions={n_acq}. (Script 53's reported baseline is N=416; a small further drop here, if any, reflects "
    "patients additionally missing `collection_date`/`admission_date` -- see log for the exact number actually "
    "fit, not assumed to be 416.)",
    "",
    "## EPV (events-per-variable)",
    "",
    f"- M1 (gene + SOFA + antibiotic + collection_day, 4 variables): EPV = {n_acq}/4 = **{epv_M1:.2f}**",
    f"- M2 (gene + SOFA + collection_day, no antibiotic, 3 variables): EPV = {n_acq}/3 = **{epv_M2:.2f}**",
    "",
    "Both are far below the conventional EPV>=10 threshold for unpenalized maximum-likelihood logistic "
    "regression. **Firth-penalized regression (R `logistf`) is the appropriate/primary estimator here; the "
    "unpenalized `sm.Logit` estimates are reported alongside for transparency only and should be treated as "
    "unreliable (plausibly inflated) at this EPV, not as the primary result.**",
    "",
    "## Results: collection-day-adjusted models",
    "",
    df_to_md_table(out[["model", "estimator", "N", "N_events", "N_predictor_pos", "EPV", "OR", "CI_low", "CI_high",
                        "p", "converged"]]),
    "",
    "## Side-by-side: script 53 (no collection-day covariate) vs this script (WITH collection-day covariate)",
    "",
    df_to_md_table(comparison_full[["source", "model_group", "estimator", "N", "N_events", "OR", "CI_low", "CI_high",
                                     "p", "converged"]]),
    "",
    "## Interpretation",
    "",
    f"{verdict} Compare the Firth `with_abx` row across the two sources above to see the shift attributable "
    "specifically to adding the collection-day covariate (all other model terms and the underlying population "
    "definition are held identical to script 53's full-population, non-landmark-restricted analysis). As with "
    "script 53, this is reported whichever way it comes out -- no covariate, threshold, or gene definition was "
    "chosen or altered after seeing this result; SOFA, antibiotic-exposure, and collection-day were all "
    "pre-specified as the confound-control set given script 53's own Part B finding.",
    "",
    "## Caveats (inherited from scripts 51/53, unchanged)",
    "",
    "- The antibiotic-exposure covariate remains a whole-admission binary flag with no day-level timing "
    "(`revision_log.md` item 10, script 51's own docstring) -- it may include doses given after the gut sample.",
    "- `day_from_admission` itself is measured with a small number of negative-gap outliers (4 of 430 patients "
    "project-wide, kept by prior explicit user decision, see `ICU_public/CLAUDE.md`) but this does not change "
    "materially at N>400.",
    "- EPV=2.75 (M1) / 3.67 (M2) means even the Firth-penalized OR magnitude should be read with substantial "
    "caution; Firth improves small-sample bias but does not manufacture power the data do not have.",
    "",
]

with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
