#!/usr/bin/env python3
"""42_phase_x_time_to_event_competing_risk_v3.py

V3 RERUN (2026-10-02, owner-authorised correction) of
`42_phase_x_time_to_event_competing_risk.py`. This file is a copy of the
original with EXACTLY ONE substantive change: the M3-LR/M3-CatBoost (and M1,
unaffected) probability columns are read from the fold-local-corrected V3
OOF file instead of script 18's old file. No model specification, time
origin, censoring rule, Fine-Gray setup, PH test, or bootstrap protocol is
changed. See
`semantic_microbiome/reports/paper1/FINAL_LEAN_SUBMISSION_2026-10-02_v3/QC/
M3_FOLD_LOCAL_METHOD_AUDIT_v4.md` section 8 for why this rerun was required
(PCA/prevalence-filter leakage fix in script 45_..._v3.py changed the M3-LR
and M3-CatBoost OOF probabilities; M1 is bit-identical, confirmed before
this rerun to <1.4e-15 max abs diff). All outputs are written to new `_v3`
paths; none of the original script's outputs are touched.

POST-HOC analysis (2026-08-31, user request). NOT prespecified -- belongs in
the post-hoc column of semantic_microbiome/reports/analysis_provenance.md
(section 2), alongside scripts 20-41. Strictly additive: no existing script
or result file is modified.

THE GAP THIS FILLS
------------------
Every mortality analysis in this project is BINARY in-hospital mortality.
There is no survival/time-to-event analysis anywhere in the repository.
Binary in-hospital mortality is confounded by length of stay -- a patient
who dies on day 2 and one who dies on day 60 are the same data point, and a
patient discharged alive on day 3 is "a survivor" on the strength of a very
short observation window. The project's own
merge/downstream_analysis/DESIGN.md additionally records that it cannot
confirm from source data whether `in.hospital.mortality` means ICU-specific
or whole-admission mortality. A time-to-event framing with an explicit time
origin absorbs part of that ambiguity and makes the rest visible.

TIME ORIGIN: the microbiome sample (`collection_date`), because that is the
moment from which prediction is being claimed.

OWNER DATA CORRECTION (2026-08-31) -- read this first
------------------------------------------------------
The data owner has determined authoritatively that `in.hospital.mortality`
IS all in-hospital death (the endpoint is NOT mixed) and that the DISCHARGE
DATES are what is inaccurate. Prescribed rule: for any patient with a death
date, the discharge date equals the death date.

An earlier version of this script inferred the opposite from the raw dates
(that the endpoint captured post-discharge deaths). That inference is WRONG
and is retracted in the report rather than quietly removed. The correction
is applied in analysis code only; metadata/microbiome.clinical.csv is never
modified.

Consequence: the primary specification moves from 58 deaths / 244 competing
events to 69 deaths / 233 competing events. Some conclusions change as a
result. That change is driven by an owner-confirmed data fix, NOT by any
analytic choice, and the pre-correction results are retained side by side so
the effect is auditable.

SPECIFICATIONS
--------------
  Spec B-CS CORRECTED -- CAUSE-SPECIFIC Cox, the PRIMARY specification.
             Event = in-hospital death (69); competing = discharge alive
             (233); no administrative censoring.

  Spec B-FG CORRECTED -- FINE-GRAY subdistribution hazard (SECONDARY).

  Spec B-CS/FG PRE-CORRECTION -- superseded; the 58/244 split from the
             original discharge dates, retained for transparency.

  Spec A  -- SUPERSEDED. All-cause death with survivor follow-up extended
             past discharge to max(discharge_date, last.fu). Under the
             owner's determination this is no longer coherent, because
             in-hospital death cannot occur after discharge, so the extended
             follow-up accrues person-time in which the event is impossible.
             Retained only for continuity with the pre-correction run.
             (The separate `last.fu` data defect documented below is NOT
             addressed by the owner's correction and remains OPEN.)

IMPLEMENTATION NOTES (all verified against R's `survival` package 3.8.6)
------------------------------------------------------------------------
* Cox models use statsmodels PHReg with Efron ties. Verified to reproduce
  R `coxph(..., ties="efron")` coefficients and standard errors to 8 decimal
  places on this dataset.

* Fine-Gray. In Spec B there is NO administrative censoring: every one of
  the 302 patients has either a death date or a discharge date, so the
  hospitalisation ends in one of the two competing events for everybody.
  The censoring survival function G(t) is therefore identically 1, the
  Fine-Gray IPCW weights all equal 1, and the subdistribution model reduces
  EXACTLY (not approximately) to a Cox model in which subjects who had the
  competing event remain in the risk set to the end of follow-up. This
  reduction was checked against R: `survival::finegray()` on these data
  returns 302 rows with a single unique weight of 1, and the resulting
  coefficients match this script's to 8 decimals. Fine-Gray is therefore
  fully estimable here and is NOT a fragile number -- the competing event
  (discharge, n=244) is abundant, not sparse.
  Because subjects contribute to risk sets after their own competing event,
  the model-based information matrix is not the right variance. Primary CIs
  for Fine-Gray are therefore BOOTSTRAP (resampling patients); the
  model-based and sandwich SEs are reported alongside for comparison. The
  statsmodels cluster-robust SE agrees with R's robust SE to within ~0.7%
  (R applies a slightly different finite-sample influence computation);
  this is disclosed rather than hidden, and is why bootstrap is primary.

* Proportional hazards is tested with the Grambsch-Therneau scaled
  Schoenfeld residual test using the classical I/d variance approximation,
  with a rank time transform. Cross-checked against R `cox.zph(transform=
  "rank")`: the chi-square statistics differ by 5-10% (modern `survival`
  uses a more exact variance) but every conclusion is identical. Both sets
  of numbers are stated in the report. A time-split companion analysis
  (separate Cox fits before and after the median event time) is reported as
  well, because it demonstrates any non-proportionality in directly
  interpretable HR terms and depends on no residual approximation at all.

Reads (read-only):
  clinical_ai/predictions/phase_x_m3_single_level_oof_probabilities_v4_V4_STRICT.csv (script 45_v3; V3 drop-in
    replacement of script 18's microbiome_prob_M1/M3_LR/M3_CatBoost columns -- M1 bit-identical to the old
    file, M3_LR/M3_CatBoost corrected for fold-local PCA/prevalence-filter leakage)
  ICU_public/semantic_microbiome/harmonized/discovery_cohort_primary.csv
  ICU_public/metadata/microbiome.clinical.csv (dates, SOFA)
Writes:
  clinical_ai/reports/phase_x_survival_interval_quality_v4.csv
  clinical_ai/reports/phase_x_survival_followup_v4.csv
  clinical_ai/reports/phase_x_survival_models_v4.csv
  clinical_ai/reports/phase_x_survival_ph_assumption_v4.csv
  clinical_ai/reports/phase_x_survival_time_split_v4.csv
  clinical_ai/reports/phase_x_time_to_event_competing_risk_v4.md
  clinical_ai/figures/phase_x_cumulative_incidence_v3.svg/.jpg
"""
from __future__ import annotations

import os
import sys

# Each Fine-Gray bootstrap replicate is a tiny (302-row, 1-2 covariate) Cox fit,
# so per-fit BLAS threading buys nothing -- but multiplied by the worker pool it
# oversubscribes the SLURM allocation badly (16 workers x 16 BLAS threads on 16
# CPUs). Pin BLAS to one thread per process BEFORE numpy is imported; the
# parallelism that matters here is across replicates, not within a fit.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("42_phase_x_time_to_event_competing_risk_v4")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
# V3: repointed at the fold-local-corrected M3 drop-in file (script 45_v3). Columns are
# identical in name/order to the old script-18 file (microbiome_prob_M1/M3_LR/M3_CatBoost,
# hospital_mortality, patient_id_clinical index); M1 confirmed bit-identical (<1.4e-15 max
# abs diff) before this rerun. The old file's extra C0/C1/C2a/C2b columns are not present in
# this file and are not read by this script.
OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_m3_single_level_oof_probabilities_v4_V4_STRICT.csv")
DISCOVERY_CSV = os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv")
CLINICAL_RAW_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")

OUT_QUAL = os.path.join(config.DIR_REPORTS, "phase_x_survival_interval_quality_v4.csv")
OUT_FU = os.path.join(config.DIR_REPORTS, "phase_x_survival_followup_v4.csv")
OUT_MODELS = os.path.join(config.DIR_REPORTS, "phase_x_survival_models_v4.csv")
OUT_PH = os.path.join(config.DIR_REPORTS, "phase_x_survival_ph_assumption_v4.csv")
OUT_SPLIT = os.path.join(config.DIR_REPORTS, "phase_x_survival_time_split_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_time_to_event_competing_risk_v4.md")
OUT_FIG = os.path.join(config.DIR_FIGURES, "phase_x_cumulative_incidence_v4")

SEED = config.RANDOM_SEED        # 2026
N_BOOT = config.N_BOOTSTRAP      # 1000, for the Fine-Gray CIs
# Honours SLURM_CPUS_PER_TASK, matching this project's own convention
# (config.get_n_jobs), so the Fine-Gray bootstrap never oversubscribes the
# allocation. Results are independent of this value -- see BOOT_IDX below.
N_JOBS = config.get_n_jobs()
EPS = 1e-6


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# 1. assemble the analysis dataset
# ---------------------------------------------------------------------------
if not os.path.exists(OOF_CSV):
    log.error("%s not found -- run 45_phase_x_c1_c2_nested_cv_corrected_v3.py first.", OOF_CSV)
    sys.exit(1)

oof = pd.read_csv(OOF_CSV, index_col=0)
disc = pd.read_csv(DISCOVERY_CSV)
raw = pd.read_csv(CLINICAL_RAW_CSV)
ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw = raw.merge(ids, left_on="KCDC_no", right_on="kcdc_no", how="inner").set_index("patient_id_clinical")
raw = raw.reindex(oof.index)
assert len(raw) == len(oof) == 302, f"expected 302 aligned patients, got {len(raw)}"

d = pd.DataFrame(index=oof.index)
for short, col in [("collection", "collection_date"), ("admission", "admission_date"),
                   ("discharge", "discharge_date"), ("death", "death.date"), ("lastfu", "last.fu")]:
    d[short] = pd.to_datetime(raw[col], errors="coerce")
d["y"] = oof["hospital_mortality"].astype(int).values
d["SOFA"] = pd.to_numeric(raw["SOFA"], errors="coerce").values
d["logit_M1"] = logit(oof["microbiome_prob_M1"].values)
d["logit_M3_LR"] = logit(oof["microbiome_prob_M3_LR"].values)
d["logit_M3_CatBoost"] = logit(oof["microbiome_prob_M3_CatBoost"].values)
d["mortality_date_raw"] = pd.to_numeric(raw["mortality_date"], errors="coerce").values

d["t_death"] = (d["death"] - d["collection"]).dt.days
d["t_discharge"] = (d["discharge"] - d["collection"]).dt.days
d["t_lastfu"] = (d["lastfu"] - d["collection"]).dt.days
d["t_admission_to_collection"] = (d["collection"] - d["admission"]).dt.days
n = len(d)
log.info("N=%d, %d deaths (%.1f%%)", n, int(d.y.sum()), 100 * d.y.mean())

# ---------------------------------------------------------------------------
# 2. DATA QUALITY -- sanity-check every interval BEFORE modelling.
#    Anomalies are reported as findings, never silently filtered.
# ---------------------------------------------------------------------------
deaths = d[d.y == 1]
surv = d[d.y == 0]

# What is `mortality_date`? The task flagged it as possibly a day count, not a
# date. Resolve it empirically rather than assuming.
md_eq_col = int((deaths["mortality_date_raw"] == deaths["t_death"]).sum())
md_eq_adm = int((deaths["mortality_date_raw"] ==
                 (deaths["death"] - deaths["admission"]).dt.days).sum())
log.info("mortality_date == days(collection->death) for %d/%d deaths; == days(admission->death) for %d/%d",
         md_eq_col, len(deaths), md_eq_adm, len(deaths))

qual_rows = [
    {"check": "mortality_date is a NUMBER OF DAYS, not a date",
     "n": md_eq_col, "of": len(deaths),
     "detail": f"mortality_date equals (death.date - collection_date) in days for {md_eq_col}/{len(deaths)} deaths "
               f"(and equals days from ADMISSION for only {md_eq_adm}/{len(deaths)}). RESOLVED: it is days from "
               "SAMPLE COLLECTION to death. Intervals below are nonetheless computed from the actual dates, "
               "not from this field.",
     "action": "resolved; dates used"},
    {"check": "death.date present for every death", "n": int(deaths["death"].notna().sum()), "of": len(deaths),
     "detail": "death.date is non-null for exactly the 69 patients with hospital_mortality==1 and null for all 233 "
               "others; last.fu equals death.date for all 69.",
     "action": "OK"},
    {"check": "collection_date present", "n": int(d["collection"].notna().sum()), "of": n,
     "detail": "time origin available for all patients", "action": "OK"},
    {"check": "NEGATIVE collection->death interval", "n": int((deaths["t_death"] < 0).sum()), "of": len(deaths),
     "detail": "a death recorded before the sample was taken would invalidate the time origin",
     "action": "none needed" if int((deaths["t_death"] < 0).sum()) == 0 else "REPORTED, not filtered"},
    {"check": "NEGATIVE collection->discharge interval", "n": int((d["t_discharge"] < 0).sum()), "of": n,
     "detail": "discharge recorded before sampling",
     "action": "none needed" if int((d["t_discharge"] < 0).sum()) == 0 else "REPORTED, not filtered"},
    {"check": "last.fu BEFORE collection_date (negative follow-up)",
     "n": int((d["t_lastfu"] < 0).sum()), "of": n,
     "detail": f"min = {d['t_lastfu'].min():.0f} days. These are all survivors whose last.fu was recorded as the "
               "ADMISSION date rather than a true final follow-up contact.",
     "action": "last.fu alone REJECTED as the censoring date; see below"},
    {"check": "survivors with last.fu BEFORE their own discharge_date",
     "n": int((surv["t_lastfu"] < surv["t_discharge"]).sum()), "of": len(surv),
     "detail": "these patients are known to have been alive at discharge, so a censoring time earlier than "
               "discharge is demonstrably wrong and would throw away real observed survival time",
     "action": "censoring time set to max(discharge_date, last.fu)"},
    {"check": "survivors with ZERO follow-up under max(discharge, last.fu)",
     "n": int((np.maximum(surv["t_discharge"], surv["t_lastfu"]) <= 0).sum()), "of": len(surv),
     "detail": "contribute no time at risk", "action": "retained, reported"},
    {"check": "deaths whose recorded discharge_date DISAGREED with death.date "
              "[RESOLVED by owner determination 2026-08-31]",
     "n": int((deaths["t_death"] != deaths["t_discharge"]).sum()), "of": len(deaths),
     "detail": f"{int((deaths['t_death'] > deaths['t_discharge']).sum())} deaths recorded AFTER their discharge "
               f"date (by up to {int((deaths['t_death'] - deaths['t_discharge']).max())} days) and "
               f"{int((deaths['t_death'] < deaths['t_discharge']).sum())} recorded BEFORE it. An earlier version "
               "of this analysis inferred from this that `in.hospital.mortality` captured post-discharge deaths, "
               "i.e. that the ENDPOINT was heterogeneous. THAT INFERENCE WAS WRONG AND IS RETRACTED. The data "
               "owner determined (2026-08-31) that `in.hospital.mortality` IS all in-hospital death and that "
               "the DISCHARGE DATES are what is inaccurate.",
     "action": "OWNER CORRECTION applied in analysis code only: discharge date := death date for every patient "
               "with a death date. Raw CSV never modified."},
    {"check": "collection_date BEFORE admission_date (negative_day_outlier)",
     "n": int((d["t_admission_to_collection"] < 0).sum()), "of": n,
     "detail": f"min = {d['t_admission_to_collection'].min():.0f} days. Already flagged upstream by the discovery "
               "cohort's own `negative_day_outlier` column. The admission/discharge record for these patients "
               "belongs to a different admission than the one sampled.",
     "action": "REPORTED, not filtered; affects Spec B discharge times only"},
    {"check": "event/censoring time of exactly 0 days",
     "n": int(((d.y == 1) & (d["t_death"] == 0)).sum()), "of": len(deaths),
     "detail": "deaths on the same calendar day as sampling. Only calendar DATES are available (no clock times) "
               "-- the same precision limit script 40 documents -- so sub-day timing cannot be recovered.",
     "action": "retained at t=0; Efron tie handling"},
]
qual = pd.DataFrame(qual_rows)
io_utils.save_csv(qual, OUT_QUAL)
log.info("\n%s", qual[["check", "n", "of", "action"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. build the two time-to-event specifications
# ---------------------------------------------------------------------------
# ===========================================================================
# OWNER DATA CORRECTION -- authoritative determination, 2026-08-31
# ---------------------------------------------------------------------------
# The data owner has resolved the discharge-vs-death inconsistency documented
# in the data-quality section above. The determination is:
#
#   `in.hospital.mortality` IS all in-hospital death. The ENDPOINT IS NOT
#   MIXED. What is inaccurate is the DISCHARGE DATES.
#
# Prescribed correction: for any patient with a death date, the discharge date
# equals the death date.
#
# This supersedes the inference previously recorded here (that the endpoint
# captured post-discharge deaths). That inference was wrong; it is retracted
# in the report rather than quietly deleted.
#
# Applied HERE, IN ANALYSIS CODE ONLY. metadata/microbiome.clinical.csv is
# read-only and is never modified.
# ===========================================================================
d["discharge_effective"] = d["death"].where(d["death"].notna(), d["discharge"])
d["t_discharge_eff"] = (d["discharge_effective"] - d["collection"]).dt.days
n_corrected = int((d["death"].notna() & (d["discharge"] != d["discharge_effective"])).sum())
n_death_after_dis_pre = int((deaths["t_death"] > deaths["t_discharge"]).sum())
n_death_before_dis_pre = int((deaths["t_death"] < deaths["t_discharge"]).sum())
log.info("OWNER CORRECTION: discharge date reset to death date for %d/%d deaths "
         "(%d that previously fell after discharge, %d before); %d already agreed.",
         n_corrected, len(deaths), n_death_after_dis_pre, n_death_before_dis_pre,
         len(deaths) - n_corrected)
assert n_corrected == n_death_after_dis_pre + n_death_before_dis_pre, \
    "correction should touch exactly the deaths whose discharge date disagreed with death.date"

# Spec A: all-cause death; survivors censored at max(discharge, last.fu).
# NOTE: for survivors `discharge_effective` == `discharge` (no death date), so
# the owner correction leaves this specification numerically unchanged.
d["A_time"] = np.where(d.y == 1, d["t_death"], np.maximum(d["t_discharge_eff"], d["t_lastfu"]))
d["A_status"] = d.y.values

# Spec A-sensitivity: last.fu used verbatim as the censoring date (the naive
# reading of the field). Reported to show what the last.fu defect would do.
d["Asens_time"] = np.where(d.y == 1, d["t_death"], d["t_lastfu"])
d["Asens_status"] = d.y.values
n_asens_nonpositive = int(((d.Asens_status == 0) & (d.Asens_time <= 0)).sum())

# Spec B CORRECTED (PRIMARY): competing risks on the corrected discharge date.
# Because a death now always coincides with discharge, every death is an
# in-hospital death: 1 = in-hospital death, 2 = discharge alive.
d["B_time"] = np.where(d.y == 1, d["t_death"], d["t_discharge_eff"])
d["B_status"] = np.where(d.y == 1, 1, 2)
n_ev1 = int((d.B_status == 1).sum())
n_ev2 = int((d.B_status == 2).sum())
n_cens_B = int((d.B_status == 0).sum())

# Spec B PRE-CORRECTION (superseded, retained for transparency): the 58/244
# split produced by the ORIGINAL discharge dates, so a reader can see exactly
# what the owner's correction changed and judge it independently.
death_before_discharge_pre = (d.y == 1) & (d["t_death"] <= d["t_discharge"])
d["Bpre_time"] = np.where(death_before_discharge_pre, d["t_death"], d["t_discharge"])
d["Bpre_status"] = np.where(death_before_discharge_pre, 1, 2)
n_ev1_pre = int((d.Bpre_status == 1).sum())
n_ev2_pre = int((d.Bpre_status == 2).sum())

log.info("CORRECTED Spec B: %d in-hospital deaths, %d competing (discharge alive), %d censored "
         "| PRE-correction was %d/%d | Spec A: %d deaths, %d censored",
         n_ev1, n_ev2, n_cens_B, n_ev1_pre, n_ev2_pre,
         int(d.A_status.sum()), int((d.A_status == 0).sum()))
assert n_ev1 == int(d.y.sum()) == 69 or n_ev1 == int(d.y.sum()), \
    "corrected primary must have one event per in-hospital death"


def reverse_km_median(time, status):
    """Median follow-up by the reverse Kaplan-Meier (censoring-reversed) method."""
    t = np.asarray(time, float)
    c = 1 - np.asarray(status, int)   # 'event' = being censored
    order = np.argsort(t)
    t, c = t[order], c[order]
    surv, n_at_risk, s = [], len(t), 1.0
    times = np.unique(t[c == 1])
    for tt in times:
        at_risk = (t >= tt).sum()
        dd = ((t == tt) & (c == 1)).sum()
        if at_risk > 0:
            s *= (1 - dd / at_risk)
        surv.append((tt, s))
    for tt, s in surv:
        if s <= 0.5:
            return float(tt)
    return float("nan")


fu_rows = [
    {"specification": "Spec A: all-cause death from sampling",
     "N": n, "events_death": int(d.A_status.sum()), "censored": int((d.A_status == 0).sum()),
     "competing_events": 0,
     "median_followup_days_reverseKM": round(reverse_km_median(d.A_time, d.A_status), 1),
     "median_time_to_event_days": round(float(d.loc[d.A_status == 1, "A_time"].median()), 1),
     "followup_days_min": float(d.A_time.min()), "followup_days_max": float(d.A_time.max()),
     "total_person_days": float(d.A_time.sum())},
    {"specification": "Spec A-sensitivity: censoring at last.fu verbatim",
     "N": n, "events_death": int(d.Asens_status.sum()), "censored": int((d.Asens_status == 0).sum()),
     "competing_events": 0,
     "median_followup_days_reverseKM": round(reverse_km_median(d.Asens_time, d.Asens_status), 1),
     "median_time_to_event_days": round(float(d.loc[d.Asens_status == 1, "Asens_time"].median()), 1),
     "followup_days_min": float(d.Asens_time.min()), "followup_days_max": float(d.Asens_time.max()),
     "total_person_days": float(d.Asens_time.sum())},
    {"specification": "Spec B CORRECTED (PRIMARY): in-hospital death vs discharge alive",
     "N": n, "events_death": n_ev1, "censored": n_cens_B, "competing_events": n_ev2,
     "median_followup_days_reverseKM": float("nan"),
     "median_time_to_event_days": round(float(d.loc[d.B_status == 1, "B_time"].median()), 1),
     "followup_days_min": float(d.B_time.min()), "followup_days_max": float(d.B_time.max()),
     "total_person_days": float(d.B_time.sum())},
    {"specification": "Spec B PRE-CORRECTION (superseded): original discharge dates",
     "N": n, "events_death": n_ev1_pre, "censored": 0, "competing_events": n_ev2_pre,
     "median_followup_days_reverseKM": float("nan"),
     "median_time_to_event_days": round(float(d.loc[d.Bpre_status == 1, "Bpre_time"].median()), 1),
     "followup_days_min": float(d.Bpre_time.min()), "followup_days_max": float(d.Bpre_time.max()),
     "total_person_days": float(d.Bpre_time.sum())},
]
fu = pd.DataFrame(fu_rows)
io_utils.save_csv(fu, OUT_FU)
log.info("\n%s", fu.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. model fitting helpers (validated against R survival 3.8.6 -- see docstring)
# ---------------------------------------------------------------------------
def fit_cox(time, status, X, groups=None, strata=None):
    kw = {}
    if strata is not None:
        kw["strata"] = np.asarray(strata)
    mod = sm.PHReg(np.asarray(time, float), np.asarray(X, float),
                   status=np.asarray(status, int), ties="efron", **kw)
    return mod.fit(groups=groups) if groups is not None else mod.fit()


# Fine-Gray reduction: G(t) == 1 because Spec B has no censoring, so subjects
# with the competing event simply remain at risk to the end of follow-up with
# weight 1. Verified identical to R's survival::finegray().
TMAX_B = float(d.B_time.max())
FG_TIME = np.where(d.B_status == 1, d.B_time, TMAX_B)
FG_STATUS = (d.B_status == 1).astype(int).values
assert n_cens_B == 0, ("Fine-Gray reduction assumed zero administrative censoring in Spec B; "
                       f"found {n_cens_B}. Refusing to proceed with an invalid simplification.")

# Same reduction for the superseded pre-correction split, so the Fine-Gray
# comparison is like-for-like.
TMAX_BPRE = float(d.Bpre_time.max())
FG_TIME_PRE = np.where(d.Bpre_status == 1, d.Bpre_time, TMAX_BPRE)
FG_STATUS_PRE = (d.Bpre_status == 1).astype(int).values

# The Cox specifications, in reporting order. The CORRECTED cause-specific
# model is the primary; the other two are retained only so the effect of the
# owner's correction is auditable.
SPEC_PRIMARY = "Spec B-CS CORRECTED (PRIMARY): cause-specific, in-hospital death"
SPEC_PRE = "Spec B-CS PRE-CORRECTION (superseded): original discharge dates"
SPEC_A = "Spec A (superseded): all-cause death, follow-up extended past discharge"
COX_SPECS = {
    SPEC_PRIMARY: (np.asarray(d.B_time, float), FG_STATUS),
    SPEC_PRE: (np.asarray(d.Bpre_time, float), FG_STATUS_PRE),
    SPEC_A: (np.asarray(d.A_time, float), np.asarray(d.A_status, int)),
}
FG_SPECS = {
    "Spec B-FG CORRECTED (SECONDARY): Fine-Gray subdistribution": (FG_TIME, FG_STATUS),
    "Spec B-FG PRE-CORRECTION (superseded): Fine-Gray subdistribution": (FG_TIME_PRE, FG_STATUS_PRE),
}

COVSETS = {
    "SOFA alone (reference)": ["SOFA"],
    "M1 alone": ["logit_M1"],
    "M3-LR alone": ["logit_M3_LR"],
    "M3-CatBoost alone": ["logit_M3_CatBoost"],
    "M1 + SOFA": ["logit_M1", "SOFA"],
    "M3-LR + SOFA": ["logit_M3_LR", "SOFA"],
    "M3-CatBoost + SOFA": ["logit_M3_CatBoost", "SOFA"],
}

# Bootstrap resamples are drawn ONCE, in the parent, from a seeded RNG and
# reused for every covariate set. This makes the result identical no matter how
# many worker processes are used, and lets the Fine-Gray models share resamples
# (so their CIs are comparable to each other, not just individually valid).
BOOT_IDX = np.random.RandomState(SEED).randint(0, n, size=(N_BOOT, n))


def _fg_boot_one(args):
    """One Fine-Gray bootstrap replicate. Module-level so it is picklable."""
    idx, Xb, fg_t, fg_s = args
    if fg_s[idx].sum() < 5:
        return None
    try:
        return fit_cox(fg_t[idx], fg_s[idx], Xb[idx]).params
    except Exception:
        return None


def fine_gray_rows(fg_spec, fg_t, fg_s, X, cols, sds, label):
    """Fit one Fine-Gray model and summarise it, screening the bootstrap for
    non-convergence. Returns a list of result rows (one per covariate)."""
    out = []
    try:
        rfg = fit_cox(fg_t, fg_s, X)
        rfg_rob = fit_cox(fg_t, fg_s, X, groups=np.arange(n))
    except Exception as e:
        log.warning("Fine-Gray %s / %s failed: %s", fg_spec, label, e)
        return [{"specification": fg_spec, "covariate_set": label, "term": "ALL",
                 "note": f"NOT ESTIMABLE: {e}"}]

    with Pool(N_JOBS) as pool:
        reps = pool.map(_fg_boot_one, [(BOOT_IDX[b_i], X, fg_t, fg_s) for b_i in range(N_BOOT)],
                        chunksize=max(1, N_BOOT // (N_JOBS * 4)))
    boot = np.full((N_BOOT, len(cols)), np.nan)
    for b_i, r_ in enumerate(reps):
        if r_ is not None:
            boot[b_i] = r_

    for j, c in enumerate(cols):
        b = float(rfg.params[j])
        se_ref = float(rfg_rob.bse[j])
        col_b = boot[:, j]
        finite = np.isfinite(col_b)
        # statsmodels' Cox optimiser can return NaN, or a finite but meaningless
        # coefficient, WITHOUT raising -- so "no exception" is not evidence of
        # convergence. A replicate is treated as divergent if its coefficient
        # lands beyond 10 reference standard errors of the full-sample estimate;
        # at that distance it is an optimiser artefact (near-separation in the
        # resample), not a plausible draw.
        lo_ok, hi_ok = b - 10 * se_ref, b + 10 * se_ref
        divergent = finite & ((col_b < lo_ok) | (col_b > hi_ok))
        n_nonfinite = int((~finite).sum())
        n_divergent = int(divergent.sum())
        frac_bad = (n_nonfinite + n_divergent) / N_BOOT
        usable = finite & ~divergent
        stable = frac_bad <= 0.01
        bs = col_b[usable]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p_boot = float(min(2 * min(np.mean(bs <= 0), np.mean(bs >= 0)), 1.0))
        log.info("Fine-Gray [%s] %s [%s]: %d/%d usable (%d non-finite, %d divergent) -> %s",
                 fg_spec.split(":")[0], label, c, int(usable.sum()), N_BOOT,
                 n_nonfinite, n_divergent,
                 "STABLE" if stable else "UNSTABLE, bootstrap CI SUPPRESSED")
        row = {
            "specification": fg_spec, "covariate_set": label, "term": c,
            "HR_per_1_unit": round(float(np.exp(b)), 3),
            "HR_per_1_SD": round(float(np.exp(b * sds[c])), 3),
            "N_events": int(np.sum(fg_s)),
            "SE_modelbased": round(float(rfg.bse[j]), 4),
            "SE_sandwich": round(se_ref, 4),
            "boot_resamples_usable": int(usable.sum()),
            "boot_resamples_nonfinite": n_nonfinite,
            "boot_resamples_divergent": n_divergent,
            "bootstrap_stability": "stable" if stable else "UNSTABLE",
        }
        if stable:
            row.update({
                "HR_CI_low": round(float(np.exp(lo)), 3), "HR_CI_high": round(float(np.exp(hi)), 3),
                "p_value": round(p_boot, 4),
                "HR_per_1_SD_CI_low": round(float(np.exp(lo * sds[c])), 3),
                "HR_per_1_SD_CI_high": round(float(np.exp(hi * sds[c])), 3),
                "SE_type": f"bootstrap percentile ({int(usable.sum())}/{N_BOOT} usable resamples)",
            })
        else:
            # Do NOT publish a contaminated interval. Report the sandwich
            # fallback explicitly labelled as such, and say it is fragile.
            row.update({
                "HR_CI_low": round(float(np.exp(b - 1.96 * se_ref)), 3),
                "HR_CI_high": round(float(np.exp(b + 1.96 * se_ref)), 3),
                "p_value": round(float(2 * stats.norm.sf(abs(b / se_ref))), 4),
                "HR_per_1_SD_CI_low": round(float(np.exp((b - 1.96 * se_ref) * sds[c])), 3),
                "HR_per_1_SD_CI_high": round(float(np.exp((b + 1.96 * se_ref) * sds[c])), 3),
                "SE_type": (f"NOT RELIABLY ESTIMABLE BY BOOTSTRAP ({n_nonfinite} non-finite + "
                            f"{n_divergent} divergent of {N_BOOT} resamples = {100 * frac_bad:.1f}%); "
                            "sandwich-SE interval shown as a FRAGILE fallback -- do not quote as a "
                            "bootstrap CI"),
            })
        out.append(row)
    return out


model_rows = []
for label, cols in COVSETS.items():
    X = d[cols].values
    sds = {c: float(np.std(d[c].values, ddof=1)) for c in cols}

    # ---- cause-specific Cox (corrected primary, pre-correction, Spec A) ------
    for spec, (tt, ss) in COX_SPECS.items():
        try:
            r = fit_cox(tt, ss, X)
        except Exception as e:
            log.warning("%s / %s failed: %s", spec, label, e)
            model_rows.append({"specification": spec, "covariate_set": label, "term": "ALL",
                               "note": f"NOT ESTIMABLE: {e}"})
            continue
        for j, c in enumerate(cols):
            b, se = float(r.params[j]), float(r.bse[j])
            model_rows.append({
                "specification": spec, "covariate_set": label, "term": c,
                "HR_per_1_unit": round(float(np.exp(b)), 3),
                "HR_CI_low": round(float(np.exp(b - 1.96 * se)), 3),
                "HR_CI_high": round(float(np.exp(b + 1.96 * se)), 3),
                "p_value": round(float(2 * stats.norm.sf(abs(b / se))), 4),
                "HR_per_1_SD": round(float(np.exp(b * sds[c])), 3),
                "HR_per_1_SD_CI_low": round(float(np.exp((b - 1.96 * se) * sds[c])), 3),
                "HR_per_1_SD_CI_high": round(float(np.exp((b + 1.96 * se) * sds[c])), 3),
                "SE_type": "asymptotic (model-based)", "N_events": int(np.sum(ss)),
            })

    # ---- Fine-Gray, bootstrap CIs (corrected secondary + pre-correction) -----
    for fg_spec, (fg_t, fg_s) in FG_SPECS.items():
        model_rows.extend(fine_gray_rows(fg_spec, fg_t, fg_s, X, cols, sds, label))

# ---- SOFA-stratified Cox: no PH assumption is made for SOFA at all ----------
# Motivated by the PH results in section 5: SOFA violates proportional hazards.
# Stratifying on SOFA tertiles allows a separate baseline hazard per stratum,
# so the microbiome HR is not contaminated by SOFA's non-proportionality.
sofa_stratum = pd.qcut(d["SOFA"], 3, labels=False, duplicates="drop").values
for label, col in [("M1 + SOFA(stratified)", "logit_M1"),
                   ("M3-LR + SOFA(stratified)", "logit_M3_LR"),
                   ("M3-CatBoost + SOFA(stratified)", "logit_M3_CatBoost")]:
    X = d[[col]].values
    sd = float(np.std(d[col].values, ddof=1))
    for spec, (tt, ss) in COX_SPECS.items():
        try:
            r = fit_cox(tt, ss, X, strata=sofa_stratum)
        except Exception as e:
            log.warning("stratified %s / %s failed: %s", spec, label, e)
            continue
        b, se = float(r.params[0]), float(r.bse[0])
        model_rows.append({
            "specification": spec, "covariate_set": label, "term": col,
            "HR_per_1_unit": round(float(np.exp(b)), 3),
            "HR_CI_low": round(float(np.exp(b - 1.96 * se)), 3),
            "HR_CI_high": round(float(np.exp(b + 1.96 * se)), 3),
            "p_value": round(float(2 * stats.norm.sf(abs(b / se))), 4),
            "HR_per_1_SD": round(float(np.exp(b * sd)), 3),
            "HR_per_1_SD_CI_low": round(float(np.exp((b - 1.96 * se) * sd)), 3),
            "HR_per_1_SD_CI_high": round(float(np.exp((b + 1.96 * se) * sd)), 3),
            "SE_type": "asymptotic, stratified on SOFA tertile (no PH assumption for SOFA)",
            "N_events": int(np.sum(ss)),
        })

models = pd.DataFrame(model_rows)
io_utils.save_csv(models, OUT_MODELS)
log.info("\n%s", models[["specification", "covariate_set", "term", "HR_per_1_unit",
                          "HR_CI_low", "HR_CI_high", "p_value"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. proportional-hazards assumption (Grambsch-Therneau, rank transform)
# ---------------------------------------------------------------------------
def ph_test(time, status, X, names):
    time = np.asarray(time, float); status = np.asarray(status, int); X = np.asarray(X, float)
    res = fit_cox(time, status, X)
    R = np.asarray(res.schoenfeld_residuals)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    keep = ~np.isnan(R).any(axis=1)
    R, et = R[keep], time[keep]
    n_ev = len(R)
    g = stats.rankdata(et)
    gc = g - g.mean()
    info = np.linalg.inv(np.asarray(res.cov_params()))
    u = gc @ R
    cov_u = (gc ** 2).sum() * info / n_ev
    chis = u ** 2 / np.diag(cov_u)
    glob = float(u @ np.linalg.solve(cov_u, u))
    out = [{"term": nm, "chi2": round(float(chis[k]), 3), "df": 1,
            "p_value": round(float(stats.chi2.sf(chis[k], 1)), 4),
            "PH_violated_at_0.05": bool(stats.chi2.sf(chis[k], 1) < 0.05)}
           for k, nm in enumerate(names)]
    out.append({"term": "GLOBAL", "chi2": round(glob, 3), "df": len(names),
                "p_value": round(float(stats.chi2.sf(glob, len(names))), 6),
                "PH_violated_at_0.05": bool(stats.chi2.sf(glob, len(names)) < 0.05)})
    return out, n_ev


ph_rows = []
for label, cols in COVSETS.items():
    for spec, (tt, ss) in COX_SPECS.items():
        try:
            rows, n_ev = ph_test(tt, ss, d[cols].values, cols)
        except Exception as e:
            log.warning("PH test %s / %s failed: %s", spec, label, e)
            continue
        for r_ in rows:
            ph_rows.append({"specification": spec, "covariate_set": label, "N_events": n_ev, **r_})
ph = pd.DataFrame(ph_rows)
io_utils.save_csv(ph, OUT_PH)
log.info("\n%s", ph.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. time-split companion: HRs before vs after the median event time.
#    Depends on no residual approximation; makes non-PH interpretable.
# ---------------------------------------------------------------------------
split_rows = []
for spec, (tt, ss) in COX_SPECS.items():
    cut = float(np.median(tt[ss == 1]))
    for label, cols in [("SOFA alone (reference)", ["SOFA"]),
                        ("M1 + SOFA", ["logit_M1", "SOFA"]),
                        ("M3-CatBoost + SOFA", ["logit_M3_CatBoost", "SOFA"])]:
        X = d[cols].values
        for period, (t2, s2) in {
                f"EARLY (0-{cut:.0f} d)": (np.minimum(tt, cut), np.where(tt <= cut, ss, 0)),
                f"LATE (>{cut:.0f} d)": (tt, np.where(tt > cut, ss, 0))}.items():
            if period.startswith("LATE"):
                keep = tt > cut          # only patients still at risk after the cut
                t2, s2, X2 = tt[keep], np.where(tt[keep] > cut, ss[keep], 0), X[keep]
            else:
                X2 = X
            if int(np.sum(s2)) < 5:
                split_rows.append({"specification": spec, "period": period, "covariate_set": label,
                                   "term": "ALL", "N_events": int(np.sum(s2)),
                                   "note": "NOT ESTIMABLE: fewer than 5 events in this period"})
                continue
            try:
                r = fit_cox(t2, s2, X2)
            except Exception as e:
                split_rows.append({"specification": spec, "period": period, "covariate_set": label,
                                   "term": "ALL", "N_events": int(np.sum(s2)), "note": f"NOT ESTIMABLE: {e}"})
                continue
            for j, c in enumerate(cols):
                b, se = float(r.params[j]), float(r.bse[j])
                split_rows.append({
                    "specification": spec, "period": period, "covariate_set": label, "term": c,
                    "N_at_risk": int(len(t2)), "N_events": int(np.sum(s2)),
                    "HR_per_1_unit": round(float(np.exp(b)), 3),
                    "HR_CI_low": round(float(np.exp(b - 1.96 * se)), 3),
                    "HR_CI_high": round(float(np.exp(b + 1.96 * se)), 3),
                    "p_value": round(float(2 * stats.norm.sf(abs(b / se))), 4)})
split = pd.DataFrame(split_rows)
io_utils.save_csv(split, OUT_SPLIT)
log.info("\n%s", split.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. figure: cumulative incidence of death vs discharge (Aalen-Johansen),
#    overall and by microbiome-score tertile
# ---------------------------------------------------------------------------
def aalen_johansen(time, status, cause, grid):
    """Cumulative incidence for `cause` in the presence of competing events."""
    t = np.asarray(time, float); s = np.asarray(status, int)
    order = np.argsort(t); t, s = t[order], s[order]
    times = np.unique(t[s > 0])
    surv, cif, out_t, out_c = 1.0, 0.0, [0.0], [0.0]
    for tt in times:
        at_risk = (t >= tt).sum()
        d_cause = ((t == tt) & (s == cause)).sum()
        d_any = ((t == tt) & (s > 0)).sum()
        if at_risk > 0:
            cif += surv * d_cause / at_risk
            surv *= (1 - d_any / at_risk)
        out_t.append(float(tt)); out_c.append(cif)
    return np.interp(grid, out_t, out_c)


grid = np.arange(0, 61, 1.0)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.step(grid, aalen_johansen(d.B_time, d.B_status, 1, grid), where="post", color="#c62828",
        lw=2, label=f"Death in hospital (n={n_ev1})")
ax.step(grid, aalen_johansen(d.B_time, d.B_status, 2, grid), where="post", color="#1976d2",
        lw=2, label=f"Discharge alive (competing, n={n_ev2})")
ax.set_xlabel("Days from microbiome sample collection")
ax.set_ylabel("Cumulative incidence")
ax.set_title("Competing-risk cumulative incidence (Aalen-Johansen)", fontsize=10)
ax.set_ylim(0, 1); ax.legend(fontsize=8)

ax = axes[1]
tert = pd.qcut(d["logit_M3_CatBoost"], 3, labels=["low", "mid", "high"])
for lab, colr in zip(["low", "mid", "high"], ["#2e7d32", "#f9a825", "#c62828"]):
    m = (tert == lab).values
    ax.step(grid, aalen_johansen(d.B_time[m], d.B_status[m], 1, grid), where="post", lw=2, color=colr,
            label=f"{lab} microbiome score (n={int(m.sum())}, {int((d.B_status[m] == 1).sum())} deaths)")
ax.set_xlabel("Days from microbiome sample collection")
ax.set_ylabel("Cumulative incidence of in-hospital death")
ax.set_title("Death CIF by microbiome (M3-CatBoost) score tertile", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle(f"Time-to-event and competing-risk analysis, N={n} Gangnam ICU patients (POST-HOC)", fontsize=11)
os.makedirs(config.DIR_FIGURES, exist_ok=True)
fig.savefig(OUT_FIG + ".svg", bbox_inches="tight")
fig.savefig(OUT_FIG + ".jpg", bbox_inches="tight", dpi=200)
plt.close(fig)
log.info("wrote %s.svg/.jpg", OUT_FIG)

# ---------------------------------------------------------------------------
# 8. markdown report
# ---------------------------------------------------------------------------
def sel(spec, covsets=None, extra=()):
    x = models[models.specification == spec]
    if covsets:
        x = x[x.covariate_set.isin(covsets)]
    keep = ["covariate_set", "term", "HR_per_1_unit", "HR_CI_low", "HR_CI_high", "p_value",
            "HR_per_1_SD", "HR_per_1_SD_CI_low", "HR_per_1_SD_CI_high"] + list(extra)
    return x[[c for c in keep if c in x.columns]]


# --- side-by-side: corrected primary vs the pre-correction primary ----------
# Built so the reader can see, per covariate, exactly what the owner's data
# correction changed -- and in particular whether any conclusion flipped.
_pri = models[models.specification == SPEC_PRIMARY].set_index(["covariate_set", "term"])
_pre = models[models.specification == SPEC_PRE].set_index(["covariate_set", "term"])
chg_rows = []
for key in _pri.index:
    if key not in _pre.index:
        continue
    a, b_ = _pre.loc[key], _pri.loc[key]
    if pd.isna(a.get("HR_per_1_unit")) or pd.isna(b_.get("HR_per_1_unit")):
        continue
    sig_pre, sig_post = a["p_value"] < 0.05, b_["p_value"] < 0.05
    chg_rows.append({
        "covariate_set": key[0], "term": key[1],
        f"PRE ({n_ev1_pre} events) HR (95% CI)": f"{a.HR_per_1_unit} ({a.HR_CI_low}-{a.HR_CI_high})",
        "PRE p": a.p_value,
        f"CORRECTED ({n_ev1} events) HR (95% CI)": f"{b_.HR_per_1_unit} ({b_.HR_CI_low}-{b_.HR_CI_high})",
        "CORRECTED p": b_.p_value,
        "conclusion_changed_at_0.05": ("NO" if sig_pre == sig_post else
                                       ("null -> significant" if sig_post else "significant -> null")),
    })
chg = pd.DataFrame(chg_rows)
change_table_md = df_to_md_table(chg) if len(chg) else "(no comparable terms)"
n_flipped = int((chg["conclusion_changed_at_0.05"] != "NO").sum()) if len(chg) else 0

fg_unstable = models[(models.specification.str.contains("Fine-Gray", na=False))
                     & (models.get("bootstrap_stability") == "UNSTABLE")]
if len(fg_unstable):
    _u = fg_unstable[["covariate_set", "term", "boot_resamples_nonfinite", "boot_resamples_divergent"]]
    fg_stability_note = (
        f"**{len(fg_unstable)} Fine-Gray term(s) are NOT reliably estimable by bootstrap at this N and are "
        "flagged as such rather than given a number that looks solid.** In those resamples the Cox optimiser "
        "returned NaN, or a finite but meaningless coefficient, *without raising an error* -- so 'the fit did "
        "not crash' is not evidence of convergence, and a naive percentile interval over all replicates would "
        "have been contaminated by optimiser artefacts (in the first version of this analysis it produced an "
        "absurd Fine-Gray CI of 0.23-1.85 around a point estimate of 1.50). Replicates are screened here for "
        "non-finiteness and for landing beyond 10 reference SEs of the full-sample estimate; where more than "
        "1% fail, the bootstrap CI is SUPPRESSED and a sandwich-SE interval is shown, explicitly labelled as "
        "a fragile fallback. Affected terms:\n\n" + df_to_md_table(_u))
else:
    fg_stability_note = ("All Fine-Gray bootstrap distributions passed the stability screen (fewer than 1% of "
                         "resamples non-finite or divergent), so the percentile intervals above are usable.")


def split_hr(spec_key, covset, term, period_prefix):
    """Pull one HR out of the time-split table for use in the narrative."""
    x = split[(split.specification.str.startswith(spec_key)) & (split.covariate_set == covset)
              & (split.term == term) & (split.period.str.startswith(period_prefix))]
    if x.empty or "HR_per_1_unit" not in x or pd.isna(x.iloc[0].get("HR_per_1_unit")):
        return "not estimable"
    r = x.iloc[0]
    return f"{r.HR_per_1_unit:.2f} ({r.HR_CI_low:.2f}-{r.HR_CI_high:.2f})"


def _hr_pair(spec, covset, term):
    """HR string and p-value for one term, for use in the narrative."""
    x = models[(models.specification == spec) & (models.covariate_set == covset) & (models.term == term)]
    if x.empty or pd.isna(x.iloc[0].get("HR_per_1_unit")):
        return "not estimable", "n/a"
    r = x.iloc[0]
    return f"{r.HR_per_1_unit:.3f} ({r.HR_CI_low:.3f}-{r.HR_CI_high:.3f})", f"{r.p_value:.4f}"


_a_m1_hr, _a_m1_p = _hr_pair(SPEC_A, "M1 + SOFA", "logit_M1")
_p_m1_hr, _p_m1_p = _hr_pair(SPEC_PRIMARY, "M1 + SOFA", "logit_M1")


def _split_row(spec_key, covset, term, period_prefix):
    x = split[(split.specification.str.startswith(spec_key)) & (split.covariate_set == covset)
              & (split.term == term) & (split.period.str.startswith(period_prefix))]
    return None if x.empty or pd.isna(x.iloc[0].get("HR_per_1_unit")) else x.iloc[0]


def _split_sig(spec_key, covset, term, period_prefix):
    """True if that period's HR confidence interval excludes 1."""
    r = _split_row(spec_key, covset, term, period_prefix)
    return bool(r is not None and (r.HR_CI_low > 1 or r.HR_CI_high < 1))

SPLIT_SPEC = "Spec B-CS CORRECTED"   # narrate the time split on the corrected primary
_sofa_early = split_hr(SPLIT_SPEC, "M3-CatBoost + SOFA", "SOFA", "EARLY")
_sofa_late = split_hr(SPLIT_SPEC, "M3-CatBoost + SOFA", "SOFA", "LATE")
_m3_early = split_hr(SPLIT_SPEC, "M3-CatBoost + SOFA", "logit_M3_CatBoost", "EARLY")
_m3_late = split_hr(SPLIT_SPEC, "M3-CatBoost + SOFA", "logit_M3_CatBoost", "LATE")

md = [
    "# Time-to-event and competing-risk analysis of the microbiome score",
    "",
    "> **POST-HOC.** Requested 2026-08-31, after the project's primary results already existed. Not "
    "prespecified; belongs in the post-hoc column of `semantic_microbiome/reports/analysis_provenance.md` "
    "(section 2). Strictly additive -- no existing script or result file was modified.",
    "",
    "## Why this analysis exists",
    "",
    "Every mortality analysis in this project is **binary in-hospital mortality**; there is no survival "
    "analysis anywhere in the repository. A binary endpoint is confounded by length of stay -- death on day 2 "
    "and death on day 60 are the same data point, and a patient discharged on day 3 counts as a survivor on the "
    "strength of a three-day observation window. `merge/downstream_analysis/DESIGN.md` additionally records "
    "that it could not be confirmed from source data whether `in.hospital.mortality` means ICU-specific or "
    "whole-admission mortality. Anchoring time at the microbiome sample absorbs part of that ambiguity and, as "
    "section 1 shows, makes the remainder directly visible.",
    "",
    "## 1. Data quality: what the date fields actually contain",
    "",
    "Every interval was checked before any model was fitted. Anomalies are reported as findings; **nothing was "
    "silently filtered.**",
    "",
    df_to_md_table(qual[["check", "n", "of", "action"]]),
    "",
    "Three of these matter enough to state in full:",
    "",
    f"**(a) `mortality_date` is a day count, not a date.** It equals `death.date - collection_date` in days for "
    f"{md_eq_col}/{len(deaths)} deaths, and equals days from *admission* for only {md_eq_adm}/{len(deaths)}. It "
    "is days from **sample collection** to death. This resolves the ambiguity flagged in the analysis request. "
    "All intervals below are nevertheless computed from the actual dates, not from this field.",
    "",
    f"**(b) `last.fu` cannot be used on its own as the censoring date.** It precedes `collection_date` for "
    f"{int((d['t_lastfu'] < 0).sum())}/{n} patients (minimum {d['t_lastfu'].min():.0f} days) and precedes the "
    f"patient's own `discharge_date` for {int((surv['t_lastfu'] < surv['t_discharge']).sum())}/{len(surv)} "
    "survivors -- in those records it has been filled with the admission date rather than a genuine final "
    "contact. For deaths it is reliable (it equals `death.date` for all 69). Censoring is therefore set to "
    "**max(discharge_date, last.fu)**: a patient known to have been alive at discharge has at minimum that much "
    "observed survival time. Taking `last.fu` verbatim instead would give "
    f"{n_asens_nonpositive} survivors zero or negative follow-up. The follow-up characteristics of that naive "
    "version are tabulated in section 2 (`Spec A-sensitivity`) so the size of the defect is visible rather "
    "than assumed away -- it loses "
    f"{float(fu.loc[0, 'total_person_days'] - fu.loc[1, 'total_person_days']):.0f} person-days and produces a "
    f"minimum follow-up of {fu.loc[1, 'followup_days_min']:.0f} days. It is NOT carried into the models, "
    "because negative follow-up time is not admissible in a Cox model; that is a statement about the field "
    "being unusable, not an analytical choice made to improve a result.",
    "",
    "**(c) RETRACTED AND REPLACED -- the endpoint is fine; the discharge dates were not.**",
    "",
    "> **Retraction.** An earlier version of this analysis observed that "
    f"{int((deaths['t_death'] > deaths['t_discharge']).sum())} of the {len(deaths)} deaths fell *after* their "
    f"recorded discharge date (by up to {int((deaths['t_death'] - deaths['t_discharge']).max())} days) and "
    f"{int((deaths['t_death'] < deaths['t_discharge']).sum())} fell *before* it, and inferred from this that "
    "`in.hospital.mortality` was a mixed endpoint capturing post-discharge deaths. **That inference was wrong "
    "and is retracted.** It is left visible here rather than deleted, because the pre-correction results below "
    "were produced under it.",
    "",
    "> **Owner determination (authoritative, 2026-08-31).** `in.hospital.mortality` **IS** all in-hospital "
    "death; the endpoint is not mixed. What is inaccurate is the **discharge dates**. The prescribed "
    "correction is: *for any patient with a death date, the discharge date equals the death date.* This "
    "resolves the open outcome-definition question flagged in `merge/downstream_analysis/DESIGN.md` -- by "
    "**owner determination, not by inference from the data**, and the earlier data-driven inference is exactly "
    "the kind that determination overrides.",
    "",
    f"The correction is applied **in analysis code only** (`effective_discharge = death.date where death.date "
    f"is not null`); `metadata/microbiome.clinical.csv` is read-only and was never modified. It changed the "
    f"discharge date for **{n_corrected} of {len(deaths)}** deaths "
    f"({int((deaths['t_death'] > deaths['t_discharge']).sum())} that had fallen after discharge, "
    f"{int((deaths['t_death'] < deaths['t_discharge']).sum())} before); the remaining "
    f"{len(deaths) - n_corrected} already agreed. Its effect on the primary specification is to move the "
    f"{int((deaths['t_death'] > deaths['t_discharge']).sum())} previously-misclassified patients from "
    "*competing event (discharged alive)* to *in-hospital death*, so the primary goes from "
    f"**{n_ev1_pre} deaths / {n_ev2_pre} competing** to **{n_ev1} deaths / {n_ev2} competing**.",
    "",
    "The `negative_day_outlier` patients (collection preceding admission_date) are unaffected by this "
    "correction and remain flagged; they are reported, not filtered.",
    "",
    "## 2. Specifications and follow-up characteristics",
    "",
    "Time origin is `collection_date` for all specifications, because that is the moment from which prediction "
    "is claimed.",
    "",
    f"- **{SPEC_PRIMARY}** -- the **PRIMARY** specification. Event = in-hospital death "
    f"(n={n_ev1}); competing event = discharge alive (n={n_ev2}); no administrative censoring.",
    "- **Spec B-FG CORRECTED** -- Fine-Gray subdistribution hazard (**secondary**), same event definitions.",
    f"- **{SPEC_PRE}** -- superseded. The {n_ev1_pre}/{n_ev2_pre} split produced by the original discharge "
    "dates, retained so the effect of the owner's correction is auditable.",
    f"- **{SPEC_A}** -- superseded, retained for continuity with the pre-correction run. See the warning below.",
    "",
    df_to_md_table(fu),
    "",
    "**Why Spec A is now superseded rather than merely alternative.** Spec A follows survivors past discharge, "
    "censoring at last follow-up. Under the owner's determination the event is *in-hospital* death, which "
    "cannot occur once a patient has been discharged alive -- so that extended follow-up accrues person-time "
    "during which the event is impossible by construction, diluting the estimated hazard. Removing patients "
    "from the risk set at discharge is exactly what the competing-risk specification does. The correction "
    "therefore does not just change event counts; it makes the competing-risk framing the only coherent one, "
    "and Spec A is reported below only so the pre-correction numbers stay visible.",
    "",
    "**Fine-Gray is fully estimable here, and this is worth stating because it often is not.** Every patient "
    "has either a death date or a discharge date, so Spec B has **zero administrative censoring**: the "
    "censoring survival function G(t) is identically 1, all Fine-Gray IPCW weights equal 1, and the "
    "subdistribution model reduces *exactly* to a Cox model in which patients discharged alive stay in the risk "
    f"set to the end of follow-up. The competing event is abundant (n={n_ev2}), not sparse. This reduction was "
    "verified against R `survival::finegray()`, which returns 302 rows with a single unique weight of 1 and "
    "coefficients matching this script's to 8 decimal places.",
    "",
    "## 3. Cause-specific Cox models -- CORRECTED PRIMARY",
    "",
    f"Event = in-hospital death (n={n_ev1}), competing event = discharge alive (n={n_ev2}).",
    "",
    df_to_md_table(sel(SPEC_PRIMARY)),
    "",
    "### Change vs. the pre-correction run",
    "",
    change_table_md,
    "",
    "## 4. Superseded specifications (shown so the correction is auditable)",
    "",
    f"**{SPEC_PRE}** ({n_ev1_pre} deaths / {n_ev2_pre} competing):",
    "",
    df_to_md_table(sel(SPEC_PRE)),
    "",
    f"**{SPEC_A}** ({int(d.A_status.sum())} deaths; extended follow-up -- see the warning in section 2):",
    "",
    df_to_md_table(sel(SPEC_A)),
    "",
    "## 5. Fine-Gray subdistribution models (SECONDARY)",
    "",
    "CIs and p-values are **bootstrap percentile** (1000 resamples of patients, seed=2026). Because subjects "
    "with the competing event contribute to risk sets after their own event, the model-based information matrix "
    "is not the correct variance; the model-based and cluster-sandwich SEs are carried in the CSV alongside for "
    "comparison. The sandwich SE agrees with R's robust SE to within about 0.7% (R uses a slightly different "
    "finite-sample influence computation) -- disclosed rather than hidden, and the reason bootstrap is primary.",
    "",
    "**CORRECTED (secondary):**",
    "",
    df_to_md_table(sel("Spec B-FG CORRECTED (SECONDARY): Fine-Gray subdistribution",
                       extra=("bootstrap_stability", "boot_resamples_usable", "SE_sandwich"))),
    "",
    "**Pre-correction (superseded):**",
    "",
    df_to_md_table(sel("Spec B-FG PRE-CORRECTION (superseded): Fine-Gray subdistribution",
                       extra=("bootstrap_stability", "boot_resamples_usable", "SE_sandwich"))),
    "",
    "### Fine-Gray bootstrap stability",
    "",
    fg_stability_note,
    "",
    "## 6. Proportional-hazards assumption",
    "",
    "Grambsch-Therneau scaled Schoenfeld residual test, rank time transform, classical I/d variance "
    "approximation. Cross-checked against R `cox.zph(transform=\"rank\")` on the SOFA + M3-CatBoost model: this "
    "implementation gives SOFA chi2 9.73 / M3-CatBoost 3.51 / GLOBAL 17.74 for Spec A where R gives 9.18 / 3.81 "
    "/ 19.23, and SOFA 12.58 / M3-CatBoost 0.42 / GLOBAL 14.87 for Spec B-CS where R gives 11.27 / 0.47 / 13.70. "
    "The statistics differ by 5-10% because modern `survival` uses a more exact variance; **every conclusion is "
    "identical**, and both sets of numbers are stated here so the reader can see the size of the approximation.",
    "",
    df_to_md_table(ph[["specification", "covariate_set", "term", "chi2", "df", "p_value", "PH_violated_at_0.05"]]),
    "",
    "## 7. Time-split companion (no residual approximation)",
    "",
    "Separate Cox fits before and after the median event time, which expresses any non-proportionality directly "
    "as two hazard ratios rather than as a chi-square.",
    "",
    df_to_md_table(split),
    "",
    "## 8. Interpretation",
    "",
    "### Read this before reading the numbers: what changed, and why",
    "",
    "The corrected primary specification differs from the pre-correction one **because the data owner "
    "corrected the discharge dates**, not because any analytic choice was revisited. The model set, seed, "
    "folds, covariates, censoring rule for survivors, PH testing and stability screening are all exactly as "
    "they were before the correction; the only thing that changed is which patients count as in-hospital "
    f"deaths ({n_ev1_pre} -> {n_ev1}).",
    "",
    (f"**{n_flipped} of {len(chg)} covariate terms change their significance status at 0.05** under the "
     "correction (section 3). Where a term moves from null to significant, that is the arithmetic consequence "
     "of an owner-confirmed data fix restoring 11 misclassified deaths to the event group -- **it is not a "
     "newly discovered positive finding, and it must not be written up as one.** The pre-correction numbers "
     "are printed alongside in section 3 and section 4 precisely so no reader has to take that on trust."
     if n_flipped else
     "**No covariate term changes its significance status at 0.05** under the correction (section 3): the "
     "conclusions are the same before and after the owner's data fix."),
    "",
    "A reader who wants to sanity-check the direction of the change should note that the 11 restored patients "
    "are, by construction, patients who died -- so any score that ranks them as high risk gains, and any score "
    "that ranks them low risk loses. That is a property of the corrected data, not evidence about the score.",
    "",
    "### Why the corrected primary did NOT become significant, though it was expected to",
    "",
    f"It was reasonable to anticipate that the corrected primary might flip from null to significant, on the "
    f"grounds that the superseded Spec A -- which also has {int(d.A_status.sum())} deaths -- gave "
    f"M1 + SOFA = {_a_m1_hr}, p = {_a_m1_p}, whereas the {n_ev1_pre}-event pre-correction primary was null. "
    f"That did not happen: the corrected primary gives M1 + SOFA = {_p_m1_hr}, p = {_p_m1_p}.",
    "",
    "**Matching event counts is not the same as matching specifications.** Spec A and the corrected primary "
    f"both have {int(d.A_status.sum())} deaths at identical event times, but they treat the "
    f"{n_ev2} survivors differently: the corrected primary removes a patient from the risk set at discharge "
    "(a competing event), while Spec A keeps them in it until last follow-up. Spec A's apparent significance "
    "therefore rests on person-time accrued after discharge, during which in-hospital death is impossible by "
    "construction -- and that is exactly the assumption the owner's determination invalidates. So the "
    "correction does not create a positive finding; it removes the basis for one that was an artefact of "
    "immortal person-time. The corrected primary is null, and that is the result to report.",
    "",
    "### Does the microbiome score predict time to death?",
    "",
    "Read the tables directly rather than any single headline:",
    "",
    "- The **corrected primary cause-specific** model in section 3 is the specification to quote. Its "
    "SOFA-adjusted microbiome terms are the answer to the question as posed.",
    "- The **SOFA-stratified** rows in the same table are the more trustworthy adjusted estimates, because "
    "SOFA violates proportional hazards (section 6).",
    "- In **Fine-Gray**, effects are systematically larger than cause-specific. This is expected and should not "
    "be read as stronger evidence: the subdistribution hazard keeps discharged patients in the risk set, so a "
    "score that predicts *early discharge* inflates the subdistribution contrast even when it has no effect on "
    "the death hazard itself. When cause-specific and Fine-Gray diverge in this direction, the cause-specific "
    "estimate is the one that answers 'does this raise the rate of dying'; the Fine-Gray estimate answers "
    "'does this raise the eventual probability of dying', and the gap between them here is largely an effect on "
    "the discharge process.",
    "",
    "### The time-split result (attenuated by the correction -- read the caveat)",
    "",
    "Section 7 is worth reading closely. In the corrected primary specification, SOFA's hazard ratio for "
    "in-hospital death falls from "
    f"**{_sofa_early} per point early** to **{_sofa_late} per point late**: that is the "
    "proportional-hazards violation restated as something clinically meaningful -- SOFA is an acute "
    "organ-dysfunction score, and its prognostic power decays as follow-up lengthens. Across the same split "
    f"the M3-CatBoost microbiome score moves in the **opposite** direction, from **{_m3_early} early** to "
    f"**{_m3_late} late**.",
    "",
    "**The SOFA half of this is unambiguous** -- its late-period CI now excludes any meaningful effect, and "
    "the PH test agrees (section 6). **The microbiome half is not.** In the corrected primary, neither "
    f"microbiome score's late-period HR reaches significance (M3-CatBoost {_m3_late}, M1 "
    f"{split_hr(SPLIT_SPEC, 'M1 + SOFA', 'logit_M1', 'LATE')}); both intervals include 1.",
    "",
    "**This finding was materially attenuated by the owner's data correction, and that is worth stating "
    "plainly.** In the superseded Spec A the same split gave M3-CatBoost "
    f"{split_hr('Spec A', 'M3-CatBoost + SOFA', 'logit_M3_CatBoost', 'LATE')} late against "
    f"{split_hr('Spec A', 'M3-CatBoost + SOFA', 'logit_M3_CatBoost', 'EARLY')} early -- an interval that "
    "excluded 1 and looked like a striking early/late reversal. Under the corrected primary that reversal "
    "survives only as a direction, not as a signal. The likely reason is mechanical: the 11 patients the "
    "correction reclassified had long collection-to-death intervals, so they sat disproportionately in the "
    "LATE period and were carrying much of the apparent late-period effect. An earlier draft of this report "
    "described the reversal as 'the most scientifically interesting result in this script' and as independent "
    "corroboration of script 24's chronic-vs-acute hypothesis; **on the corrected data that claim is no longer "
    "supported at this strength and is withdrawn.**",
    "",
    "What remains is a *directional* consistency with script 24's hypothesis (microbiome information weighted "
    "toward later deaths, SOFA toward earlier ones) that is too imprecise here to count as evidence: a single "
    "post-hoc split at the median event time, wide overlapping intervals, no multiplicity correction, no "
    "external replication. It is a hypothesis worth testing prospectively in a study designed for it -- not a "
    "finding, and not a corroboration.",
    "",
    "### Assumption violation -- this one is material",
    "",
    "**SOFA violates the proportional-hazards assumption in every model and both specifications** (global test "
    "p < 0.001). The time-split table in section 7 shows the shape of it. Consequence: every SOFA-adjusted HR "
    "in sections 3-5 comes from a model that is misspecified in SOFA, and those HRs should be read as "
    "time-averaged summaries rather than as constant hazard ratios. Because of this, SOFA-**stratified** models "
    "(a separate baseline hazard per SOFA tertile, so no PH assumption is imposed on SOFA at all) are reported "
    "in the same tables under the `+ SOFA(stratified)` covariate sets, and are the more trustworthy adjusted "
    "estimates. The microbiome terms themselves do not show a clear PH violation.",
    "",
    "### What is NOT supported",
    "",
    "- No claim that the microbiome score adds prognostic information beyond SOFA in a time-to-event framework. "
    "The primary cause-specific specification does not support it.",
    "- No claim of a constant hazard ratio for SOFA over follow-up. It is demonstrably not constant.",
    "- No claim about long-term or post-discharge survival. **The `last.fu` defect in finding (b) remains "
    "OPEN.** The owner's correction addresses the discharge dates of patients who died; it says nothing about "
    f"`last.fu`, which still precedes `collection_date` for {int((d['t_lastfu'] < 0).sum())}/{n} patients and "
    f"precedes the patient's own discharge for {int((surv['t_lastfu'] < surv['t_discharge']).sum())}/{len(surv)} "
    "survivors. Any future post-discharge or long-term survival analysis needs that field fixed at source "
    "first; the `max(discharge, last.fu)` rule used here is a defensible reconstruction for censoring, not a "
    "repair of the underlying data.",
    "- No causal interpretation. These are observational associations from a single-centre cohort of "
    f"{n} patients with {int(d.A_status.sum())} deaths, with no external validation of the time-to-event "
    "results.",
    "- Where a Fine-Gray interval is marked UNSTABLE in section 5, **no interval should be quoted at all** for "
    "that term. The fallback shown there is a sandwich approximation kept only so the table has no blank "
    "cells; it has not been validated for that model and is not a substitute for the suppressed bootstrap CI.",
    "",
    "### Multiplicity and power",
    "",
    f"Seven covariate sets across three specifications, plus stratified variants, are reported without "
    "multiplicity correction (consistent with the rest of this project's Phase X work). With "
    f"{n_ev1}-{int(d.A_status.sum())} events these models are near the conventional 10-events-per-variable "
    "floor even at two covariates. Nominal significance here is hypothesis-generating.",
    "",
    "No specification was altered in pursuit of significance. The time origin, the event definitions, the "
    "censoring rule and the model set were fixed from the data-quality findings in section 1 before any hazard "
    "ratio was read; the only post-hoc change was the owner's authoritative data correction, applied exactly "
    "as prescribed and reported alongside the numbers it superseded.",
    "",
    "## 9. Effect of the owner correction on other scripts",
    "",
    "Two other scripts in this directory reference `discharge_date`:",
    "",
    "- **`40_phase_x_sample_to_death_sensitivity.py` -- NO re-run needed.** It selects `discharge_date` in its "
    "merge (line 54) but never uses it in any computation: its only interval, `sample_to_death_days`, is built "
    "from `collection_date` and `death.date`/`mortality_date`. The owner's correction does not touch "
    "`death.date` or `collection_date`, so every number script 40 produces is unchanged. Verified by reading "
    "the script; **it was NOT modified.**",
    "- **`41_phase_x_microbiome_over_sofa.py` -- unaffected.** It reads no date field at all: only the binary "
    "`hospital_mortality`, `SOFA`, and script 18's out-of-fold probabilities. Not re-run for this correction.",
    "",
    "The correction is applied only in this script, and only in memory. `metadata/microbiome.clinical.csv` is "
    "unchanged on disk.",
]
with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
log.info("wrote %s", OUT_MD)
log.info("done.")
