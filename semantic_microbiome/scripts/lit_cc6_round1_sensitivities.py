"""
lit_cc6_round1_sensitivities.py -- review-triggered analyses PR-1, PR-4, PR-5 (POST_HOC_EXPLORATORY).
Pre-registered in semantic_microbiome/reports/paper1/ROUND1_PREREGISTRATION.md BEFORE this ran.

PR-1  external endpoint harmonisation: percentile-bootstrap CI (2000, seed 2026) for the frozen M1 AUROC in PRJNA842425 under the
      in-hospital-death endpoint (27 events) AND under the vital-status-at-follow-up endpoint (32 events); SOFA likewise.
PR-4  index-time sensitivity: ΔAUROC for C1 (vs C0) and S1 (vs SOFA) on saved out-of-fold predictions restricted to patients admitted
      directly to the ICU (hospital-to-ICU interval 0 days) -- no refit.
PR-5  ΔAUROC for C1 and S1 excluding the 18 patients whose ICU day 0 is an owner attestation (N = 284) -- no refit.
Bootstrap algorithm = utils.nested_meta.paired_bootstrap_delta_auroc (fresh RandomState(2026), same resample index for both arms,
single-class resamples skipped, 2000). Reproduction gate: full-cohort ΔAUROC/interval must match Table 2 before subset rows are emitted.
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('ICU_DAY0_ATTESTED_IDS', '').strip():
    raise SystemExit('ICU_DAY0_ATTESTED_IDS must be set (see README).')
_pre_ids = [i.strip() for i in _pre_os.environ['ICU_DAY0_ATTESTED_IDS'].split(',')]
if any(not i for i in _pre_ids) or len(_pre_ids) != 18 or len(set(_pre_ids)) != 18:
    raise SystemExit('ICU_DAY0_ATTESTED_IDS must list exactly 18 unique, non-empty IDs (see README).')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import SEED, SEMANTIC, ICU_PUBLIC
import os
_ATT_RAW = os.environ.get("ICU_DAY0_ATTESTED_IDS")  # [PUBLIC RELEASE] required; validated before any data is read
_ATT_LIST = [i.strip() for i in (_ATT_RAW or "").split(",")]
if not _ATT_RAW or any(not i for i in _ATT_LIST) or len(_ATT_LIST) != 18 or len(set(_ATT_LIST)) != 18:
    raise SystemExit("ICU_DAY0_ATTESTED_IDS must be set to a comma-separated list of exactly 18 unique, non-empty "
                     f"pseudonymised patient IDs (got {0 if not _ATT_RAW else len(_ATT_LIST)} entries, "
                     f"{len(set(i for i in _ATT_LIST if i))} unique non-empty). See README.")

OUT = SEMANTIC / "reports" / "paper1" / "critical_care_external" / "addendum2"
OUT.mkdir(parents=True, exist_ok=True)
PRED = ICU_PUBLIC / "clinical metadata" / "clinical_ai" / "predictions"


def boot(y, ref, cand, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], cand[idx]) - roc_auc_score(y[idx], ref[idx]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0)
    return roc_auc_score(y, cand) - roc_auc_score(y, ref), lo, hi, p, len(d)


def boot_auc(y, s, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    b = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        b.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi


rows_pr1, rows_sens = [], []

# ---------------- PR-1 ----------------
q = pd.read_csv(SEMANTIC / "public_validation" / "covid_prjna842425_concordance_groups.csv")
sm = pd.read_csv(ICU_PUBLIC / "merge" / "output" / "sample_metadata.tsv", sep="\t", dtype=str)
idm = sm[sm.subcohort == "PRJNA842425"][["sample_id", "patient_id"]].copy()
idm["covid_id"] = idm["patient_id"].str.replace("PRJNA842425_", "", regex=False)
q = q.merge(idm[["sample_id", "covid_id"]], on="sample_id", how="left", validate="one_to_one")
xl = pd.read_excel(ICU_PUBLIC / "metadata.xlsx", sheet_name="PRJNA842425").rename(columns={"patient_ID": "covid_id"})
q = q.merge(xl[["covid_id", "DischargeLocation"]], on="covid_id", how="left", validate="one_to_one")
assert q["DischargeLocation"].notna().all()
ends = {"in_hospital_death (primary, 27 events)": (q["DischargeLocation"] == "Death").astype(int).values,
        "vital_status_at_follow_up (secondary, 32 events)": q["mortality"].astype(int).values}
m1 = q["microbiome_prob_M1"].astype(float).values
sofa = q["SOFA"].astype(float).values
for en, y in ends.items():
    for lab, s in (("M1 (frozen)", m1), ("SOFA", sofa)):
        a, lo, hi = boot_auc(y, s)
        rows_pr1.append(dict(endpoint=en, score=lab, N=len(y), events=int(y.sum()), AUROC=round(a, 4),
                             ci_low=round(lo, 4), ci_high=round(hi, 4)))
assert abs(rows_pr1[0]["AUROC"] - 0.6271) < 5e-5 and rows_pr1[0]["events"] == 27, rows_pr1[0]
assert abs(rows_pr1[2]["AUROC"] - 0.5665) < 5e-5 and rows_pr1[2]["events"] == 32, rows_pr1[2]
pd.DataFrame(rows_pr1).to_csv(OUT / "PR1_external_endpoint_M1_AUROC.csv", index=False)

# ---------------- PR-4 / PR-5 ----------------
o1 = pd.read_csv(PRED / "phase_x_c1_c2_nested_oof_probabilities.csv")
o2 = pd.read_csv(PRED / "phase_x_sofa_nested_oof_probabilities.csv")
assert list(o1["patient_id_clinical"]) == list(o2["patient_id_clinical"])
pid = o1["patient_id_clinical"].values
y = o1["hospital_mortality"].values
disc = pd.read_csv(SEMANTIC / "harmonized" / "discovery_cohort_primary.csv")
disc["kcdc_no"] = disc["kcdc_no"].astype(str)
nm = pd.read_csv(ICU_PUBLIC / "new_meta.csv", dtype=str)
nm["KCDC_no"] = nm["KCDC_no"].astype(str)
d = disc.merge(nm[["KCDC_no", "ICU입원일"]], left_on="kcdc_no", right_on="KCDC_no", how="left").set_index("patient_id_clinical").reindex(pid)
hosp_adm = pd.to_datetime(pd.read_csv(ICU_PUBLIC / "metadata" / "microbiome.clinical.csv").assign(
    KCDC_no=lambda x: x["KCDC_no"].astype(str)).merge(disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates(),
    left_on="KCDC_no", right_on="kcdc_no").set_index("patient_id_clinical").reindex(pid)["admission_date"], errors="coerce")
icu_dt = pd.to_datetime(d["ICU입원일"], errors="coerce")
h2i = (icu_dt - hosp_adm).dt.days
ATT = frozenset(_ATT_LIST)  # [REDACTED for public release] validated at the top of this script
# 18 hard-coded pseudonymised patient IDs (owner ICU-day-0 attestation) removed; supply locally via env var ICU_DAY0_ATTESTED_IDS.
attested = pd.Index(pid).isin(ATT)
if int(attested.sum()) != 18:  # [PUBLIC RELEASE] all 18 attested IDs must be in the 302-patient index
    raise SystemExit(f"ICU_DAY0_ATTESTED_IDS: only {int(attested.sum())} of 18 IDs are in the 302-patient index.")
direct = ((h2i == 0) & icu_dt.notna()).values
print("hospital-to-ICU interval (recorded dates):", h2i.value_counts(dropna=False).sort_index().to_dict())
print("direct ICU admission n =", int(direct.sum()), "| attested n =", int(attested.sum()))

C = {"C1 vs C0": (o1["oof_prob_C0_clinical_alone"].values, o1["oof_prob_C1_clinical_plus_M1_LiteratureCore"].values, y),
     "S1 vs SOFA": (o2["oof_prob_S0_SOFA_alone"].values, o2["oof_prob_S1_SOFA_plus_M1"].values, o2["hospital_mortality"].values)}
subsets = {"primary_302 (reference)": np.ones(len(y), bool),
           "direct_ICU_admission (interval 0 d)": direct,
           "exclude_owner_attested_18": ~attested}
pub = {"C1 vs C0": (0.0082, -0.0129, 0.0286), "S1 vs SOFA": (0.0113, -0.0123, 0.0341)}
for sname, mask in subsets.items():
    for cname, (ref, cand, yy) in C.items():
        pt, lo, hi, p, nb = boot(yy[mask], ref[mask], cand[mask])
        if sname.startswith("primary"):
            assert abs(pt - pub[cname][0]) < 6e-5 and abs(lo - pub[cname][1]) < 2e-4 and abs(hi - pub[cname][2]) < 2e-4, ("REPRO GATE", cname, pt, lo, hi)
        rows_sens.append(dict(subset=sname, contrast=cname, N=int(mask.sum()), events=int(yy[mask].sum()), delta_AUROC=round(pt, 4),
                              ci_low=round(lo, 4), ci_high=round(hi, 4), boot_p=round(p, 3),
                              AUROC_reference=round(roc_auc_score(yy[mask], ref[mask]), 4), AUROC_candidate=round(roc_auc_score(yy[mask], cand[mask]), 4)))
res = pd.DataFrame(rows_sens)
res.to_csv(OUT / "PR4_PR5_index_time_and_attested_sensitivity.csv", index=False)
md = ["# PR-1, PR-4, PR-5 (POST_HOC_EXPLORATORY; pre-registered in ROUND1_PREREGISTRATION.md)", "",
      "## PR-1: frozen M1 and SOFA in PRJNA842425 by endpoint", "", pd.DataFrame(rows_pr1).to_markdown(index=False), "",
      "## PR-4/PR-5: ΔAUROC on saved out-of-fold predictions, subsets of the same 302 (no refit)", "", res.to_markdown(index=False), "",
      "Reference rows reproduce Table 2 (gate). Subset results describe stability of the increment; they are not primary estimates."]
(OUT / "PR1_PR4_PR5_round1_sensitivities.md").write_text("\n".join(md))
print(pd.DataFrame(rows_pr1).to_string(index=False)); print(res.to_string(index=False))
