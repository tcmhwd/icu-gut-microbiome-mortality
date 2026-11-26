"""
03_define_discovery_cohort.py

Builds the Gangnam Severance primary discovery cohort from the authoritative
clinical file (common.CLINICAL_CSV) and reports sampling-time distribution
and hospital-mortality outcome audit, per Phase-1 spec sections 13/14/44/45.

Key facts established during interactive audit (2026-08-13, confirmed with
user via AskUserQuestion) and encoded here:

  * CLINICAL_CSV already has ONE ROW PER PATIENT (all of ID/No/list/KCDC_no/
    sample_list are unique across all 430 rows) — user confirmed this file is
    final and already deduplicated to one representative sample per patient.
    We therefore do NOT re-run first-sample-selection logic here; each row
    IS the discovery-cohort candidate for that patient.
  * Identifier-column audit (IMPORTANT, discovered while writing this
    script): CLINICAL_CSV's `ID` column is NOT the "ICU####" pseudonymized
    code — it holds a distinct all-numeric value (e.g. "<7-digit number, redacted>") that looks
    like a raw hospital chart/record number, i.e. potential PHI. It is
    NEVER used as a patient identifier anywhere in this pipeline.
    The actual "ICU####" pseudonymized patient code lives in column `No`,
    which is populated and unique for all 430 rows. Columns `list` and
    `sample_list` are UNRELIABLE as a patient key: for 180/430 rows they
    mirror `No` (ICU####), but for the other 250/430 rows they instead hold
    the KCDC_no-style sequencing id (e.g. "A0824Gxxxx" / "A0824Gxxxx_WM")
    — i.e. they silently switch schemes row-to-row. `No` is therefore the
    only column used here as `patient_id_clinical`; `KCDC_no` is used
    separately as the sequencing/sample-matrix join key (see script 02).
  * `collection_time` is NOT a continuous sampling day. It is a pre-computed
    binary flag: 0 = candidate falls in the intended primary
    admission-window sample, 1 = later/follow-up sample. We verified this
    against collection_date - admission_date:
      collection_time==0 (n=302): day0=116, day1=97, day2=59, day3=26,
                                   plus 4 rows with NEGATIVE day-gap
                                   (-274, -191, -18, -1)
      collection_time==1 (n=128): day4..day193 (wide spread)
  * User decision on the 4 negative-day outliers within collection_time==0:
    "그대로 포함, QC 리포트에 별도 명시" -> keep them in the primary cohort
    but flag them explicitly in sampling_time_distribution.csv and the
    discovery cohort summary so they are never silently miscounted as
    clean day0-3 samples.

Outputs (per spec sec. 43-45):
  reports/discovery_cohort_summary.csv
  reports/sampling_time_distribution.csv
"""
from __future__ import annotations

import csv
import datetime

import pandas as pd

from common import CLINICAL_CSV, REPORTS_DIR, HARMONIZED_DIR

df = pd.read_csv(CLINICAL_CSV, dtype=str)
df["collection_time"] = df["collection_time"].astype(int)
df["in.hospital.mortality"] = pd.to_numeric(df["in.hospital.mortality"], errors="coerce")

# -- day-gap (raw calculated timing), independent of the collection_time flag --
def day_gap(row):
    cd, ad = row["collection_date"], row["admission_date"]
    if pd.isna(cd) or pd.isna(ad) or cd in ("", "NA") or ad in ("", "NA"):
        return None
    try:
        return (datetime.date.fromisoformat(cd) - datetime.date.fromisoformat(ad)).days
    except ValueError:
        return None

df["days_admission_to_collection_raw"] = df.apply(day_gap, axis=1)
df["negative_day_outlier"] = (df["collection_time"] == 0) & (df["days_admission_to_collection_raw"] < 0)

n_total_patients = len(df)
n_missing_date = df["days_admission_to_collection_raw"].isna().sum()

# -- primary eligibility: collection_time == 0 (existing recoded timing), per user instruction --
primary = df[df["collection_time"] == 0].copy()
n_primary = len(primary)
n_negative_outliers = int(primary["negative_day_outlier"].sum())

n_outcome_known = primary["in.hospital.mortality"].notna().sum()
n_outcome_missing = primary["in.hospital.mortality"].isna().sum()
n_survivors = int((primary["in.hospital.mortality"] == 0).sum())
n_deaths = int((primary["in.hospital.mortality"] == 1).sum())
mortality_pct = 100 * n_deaths / n_outcome_known if n_outcome_known else float("nan")

summary_rows = [
    ("original_patients_in_clinical_csv", n_total_patients),
    ("patients_with_multiple_samples", 0),  # CLINICAL_CSV is already 1 row/patient; see docstring
    ("patients_with_valid_sampling_timing", n_total_patients - int(n_missing_date)),
    ("patients_with_missing_sampling_dates", int(n_missing_date)),
    ("patients_at_least_one_day0_3_sample_raw_calc", int(((df["days_admission_to_collection_raw"] >= 0) & (df["days_admission_to_collection_raw"] <= 3)).sum())),
    ("final_primary_cohort_N_collection_time_eq_0", n_primary),
    ("primary_cohort_negative_day_outliers_included", n_negative_outliers),
    ("hospital_mortality_available_N", int(n_outcome_known)),
    ("hospital_mortality_missing_N", int(n_outcome_missing)),
    ("hospital_survivors", n_survivors),
    ("hospital_deaths", n_deaths),
    ("hospital_mortality_pct", round(mortality_pct, 2)),
    ("ICU_mortality_available_N", 0),  # see note below
    ("ICU_mortality_note", "no dedicated ICU-specific mortality column found in CLINICAL_CSV (only in.hospital.mortality, mortality.30d, mortality_date, death.date, last.fu)"),
]
pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
    REPORTS_DIR / "discovery_cohort_summary.csv", index=False
)

# -- sampling time distribution (raw day-gap, both flag groups, negative days shown separately) --
dist = (
    df.dropna(subset=["days_admission_to_collection_raw"])
    .groupby("days_admission_to_collection_raw")
    .agg(n_samples=("No", "count"), n_unique_patients=("No", "nunique"), n_collection_time_eq_0=("collection_time", lambda s: int((s == 0).sum())))
    .reset_index()
    .rename(columns={"days_admission_to_collection_raw": "sampling_day_raw"})
    .sort_values("sampling_day_raw")
)
dist.to_csv(REPORTS_DIR / "sampling_time_distribution.csv", index=False)

# cumulative windows summary appended as a second small table for convenience
windows = []
for label, lo, hi in [("day0", 0, 0), ("day0-1", 0, 1), ("day0-2", 0, 2), ("day0-3", 0, 3), ("day0-7", 0, 7), ("all_nonneg", 0, 10**6), ("negative_days", -(10**6), -1)]:
    sub = df[(df["days_admission_to_collection_raw"] >= lo) & (df["days_admission_to_collection_raw"] <= hi)]
    windows.append((label, len(sub), sub["No"].nunique()))
pd.DataFrame(windows, columns=["window", "n_samples", "n_unique_patients"]).to_csv(
    REPORTS_DIR / "sampling_time_distribution_windows.csv", index=False
)

# -- write the primary cohort id list for downstream scripts --
# NOTE: `sample_list`/`list` are deliberately excluded — audit showed they
# switch between the No-scheme and KCDC_no-scheme row-to-row (see docstring)
# and are therefore unsafe as an identifier. `ID` (raw chart-number-like
# PHI) is also deliberately excluded.
primary_out = primary[["No", "KCDC_no", "collection_date", "admission_date",
                        "days_admission_to_collection_raw", "negative_day_outlier",
                        "sample_type", "sex", "age", "in.hospital.mortality"]].copy()
primary_out = primary_out.rename(columns={"No": "patient_id_clinical", "KCDC_no": "kcdc_no",
                                           "in.hospital.mortality": "hospital_mortality"})
primary_out.to_csv(HARMONIZED_DIR / "discovery_cohort_primary.csv", index=False)

print(f"[03] N_total_clinical={n_total_patients}  N_primary(day0-3 flag)={n_primary}  "
      f"deaths={n_deaths}  survivors={n_survivors}  mortality%={mortality_pct:.1f}  "
      f"negative_day_outliers_in_primary={n_negative_outliers}")
print(f"[03] sample_type distribution within primary cohort:\n{primary['sample_type'].value_counts()}")
