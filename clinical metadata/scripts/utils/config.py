"""
config.py — Central configuration for the ICU Clinical Mortality AI pipeline.

Every variable-inclusion / variable-exclusion decision made while inspecting
ICU.csv and mimic.csv is encoded here, WITH the reasoning, so nothing is a
silent choice buried in analysis code. Downstream scripts (01-06) must import
from this module rather than re-deriving these lists.

Do not edit the raw CSVs. This module only describes them.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/home/tcmhwd/microbiome/ICU_public/clinical metadata"
ICU_CSV = os.path.join(BASE_DIR, "ICU.csv")
MIMIC_CSV = os.path.join(BASE_DIR, "mimic.csv")

AI_DIR = os.path.join(BASE_DIR, "clinical_ai")
DIR_PREPROCESSING = os.path.join(AI_DIR, "preprocessing")
DIR_MISSINGNESS = os.path.join(AI_DIR, "missingness")
DIR_ML_SCREENING = os.path.join(AI_DIR, "ml_screening")
DIR_OPTUNA = os.path.join(AI_DIR, "optuna")
DIR_OPTUNA_ML = os.path.join(DIR_OPTUNA, "ML")
DIR_OPTUNA_DL = os.path.join(DIR_OPTUNA, "DL")
DIR_TRAINED_MODELS = os.path.join(AI_DIR, "trained_models")
DIR_PREDICTIONS = os.path.join(AI_DIR, "predictions")
DIR_PERFORMANCE = os.path.join(AI_DIR, "performance")
DIR_FIGURES = os.path.join(AI_DIR, "figures")
DIR_REPORTS = os.path.join(AI_DIR, "reports")

ALL_OUTPUT_DIRS = [
    DIR_PREPROCESSING, DIR_MISSINGNESS, DIR_ML_SCREENING,
    DIR_OPTUNA_ML, DIR_OPTUNA_DL, DIR_TRAINED_MODELS, DIR_PREDICTIONS,
    DIR_PERFORMANCE, DIR_FIGURES, DIR_REPORTS,
]


def ensure_output_dirs():
    for d in ALL_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 2026
N_CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_OPTUNA_TRIALS_ML = 100
# Reduced from the original brief's 50 (Section 33) to 20, per user decision
# 2026-08-12: on CPU (GPU temporarily unavailable), FT-Transformer/TabNet
# trials were observed taking ~1-7.5h each (attention/embedding cost), making
# 50 trials impractical within a reasonable calendar time. Applied uniformly
# across all 3 DL families for a fair, consistent trial budget. This is a
# documented deviation from the original spec -- note it in the Stage 1
# summary. Override per-run with `05_tune_dl.py --n-trials N`.
N_OPTUNA_TRIALS_DL = 20

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
DATASETS = ("ICU", "MIMIC")

# ---------------------------------------------------------------------------
# Identifiers (never predictors; used only for record-keeping / prediction files)
# ---------------------------------------------------------------------------
# 'No' is a unique per-row admission counter in both files (verified: nunique ==
# nrows in both ICU.csv and mimic.csv). Neither file provides a separate
# patient-level identifier, so repeat ICU stays by the same patient cannot be
# detected from the source data. This is documented explicitly (see
# clinical_ai/reports/dataset_summary.csv notes) and each row is therefore
# treated as an independent admission/ICU stay for CV purposes (plain
# stratified CV, not patient-grouped CV).
ID_COL_ICU = "No"
ID_COL_MIMIC = "No"
MIMIC_PROVENANCE_COLS = ["source_dataset", "source_subject_id", "source_hadm_id", "source_stay_id"]

# ---------------------------------------------------------------------------
# Outcome / target
# ---------------------------------------------------------------------------
# CORRECTED 2026-08-12 (user decision, overrides the original brief's Section 4
# text verbatim): the primary outcome is HOSPITAL mortality, not ICU-stay
# mortality. The original brief said "ICU_mortality NOT hospital mortality";
# the user explicitly reversed that after reviewing the pipeline output.
#
# Verified encodings (2026-08-12 inspection):
#   ICU.csv   hospital_mortality: '0' (survived) / '1' (died) [row counts withheld in public release]. No missing.
#   mimic.csv hospital_mortality: 'No' (survived) / 'Yes' (died) [row counts withheld in public release]. No missing.
# Harmonized target: 0 = survived the hospitalization, 1 = died in-hospital.
#
# Relationship to ICU_mortality (now a leakage-excluded predictor, see
# OUTCOME_LEAKAGE_COLS below): in ICU.csv, ICU_mortality==1 is a strict subset
# of hospital_mortality==1 (every ICU death [count withheld] also has
# hospital_mortality==1); hospital_mortality adds the patients who survived
# the ICU stay but died later in the same hospitalization. This is why the
# 'lenth_of_stay' missing-at-death pattern (see LOS section below) persists
# under this target too, just at a smaller share (most, not all, deaths,
# vs 100% under the old ICU_mortality target).
TARGET_RAW_COL = "hospital_mortality"
TARGET_COL = "hospital_mortality"
TARGET_ENCODING = {
    "ICU": {"0": 0, "1": 1},
    "MIMIC": {"No": 0, "Yes": 1},
}

# ---------------------------------------------------------------------------
# ICU length-of-stay (landmark eligibility variable) — NEVER a predictor.
# ---------------------------------------------------------------------------
# Only one length-of-stay-like field exists per row: 'lenth_of_stay' [sic].
#   - ICU.csv: integer values (0, 1, 2, 3, ... [maximum withheld]). [share withheld] of rows == 0.
#   - mimic.csv: continuous fractional-day values (example values withheld), which
#     matches MIMIC-IV's icustays.los convention (ICU length of stay in
#     fractional days computed from ICU intime/outtime). Since both files
#     share one harmonized schema, 'lenth_of_stay' is inferred to represent
#     ICU length of stay in days in both cohorts (ICU.csv's own values appear
#     coarser/rounded to whole days, consistent with chart-abstraction rather
#     than EHR-timestamp derivation). This is a data-driven inference, not an
#     administrative confirmation -- flagged for clinical-team sign-off in
#     clinical_AI_stage1_summary.md, matching the existing project convention
#     of flagging unresolved semantics (see merge/downstream_analysis/PLAN.md
#     re: in.hospital.mortality).
#   - CRITICAL FINDING: in ICU.csv, 'lenth_of_stay' is missing for exactly
#     100% of ICU deaths (the subset of hospital deaths who
#     died IN the ICU) and 0% of survivors, consistently across both hospital
#     sites and all years 2006-2020. There is no ICU admission timestamp
#     field to reconstruct LOS for the deceased from death_date. Under the
#     current hospital_mortality target this means most [count withheld] of
#     target==1 rows have missing LOS (the remaining hospital deaths
#     survived their ICU stay and do have a recorded LOS). In mimic.csv the
#     same field is missing for only ~1% of hospital deaths and ~0.02% of
#     survivors (unremarkable / plausible random missingness).
#   - USER DECISION (2026-08-12): for ICU.csv, a missing 'lenth_of_stay'
#     combined with a positive outcome (hospital_mortality == 1) is treated
#     as "presumed >= 24h / landmark-eligible" rather than excluded. This
#     preserves the death signal (all hospital deaths remain usable) at
#     the cost of not being able to verify a small unknown fraction who may
#     have died within the first 24h. This assumption is applied ONLY to
#     ICU.csv rows with a positive outcome and missing LOS; it is documented
#     per-row in the landmark cohort output (landmark_basis column) and
#     reported in aggregate in landmark_cohort_summary.csv so the assumption
#     is fully auditable. The rule is expressed generically in
#     harmonize.compute_landmark_eligibility() against whatever config.TARGET_COL
#     is (not hardcoded to a specific mortality definition), so it applies
#     correctly under this hospital_mortality target without code changes.
LOS_COL = "lenth_of_stay"
LOS_UNIT_ASSUMED = "days"
LANDMARK_DAYS = 1.0  # >= 24h
LOS_MISSING_DEATH_ASSUMPTION = "presumed_eligible"  # applies to ICU.csv only, per user decision

# ---------------------------------------------------------------------------
# Explicitly excluded predictors (given, not to be re-litigated)
# ---------------------------------------------------------------------------
EXPLICITLY_EXCLUDED_PREDICTORS = [
    "admission",
    "age_related_cci",
    "cardiac arrest",
    "pitt bacteremia score",
    "Before amission to ICU",
    "before the ICU admission LOS",
]

# ---------------------------------------------------------------------------
# 'op' — retained in the harmonized dataset, never used as an AI predictor.
# ---------------------------------------------------------------------------
OP_COL = "op"

# ---------------------------------------------------------------------------
# Conventional severity scores — comparators only, never AI predictors.
# ---------------------------------------------------------------------------
# apach2, SOFA, qSOFA, SAPS2 are available (0% missing) in BOTH ICU.csv and
# mimic.csv and are the four requested comparators.
SEVERITY_SCORE_COMPARATORS = ["apach2", "SOFA", "qSOFA", "SAPS2"]
SEVERITY_SCORE_DISPLAY_NAMES = {
    "apach2": "APACHE II",
    "SOFA": "SOFA",
    "qSOFA": "qSOFA",
    "SAPS2": "SAPS II",
}
# apache3, SAPS3, MPM02, MPM03 exist in ICU.csv but are 100% missing in
# mimic.csv -> cannot support bidirectional comparison, not part of the
# requested comparator set (Section 39), excluded entirely from both the
# predictor matrix and the comparator set. Documented in the leakage manifest.
SEVERITY_SCORES_ICU_ONLY_EXCLUDED = ["apache3", "SAPS3", "MPM02", "MPM03"]

# ---------------------------------------------------------------------------
# Dataset/era-identifying columns — excluded from predictors (transportability)
# ---------------------------------------------------------------------------
# 'hospital' / 'No': raw identifiers.
# 'year': ICU.csv spans 2006-2020 (real calendar years); mimic.csv spans
#   2110-2213 (MIMIC-IV's standard date-shifted de-identification years).
#   The two ranges are entirely non-overlapping, so 'year' would let a model
#   trivially learn dataset identity instead of physiology -> excluded.
# 'icu_type': vocabularies are completely disjoint and institution-specific
#   (ICU.csv: CCU/ICU/NCU; mimic.csv: MICU/SICU/CVICU/... full MIMIC unit
#   names) and it is not in the Section 11 candidate-predictor list ->
#   excluded (not requested, and would harm transportability if included).
DATASET_IDENTIFYING_EXCLUDED = ["hospital", "year", "icu_type"]

# ---------------------------------------------------------------------------
# Leakage-risk exclusions (Section 12)
# ---------------------------------------------------------------------------
# 'hospital_mortality' itself is the target (TARGET_RAW_COL) and is handled
# separately, not listed here. 'ICU_mortality' is now the excluded one: it is
# a near-perfect subset flag of hospital_mortality (see Outcome/target note
# above) and would leak the target almost completely if used as a predictor.
OUTCOME_LEAKAGE_COLS = ["death_date", "x30dmortality", "ICU_mortality"]

CULTURE_LEAKAGE_COLS = [
    "first_infection_site", "BCx_at_ICU_admission", "initial BCx day",
    "ABA.B", "CAL.B", "ECO.B", "EFA.B", "EFM.B", "PMA.B", "KPN.B",
    "other candida.B", "Other.B", "PAE.B", "SAU.B",
    "UCx_at_ICU_admission", "initial UCx day",
    "ABA.U", "CAL.U", "ECO.U", "EFA.U", "EFM.U", "KPN.U",
    "other candida.U", "Other.U", "PAE.U", "PMA.U", "SAU.U",
    "pneumonia", "initial RCx day",
    "ABA.rx", "CAL.rx", "ECO.rx", "KPN.rx", "EFA.rx", "EFM.rx",
    "Other.rx", "Candida.rx", "Other yeast.rx", "PAE.rx", "PMA.rx", "SAU.rx",
    "BFCX", "initial culture positive date", "Type_of_infection",
]

# Antibiotic-class '.use' flags and antibiotic_use/anti_start_date have no
# confirmed first-24h time window in the source data (only anti_start_date
# gives a day offset, and it is frequently 'Not use'/absent) -> conservatively
# treated as unconfirmed-timing / potential full-stay exposure -> excluded.
ANTIBIOTIC_LEAKAGE_COLS = [
    "antibiotic_use", "anti_start_date",
    "1st cepha.use", "2nd cepha.use", "3rd cepha.use", "4th cepha.use",
    "aminoglycoside.use", "antifungal.use", "betalactam/lactamase.use",
    "carbapenem.use", "colistin.use", "cotrimoxazole.use",
    "erythromycin.use", "glycopeptide.use", "macrolide.use",
    "Metronidazole.use", "penicillin.use", "Quinolone.use", "tetracycline.use",
]

ALL_LEAKAGE_COLS = OUTCOME_LEAKAGE_COLS + CULTURE_LEAKAGE_COLS + ANTIBIOTIC_LEAKAGE_COLS

# ---------------------------------------------------------------------------
# Candidate AI predictors (Section 11) — exact source column names
# ---------------------------------------------------------------------------
DEMOGRAPHIC_PREDICTORS = ["age", "sex"]
ADMISSION_TYPE_PREDICTOR = ["type"]  # medical / surgical

COMORBIDITY_PREDICTORS = [
    "cancer", "CHD", "connective tissue ds", "COPD", "CVD", "dementia", "DM",
    "ESRD", "hematologic mal", "hemiplegia", "HTN", "leukemia", "liver ds",
    "lung ds", "lymphoma", "MI", "moderate/severe liver ds", "PAOD",
    "renal ds", "ulcer", "transplantation",
]

VITAL_PREDICTORS = [
    "BT min", "BT max", "MAP min", "MAP max", "HR min", "HR max",
    "RR min", "RR max", "SBP_min", "SBP_max", "GCS.score",
]

LAB_PREDICTORS = [
    "PH min", "PH max", "Na min", "Na max", "K min", "K max",
    "Cr min", "Cr max", "Hct min", "Hct max", "WBC min", "WBC max",
    "PLT min", "PLT max", "bilirubin min", "bilirubin max",
    "BUN min", "BUN max", "Glucose min", "Glucose max",
    "pCO2 min", "pCO2 max", "albumin min", "albumin max",
    "HCO3 min", "HCO3 max",
]

ORGAN_SUPPORT_PREDICTORS = ["ventilator", "vasopressor"]
URINE_OUTPUT_PREDICTOR = ["Urine_output"]

# Binary Yes/No (or M/F, medical/surgical) predictors -> encoded 0/1 directly
BINARY_CATEGORICAL_PREDICTORS = (
    ["sex", "type"] + COMORBIDITY_PREDICTORS + ORGAN_SUPPORT_PREDICTORS
)

# Continuous/count numeric predictors -> Universal pipeline (median impute +
# missingness indicator [+ scaling for models that need it]) or native-NaN
# for tree models in Pipeline B.
NUMERIC_PREDICTORS = (
    ["age"] + VITAL_PREDICTORS + LAB_PREDICTORS + URINE_OUTPUT_PREDICTOR
)

ALL_CANDIDATE_PREDICTORS = sorted(set(
    DEMOGRAPHIC_PREDICTORS + ADMISSION_TYPE_PREDICTOR + COMORBIDITY_PREDICTORS
    + VITAL_PREDICTORS + LAB_PREDICTORS + ORGAN_SUPPORT_PREDICTORS
    + URINE_OUTPUT_PREDICTOR
))

# Binary encoding maps (verified encodings; unseen values raise, not silently coerced)
YESNO_MAP = {"Yes": 1, "No": 0}
SEX_MAP = {"M": 1, "F": 0}  # sex_male = 1
TYPE_MAP = {"surgical": 1, "medical": 0}  # type_surgical = 1

# ---------------------------------------------------------------------------
# DL model families (Section 31)
# ---------------------------------------------------------------------------
DL_MODEL_FAMILIES = ["MLP", "FTTransformer", "TabNet"]

# ---------------------------------------------------------------------------
# Conda env expected for all stages of this pipeline (verified installed):
# pycaret 3.3.2, optuna 4.6.0, scikit-learn 1.4.2, pandas 2.1.4, numpy 1.26.4,
# torch 2.5.1 (cu118), catboost 1.2.8, lightgbm 4.6.0, xgboost 2.0.3,
# pytorch-tabnet 4.1.0, shap 0.44.1, imbalanced-learn 0.14.0
# (checked 2026-08-12: not present in 'bioinfo_ml'; present in 'gpt')
# ---------------------------------------------------------------------------
CONDA_ENV = "gpt"


def get_n_jobs() -> int:
    """CPU budget for this job. Honors SLURM_CPUS_PER_TASK (matches the
    convention already used elsewhere in this repo, e.g.
    downstream_analysis/slurm/03_*.sh / 04_*.sh 'SLURM_CPUS_PER_TASK n_cores
    fix') so total threads never exceed what SLURM actually allocated."""
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        return max(int(v), 1)
    return max(os.cpu_count() or 4, 1)
