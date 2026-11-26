#!/usr/bin/env python3
"""lit53_excluded67_scoring.py — scores the 67 patients excluded by the
current hospital-admission-anchored cohort flag, per
`reports/paper1/EXCLUDED67_SCORING_DESIGN.md` (design memo, 2026-09-04) and
frozen, before this script is ever executed, by
`reports/paper1/EXCLUDED67_PRESPECIFICATION.md`.

*** THIS SCRIPT MUST NOT BE RUN UNTIL EXCLUDED67_PRESPECIFICATION.md IS
COMMITTED. Its own integrity check (step 0 below) enforces this: it reads
the pinned SHA-256 for this file out of that prespecification and refuses
to proceed if this file has been edited since. ***

WHAT THIS IS. A post-hoc, outcome-motivated sensitivity/disclosure analysis
(NOT a change to the primary 302-patient result) that asks whether the
paper's null survives in the 67 patients who are excluded from the primary
302-patient cohort purely by an artefact of the cohort-definition window
(hospital-admission-anchored day 0-3) even though they are ICU-day-0/1
patients by the same anchor the primary cohort itself uses for its own
ICU-admission date. The 67 have 41.8% mortality (vs 23.5% in the 302) --
see EXCLUDED67_PRESPECIFICATION.md sec 0.1 for the one non-blind number in
this whole exercise (AUROC(SOFA) in the 67 = 0.7537, computed during design
feasibility, NOT reproduced as though it were a blind result).

SINGLE-STAGE, NOT TWO (deviation from the design's own default, exercising
the design's own permission to do this -- sec 7 item 6 / sec 5.1). The
design defaulted to a two-stage split (semantic_microbiome under `base`
sklearn 1.8.0; clinical metadata under `gpt` sklearn 1.4.2) because
unpickling the frozen M1 joblib (written under 1.8.0) under `gpt`'s 1.4.2
is a downgrade and raises/misbehaves. The design's own sec 7 item 6 flagged
that it had NOT verified whether the *clinical* CatBoost bundle loads
cleanly under `base`, and instructed: if it does, prefer a single stage.
It was verified before this script was written (read-only `joblib.load()` +
a dummy `predict_proba()` call under `base`, sklearn 1.8.0, catboost
1.2.10): the bundle loads with zero warnings and predicts correctly. So
this ONE script does everything Stage A + Stage B would have done, under
`base` only. No `clinical metadata/scripts/59_*.py` sibling exists; this
file is the whole analysis. `clinical metadata/scripts/utils/{model_io,
preprocessing_pipeline}.py` are imported directly (sys.path insert), same
pattern already used one level down in these files' own docstrings ("copied
in / imported, not reinvented"). `10_apply_reduced_model_to_microbiome_
cohort.py`'s `build_mapped_X()` is COPIED here (not imported -- that module
is digit-prefixed and executes network-scan side effects if imported as
__main__; copying a 15-line pure function is safer and is this project's
own established idiom, see script 41's docstring), byte-identical, credited.

FIVE TRAPS THIS SCRIPT IMPLEMENTS DELIBERATELY (design memo sec headers):

  1. THE ID JOIN (design sec 1.1). `metadata/microbiome.clinical.csv`'s
     `list`/`No` columns are a MIXED scheme (180/430 rows carry the
     `ICUxxxx`-style patient_id_clinical directly in `list`; the other
     250/430 carry the raw KCDC number in both `list` and `No`, with the
     clinical ID only in `No`... except `No` for those same 250 rows
     ALSO reads a KCDC-style value in a few, so `No`/`list` cannot be
     trusted directly). Root fix (documented once already for a related
     defect in `lit08`'s bugfix docstring, 2026-08-14): join `KCDC_no` to
     `harmonized/sample_id_mapping.csv` rows where
     `match_method == 'kcdc_no_strip_wm'` (covers 250/430), then
     `.fillna()` with `microbiome.clinical.csv['list']` (covers the other
     180/430, which already hold `ICUxxxx` directly). Verified BEFORE this
     script was written (read-only join check, not a model computation):
     430/430 unique, all match `^ICU\\d+$`, 302/302 overlap with
     `discovery_cohort_primary.csv`. Asserted again below, unconditionally.

  2. THE SKLEARN ENVIRONMENT SPLIT (design sec 5.1). See "SINGLE-STAGE,
     NOT TWO" above -- resolved by verifying the clinical bundle loads
     under `base`, eliminating the need for a cross-environment CSV
     handoff entirely.

  3. NEVER APPEND TO `structured/literature_core_features.csv`. This
     script writes a NEW file, `structured/literature_core_features_
     excluded67.csv`, and never opens the frozen-adjacent file in "a" or
     "w" mode -- only "r" (via pandas.read_csv), for the reproduction gate.

  4. BLAS EXPORTS MUST BE 1, NOT `${SLURM_CPUS_PER_TASK}`. See the SLURM
     wrapper `semantic_microbiome/slurm/run_lit53_excluded67_scoring.sh`:
     the outer parallelism is a `joblib.Parallel` loop over the 2000 R1
     draws (`R1_N_JOBS` below, from `SLURM_CPUS_PER_TASK`), and EACH
     worker's `LogisticRegressionCV` is called with `n_jobs=1` -- nesting
     both on a 256-core node would oversubscribe by outer_workers x
     inner_fits x BLAS_threads and run slower than serial. Copied script
     57's wrapper *structure*, changed the *value*, documented why (design
     sec 5.4(a)/(b)) -- NOT copied from 41_42/45's sh files, which predate
     the BLAS-export fix.

  5. THE REPRODUCTION GATE RUNS BEFORE ANY NEW NUMBER IS EMITTED (design
     sec 5.2), as an unconditional `assert`, never a warning: rebuild the
     16 Core features for AGI_BGISEQ (N=210) and PRJNA842425 (N=71) from
     the raw MetaPhlAn/CoverM matrices via THIS SCRIPT'S OWN feature-
     construction code path (`build_literature_core()`, below), assert
     they match the frozen `structured/literature_core_features.csv` to
     float tolerance, then score the frozen M1 joblib on them and assert
     the AUROCs reproduce 0.618 (AGI) and 0.567 (PRJNA842425) exactly as
     recorded in `public_validation/internal_external_auroc_significance_
     test.md` / `model_performance_M1_M3.tsv`. (Correction to the design
     memo's own prose here, not a substantive change: the design text says
     "128 PRJNA842425 samples" -- the actual PRJNA842425 subcohort in
     `merge/output/sample_metadata.tsv` is N=71 [32 deaths/39 survivors];
     128 is 71 + the two 100%-death "sanity check only" cohorts [33+24],
     which carry no AUROC and are not part of this gate. N=71 is what
     `model_performance_M1_M3.tsv` reports 0.567 against, and is what this
     script reproduces.) Any failure here aborts via a plain Python
     AssertionError before the 67 are ever touched.

ESTIMANDS COMPUTED (design sec 3.2 / EXCLUDED67_PRESPECIFICATION.md sec 2 --
frozen there, not redefined here):
  E1  AUROC(SOFA) in the 67                                    [not blind]
  E2  AUROC(M1-frozen probability) in the 67                   [blind]
  E3  mortality ~ SOFA + logit(M1-frozen prob) in the 67: adjusted OR
      (profile-likelihood 95% CI) + LR test vs SOFA-only + paired
      Delta-AUROC (2000-resample percentile bootstrap CI)       [blind]
  E4  AUROC(clinical_AI_probability_PRIMARY), direct application
      (the 0.7560-family C0, NOT the 0.7528 OOF family)         [blind]
  R1  procedurally matched reference band for E2 (2000 draws of 28
      deaths/39 survivors from the 302, M1 spec refit on the 235-patient
      complement each time -- the 67 never enter any fit); the same
      resampling done fit-free for E1/E4 against the 302's existing scores
  E5  the pooled ICU-day-0/1 population (302's own icu_day0_1 subset,
      N=281, PRE the item-50 18-patient attestation -- see note below --
      plus the 67, N=348 total, matching the design's own stated figure):
      348-naive frozen-M1 AUROC, computed and written with a
      CONTAMINATED -- DO NOT QUOTE label, confined to a supplementary
      output; SOFA and C0-direct on the 348 are clean and reported
      normally.
  E6  descriptive comparison, 67 vs 302 (SOFA/age/sex/ventilator/specimen
      type/16 M1 markers/hospital-day and ICU-day gaps)

NOTE on E5's N=281 (302-side): `clinical metadata/scripts/58_phase_x_
landmark_anchor_sensitivity.py`'s `icu_day0_1` mask, as amended by
revision_log.md item 50 (2026-09-04 owner attestation for 18 patients with
no recorded ICU admission date), gives N=299, not 281. The design memo
(also dated 2026-09-04) states E5's population as "N=348 = 281 + 67" without
the attestation. Because the prespecification is frozen against the design
memo's LITERAL stated quantities, this script reproduces the PRE-attestation
mask (no `ICU_DAY0_ATTESTED_2026_09_04` override) so E5 = 281 + 67 = 348, as
declared, and logs the post-attestation alternative (299 + 67 = 366) as an
explicit note rather than silently using it -- changing which population E5
means is a redefinition, and EXCLUDED67_PRESPECIFICATION.md sec 6
(escalation pre-commitment) forbids redefining any estimand after the fact.
E5 is secondary and contaminated-labelled throughout regardless.

Reads (read-only -- nothing here is ever opened in "a"/"w" mode except this
script's own new output files):
  semantic_microbiome/supervised_semantic/frozen_models/M1_literature_core_LR.joblib
  semantic_microbiome/structured/literature_core_features.csv           (reproduction gate only)
  semantic_microbiome/harmonized/{sample_id_mapping,discovery_cohort_primary}.csv
  merge/output/metaphlan3/metaphlan3_{species,genus,family,phylum}_merged.tsv
  merge/output/mag_summary/{gtdb/gtdb_mag_taxonomy,coverm/coverm_abundance_matrix}.tsv
  merge/output/sample_metadata.tsv
  ICU_public/metadata/microbiome.clinical.csv
  ICU_public/new_meta.csv
  ICU_public/clinical metadata/ICU.csv                                  (usecols=['year'] only, overlap gate)
  ICU_public/clinical metadata/clinical_ai/predictions/phase_x_c1_c2_oof_probabilities.csv
  ICU_public/clinical metadata/clinical_ai/trained_models/ML_reduced/Gangnam/catboost/ (bundle)
  reports/paper1/EXCLUDED67_PRESPECIFICATION.md                          (self-hash pin only)
Writes:
  structured/literature_core_features_excluded67.csv
  public_validation/excluded67_m1_probabilities.csv
  public_validation/excluded67_m1_reference_band.csv
  reports/paper1/excluded67_scoring_results.csv / .md      (E1-E4, E6 -- TS9)
  reports/paper1/excluded67_e5_pooled348_supplement.csv / .md (E5, CONTAMINATED-labelled)
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from scipy.optimize import brentq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEMANTIC, ICU_PUBLIC, HARMONIZED_DIR, STRUCTURED_DIR, METAPHLAN_DIR, SEED, today_str

# ---------------------------------------------------------------------------
# 0. paths and constants
# ---------------------------------------------------------------------------
MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
MAG_SUMMARY = MERGE_OUTPUT / "mag_summary"
FROZEN_DIR = SEMANTIC / "supervised_semantic" / "frozen_models"
M1_JOBLIB = FROZEN_DIR / "M1_literature_core_LR.joblib"

CLINICAL_CSV = ICU_PUBLIC / "metadata" / "microbiome.clinical.csv"
NEW_META_CSV = ICU_PUBLIC / "new_meta.csv"
SID_MAP_CSV = HARMONIZED_DIR / "sample_id_mapping.csv"
DISCOVERY_CSV = HARMONIZED_DIR / "discovery_cohort_primary.csv"

CLINICAL_META_BASE = ICU_PUBLIC / "clinical metadata"
CLINICAL_SCRIPTS_DIR = CLINICAL_META_BASE / "scripts"
ICU_CSV_RAW = CLINICAL_META_BASE / "ICU.csv"
OOF_302_CSV = CLINICAL_META_BASE / "clinical_ai" / "predictions" / "phase_x_c1_c2_oof_probabilities.csv"
CLINICAL_BUNDLE_DIR = (CLINICAL_META_BASE / "clinical_ai" / "trained_models" / "ML_reduced" / "Gangnam" / "catboost")

PAPER1_DIR = SEMANTIC / "reports" / "paper1"
PUBVAL_DIR = SEMANTIC / "public_validation"
PRESPEC_MD = PAPER1_DIR / "EXCLUDED67_PRESPECIFICATION.md"

OUT_CORE67_CSV = STRUCTURED_DIR / "literature_core_features_excluded67.csv"
OUT_PROB67_CSV = PUBVAL_DIR / "excluded67_m1_probabilities.csv"
OUT_R1_CSV = PUBVAL_DIR / "excluded67_m1_reference_band.csv"
OUT_RESULTS_CSV = PAPER1_DIR / "excluded67_scoring_results.csv"
OUT_RESULTS_MD = PAPER1_DIR / "excluded67_scoring_results.md"
OUT_E5_CSV = PAPER1_DIR / "excluded67_e5_pooled348_supplement.csv"
OUT_E5_MD = PAPER1_DIR / "excluded67_e5_pooled348_supplement.md"

EXPECTED_M1_JOBLIB_SHA256 = "f0d7ec63524068da04f157344452318243e973ce6c03bde7a753dc3df710c8d5"

N_DRAWS_R1 = 2000          # design sec 3.4
N_BOOT_E3 = 2000           # design sec 3.2 ("2000-resample percentile bootstrap CI")
N_BOOT_AUROC = 2000        # for the plain AUROC CIs reported alongside E1/E2/E4 (this project's lit15 convention)
R1_N_JOBS = max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1)
EPS = 1e-6

print(f"[lit53] host job cpus (SLURM_CPUS_PER_TASK)={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}, "
      f"R1 outer joblib workers={R1_N_JOBS} (each worker's LogisticRegressionCV runs n_jobs=1 -- design sec 5.4)")

sys.path.insert(0, str(CLINICAL_SCRIPTS_DIR))
from utils import model_io  # noqa: E402  (only cross-project import: the frozen clinical bundle loader)


# ---------------------------------------------------------------------------
# 0.1 self-integrity check -- refuse to run if this file has been edited
#     since EXCLUDED67_PRESPECIFICATION.md pinned its SHA-256 (design sec
#     4.3 item 8: "re-verified by the script at run time"). Non-circular:
#     the pinned hash lives in a DIFFERENT, already-committed file, never
#     inside this one.
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_self_hash():
    this_file = Path(__file__).resolve()
    this_hash = sha256_file(this_file)
    if not PRESPEC_MD.exists():
        raise RuntimeError(
            f"{PRESPEC_MD} does not exist -- the prespecification must be committed BEFORE this script is "
            f"ever run (EXCLUDED67_SCORING_DESIGN.md sec 4.3; task instruction). Refusing to proceed.")
    text = PRESPEC_MD.read_text()
    m = re.search(rf"`{re.escape(this_file.name)}`\s*\|\s*`([0-9a-f]{{64}})`", text)
    if not m:
        raise RuntimeError(
            f"could not find a pinned SHA-256 for {this_file.name!r} in {PRESPEC_MD} -- refusing to proceed "
            f"without a verifiable integrity pin.")
    expected = m.group(1)
    if this_hash != expected:
        raise RuntimeError(
            f"SCRIPT INTEGRITY CHECK FAILED: {this_file} sha256={this_hash} does not match the hash pinned in "
            f"{PRESPEC_MD} ({expected}). This file was edited after the prespecification was frozen -- "
            f"see EXCLUDED67_PRESPECIFICATION.md sec 0 for why this check exists. Update the prespecification's "
            f"pinned hash (a new, dated amendment, not a silent edit) before running the changed script.")
    print(f"[lit53] self-integrity check PASSED: {this_file.name} sha256={this_hash} matches the hash pinned "
          f"in {PRESPEC_MD.name}.")


# ---------------------------------------------------------------------------
# 1. shared helpers
# ---------------------------------------------------------------------------
def logit(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def bootstrap_auroc(y, score, n_boot=N_BOOT_AUROC, seed=SEED):
    """Percentile bootstrap CI for a plain AUROC -- byte-identical
    resampling convention to 58_phase_x_landmark_anchor_sensitivity.py /
    56_phase_x_panelA_discrimination_ci.py: np.random.RandomState(seed),
    rng.randint(0, n, n) with-replacement patient resampling, single-class
    resamples skipped, 2.5/97.5 percentiles."""
    y = np.asarray(y)
    score = np.asarray(score)
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
    return float(point), float(lo), float(hi), len(boots)


def paired_bootstrap_delta_auroc(y, p_ref, p_new, n_boot=N_BOOT_E3, seed=SEED):
    """Paired bootstrap CI for AUROC(p_new) - AUROC(p_ref), same resampling
    convention as bootstrap_auroc() above (single shared per-resample
    patient index, so the two AUROCs in each resample are on the SAME
    resampled patients -- the 'paired' part)."""
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    p_new = np.asarray(p_new)
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_ref)
    return float(point), float(lo), float(hi), len(deltas)


def profile_likelihood_ci(y, X_with_const, target_idx, level=0.95, max_expand=40):
    """Profile-likelihood 95% CI for one coefficient of a binomial GLM, via
    offset-refit + chi-square(1) root-finding. Not built into statsmodels
    directly (Wald CIs are its default); implemented here because
    EXCLUDED67_PRESPECIFICATION.md/design sec 3.2 explicitly calls for a
    profile-likelihood CI on the M1 term at n=67 (Wald normal-approximation
    CIs are least reliable exactly in this small-N regime).

    X_with_const: design matrix INCLUDING the constant column (as returned
    by sm.add_constant). target_idx: column index (0=const) of the
    coefficient to profile. Returns (ci_low, ci_high, beta_hat, converged).
    """
    full_fit = sm.GLM(y, X_with_const, family=sm.families.Binomial()).fit()
    beta_hat = float(full_fit.params[target_idx])
    ll_full = float(full_fit.llf)
    crit = scipy_stats.chi2.ppf(level, 1)
    other_idx = [j for j in range(X_with_const.shape[1]) if j != target_idx]
    X_other = X_with_const[:, other_idx]
    x_target = X_with_const[:, target_idx]

    def profile_deviance(beta):
        offset = x_target * beta
        fit = sm.GLM(y, X_other, family=sm.families.Binomial(), offset=offset).fit()
        return 2 * (ll_full - fit.llf) - crit

    se_wald = float(full_fit.bse[target_idx])
    step0 = max(se_wald, abs(beta_hat) * 0.25, 0.1)

    def _find_bound(direction):
        bound = beta_hat
        step = step0
        for _ in range(max_expand):
            bound = bound + direction * step
            if profile_deviance(bound) > 0:
                lo, hi = (beta_hat, bound) if direction > 0 else (bound, beta_hat)
                try:
                    return brentq(profile_deviance, lo, hi), True
                except Exception:
                    return bound, False
            step *= 1.6
        return bound, False

    ci_hi, ok_hi = _find_bound(+1)
    ci_lo, ok_lo = _find_bound(-1)
    return float(ci_lo), float(ci_hi), beta_hat, bool(ok_lo and ok_hi)


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"[lit53] wrote {path} ({df.shape[0]} rows x {df.shape[1]} cols)")


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[lit53] wrote {path}")


# ---------------------------------------------------------------------------
# 2. feature construction -- byte-identical formulas to
#    lit08_build_literature_core_extended.py, generalized so the SAME code
#    path builds features for (a) the reproduction-gate public cohorts
#    (no renaming needed -- their raw sample_id IS the target id, exactly
#    as lit08 leaves public columns untouched) and (b) the excluded 67
#    (renamed from raw view sample_ids to patient_id_clinical via
#    sample_id_mapping.csv, exactly the mechanism lit08 calls
#    `gangnam_rename_dict` for the 302 -- generalized here to any target ID
#    set, not hardcoded to the 302). This is "trap 5"'s reproduction gate:
#    running this SAME function on AGI/PRJNA842425 (rename_ids=None) must
#    reproduce literature_core_features.csv's existing values exactly.
# ---------------------------------------------------------------------------
def build_literature_core(target_ids, rename_ids=None) -> pd.DataFrame:
    rename_ids = set(rename_ids or [])
    sid_map = pd.read_csv(SID_MAP_CSV)

    def rename_dict(view_name):
        m = sid_map[(sid_map.view == view_name) & (sid_map.match_method != "unmatched")]
        m = m[m.patient_id_clinical.isin(rename_ids)]
        return dict(zip(m.sample_id_in_view, m.patient_id_clinical))

    levels = {}
    for level in ["species", "genus", "family", "phylum"]:
        raw = pd.read_csv(METAPHLAN_DIR / f"metaphlan3_{level}_merged.tsv", sep="\t", index_col=0).T
        levels[level] = raw.rename(index=rename_dict("MetaPhlAn")) if rename_ids else raw

    def find_col(level, leaf_name):
        hits = [c for c in levels[level].columns
                if c.lower().split("|")[-1].split("__")[-1].replace("_", " ") == leaf_name.lower()]
        return hits[0] if hits else None

    alpha_shannon = levels["species"].apply(
        lambda r: (lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum()))(r.values / 100.0), axis=1)

    genus_targets = {"Enterococcus": "harmful", "Klebsiella": "harmful", "Escherichia": "harmful",
                      "Faecalibacterium": "protective", "Prevotella": "protective", "Bacteroides": "harmful"}
    family_targets = {"Christensenellaceae": "harmful", "Erysipelotrichaceae": "harmful"}
    phylum_targets = {"Proteobacteria": "harmful"}
    species_targets = {"Clostridium leptum": "harmful"}

    core = pd.DataFrame(index=list(target_ids))
    core["Literature_alpha_diversity_shannon"] = alpha_shannon.reindex(target_ids)

    for name in genus_targets:
        col = find_col("genus", name)
        core[f"Literature_{name}_genus_relabund"] = levels["genus"][col].reindex(target_ids) if col else np.nan
    for name in family_targets:
        col = find_col("family", name)
        core[f"Literature_{name}_family_relabund"] = levels["family"][col].reindex(target_ids) if col else np.nan
    for name in phylum_targets:
        col = find_col("phylum", name)
        core[f"Literature_{name}_phylum_relabund"] = levels["phylum"][col].reindex(target_ids) if col else np.nan
    for name in species_targets:
        col = find_col("species", name)
        core[f"Literature_{name.replace(' ', '_')}_species_relabund"] = (
            levels["species"][col].reindex(target_ids) if col else np.nan)

    firm_col, bact_col = find_col("phylum", "Firmicutes"), find_col("phylum", "Bacteroidetes")
    core["Literature_Bacteroidetes_Firmicutes_ratio"] = (
        ((levels["phylum"][bact_col] + 1e-6) / (levels["phylum"][firm_col] + 1e-6)).reindex(target_ids)
        if (firm_col and bact_col) else np.nan)

    pathobiont_terms = ["Enterococcus", "Klebsiella", "Escherichia"]
    anaerobe_terms = ["Faecalibacterium", "Prevotella"]
    core["Literature_pathobiont_burden"] = core[[f"Literature_{n}_genus_relabund" for n in pathobiont_terms]].sum(
        axis=1, skipna=True)
    core["Literature_obligate_anaerobe_preservation"] = core[[f"Literature_{n}_genus_relabund"
                                                                for n in anaerobe_terms]].sum(axis=1, skipna=True)
    core["Literature_facultative_to_obligate_anaerobe_balance"] = (
        (core["Literature_pathobiont_burden"] + EPS) / (core["Literature_obligate_anaerobe_preservation"] + EPS))

    gtdb = pd.read_csv(MAG_SUMMARY / "gtdb" / "gtdb_mag_taxonomy.tsv", sep="\t")
    pathobiont_genera = {"Enterococcus", "Klebsiella", "Escherichia"}
    pathobiont_mags = set(gtdb.loc[gtdb["genus"].isin(pathobiont_genera), "MAG"])
    coverm_raw = pd.read_csv(MAG_SUMMARY / "coverm" / "coverm_abundance_matrix.tsv", sep="\t", index_col="MAG")
    coverm_raw = coverm_raw.drop(index="unmapped", errors="ignore")
    coverm = coverm_raw.rename(columns=rename_dict("MAG_abundance_CoverM")) if rename_ids else coverm_raw
    mag_pathobiont_burden = coverm.loc[coverm.index.isin(pathobiont_mags)].sum(axis=0)
    core["Genome_resolved_pathobiont_MAG_burden"] = mag_pathobiont_burden.reindex(target_ids)

    core.index.name = "sample_id"
    return core


def m1_prefixed(core: pd.DataFrame) -> pd.DataFrame:
    out = core.copy()
    out.columns = [f"LitCore_{c}" for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# 10_apply_reduced_model_to_microbiome_cohort.py's build_mapped_X(), COPIED
# verbatim (design sec 5.1: single-stage means this project's clinical
# feature-mapping logic is reused here rather than re-derived; copying a
# small pure function, not importing a digit-prefixed __main__ module, is
# this project's own established idiom -- see script 41's docstring).
# ---------------------------------------------------------------------------
COMORBIDITY_RENAME = {
    "connective.tissue.ds": "connective_tissue_ds",
    "liver.ds": "liver_ds",
    "moderate.to.severe.liver.ds": "moderate_severe_liver_ds",
}
DIRECT_COMORBIDITIES = ["cancer", "CHD", "COPD", "dementia", "DM", "hemiplegia", "HTN",
                         "leukemia", "lymphoma", "MI", "ulcer", "transplantation"]


def build_mapped_X(clin: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=clin.index)
    X["age"] = clin["age"]
    X["sex_male"] = clin["sex"].map({"M": 1, "F": 0})
    X["BT"] = clin["BT"]
    X["MAP"] = clin["MBP"]
    X["HR"] = clin["PR"]
    X["RR"] = clin["RR"]
    X["SBP"] = clin["SBP"]
    X["GCS_score"] = clin["GCS"]
    X["Cr"] = clin["Cr"]
    X["WBC"] = clin["WBC"]
    X["PLT"] = clin["PLT"]
    X["bilirubin"] = clin["T.bil"]
    X["albumin"] = clin["Alb"]
    X["ventilator"] = clin["ventilator"]
    for c in DIRECT_COMORBIDITIES:
        X[c] = clin[c]
    for src, dst in COMORBIDITY_RENAME.items():
        X[dst] = clin[src]
    return X


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    _verify_self_hash()

    print(f"[lit53] === trap 5: reproduction gate (must PASS before any new number is produced) ===")
    assert sha256_file(M1_JOBLIB) == EXPECTED_M1_JOBLIB_SHA256, (
        f"M1 joblib SHA-256 mismatch at {M1_JOBLIB} -- the frozen model has changed since this script was "
        f"written. STOP.")
    m1_pipe = joblib.load(M1_JOBLIB)

    sm_meta = pd.read_csv(MERGE_OUTPUT / "sample_metadata.tsv", sep="\t", dtype=str).set_index("sample_id")
    sm_meta["mortality"] = pd.to_numeric(sm_meta["mortality"], errors="coerce")
    agi_ids = sm_meta.index[sm_meta.subcohort == "AGI_BGISEQ"].tolist()
    prjna_ids = sm_meta.index[sm_meta.subcohort == "PRJNA842425"].tolist()
    assert len(agi_ids) == 210 and len(prjna_ids) == 71, (
        f"reproduction-gate cohort sizes changed: AGI={len(agi_ids)} (want 210), "
        f"PRJNA842425={len(prjna_ids)} (want 71). STOP.")

    repro_core = build_literature_core(agi_ids + prjna_ids, rename_ids=None)
    existing_core = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id")
    existing_core = existing_core.reindex(agi_ids + prjna_ids)[repro_core.columns]
    max_abs_diff = float(np.nanmax(np.abs(repro_core.values.astype(float) - existing_core.values.astype(float))))
    assert np.allclose(repro_core.values.astype(float), existing_core.values.astype(float),
                        atol=1e-8, rtol=1e-6, equal_nan=True), (
        f"REPRODUCTION GATE FAILED: rebuilt AGI/PRJNA842425 Core features do not match "
        f"literature_core_features.csv (max abs diff={max_abs_diff:.3e}). The feature-construction code path "
        f"in this script has drifted from lit08's. STOP -- do not proceed.")
    print(f"[lit53] feature-reconstruction check PASSED (max abs diff={max_abs_diff:.2e} vs "
          f"literature_core_features.csv, N={len(agi_ids) + len(prjna_ids)})")

    p_agi = m1_pipe.predict_proba(m1_prefixed(repro_core.reindex(agi_ids)))[:, 1]
    y_agi = sm_meta.loc[agi_ids, "mortality"].astype(int).values
    auc_agi = roc_auc_score(y_agi, p_agi)
    p_prjna = m1_pipe.predict_proba(m1_prefixed(repro_core.reindex(prjna_ids)))[:, 1]
    y_prjna = sm_meta.loc[prjna_ids, "mortality"].astype(int).values
    auc_prjna = roc_auc_score(y_prjna, p_prjna)
    print(f"[lit53] reproduced AUROC: AGI={auc_agi:.4f} (want 0.618), PRJNA842425={auc_prjna:.4f} (want 0.567)")
    assert round(auc_agi, 3) == 0.618, f"AGI AUROC reproduction FAILED: got {auc_agi:.4f}, want 0.618. STOP."
    assert round(auc_prjna, 3) == 0.567, f"PRJNA842425 AUROC reproduction FAILED: got {auc_prjna:.4f}, want 0.567. STOP."
    print("[lit53] REPRODUCTION GATE PASSED (feature path + M1 pickle + scoring path all verified end-to-end).")

    # -----------------------------------------------------------------------
    # trap 1: the ID join
    # -----------------------------------------------------------------------
    print("[lit53] === trap 1: the ID join (KCDC_no -> patient_id_clinical via sample_id_mapping, "
          "fillna via 'list') ===")
    clin = pd.read_csv(CLINICAL_CSV)
    sid_map = pd.read_csv(SID_MAP_CSV)
    disc = pd.read_csv(DISCOVERY_CSV)

    kmap = (sid_map[sid_map.match_method == "kcdc_no_strip_wm"]
            .drop_duplicates("matched_kcdc_no_or_id")
            .set_index("matched_kcdc_no_or_id")["patient_id_clinical"])
    pid = clin["KCDC_no"].map(kmap).fillna(clin["list"])
    assert pid.nunique() == 430, f"ID join did not resolve 430 unique patient IDs (got {pid.nunique()}). STOP."
    assert pid.astype(str).str.match(r"^ICU\d+$").all(), "ID join produced a non-ICU\\d+ value. STOP."
    n_overlap_302 = len(set(pid) & set(disc["patient_id_clinical"]))
    assert n_overlap_302 == 302, f"ID join did not resolve exactly the 302 discovery-cohort patients (got {n_overlap_302}). STOP."
    clin = clin.assign(patient_id_clinical=pid)
    print(f"[lit53] ID join PASSED: 430/430 unique ICU\\d+ IDs, {n_overlap_302}/302 overlap with "
          f"discovery_cohort_primary.csv.")

    # -----------------------------------------------------------------------
    # cohort rule (design sec 1.1) -- the excluded 67
    # -----------------------------------------------------------------------
    print("[lit53] === cohort rule: ICU-day-0/1 patients excluded by the current hospital-anchor flag ===")
    new_meta = pd.read_csv(NEW_META_CSV, dtype=str)
    nm_small = new_meta[["KCDC_no", "ICU입원일"]].drop_duplicates("KCDC_no")
    merged = clin.merge(nm_small, on="KCDC_no", how="left")
    merged["collection_date"] = pd.to_datetime(merged["collection_date"], errors="coerce")
    merged["ICU입원일"] = pd.to_datetime(merged["ICU입원일"], errors="coerce")
    merged["icu_gap_days"] = (merged["collection_date"] - merged["ICU입원일"]).dt.days

    excl67 = merged[(merged["icu_gap_days"].isin([0, 1])) & (merged["collection_time"] != 0)].copy()
    n67, n67_deaths = len(excl67), int(excl67["in.hospital.mortality"].sum())
    assert n67 == 67, f"excluded-67 cohort size is {n67}, not 67. STOP -- cohort rule has drifted."
    assert n67_deaths == 28, f"excluded-67 death count is {n67_deaths}, not 28. STOP."
    assert (excl67["collection_time"] == 1).all(), "not all 67 have collection_time==1. STOP."
    assert (excl67["sample_type"] == "swab").all(), "not all 67 are swab specimens. STOP."
    assert excl67["SOFA"].notna().all(), "SOFA is not complete for all 67. STOP."
    excl67_ids = excl67["patient_id_clinical"].tolist()
    assert len(set(excl67_ids)) == 67, "duplicate patient_id_clinical among the 67. STOP."
    assert len(set(excl67_ids) & set(disc["patient_id_clinical"])) == 0, (
        "overlap between the 67 and the 302 discovery cohort -- STOP, this must be zero by construction.")
    print(f"[lit53] cohort rule PASSED: N=67, deaths=28 (41.8%), all collection_time==1, all swab, "
          f"SOFA complete, 0 overlap with the 302.")

    # -----------------------------------------------------------------------
    # overlap gate (re-asserted, not inherited): the 67's admission dates
    # must fall inside 2024-05..2025-03 (month precision in the public copy), disjoint from ICU.csv's
    # 2006-2020 (config.py's documented, verified range).
    # -----------------------------------------------------------------------
    print("[lit53] === overlap gate: 67's admission dates vs ICU.csv's calendar window ===")
    excl67["admission_date"] = pd.to_datetime(excl67["admission_date"], errors="coerce")
    admit_min, admit_max = excl67["admission_date"].min(), excl67["admission_date"].max()
    assert excl67["admission_date"].notna().all(), "missing admission_date among the 67. STOP."
    assert admit_min >= pd.Timestamp("2024-05-01") and admit_max < pd.Timestamp("2025-04-01"), (
        f"67's admission-date range [{admit_min.date()}, {admit_max.date()}] falls outside the expected "
        f"2024-05..2025-03 window. STOP.")
    icu_years = pd.read_csv(ICU_CSV_RAW, usecols=["year"], dtype=str)["year"].astype(int)
    icu_year_min, icu_year_max = int(icu_years.min()), int(icu_years.max())
    assert icu_year_max < admit_min.year, (
        f"ICU.csv's year range [{icu_year_min}, {icu_year_max}] is NOT disjoint from the 67's admission window "
        f"(starts {admit_min.date()}). STOP -- the no-overlap argument does not hold.")
    print(f"[lit53] overlap gate PASSED: 67's admissions span [{admit_min.date()}, {admit_max.date()}], "
          f"ICU.csv spans [{icu_year_min}, {icu_year_max}] -- disjoint.")

    # -----------------------------------------------------------------------
    # M1 Core features for the 67 (design sec 1.2: zero re-estimation) + E2
    # -----------------------------------------------------------------------
    print("[lit53] === building 16 M1 Core features for the 67 (pure per-sample arithmetic) ===")
    mp_cov = sid_map[(sid_map.view == "MetaPhlAn") & (sid_map.match_method != "unmatched")]
    cm_cov = sid_map[(sid_map.view == "MAG_abundance_CoverM") & (sid_map.match_method != "unmatched")]
    assert len(set(mp_cov.patient_id_clinical) & set(excl67_ids)) == 67, "MetaPhlAn coverage of the 67 is not 67/67. STOP."
    assert len(set(cm_cov.patient_id_clinical) & set(excl67_ids)) == 67, "CoverM coverage of the 67 is not 67/67. STOP."

    core67 = build_literature_core(excl67_ids, rename_ids=excl67_ids)
    assert core67.notna().all(axis=1).all(), "M1 Core features for the 67 contain unexpected NaNs. STOP."
    save_csv(core67.reset_index(), OUT_CORE67_CSV)

    y67 = excl67.set_index("patient_id_clinical")["in.hospital.mortality"].astype(int).reindex(excl67_ids).values
    SOFA67 = excl67.set_index("patient_id_clinical")["SOFA"].astype(float).reindex(excl67_ids).values

    p_m1_67 = m1_pipe.predict_proba(m1_prefixed(core67))[:, 1]
    e2_auc, e2_lo, e2_hi, e2_nboot = bootstrap_auroc(y67, p_m1_67)
    print(f"[lit53] E2 (blind): AUROC(M1-frozen) in the 67 = {e2_auc:.4f} [{e2_lo:.4f}, {e2_hi:.4f}]")

    # -----------------------------------------------------------------------
    # E1 -- SOFA (already disclosed as 0.7537 at design-feasibility time,
    # NOT reproduced here as though blind; recomputed only because the
    # script needs the value programmatically for E3/R1/E5).
    # -----------------------------------------------------------------------
    e1_auc, e1_lo, e1_hi, e1_nboot = bootstrap_auroc(y67, SOFA67)
    print(f"[lit53] E1 (NOT blind -- disclosed 2026-09-04 during design feasibility as 0.7537): "
          f"AUROC(SOFA) in the 67 = {e1_auc:.4f} [{e1_lo:.4f}, {e1_hi:.4f}]")

    # -----------------------------------------------------------------------
    # E4 -- clinical bundle, direct application (the 0.7560-family C0)
    # -----------------------------------------------------------------------
    print("[lit53] === E4: frozen clinical CatBoost bundle, direct application to the 67 ===")
    bundle = model_io.load_bundle(str(CLINICAL_BUNDLE_DIR))
    clin_cols = bundle.metadata["raw_feature_columns"]
    X67_clin = build_mapped_X(excl67.set_index("patient_id_clinical").reindex(excl67_ids))
    n_missing_e4 = int(X67_clin[clin_cols].isna().sum().sum())
    print(f"[lit53] E4 input completeness: {n_missing_e4} missing values across {len(clin_cols)} features x 67 "
          f"patients (design sec 1.3 expects 8, from one patient)")
    p_c0_67 = bundle.predict_proba(X67_clin[clin_cols])
    e4_auc, e4_lo, e4_hi, e4_nboot = bootstrap_auroc(y67, p_c0_67)
    print(f"[lit53] E4 (blind): AUROC(clinical_AI_probability_PRIMARY, direct) in the 67 = "
          f"{e4_auc:.4f} [{e4_lo:.4f}, {e4_hi:.4f}]")

    save_csv(pd.DataFrame({
        "patient_id_clinical": excl67_ids,
        "hospital_mortality": y67,
        "SOFA": SOFA67,
        "m1_frozen_prob": p_m1_67,
        "clinical_AI_prob_direct": p_c0_67,
    }), OUT_PROB67_CSV)

    # -----------------------------------------------------------------------
    # E3 -- SOFA-adjusted M1 term: adjusted OR (profile-likelihood CI),
    # LR test vs SOFA-only, paired Delta-AUROC bootstrap CI. Full-data
    # (in-sample) fit -- N=67 is far too small for any meaningful CV split
    # at 2 covariates (EPV 14.0), matching this project's own convention
    # (script 41) of full-data adjusted OR/LRT fits everywhere else.
    # -----------------------------------------------------------------------
    print("[lit53] === E3 (blind): mortality ~ SOFA + logit(M1-frozen prob) in the 67 ===")
    logit_m1_67 = logit(p_m1_67)
    X_full = sm.add_constant(np.column_stack([SOFA67, logit_m1_67]))  # [const, SOFA, logit_M1]
    X_ref = sm.add_constant(SOFA67.reshape(-1, 1))                    # [const, SOFA]
    fit_full = sm.Logit(y67, X_full).fit(disp=0)
    fit_ref = sm.Logit(y67, X_ref).fit(disp=0)
    lr_stat = 2 * (fit_full.llf - fit_ref.llf)
    lr_df = X_full.shape[1] - X_ref.shape[1]
    lr_p = float(scipy_stats.chi2.sf(lr_stat, lr_df))

    coef_m1 = float(fit_full.params[2])
    se_wald_m1 = float(fit_full.bse[2])
    sd_logit_m1 = float(np.std(logit_m1_67, ddof=1))
    or_per_sd_wald = float(np.exp(coef_m1 * sd_logit_m1))

    ci_lo_beta, ci_hi_beta, beta_hat_check, profile_converged = profile_likelihood_ci(y67, X_full, target_idx=2)
    or_per_sd_profile_lo = float(np.exp(ci_lo_beta * sd_logit_m1))
    or_per_sd_profile_hi = float(np.exp(ci_hi_beta * sd_logit_m1))

    p_sofa_plus_m1_insample = fit_full.predict(X_full)  # in-sample fitted probabilities, N=67, 2 covariates
    e3_delta_auroc, e3_delta_lo, e3_delta_hi, e3_nboot = paired_bootstrap_delta_auroc(
        y67, SOFA67, p_sofa_plus_m1_insample.values if hasattr(p_sofa_plus_m1_insample, "values")
        else p_sofa_plus_m1_insample)

    print(f"[lit53] E3: OR per SD(logit M1) = {or_per_sd_wald:.3f} (Wald), profile-likelihood 95% CI = "
          f"[{or_per_sd_profile_lo:.3f}, {or_per_sd_profile_hi:.3f}] (converged={profile_converged}); "
          f"LRT chi2={lr_stat:.3f} df={lr_df} p={lr_p:.4f}; "
          f"paired delta-AUROC(SOFA+M1 vs SOFA) = {e3_delta_auroc:.4f} [{e3_delta_lo:.4f}, {e3_delta_hi:.4f}] "
          f"(n_boot={e3_nboot}/{N_BOOT_E3})")

    # -----------------------------------------------------------------------
    # R1 -- procedurally matched reference band (required companion to E2,
    # and the fit-free analogue for E1/E4). Embarrassingly parallel over
    # the 2000 draws (trap 4): deterministic per-draw seeds derived from a
    # single master RandomState(SEED), so the result is reproducible
    # regardless of worker count.
    # -----------------------------------------------------------------------
    print(f"[lit53] === R1: {N_DRAWS_R1} draws of 28 deaths/39 survivors from the 302, M1 spec refit on the "
          f"235-patient complement each time ({R1_N_JOBS} parallel workers) ===")
    oof302 = pd.read_csv(OOF_302_CSV, index_col=0)
    y302 = oof302["hospital_mortality"].values
    c0_302_direct = oof302["clinical_AI_probability_PRIMARY"].values
    assert int(y302.sum()) == 69 and len(y302) == 302, f"302-cohort deaths/N drifted (got {int(y302.sum())}/{len(y302)}). STOP."

    disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
    raw_clin_302 = (clin.drop(columns=["patient_id_clinical"])
                     .merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner")
                     .set_index("patient_id_clinical").reindex(oof302.index))
    SOFA_302 = pd.to_numeric(raw_clin_302["SOFA"], errors="coerce").values

    lit_core_302 = pd.read_csv(STRUCTURED_DIR / "literature_core_features.csv", index_col="sample_id").reindex(oof302.index)
    X_m1_302 = m1_prefixed(lit_core_302)

    deaths_idx = np.where(y302 == 1)[0]
    survivors_idx = np.where(y302 == 0)[0]
    assert len(deaths_idx) == 69 and len(survivors_idx) == 233, "302-cohort death/survivor split drifted. STOP."

    def make_m1_spec_pipe():
        # byte-identical spec to lit14_freeze_pipeline.py:78-80 / clinical
        # metadata/scripts/utils/nested_meta.py:make_lr_pipe() -- ONLY
        # n_jobs differs (1 here, since the outer joblib.Parallel loop over
        # R1 draws is where this script's parallelism lives -- trap 4).
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                          ("clf", LogisticRegressionCV(Cs=8, cv=3, penalty="elasticnet", solver="saga",
                                                        l1_ratios=[0.1, 0.5, 0.9], max_iter=3000, scoring="roc_auc",
                                                        random_state=SEED, n_jobs=1))])

    master_rng = np.random.RandomState(SEED)
    draw_seeds = master_rng.randint(0, 2**31 - 1, size=N_DRAWS_R1)

    def one_draw(b):
        rng = np.random.RandomState(draw_seeds[b])
        draw_deaths = rng.choice(deaths_idx, 28, replace=False)
        draw_survivors = rng.choice(survivors_idx, 39, replace=False)
        draw_idx = np.concatenate([draw_deaths, draw_survivors])
        complement_idx = np.setdiff1d(np.arange(len(y302)), draw_idx)
        pipe = make_m1_spec_pipe().fit(X_m1_302.iloc[complement_idx], y302[complement_idx])
        p_draw = pipe.predict_proba(X_m1_302.iloc[draw_idx])[:, 1]
        auc_m1 = roc_auc_score(y302[draw_idx], p_draw)
        auc_sofa = roc_auc_score(y302[draw_idx], SOFA_302[draw_idx])
        auc_c0 = roc_auc_score(y302[draw_idx], c0_302_direct[draw_idx])
        return auc_m1, auc_sofa, auc_c0

    r1_results = Parallel(n_jobs=R1_N_JOBS)(delayed(one_draw)(b) for b in range(N_DRAWS_R1))
    r1_band = pd.DataFrame(r1_results, columns=["auroc_m1_spec", "auroc_sofa_fitfree", "auroc_c0_direct_fitfree"])
    r1_band.insert(0, "draw", range(N_DRAWS_R1))
    r1_band.insert(1, "seed", draw_seeds)
    save_csv(r1_band, OUT_R1_CSV)

    def band_summary(col, observed):
        vals = r1_band[col].values
        pct = float((vals < observed).mean() * 100)
        return {"band_median": float(np.median(vals)), "band_ci_low": float(np.percentile(vals, 2.5)),
                "band_ci_high": float(np.percentile(vals, 97.5)), "observed_percentile": pct}

    r1_e2 = band_summary("auroc_m1_spec", e2_auc)
    r1_e1 = band_summary("auroc_sofa_fitfree", e1_auc)
    r1_e4 = band_summary("auroc_c0_direct_fitfree", e4_auc)
    print(f"[lit53] R1 band (M1 spec, 235-refit): median={r1_e2['band_median']:.4f} "
          f"[{r1_e2['band_ci_low']:.4f}, {r1_e2['band_ci_high']:.4f}]; observed E2 percentile="
          f"{r1_e2['observed_percentile']:.1f}. KNOWN ASYMMETRY (design sec 3.4): the band's models train on "
          f"235 patients, M1-frozen trained on 302 -- a small advantage FOR the observed E2, i.e. the band is "
          f"mildly conservative against the null. Disclosed, not corrected.")

    # -----------------------------------------------------------------------
    # E6 -- descriptive comparison, 67 vs 302
    # -----------------------------------------------------------------------
    print("[lit53] === E6: descriptive comparison, 67 vs 302 ===")
    clin_67_full = excl67.set_index("patient_id_clinical").reindex(excl67_ids)
    e6_rows = []
    for label, v67, v302 in [
        ("SOFA", SOFA67, SOFA_302),
        ("age", pd.to_numeric(clin_67_full["age"], errors="coerce").values,
         pd.to_numeric(raw_clin_302["age"], errors="coerce").values),
        ("ventilator", pd.to_numeric(clin_67_full["ventilator"], errors="coerce").values,
         pd.to_numeric(raw_clin_302["ventilator"], errors="coerce").values),
    ]:
        e6_rows.append({"variable": label, "N_67": int(pd.notna(v67).sum()), "median_67": float(np.nanmedian(v67)),
                         "IQR_67": f"{np.nanpercentile(v67, 25):.2f}-{np.nanpercentile(v67, 75):.2f}",
                         "N_302": int(pd.notna(v302).sum()), "median_302": float(np.nanmedian(v302)),
                         "IQR_302": f"{np.nanpercentile(v302, 25):.2f}-{np.nanpercentile(v302, 75):.2f}"})
    e6_rows.append({"variable": "sex_male_pct", "N_67": 67,
                     "median_67": round(100 * (clin_67_full["sex"] == "M").mean(), 1), "IQR_67": "",
                     "N_302": 302, "median_302": round(100 * (raw_clin_302["sex"] == "M").mean(), 1), "IQR_302": ""})
    for col in core67.columns:
        v67 = core67[col].values
        v302 = lit_core_302[col].values
        e6_rows.append({"variable": f"M1_marker__{col}", "N_67": int(pd.notna(v67).sum()),
                         "median_67": float(np.nanmedian(v67)),
                         "IQR_67": f"{np.nanpercentile(v67, 25):.4g}-{np.nanpercentile(v67, 75):.4g}",
                         "N_302": int(pd.notna(v302).sum()), "median_302": float(np.nanmedian(v302)),
                         "IQR_302": f"{np.nanpercentile(v302, 25):.4g}-{np.nanpercentile(v302, 75):.4g}"})
    e6_df = pd.DataFrame(e6_rows)

    # -----------------------------------------------------------------------
    # results table (E1-E4, E6) -- TS9
    # -----------------------------------------------------------------------
    results = pd.DataFrame([
        {"estimand": "E1", "label": "AUROC(SOFA) in the 67", "blind": False, "N": 67, "N_events": 28,
         "AUROC": round(e1_auc, 4), "CI_low": round(e1_lo, 4), "CI_high": round(e1_hi, 4),
         "R1_band_median": round(r1_e1["band_median"], 4), "R1_band_low": round(r1_e1["band_ci_low"], 4),
         "R1_band_high": round(r1_e1["band_ci_high"], 4), "R1_observed_percentile": round(r1_e1["observed_percentile"], 1)},
        {"estimand": "E2", "label": "AUROC(M1-frozen) in the 67", "blind": True, "N": 67, "N_events": 28,
         "AUROC": round(e2_auc, 4), "CI_low": round(e2_lo, 4), "CI_high": round(e2_hi, 4),
         "R1_band_median": round(r1_e2["band_median"], 4), "R1_band_low": round(r1_e2["band_ci_low"], 4),
         "R1_band_high": round(r1_e2["band_ci_high"], 4), "R1_observed_percentile": round(r1_e2["observed_percentile"], 1)},
        {"estimand": "E4", "label": "AUROC(clinical_AI_probability_PRIMARY, direct) in the 67", "blind": True,
         "N": 67, "N_events": 28, "AUROC": round(e4_auc, 4), "CI_low": round(e4_lo, 4), "CI_high": round(e4_hi, 4),
         "R1_band_median": round(r1_e4["band_median"], 4), "R1_band_low": round(r1_e4["band_ci_low"], 4),
         "R1_band_high": round(r1_e4["band_ci_high"], 4), "R1_observed_percentile": round(r1_e4["observed_percentile"], 1)},
    ])
    e3_row = pd.DataFrame([{
        "estimand": "E3", "label": "mortality ~ SOFA + logit(M1-frozen prob), M1 term", "blind": True, "N": 67,
        "N_events": 28, "OR_per_SD_M1_wald": round(or_per_sd_wald, 3),
        "OR_per_SD_M1_profile_CI_low": round(or_per_sd_profile_lo, 3),
        "OR_per_SD_M1_profile_CI_high": round(or_per_sd_profile_hi, 3), "profile_CI_converged": profile_converged,
        "LRT_chi2": round(lr_stat, 3), "LRT_df": lr_df, "LRT_p_value": round(lr_p, 4),
        "delta_AUROC_SOFA_plus_M1_minus_SOFA": round(e3_delta_auroc, 4),
        "delta_AUROC_CI_low": round(e3_delta_lo, 4), "delta_AUROC_CI_high": round(e3_delta_hi, 4),
        "delta_AUROC_n_boot": e3_nboot,
    }])

    save_csv(results, OUT_RESULTS_CSV.with_suffix(".e1e2e4.csv"))
    save_csv(e3_row, OUT_RESULTS_CSV.with_suffix(".e3.csv"))
    save_csv(e6_df, OUT_RESULTS_CSV.with_suffix(".e6_descriptive.csv"))
    save_csv(results, OUT_RESULTS_CSV)  # primary TS9 file = E1/E2/E4 + R1 band summary

    md = [
        "# TS9 -- excluded-67 scoring results (E1/E2/E4 + R1 reference band; E3 in a companion table; "
        "E6 descriptive)", "",
        f"Generated {today_str()}. Post-hoc, outcome-motivated sensitivity/disclosure analysis on the 67 "
        "patients excluded from the primary 302-patient cohort by the hospital-admission-anchor artefact. "
        "**Not a change to the primary result.** See `EXCLUDED67_PRESPECIFICATION.md` for the full frozen "
        "specification, including the E1 non-blind disclosure (sec 0.1) and the escalation pre-commitment "
        "(sec 6).", "",
        "## E1 / E2 / E4 + R1 reference band", "", df_to_md_table(results), "",
        "## E3 -- SOFA-adjusted M1 term (primary contrast)", "", df_to_md_table(e3_row), "",
        "## E6 -- descriptive comparison, 67 vs 302", "", df_to_md_table(e6_df), "",
        "## Interpretation", "",
        "Apply the interpretation rule of `EXCLUDED67_PRESPECIFICATION.md` sec 5 (CI positions, not point "
        "estimates) mechanically to the numbers above; do not re-derive or re-word it here.",
    ]
    save_text("\n".join(md), OUT_RESULTS_MD)

    # -----------------------------------------------------------------------
    # E5 -- pooled 348 (302's icu_day0_1 subset, N=281, PRE-attestation +
    # the 67). CONTAMINATED frozen-M1 AUROC, clean SOFA/C0.
    # -----------------------------------------------------------------------
    print("[lit53] === E5: pooled ICU-day-0/1 population (302's icu_day0_1 subset + the 67), secondary, "
          "M1 CONTAMINATED ===")
    icu_302 = disc.merge(new_meta[["KCDC_no", "ICU입원일"]], left_on="kcdc_no", right_on="KCDC_no", how="left")
    icu_302 = icu_302.set_index("patient_id_clinical").reindex(oof302.index)
    icu_dt_302 = pd.to_datetime(icu_302["ICU입원일"], errors="coerce")
    coll_dt_302 = pd.to_datetime(icu_302["collection_date"], errors="coerce")
    icu_gap_302 = (coll_dt_302 - icu_dt_302).dt.days
    icu_available_302 = icu_dt_302.notna()  # deliberately WITHOUT the item-50 18-patient attestation -- see docstring
    mask_day01_302 = (icu_available_302 & (icu_gap_302 >= 0) & (icu_gap_302 <= 1)).values
    ids_302_icu_day01 = oof302.index[mask_day01_302].tolist()
    n_302_side = len(ids_302_icu_day01)
    if n_302_side != 281:
        print(f"[lit53] WARNING: E5's 302-side count is {n_302_side}, not the design memo's stated 281 "
              f"(pooled N would be {n_302_side + 67}, not 348). E5 is secondary/supplementary-only; proceeding "
              f"and reporting the actual count rather than hard-aborting.")

    pooled_ids = ids_302_icu_day01 + excl67_ids
    assert len(set(ids_302_icu_day01) & set(excl67_ids)) == 0, "E5 pooling produced a duplicate ID. STOP."
    n_pooled = len(pooled_ids)

    core_pooled_302_side = lit_core_302.reindex(ids_302_icu_day01)
    core_pooled = pd.concat([core_pooled_302_side, core67.reindex(excl67_ids)], axis=0)
    p_pooled_m1_CONTAMINATED = m1_pipe.predict_proba(m1_prefixed(core_pooled))[:, 1]
    y_pooled = np.concatenate([y302[mask_day01_302], y67])
    SOFA_pooled = np.concatenate([SOFA_302[mask_day01_302], SOFA67])

    disc_ids_302side = pd.Index(ids_302_icu_day01)
    c0_pooled = np.concatenate([c0_302_direct[mask_day01_302], p_c0_67])

    auc_pooled_m1_CONTAMINATED = roc_auc_score(y_pooled, p_pooled_m1_CONTAMINATED)
    auc_pooled_sofa = roc_auc_score(y_pooled, SOFA_pooled)
    auc_pooled_c0 = roc_auc_score(y_pooled, c0_pooled)
    print(f"[lit53] E5 (N={n_pooled}): SOFA={auc_pooled_sofa:.4f} (clean), C0-direct={auc_pooled_c0:.4f} (clean), "
          f"M1-frozen={auc_pooled_m1_CONTAMINATED:.4f} -- CONTAMINATED, DO NOT QUOTE "
          f"({n_302_side} of {n_pooled} rows are in-sample for M1-frozen, which was fit on all 302).")

    e5_df = pd.DataFrame([
        {"model": "SOFA", "N": n_pooled, "N_events": int(y_pooled.sum()), "AUROC": round(auc_pooled_sofa, 4),
         "status": "CLEAN -- reportable normally"},
        {"model": "clinical_AI_probability_PRIMARY (direct)", "N": n_pooled, "N_events": int(y_pooled.sum()),
         "AUROC": round(auc_pooled_c0, 4), "status": "CLEAN -- reportable normally"},
        {"model": "M1_literature_core_LR (frozen)", "N": n_pooled, "N_events": int(y_pooled.sum()),
         "AUROC": round(auc_pooled_m1_CONTAMINATED, 4),
         "status": "CONTAMINATED -- DO NOT QUOTE (302-side rows are in-sample for this frozen model)"},
    ])
    save_csv(e5_df, OUT_E5_CSV)
    save_text("\n".join([
        "# E5 -- pooled ICU-day-0/1 supplement (SECONDARY; no E5 quantity may enter the abstract)", "",
        f"N={n_pooled} = {n_302_side} (302's icu_day0_1 subset) + 67 (excluded). Generated {today_str()}.", "",
        df_to_md_table(e5_df), "",
        "**The M1-frozen row is CONTAMINATED and MUST NOT be quoted or cited anywhere in the manuscript.** "
        f"{n_302_side} of the {n_pooled} pooled rows were part of the 302 patients `M1_literature_core_LR."
        "joblib` was fit on, so this AUROC blends in-sample optimism with genuine out-of-sample performance "
        "on the 67 -- exactly the leakage class already documented for the (separately excluded) M4 Hybrid "
        "arm. It is computed and recorded here for exactly one reason: so that nobody else computes it and "
        "quotes it (design memo sec 3.5).",
    ]), OUT_E5_MD)

    print("[lit53] === lit53_excluded67_scoring.py complete. All outputs written; no frozen artefact touched. ===")


if __name__ == "__main__":
    main()
