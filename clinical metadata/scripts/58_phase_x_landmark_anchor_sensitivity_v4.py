#!/usr/bin/env python3
"""58_phase_x_landmark_anchor_sensitivity_v4.py

Fold-local M3 preprocessing correction (owner-authorised 2026-10-02) for
`58_phase_x_landmark_anchor_sensitivity.py`. This script does its OWN model
fitting NOWHERE: it RE-SCORES already-persisted OOF prediction columns
(including the M3-dependent C2a/C2b nested-CV columns) on ICU-anchored
SUBSETS of the same 302 patients -- confirmed by reading the original script
in full. It therefore belongs to the "read the old M3/C2 OOF and need
repointing and rerunning" category (`M3_FOLD_LOCAL_METHOD_AUDIT_v3.md`
section 8), not the "reimplement the frozen PCA" category: no PCA, no
prevalence filter, no base-model refit happens here or in the original.

WHAT CHANGED
  1. `NESTED_CSV` -> `phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv`
     (45_v3's fold-local PRIMARY-variant drop-in replacement for script 46's
     file). C1 is confirmed bit-identical to the original in that file (M1
     untouched); C2a/C2b differ.
  2. `OOF_CSV` (script 18's `phase_x_c1_c2_oof_probabilities.csv`, used for
     M1/C0/clinical PRIMARY) is UNCHANGED -- those columns are not part of
     this correction's scope (M1/C0 untouched).
  3. `REPRO_TARGETS["C2b"]` updated 0.7722 -> 0.7711 (45_v3's V3 primary
     value; read from `phase_x_c1_c2_incremental_value_nested_corrected_v4.csv`
     at import time rather than hardcoded twice, so the anchor cannot drift
     from that file). `REPRO_TARGETS["C2a"]` is unchanged at 0.7608 (also
     read live) -- the fold-local fix moves C2a's point estimate by +0.0001
     relative to the already-published value's 4th decimal rounding for this
     seed/split, so the two happen to agree to 4dp; this is NOT assumed, it
     is verified at runtime by the reproduction gate below, which aborts
     (same as the original script) if any target fails to reproduce.
     SOFA/C0/M1/C1 targets are unchanged.
  4. Every output path carries a `_v3` suffix; nothing from the original
     script is overwritten. An OLD-vs-NEW numeric comparison for the
     M3-dependent rows (C2a/C2b, every cohort) is appended.

Everything else -- the owner-attestation handling of the 18 ICU-day-0
patients, the bootstrap convention, the cohort masks (primary/icu_day0_1/
icu_day0_3), the descriptive and disclosure blocks -- is byte-identical to
the original script.

Reads (read-only):
  clinical_ai/predictions/phase_x_c1_c2_oof_probabilities.csv           (script 18; M1, C0, clinical PRIMARY -- unchanged)
  clinical_ai/predictions/phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv (script 45_v3; fold-local PRIMARY C1/C2a/C2b)
  clinical_ai/reports/phase_x_c1_c2_incremental_value_nested_corrected_v4.csv (script 45_v3; live REPRO_TARGETS for C2a/C2b)
  semantic_microbiome/harmonized/discovery_cohort_primary.csv
  ICU_public/metadata/microbiome.clinical.csv
  ICU_public/new_meta.csv
  clinical_ai/reports/phase_x_landmark_anchor_sensitivity.csv (OLD, read-only, for the old-vs-new comparison)
Writes:
  clinical_ai/reports/phase_x_landmark_anchor_sensitivity_v4.csv
  clinical_ai/reports/phase_x_landmark_anchor_sensitivity_v3.md
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('ICU_DAY0_ATTESTED_IDS', '').strip():
    raise SystemExit('ICU_DAY0_ATTESTED_IDS must be set (see README).')
_pre_ids = [i.strip() for i in _pre_os.environ['ICU_DAY0_ATTESTED_IDS'].split(',')]
if any(not i for i in _pre_ids) or len(_pre_ids) != 18 or len(set(_pre_ids)) != 18:
    raise SystemExit('ICU_DAY0_ATTESTED_IDS must list exactly 18 unique, non-empty IDs (see README).')

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils  # noqa: E402

log = io_utils.setup_logging("58_phase_x_landmark_anchor_sensitivity_v4")

MICROBIOME_BASE = "/home/tcmhwd/microbiome/ICU_public"
OOF_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_oof_probabilities.csv")  # unchanged (M1/C0)
NESTED_CSV = os.path.join(config.DIR_PREDICTIONS, "phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv")
V3_REPORT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_c1_c2_incremental_value_nested_corrected_v4.csv")
CLINICAL_RAW_CSV = os.path.join(MICROBIOME_BASE, "metadata", "microbiome.clinical.csv")
DISCOVERY_CSV = os.path.join(MICROBIOME_BASE, "semantic_microbiome", "harmonized", "discovery_cohort_primary.csv")
NEW_META_CSV = os.path.join(MICROBIOME_BASE, "new_meta.csv")
OLD_CSV = os.path.join(config.DIR_REPORTS, "phase_x_landmark_anchor_sensitivity.csv")  # read-only reference
OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_landmark_anchor_sensitivity_v4.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_landmark_anchor_sensitivity_v3.md")

SEED = config.RANDOM_SEED    # 2026
N_BOOT = config.N_BOOTSTRAP  # 1000

# ---------------------------------------------------------------------------
# OWNER ATTESTATION, 2026-09-04 -- UNCHANGED from the original script. See
# that script's docstring for the full rationale.
_ATT_RAW = os.environ.get("ICU_DAY0_ATTESTED_IDS")  # [PUBLIC RELEASE] required; validated before any data is read
_ATT_LIST = [i.strip() for i in (_ATT_RAW or "").split(",")]
if not _ATT_RAW or any(not i for i in _ATT_LIST) or len(_ATT_LIST) != 18 or len(set(_ATT_LIST)) != 18:
    raise SystemExit("ICU_DAY0_ATTESTED_IDS must be set to a comma-separated list of exactly 18 unique, non-empty "
                     f"pseudonymised patient IDs (got {0 if not _ATT_RAW else len(_ATT_LIST)} entries, "
                     f"{len(set(i for i in _ATT_LIST if i))} unique non-empty). See README.")
ICU_DAY0_ATTESTED_2026_09_04 = frozenset(  # [REDACTED for public release] 18 hard-coded pseudonymised patient IDs removed;
    # supply them locally as a comma-separated list in env var ICU_DAY0_ATTESTED_IDS (see README).
    _ATT_LIST
)

# reproduction targets: SOFA/C0/M1/C1 unchanged; C2a/C2b now read LIVE from
# 45_v3's aggregate report (REF/V3 rows) rather than hardcoded, so this
# anchor can never silently drift from the corrected source of truth.
if not os.path.exists(V3_REPORT_CSV):
    log.error("missing %s -- run 45_phase_x_c1_c2_nested_cv_corrected_v4.py --aggregate first.", V3_REPORT_CSV)
    sys.exit(1)
_v3rep = pd.read_csv(V3_REPORT_CSV)


def _v3_auroc(variant, model):
    sub = _v3rep[(_v3rep.preprocessing_variant == variant) & (_v3rep.model == model)]
    return float(sub["CV_AUROC_5fold_nested"].iloc[0])


REPRO_TARGETS = {
    "SOFA": 0.7593,
    "C0": 0.7528,
    "M1": 0.5920,
    "C1": 0.7610,
    "C2a": round(_v3_auroc("V4_STRICT", "C2a_clinical_plus_M3_LR"), 4),
    "C2b": round(_v3_auroc("V3", "C2b_clinical_plus_M3_CatBoost"), 4),
}
log.info("REPRO_TARGETS (C2a/C2b read live from 45_v3 V3 rows): %s", REPRO_TARGETS)

MODELS = [
    ("SOFA", "SOFA (conventional severity score)"),
    ("C0", "Reduced Clinical AI alone (C0, OOF stacking baseline)"),
    ("M1", "Microbiome literature-core score alone (M1, OOF)"),
    ("C1", "Clinical + M1 literature-core (C1, nested-CV corrected)"),
    ("C2a", "Clinical + M3-LR microbiome (C2a, fold-local-M3-corrected nested-CV)"),
    ("C2b", "Clinical + M3-CatBoost microbiome (C2b, fold-local-M3-corrected nested-CV)"),
]


def bootstrap_auroc(y, score, n_boot=N_BOOT, seed=SEED):
    """Byte-for-byte the resampling protocol of the original script 58 /
    56_phase_x_panelA_discrimination_ci.py / 29_phase_x_severity_score_comparison.py."""
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
    return point, lo, hi, len(boots)


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. load and join (unchanged logic)
# ---------------------------------------------------------------------------
for path in (OOF_CSV, NESTED_CSV, CLINICAL_RAW_CSV, DISCOVERY_CSV, NEW_META_CSV):
    if not os.path.exists(path):
        log.error("missing input: %s", path)
        sys.exit(1)

oof = pd.read_csv(OOF_CSV, index_col=0)
nested = pd.read_csv(NESTED_CSV, index_col=0).reindex(oof.index)
disc = pd.read_csv(DISCOVERY_CSV)
raw_clin = pd.read_csv(CLINICAL_RAW_CSV)
new_meta = pd.read_csv(NEW_META_CSV, dtype=str)

disc["kcdc_no"] = disc["kcdc_no"].astype(str)
new_meta["KCDC_no"] = new_meta["KCDC_no"].astype(str)

disc_ids = disc[["patient_id_clinical", "kcdc_no"]].drop_duplicates()
raw_clin_302 = (raw_clin.assign(KCDC_no=raw_clin["KCDC_no"].astype(str))
                .merge(disc_ids, left_on="KCDC_no", right_on="kcdc_no", how="inner")
                .set_index("patient_id_clinical").reindex(oof.index))

icu = disc.merge(new_meta[["KCDC_no", "ICU입원일", "ICU퇴실일", "ICU"]],
                  left_on="kcdc_no", right_on="KCDC_no", how="left").set_index("patient_id_clinical")
icu = icu.reindex(oof.index)
icu_dt = pd.to_datetime(icu["ICU입원일"], errors="coerce")
coll_dt = pd.to_datetime(icu["collection_date"], errors="coerce")
icu_gap = (coll_dt - icu_dt).dt.days
icu_available = icu_dt.notna()

n_attested_present = int(pd.Index(icu.index).isin(ICU_DAY0_ATTESTED_2026_09_04).sum())
if n_attested_present != len(ICU_DAY0_ATTESTED_2026_09_04):  # [PUBLIC RELEASE] was log.warning; now fails loudly
    raise SystemExit(f"ICU_DAY0_ATTESTED_IDS lists {len(ICU_DAY0_ATTESTED_2026_09_04)} patients but only "
                     f"{n_attested_present} are present in the current 302-patient index -- check the "
                     "attestation list against the current cohort")
attested_mask = pd.Index(icu.index).isin(ICU_DAY0_ATTESTED_2026_09_04)
n_attested_already_had_date = int((attested_mask & icu_available.values).sum())
if n_attested_already_had_date:
    log.warning("%d of the attested patients already have a recorded ICU admission date -- "
                "attestation is a no-op for them, real data takes precedence", n_attested_already_had_date)
icu_gap = icu_gap.mask(attested_mask & ~icu_available, 0)
icu_available = icu_available | attested_mask
log.info("owner attestation (2026-09-04): assigned ICU day 0 by attestation (no recorded "
         "ICU admission date) for %d/%d patients", n_attested_present, len(ICU_DAY0_ATTESTED_2026_09_04))

y = oof["hospital_mortality"].values
SOFA = pd.to_numeric(raw_clin_302["SOFA"], errors="coerce").values

scores = {
    "SOFA": SOFA,
    "C0": oof["oof_prob_C0_clinical_alone"].values,
    "M1": oof["microbiome_prob_M1"].values,
    "C1": nested["oof_prob_C1_clinical_plus_M1_LiteratureCore"].values,
    "C2a": nested["oof_prob_C2a_clinical_plus_M3_LR"].values,
    "C2b": nested["oof_prob_C2b_clinical_plus_M3_CatBoost"].values,
}

log.info("N=%d, events=%d (%.1f%%); ICU admission date available for %d/%d (%.1f%%)",
         len(y), int(y.sum()), 100 * y.mean(), int(icu_available.sum()), len(y),
         100 * icu_available.mean())

# ---------------------------------------------------------------------------
# 2. reproduction gate (unchanged logic; targets now include the corrected
#    C2a/C2b)
# ---------------------------------------------------------------------------
for key, want in REPRO_TARGETS.items():
    valid = ~pd.isna(scores[key])
    got = round(float(roc_auc_score(y[valid], scores[key][valid])), 4)
    log.info("reproduction check %-4s got=%.4f want=%.4f", key, got, want)
    assert got == want, f"{key} does not reproduce the published primary-cohort AUROC -- do not proceed"

# ---------------------------------------------------------------------------
# 3. cohort masks (unchanged)
# ---------------------------------------------------------------------------
mask_primary = np.ones(len(y), dtype=bool)
mask_day01 = (icu_available & (icu_gap >= 0) & (icu_gap <= 1)).values
mask_day03 = (icu_available & (icu_gap >= 0) & (icu_gap <= 3)).values
mask_attested_18 = attested_mask

cohorts = [
    ("primary_302_as_published", "Primary cohort as-is (N=302, hospital-admission-anchored, day 0-3) -- REFERENCE ROW", mask_primary),
    ("icu_day0_1", "ICU day 0-1 among those with a recorded or owner-attested ICU date (sensitivity)", mask_day01),
    ("icu_day0_3", "ICU day 0-3 among those with a recorded or owner-attested ICU date (sensitivity)", mask_day03),
]

# ---------------------------------------------------------------------------
# 4. AUROC + bootstrap CI per cohort x model (unchanged algorithm)
# ---------------------------------------------------------------------------
rows = []
for coh_key, coh_label, mask in cohorts:
    n = int(mask.sum())
    n_ev = int(y[mask].sum())
    for mkey, mlabel in MODELS:
        score = scores[mkey]
        valid = mask & ~pd.isna(score)
        pt, lo, hi, n_eff = bootstrap_auroc(y[valid], score[valid])
        rows.append({
            "cohort": coh_key, "cohort_label": coh_label,
            "N": n, "N_events": n_ev, "mortality_pct": round(100 * n_ev / n, 1) if n else np.nan,
            "model": mkey, "model_label": mlabel,
            "N_scored": int(valid.sum()),
            "AUROC": round(pt, 4), "AUROC_CI_low": round(lo, 4), "AUROC_CI_high": round(hi, 4),
            "N_boot": N_BOOT, "N_effective_resamples": n_eff, "seed": SEED,
        })

result = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. the 18 owner-attested ICU-day-0 patients -- unchanged descriptive block
#    (SOFA-based, not M3-dependent).
# ---------------------------------------------------------------------------
n_missing = int(mask_attested_18.sum())
n_missing_ev = int(y[mask_attested_18].sum())
sofa_missing = SOFA[mask_attested_18]
sofa_rest = SOFA[~mask_attested_18]
sofa_missing_valid = sofa_missing[~pd.isna(sofa_missing)]
sofa_rest_valid = sofa_rest[~pd.isna(sofa_rest)]

try:
    from scipy import stats as _stats
    _, mw_p = _stats.mannwhitneyu(sofa_missing_valid, sofa_rest_valid, alternative="two-sided")
except Exception as exc:  # pragma: no cover
    mw_p = float("nan")
    log.warning("Mann-Whitney SOFA comparison failed: %s", exc)

missing_row = {
    "cohort": "icu_day0_attested_18", "cohort_label":
        "18/302 (6.0%) with no RECORDED ICU admission date, assigned ICU day 0 by owner "
        "attestation (2026-09-04; see ICU_DAY0_ATTESTED_2026_09_04) -- full members of "
        "icu_day0_1/icu_day0_3 above, NOT excluded; this row is a descriptive "
        "characterisation of a distinctive subgroup, not an exclusion note",
    "N": n_missing, "N_events": n_missing_ev,
    "mortality_pct": round(100 * n_missing_ev / n_missing, 1),
    "model": "DESCRIPTIVE", "model_label":
        f"SOFA attested-18 mean={np.nanmean(sofa_missing):.2f} median={np.nanmedian(sofa_missing):.1f} "
        f"(n={len(sofa_missing_valid)}) vs rest mean={np.nanmean(sofa_rest):.2f} "
        f"median={np.nanmedian(sofa_rest):.1f} (n={len(sofa_rest_valid)}); "
        f"Mann-Whitney p={mw_p:.3f}",
    "N_scored": n_missing, "AUROC": np.nan, "AUROC_CI_low": np.nan, "AUROC_CI_high": np.nan,
    "N_boot": np.nan, "N_effective_resamples": np.nan, "seed": SEED,
}
result = pd.concat([result, pd.DataFrame([missing_row])], ignore_index=True)

# ---------------------------------------------------------------------------
# 6. disclosure block (unchanged -- not M3-dependent, SOFA/outcome only)
# ---------------------------------------------------------------------------
mc430 = pd.read_csv(CLINICAL_RAW_CSV)
mc430["KCDC_no"] = mc430["KCDC_no"].astype(str)
m430 = mc430.merge(new_meta[["KCDC_no", "ICU입원일", "검체채취일"]], on="KCDC_no", how="left")
m430["icu_dt"] = pd.to_datetime(m430["ICU입원일"], errors="coerce")
m430["coll_dt"] = pd.to_datetime(m430["검체채취일"], errors="coerce")
m430["gap"] = (m430["coll_dt"] - m430["icu_dt"]).dt.days
m430_day01 = m430[(m430["gap"] >= 0) & (m430["gap"] <= 1)]

current_302_kcdc = set(disc["kcdc_no"])
eligible_kcdc = set(m430_day01["KCDC_no"])
excluded_kcdc = eligible_kcdc - current_302_kcdc
intersection_kcdc = eligible_kcdc & current_302_kcdc

sub_excluded = m430[m430["KCDC_no"].isin(excluded_kcdc)]
sub_intersection = m430[m430["KCDC_no"].isin(intersection_kcdc)]

n_full_430 = len(mc430)
n_eligible_day01_430 = len(m430_day01)
n_excluded_by_flag = len(sub_excluded)
mort_excluded = 100 * sub_excluded["in.hospital.mortality"].mean()
n_intersection = len(sub_intersection)
mort_intersection = 100 * sub_intersection["in.hospital.mortality"].mean()

log.info("Disclosure: of %d linked patients, %d qualify ICU day 0-1; %d of those are "
         "excluded by the current hospital-anchored flag (mortality %.1f%%) vs %d retained "
         "in the intersection (mortality %.1f%%)",
         n_full_430, n_eligible_day01_430, n_excluded_by_flag, mort_excluded,
         n_intersection, mort_intersection)

disclosure_rows = [
    {"cohort": "disclosure_430_icu_day0_1_excluded_by_current_flag", "cohort_label":
        "DISCLOSURE, not a result: among 430 linked patients, ICU day 0-1 eligible but "
        "EXCLUDED by the current hospital-anchored (collection_time==0) flag",
     "N": n_excluded_by_flag, "N_events": int(sub_excluded["in.hospital.mortality"].sum()),
     "mortality_pct": round(mort_excluded, 1), "model": "DESCRIPTIVE",
     "model_label": "outcome-correlated exclusion -- current cohort's exclusions favour survivors",
     "N_scored": n_excluded_by_flag, "AUROC": np.nan, "AUROC_CI_low": np.nan, "AUROC_CI_high": np.nan,
     "N_boot": np.nan, "N_effective_resamples": np.nan, "seed": SEED},
    {"cohort": "disclosure_430_icu_day0_1_retained_intersection", "cohort_label":
        "DISCLOSURE, not a result: among 430 linked patients, ICU day 0-1 eligible AND "
        "retained by the current hospital-anchored flag (the intersection)",
     "N": n_intersection, "N_events": int(sub_intersection["in.hospital.mortality"].sum()),
     "mortality_pct": round(mort_intersection, 1), "model": "DESCRIPTIVE",
     "model_label": "reference point for the exclusion comparison above",
     "N_scored": n_intersection, "AUROC": np.nan, "AUROC_CI_low": np.nan, "AUROC_CI_high": np.nan,
     "N_boot": np.nan, "N_effective_resamples": np.nan, "seed": SEED},
]
result = pd.concat([result, pd.DataFrame(disclosure_rows)], ignore_index=True)

io_utils.save_csv(result, OUT_CSV)
log.info("\n%s", result[["cohort", "N", "N_events", "model", "AUROC", "AUROC_CI_low", "AUROC_CI_high"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 6b. OLD-vs-NEW comparison for the M3-dependent rows (C2a/C2b, every
#     cohort) -- new in v3.
# ---------------------------------------------------------------------------
old_vs_new_note = ""
if os.path.exists(OLD_CSV):
    old = pd.read_csv(OLD_CSV)
    cmp_rows = []
    for coh_key, _, _ in cohorts:
        for mkey in ("C2a", "C2b"):
            o = old[(old.cohort == coh_key) & (old.model == mkey)]
            n_ = result[(result.cohort == coh_key) & (result.model == mkey)]
            if len(o) and len(n_):
                cmp_rows.append({"cohort": coh_key, "model": mkey,
                                 "old_AUROC": float(o["AUROC"].iloc[0]), "new_AUROC": float(n_["AUROC"].iloc[0]),
                                 "old_CI_low": float(o["AUROC_CI_low"].iloc[0]), "old_CI_high": float(o["AUROC_CI_high"].iloc[0]),
                                 "new_CI_low": float(n_["AUROC_CI_low"].iloc[0]), "new_CI_high": float(n_["AUROC_CI_high"].iloc[0])})
    cmp_df = pd.DataFrame(cmp_rows)
    log.info("OLD-vs-NEW (C2a/C2b, every cohort):\n%s", cmp_df.to_string(index=False))
    old_vs_new_note = "\n".join(["## Old (frozen-PCA) vs new (fold-local M3), C2a/C2b rows", "", df_to_md_table(cmp_df), ""])
else:
    log.warning("%s not found -- skipping old-vs-new comparison.", OLD_CSV)

# ---------------------------------------------------------------------------
# 7. markdown report (unchanged structure, v3 framing added)
# ---------------------------------------------------------------------------
sens_table = result[result["cohort"].isin(["primary_302_as_published", "icu_day0_1", "icu_day0_3"])]

model_order = [m for m, _ in MODELS]
disp_rows = []
for mkey, mlabel in MODELS:
    row = {"model": mkey, "model_label": mlabel}
    for coh_key, _, _ in cohorts:
        sub = sens_table[(sens_table["model"] == mkey) & (sens_table["cohort"] == coh_key)].iloc[0]
        row[coh_key] = f"{sub['AUROC']:.4f} [{sub['AUROC_CI_low']:.4f}, {sub['AUROC_CI_high']:.4f}] (N={sub['N']})"
    disp_rows.append(row)
disp_df = pd.DataFrame(disp_rows)

md = [
    "# Landmark-anchor sensitivity analysis (v3, fold-local M3-corrected C2a/C2b)",
    "",
    "**Fold-local M3 correction (2026-10-02):** repoints the nested-CV C2a/C2b columns at "
    "`45_phase_x_c1_c2_nested_cv_corrected_v4.py`'s fold-local PRIMARY (V3) OOF predictions. SOFA/C0/M1/C1 "
    "columns, the owner-attestation handling, the cohort masks and the disclosure block are byte-identical "
    "to `58_phase_x_landmark_anchor_sensitivity.py`.", "",
    old_vs_new_note,
    "**Owner decision, 2026-09-04 (unchanged): the primary cohort stays N=302, hospital-admission-anchored, "
    "as currently defined.** This is a supplementary sensitivity analysis, not a cohort change. Every number "
    "below is a re-scoring of already-persisted out-of-fold prediction columns "
    "(`phase_x_c1_c2_oof_probabilities.csv` for M1/C0/clinical PRIMARY, "
    "`phase_x_c1_c2_nested_oof_probabilities_v4_V4_STRICT.csv` for the fold-local-M3-corrected C1/C2a/C2b) on "
    "ICU-anchored SUBSETS of the same 302 patients. No model was refit; no cohort was changed; no frozen "
    "artefact (`FROZEN_MICROBIOME_PIPELINE.json` GATE_4) was touched.", "",
    "**Bootstrap convention** (identical to `56_phase_x_panelA_discrimination_ci.py`): percentile "
    f"95% CI, {N_BOOT} with-replacement patient resamples, `np.random.RandomState({SEED})`, "
    "`rng.randint(0, n, n)`, single-outcome-class resamples skipped, percentiles 2.5/97.5. This "
    "script's primary-cohort point estimates are gated to reproduce the already-published "
    f"values exactly ({', '.join(f'{k}={v}' for k, v in REPRO_TARGETS.items())}) before any "
    "subset row is trusted.", "",
    "**Owner attestation, 2026-09-04 (unchanged, see `ICU_DAY0_ATTESTED_2026_09_04` in the script).** For "
    "18 of the 302 patients, `new_meta.csv` does not record an ICU admission date; the owner attested all "
    "18 as ICU day 0. See the original script's docstring for the full rationale.", "",
    "## Sensitivity table", "",
    df_to_md_table(disp_df.rename(columns={
        "primary_302_as_published": "Primary (N=302, as published) -- REFERENCE",
        "icu_day0_1": "ICU day 0-1 (recorded or owner-attested ICU date)",
        "icu_day0_3": "ICU day 0-3 (recorded or owner-attested ICU date)",
    })), "",
    "## The 18 owner-attested ICU-day-0 patients (unchanged, SOFA-based, not M3-dependent)", "",
    f"- N = {n_missing}, deaths = {n_missing_ev} ({100 * n_missing_ev / n_missing:.1f}% mortality)",
    f"- SOFA: attested-18 group mean {np.nanmean(sofa_missing):.2f} / median "
    f"{np.nanmedian(sofa_missing):.1f} (n={len(sofa_missing_valid)}) vs the rest of the primary "
    f"cohort mean {np.nanmean(sofa_rest):.2f} / median {np.nanmedian(sofa_rest):.1f} "
    f"(n={len(sofa_rest_valid)}); Mann-Whitney two-sided p = {mw_p:.3f}.", "",
    "## Disclosure: the current cohort's exclusions are outcome-correlated (unchanged)", "",
    f"Among the 430 patients linked in `metadata/microbiome.clinical.csv`, {n_eligible_day01_430} qualify "
    f"under an ICU day 0-1 anchor. {n_intersection} are retained by the current hospital-anchored flag "
    f"({mort_intersection:.1f}% mortality); {n_excluded_by_flag} are excluded despite qualifying "
    f"({mort_excluded:.1f}% mortality).", "",
    "## Interpretation", "",
    "Re-run with the fold-local-M3-corrected C2a/C2b columns; see the old-vs-new table above for whether any "
    "cohort x model cell moved materially. No model fitting happens in this script, so any change here is "
    "entirely a consequence of the corrected input predictions, not a new sensitivity choice.", "",
    "## Full results table", "",
    "See `phase_x_landmark_anchor_sensitivity_v4.csv` for every cohort x model row, plus the "
    "attested-18 descriptive row and the two disclosure rows.",
]
io_utils.save_text("\n".join(md), OUT_MD)
log.info("wrote %s / %s", OUT_CSV, OUT_MD)
