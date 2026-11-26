# Prespecification — scoring the 67 patients excluded by the hospital-admission anchor

Status: **FROZEN. Committed before `lit53_excluded67_scoring.py` is ever executed**, so that the
ordering (prespecification → execution) is verifiable in git history, per the design memo
(`EXCLUDED67_SCORING_DESIGN.md`, 2026-09-04) sec 4.3 and the task instruction that produced this
file. `lit53_excluded67_scoring.py` refuses to run (step 0, `_verify_self_hash()`) unless its own
SHA-256 matches the pin recorded in sec 7 below — so a script edited after this document was frozen
cannot silently execute against a stale prespecification.

This document does not re-derive anything. It restates, verbatim where the source says "verbatim,"
the content `EXCLUDED67_SCORING_DESIGN.md` sec 4.3 requires to exist here before execution. Section
numbers below are this document's own; cross-references to the design memo are called out
explicitly as "(design sec X)".

No result in this document exists yet. Every numeric cell in sec 4 (the reporting template) is
`___`. Sections 1–3 and 6–10 state *design* quantities (cohort counts, CI-width arithmetic, EPV,
seeds) that are fixed by definition or by feasibility work already disclosed in sec 0.1 — these are
not blind results and are not blanked.

---

## 0. Disclosures that come before everything else (design sec 0)

### 0.1 The SOFA result is already known and is therefore not blind

While establishing feasibility for the design memo, `roc_auc_score(mortality, SOFA)` within the 67
was computed on the login node, on 2026-09-04. **It is 0.7537**, against 0.7593 in the 302. This is
estimand **E1** below. **It is on the record as of 2026-09-04 and is not a blind prespecified
result.** Presenting it later as though it had been produced by the frozen script for the first
time would be false, and this document states that plainly rather than smoothing it over.

**Every other quantity in this analysis — E2 (M1-frozen in the 67), E3 (the SOFA-adjusted M1 term),
E4 (the clinical bundle in the 67), and R1 (the reference band) — was NOT computed before this
prespecification was frozen and remains blind.** They could not be computed without first building
a feature matrix that does not exist before `lit53_excluded67_scoring.py` runs (design sec 1), which
is why the blindness held for everything except E1.

### 0.2 What this analysis can and cannot do

It can show whether the frozen microbiome model discriminates in the excluded stratum, and whether
the severity comparator (SOFA) does. It **cannot** overturn the paper's primary result (defined on
the 302), and it **cannot** establish what the microbiome would have shown had the cohort been
re-anchored, because that would require refitting the frozen pipeline (forbidden — see sec 1.3, the
freeze-preserving boundary). The analysis is asymmetric by construction: it can add reassurance or
add a caveat, and it cannot do more than that. This asymmetry is stated wherever any result from
this analysis is later quoted.

---

## 1. Cohort rule (design sec 1.1) — verbatim, with its assertions

**Rule.** Patients in `metadata/microbiome.clinical.csv` (430 linked, one row per patient) whose
`collection_date − ICU입원일` (ICU admission date, from `new_meta.csv`, joined on `KCDC_no`) is 0 or
1 calendar days, **and** whose `collection_time != 0` (i.e., excluded by the current
hospital-admission-anchored flag that defines the primary 302-patient cohort).

**Expected, reproduced independently before this document was frozen (descriptive cohort facts,
not model estimates — disclosed alongside E1 in sec 0.1):** N = 67, 28 deaths (41.8% mortality);
all 67 have `collection_time == 1`; all 67 are swab specimens (0 stool); SOFA present for 67/67
(median 11, IQR 8–13, vs 10, IQR 7–14 in the 302); overlap with the 302 discovery cohort = 0;
hospital admission dates span May 2024 to March 2025 (day precision redacted in the public copy).

**The ID-join trap (design sec 1.1, "Implementation trap").** `microbiome.clinical.csv`'s `list`/
`No` columns use a mixed scheme (180/430 rows carry the clinical `ICUxxxx` ID directly in `list`;
the other 250/430 carry the raw KCDC number). The only recipe verified to resolve 430/430 uniquely
to `^ICU\d+$` and 302/302 against `discovery_cohort_primary.csv` is:

```
pid = KCDC_no  ->  patient_id_clinical   via sample_id_mapping.csv rows with
                   match_method == 'kcdc_no_strip_wm'   (covers 250)
      fillna( microbiome.clinical.csv['list'] )         (covers the other 180)
```

**Executable assertions this maps to (`lit53_excluded67_scoring.py`, unconditional `assert`, no
warnings-only path):**

| # | assertion | expected |
|---|---|---|
| 1 | `pid.nunique() == 430` | 430 |
| 2 | `pid.astype(str).str.match(r'^ICU\d+$').all()` | True |
| 3 | `len(set(pid) & set(discovery_cohort_primary.patient_id_clinical)) == 302` | 302 |
| 4 | `len(excl67) == 67` | 67 |
| 5 | `int(excl67['in.hospital.mortality'].sum()) == 28` | 28 |
| 6 | `(excl67['collection_time'] == 1).all()` | True |
| 7 | `(excl67['sample_type'] == 'swab').all()` | True |
| 8 | `excl67['SOFA'].notna().all()` | True |
| 9 | `len(set(excl67_ids)) == 67` (no duplicate patients) | 67 |
| 10 | `len(set(excl67_ids) & set(discovery_cohort_primary.patient_id_clinical)) == 0` | 0 |
| 11 | MetaPhlAn coverage of the 67 == 67/67 | 67 |
| 12 | MAG_abundance_CoverM coverage of the 67 == 67/67 | 67 |

**Overlap gate (re-asserted, not inherited — design task instruction):** all 67 admission dates lie
in May 2024 to March 2025 (month precision in the public copy); `ICU.csv`'s `year` column (the primary clinical AI training cohort)
spans 2006–2020. Assertion: `icu_year_max < admission_date_min.year` — a strict calendar disjointness
check computed at runtime, not copied from `clinical_microbiome_patient_overlap_audit.md`, whose own
text scopes it to the 302 only.

**What building the M1 features for the 67 requires (design sec 1.2 — decisive finding, corrects
the state audit that flagged this as an open risk): zero re-estimation.** All 16 M1 Core features
are pure per-sample arithmetic on `metaphlan3_{species,genus,family,phylum}_merged.tsv` and
`coverm_abundance_matrix.tsv` — no prevalence filter, no PCA, no scaler, no imputer, so `GATE_4`
(`FROZEN_MICROBIOME_PIPELINE.json`) is untouched. The one cohort-level object involved (the 146-MAG
pathobiont list from the pooled 887-sample dRep catalogue) was built before the 302 was defined, is
outcome-blind, and is identical whether or not the 67 are scored.

---

## 2. Frozen estimand table (design sec 3.2) — verbatim; no estimand may be added later

| id | estimand | population | model objects | freeze contact | status |
|---|---|---|---|---|---|
| **E1** | AUROC of SOFA for in-hospital mortality | the 67 | none | none | **PRIMARY ANCHOR** (not blind — sec 0.1) |
| **E2** | AUROC of M1-frozen predicted probability | the 67 | `M1_literature_core_LR.joblib`, applied unchanged | none | **PRIMARY TEST (co-primary)** (blind) |
| **E3** | within-67 SOFA-adjusted association of `logit(M1-frozen prob)` with mortality: adjusted OR, LR-test p, and ΔAUROC(SOFA + M1 vs SOFA) | the 67 | as E2 | none | **PRIMARY CONTRAST** (blind) |
| **E4** | AUROC of `clinical_AI_probability_PRIMARY` (direct application, the 0.7560 family — NOT the 0.7528 OOF family) | the 67 | frozen CatBoost bundle | none | secondary (blind) |
| **R1** | procedurally matched reference band for E2, and its fit-free analogue for E1/E4 | the 302 only | M1 *specification* refit on 302-subsets | none | **interpretive companion to E2 — required, not optional** |
| **E5** | the same quantities on the pooled ICU-day-0/1 population, N = 348 (281 from the 302's own icu_day0_1 subset + 67) | 281 + 67 | frozen M1 (CONTAMINATED for the 302-side rows), SOFA, C0-direct | none | secondary, hard caveat; **no E5 quantity may enter the abstract** |
| **E6** | descriptive comparison of the 67 vs the 302: SOFA, age, sex, ventilator, specimen type, the 16 M1 marker distributions, hospital-day and ICU-day gaps | both | none | none | descriptive |

**Primary contrast = E3.** E1 anchors it, E2 supplies its covariate, and E3 is the only estimand
that speaks to the paper's actual claim (the microbiome adds nothing over severity assessment,
not merely "M1's own AUROC is unimpressive"). E2 is co-primary because it is the number a reviewer
looks at first and must be pre-declared rather than produced on request.

**E3's exact specification, fixed now, never extended.** Logistic regression in the 67:
`mortality ~ SOFA + logit(M1_frozen_prob)`. Report the M1 coefficient's adjusted OR per SD with a
**profile-likelihood 95% CI** (not Wald — Wald is reported alongside for comparison only), the
likelihood-ratio test p-value against the SOFA-only model, and the paired ΔAUROC with a
2000-resample percentile bootstrap CI. **Exactly two covariates. No age, no sex, no ventilator, no
interaction terms, no alternative link.** A third covariate would drop EPV from 14.0 to 9.3 and
cross the project's own 10-EPV convention (`42_phase_x_time_to_event_competing_risk.py:1151`).

---

## 3. Precision — pre-declared, before any number exists (design sec 3.3, verbatim)

With 28 events and 39 non-events, by Hanley–McNeil:

| quantity | expected 95% CI half-width | expected CI width |
|---|---|---|
| AUROC at 0.50 | ±0.141 | **0.282** |
| AUROC at 0.59 | ±0.140 | **0.280** |
| AUROC at 0.65 | ±0.136 | 0.271 |
| AUROC at 0.75 | ±0.122 | **0.245** |
| AUROC at 0.80 | ±0.112 | 0.225 |

For reference, the same quantities in the 302 (69/233) have widths 0.157 (at 0.592) and 0.141
(at 0.759), and in the pooled 348 (94/254) 0.138 and 0.124.

**Pre-declared consequence for E2: an AUROC anywhere in roughly [0.45, 0.73] will have a 95% CI
containing both 0.5 and 0.592.** E2 alone is *incapable* of distinguishing "chance" from "the
published internal value," and must never be reported as though it could. This is why R1 is
mandatory, not a nicety.

For the **paired** increment E3, scaling the published nested-CV bootstrap SEs by √(69/28):

| contrast | published SE (302) | implied SE (67) | MDE at 80% power | implied CI width |
|---|---|---|---|---|
| C1 vs C0 (script 46, nested) | 0.0106 | 0.0166 | **0.0466** | 0.065 |
| SOFA+M1 vs SOFA (script 41) | 0.0128 | 0.0201 | **0.0563** | 0.079 |

The observed increments in the 302 are **+0.0082** and **+0.0133**. The within-67 MDE is **4–6×
larger than the effect the study is looking for.**

**E3 cannot detect an increment of the magnitude observed in the primary cohort. It is powered only
to exclude a large increment (> ≈0.05). It is designated descriptive, its p-value is reported for
completeness and is not a confirmatory test, and a non-significant result will be reported as
"uninformative about small increments," never as "confirms the null."** By the project's own
≥10-event evaluability bar (`50_phase_x_surveillance_feasibility.py`,
`51_phase_x_mdro_acquisition_prediction.py:280`), 28 events clears the threshold for *reporting* —
clearing an evaluability bar is not the same as being powered, and this document states both.

**Flagged as not confident (design sec 3.3/7):** the SE-scaling assumption (paired bootstrap SE ∝
1/√events) is approximate for a paired, correlated statistic. `lit53_excluded67_scoring.py` reports
the *actual* bootstrap CI width alongside this predicted one, so the prediction is auditable, not
assumed.

---

## 4. Blank reporting template — every numeric cell is `___`

The sentences and table below are the exact write-up that will be filled in once
`lit53_excluded67_scoring.py` has been reviewed to have run correctly (reproduction gate passed).
No wording is to change when the numbers are filled in; only the `___` placeholders are replaced.

> In the 67 patients excluded from the primary cohort by the hospital-admission anchor (28/67
> deaths, 41.8% mortality vs 23.5% in the primary 302-patient cohort — see sec 0.1 for the
> non-blind disclosure of the SOFA figure below), SOFA discriminated in-hospital mortality with
> AUROC ___ (95% CI ___–___), against ___ (95% CI ___–___) in the primary cohort. The frozen
> literature-core microbiome model (M1) achieved AUROC ___ (95% CI ___–___) — falling at the
> ___th percentile of a procedurally matched reference band (median ___, 95% band ___–___)
> constructed from 2000 same-domain, same-size, same-event-count draws from the primary cohort.
> Adjusting for SOFA, the M1 term's adjusted odds ratio per SD was ___ (profile-likelihood 95% CI
> ___–___; Wald 95% CI ___–___ for comparison), a likelihood-ratio test against the SOFA-only model
> gave χ² = ___ (df = 1, p = ___), and the paired ΔAUROC(SOFA + M1 vs SOFA) was ___ (95% CI
> ___–___). The reduced clinical AI model, applied directly with no refitting, achieved AUROC ___
> (95% CI ___–___) in this stratum. [Interpretation rule sentence from sec 5 goes here, filled in
> mechanically from the CI positions above, not re-derived.]

### Blank results table

| estimand | point estimate | 95% CI low | 95% CI high | N | N events | R1 band median | R1 band 95% low | R1 band 95% high | observed percentile in R1 |
|---|---|---|---|---|---|---|---|---|---|
| E1 (SOFA) | ___ | ___ | ___ | 67 | 28 | ___ | ___ | ___ | ___ |
| E2 (M1-frozen) | ___ | ___ | ___ | 67 | 28 | ___ | ___ | ___ | ___ |
| E4 (C0-direct) | ___ | ___ | ___ | 67 | 28 | ___ | ___ | ___ | ___ |

| E3 quantity | value |
|---|---|
| M1 term, OR per SD (Wald) | ___ |
| M1 term, OR per SD (profile-likelihood 95% CI) | ___ – ___ |
| LR test χ² (df=1) | ___ |
| LR test p-value | ___ |
| ΔAUROC(SOFA+M1 − SOFA) | ___ |
| ΔAUROC 95% CI | ___ – ___ |

| E5 (secondary, N=___) | AUROC | status |
|---|---|---|
| SOFA | ___ | clean |
| C0 (direct) | ___ | clean |
| M1-frozen | ___ | **CONTAMINATED — DO NOT QUOTE** |

### Blank interpretation call

Per sec 5's rule, mechanically applied once the numbers above exist:
**[ Consistent with the primary null / Indeterminate / Inconsistent — escalate ]** — reasoning:
___

---

## 5. Interpretation rule — fixed in advance, expressed as CI positions (design sec 4.4)

| outcome | rule |
|---|---|
| **Consistent with the primary null** | E2's 95% CI contains 0.592 **and** E3's ΔAUROC CI contains 0 **and** its upper bound is < 0.05 **and** E1 ≥ 0.70 |
| **Indeterminate** | E2's CI contains 0.592 but E3's ΔAUROC upper bound is ≥ 0.05 (a clinically meaningful increment cannot be excluded) — **the expected outcome given sec 3's arithmetic** |
| **Inconsistent — escalate** | E2's entire 95% CI lies above 0.592, **or** E3's ΔAUROC CI excludes 0 on the positive side, **or** E1 < 0.65 (comparator collapse) |

**"Indeterminate" is the most likely outcome by design arithmetic, not a failure.** It is reported
as planned, not spun in either direction, if it occurs.

---

## 6. Escalation pre-commitment (design sec 4.5, verbatim)

> If the result is **Inconsistent**, the paper's primary claim does **not** change. The finding is
> reported in Results and Limitations as an unexplained discrepancy in an excluded stratum, is
> explicitly labelled hypothesis-generating, and is named as motivating a properly ICU-anchored
> re-analysis with a re-established pre-outcome freeze in future work. **We commit in advance to
> reporting E1–E5 whichever way they come out, and to not adding, removing, or redefining any
> estimand after seeing them.**

This commitment is what makes the analysis publishable as a post-hoc sensitivity rather than as
fishing. **Without a signed instance of this commitment, the analysis is not to be run at all**
(design sec 6.3/6.5) — this document's existence, committed before execution, is that instance.

---

## 7. Integrity pins — SHA-256, re-verified by the script at run time

| file | SHA-256 |
|---|---|
| `lit53_excluded67_scoring.py` | `e1157b9b28f8e507948050fb49cb8fb4471315c8023a20286902b32ea537c238` |
| `M1_literature_core_LR.joblib` | `f0d7ec63524068da04f157344452318243e973ce6c03bde7a753dc3df710c8d5` |

`lit53_excluded67_scoring.py` step 0 (`_verify_self_hash()`) reads its own SHA-256 pin out of THIS
file at run time and refuses to proceed if the on-disk script does not match — non-circular, since
the pin lives in a different, already-committed file. It independently re-hashes and asserts the
M1 joblib pin inside the reproduction gate (design sec 5.2 step 1), also as an unconditional
`assert`. If either file must change after this document is frozen, that is a **new, dated
amendment to this document** (a new pinned hash with a note of why), never a silent edit.

---

## 8. Multiplicity rule (design sec 4.2)

**Exactly one inferential test in the whole analysis: the pre-declared likelihood-ratio test for
`logit(M1-frozen prob)` in the SOFA-adjusted model within the 67 (E3).** Every other quantity in
this analysis is a point estimate with a 95% CI and no p-value. This is consistent with how the
project already reports discrimination (`56_phase_x_panelA_discrimination_ci.py`) and makes the
multiplicity question nearly vacuous — the family of {SOFA, M1-frozen, C0} × {the 67, the 348} ×
{AUROC, increment} is ≈12 cells, and correcting 12 cells for one already-singular test would be
theatre.

**Explicitly forbidden:** any subgroup of the 67 (by SOFA stratum, ward-LOS, specimen, organism);
any threshold selection within the 67; any alternative feature panel; any "sensitivity" that varies
the ICU-day window after seeing E2. The 67 are 67 patients and there is no second look available in
them.

---

## 9. Stated expectation (design sec 6.1, for the record — not a target to be met)

| quantity | expectation | reasoning |
|---|---|---|
| E1, SOFA in the 67 | 0.7537 (known, sec 0.1) | measured |
| E4, C0 in the 67 | 0.70–0.78 | same variables plus comorbidities, same frozen bundle, complete inputs |
| E2, M1-frozen in the 67 | 0.50–0.62, **modal guess ≈ 0.55** | M1's own transfers are 0.567/0.618, its same-domain held-out value is 0.592, its 25-fold nested value is 0.624 — no mechanism predicts a jump |
| E3, ΔAUROC over SOFA | [−0.03, +0.06], CI containing 0, width ≈ 0.08 | sec 3 |
| E5, SOFA on the 348 | ≈ 0.75–0.76 | pooling two populations with near-identical SOFA discrimination |

**A specific, falsifiable, pre-registered prediction.** M1 is expected to perform *worse* in the 67
than in the 302, not better. The 67 spent days to weeks (range redacted in the public copy) on a general ward before ICU transfer — the
regime of maximal cumulative antibiotic and nosocomial exposure — and M1's dominant features are
`pathobiont_burden` and `facultative_to_obligate_anaerobe_balance`. If pathobiont domination is
close to universal in the 67, these features **saturate**, compressing the discriminating variance
and moving AUROC toward 0.5. E6's marker-distribution comparison tests this directly and is to be
read alongside E2, not after it.

**The competing hypothesis — that heavy microbiome perturbation makes the microbiome *more*
informative in this stratum — remains live, at roughly 20% prior probability**, per the design
memo. Neither prediction is privileged in how the results are reported; both are stated here so
that whichever direction the data go, this document shows the prediction was made in advance.

---

## 10. Explicit refusal of the half-measure (design sec 2.4 point 4 / sec 6.4 point 5)

**Running E1 and/or E4 (the comparators) without E2 and E3 (the model under test) is selective
reporting and is forbidden.** SOFA does not fully explain the 67's excess mortality (median SOFA 11
vs 10 cannot produce 41.8% vs 22.8% mortality on its own) — that residual, unexplained excess risk
is precisely the gap the microbiome is claimed to fill, so declining to look at M1 while reporting
only the comparators would mean looking exactly where the paper's claim is safest and refusing to
look where it is actually tested. **Either the full estimand set (E1–E6, R1) is computed and
reported, or none of it is.**

---

## 11. Scope note — the abstract's Conclusions clause

`ABSTRACT_and_INTRO_skeleton_v3.md`'s Conclusions clause already reads "a supplementary
ICU-anchored sensitivity analysis **within the same 302 patients** confirmed the same conclusion"
(added by `revision_log.md` item 51, S9, 2026-09-04, predating this task) — correctly scoped: that
sensitivity (`58_phase_x_landmark_anchor_sensitivity.py`) is a subset re-scoring of the same 302
patients and covers none of the 67. Verified current at the time this document was frozen; no
further edit was needed or made.

---

## 12. What this document does not do

It does not run `lit53_excluded67_scoring.py`. It does not produce a single new AUROC, OR, or
p-value beyond the already-disclosed, non-blind E1 = 0.7537 (sec 0.1). Every numeric cell in sec 4
is `___`. Execution is a separate, later, explicit step, gated on this document's own commit
existing first in git history.
