# Limited incremental value of early gut microbiome profiles for predicting in-hospital mortality in critically ill adults

Analysis code for the manuscript of the same title (Kwon J-G, Lee JH, Shin JH, Choi MH; submitted to *Critical Care*).

Release **v1.0.1**. This repository contains the final analysis scripts behind the numbers reported in the manuscript, its Supplementary Methods, Tables S1 to S9 and Figures S1 to S5, plus the superseded script versions whose outputs those final scripts still read. It is not a complete push-button reproduction: Hospital A per-sample processing ran outside this project, a few reported numbers have no producing script, and some inputs are curated tables (see "Known gaps"). Scripts are released **verbatim** as they were run; the only edits are the redactions and fail-loud checks listed under "Redactions" below, and 1 file written for this release is marked as such in MANIFEST.sha256. No data are included.

## What is included and what is not

Included

- Shotgun metagenomic processing for the public external cohorts and the cohort-level MAG catalogue (trimming and host removal, MetaPhlAn 3, HUMAnN 3, MEGAHIT, BWA, binning, CheckM2, dRep, Prokka, eggNOG-mapper, antiSMASH, RGI/CARD, VFDB, GTDB-Tk, CoverM, contig-level RGI and RPKM).
- Merging of per-sample outputs into cohort feature matrices (`merge/scripts/`).
- Cohort definition (302-patient hospital-day 0 to 3 landmark cohort) and outcome-blind structured feature views.
- Literature-derived 16-marker panel: PubMed search, article retrieval, GPT-4o-mini passage-grounded extraction, ontology mapping, evidence tiering, freeze, marker construction, final M1 refit.
- The 29-variable clinical prediction model pipeline (development on Hospital A, evaluation on Hospital B and MIMIC-IV, selection, application to the microbiome cohort).
- Incremental-value analyses: nested 5 x 5 cross-validation with two-stage stacking, paired bootstrap, decision curves, calibration, SOFA-referenced comparisons, repeated nested CV, landmark sensitivity, time-to-event and competing-risk models, influence analysis.
- External benchmarking (AGI cohort PRJCA011331 and PRJNA842425), leave-one-cohort-out corpus masking (LOCO), cross-cohort marker consistency.
- Exploratory MDRO and VRE analyses.
- Two R figure scripts that also compute reported numbers (Holm-adjusted decision-curve P values; one-sided binomial concordance tests), plus their shared theme file and the generic plotting theme it sources (`R/house_style.R`, the owner's cross-project ggplot2 theme; palette, fonts and save helpers only, no data; copy it to `~/R/house_style.R` or edit the `source()` line in `00_setup_theme.R`). The copy included is the version on disk at release time (last modified 2026-09-23), which may differ cosmetically from the version used for earlier renders.
- **Superseded versions included only because later scripts read their outputs** (`clinical metadata/scripts/` and matching `slurm/` wrappers): `45_phase_x_c1_c2_nested_cv_corrected.py` (original 45), `45_..._v3.py` (and `45_..._v3.sh`, `45_..._v3_aggregate.sh`), `46_phase_x_c1_c2_nested_bootstrap_ci.py`, `48_phase_x_minimum_detectable_effect.py` (original 48), `61_phase_x_microbiome_over_sofa_nested.py` (original 61) and `61_..._v3.py`. Their results are not the reported ones; they are here so that the file lineage of `45_v4`, `56`, `61_v4`, `lit_cc5`, `lit_cc6`, `audit_outlier_sensitivity.py` and `15b_paper1_figures_final_v4.R` can be followed. Read `45_v4`/`61_v4` for the reported analysis.
- Two small scripts written for this release (not part of the original run): `analysis_extras/holm_table2.py` (post hoc Holm across the three Table 2 clinical-model contrasts) and, copied from the Supplementary Methods working folder, `semantic_microbiome/reports/paper1/supp_methods_pass_2026-10-30/work/qc_read_depth_from_bwa_logs.py` (sequencing-depth summary from the BWA mapping logs).
- Data-free reference files: the frozen literature feature library and its manifest (`semantic_microbiome/literature_prior/final/`), the microbiome pipeline freeze record (`semantic_microbiome/FROZEN_MICROBIOME_PIPELINE.json`) and the pre-specification for the excluded-67 analysis (`semantic_microbiome/reports/paper1/EXCLUDED67_PRESPECIFICATION.md`).

Not included

- **Any patient-level data**: clinical tables, metadata CSVs, abundance tables, out-of-fold prediction files, MIMIC-IV-derived files.
- **Trained model bundles.** The frozen clinical model and the literature-derived microbiome model are served from the Hugging Face Space [`tcmhwd/evidence-grounded-icu-risk`](https://huggingface.co/spaces/tcmhwd/evidence-grounded-icu-risk), which matches the manuscript.
- Other superseded script versions (apart from the producers listed above, only the final `_v4` version, or the highest version, of each analysis is included), and figure scripts that only render already-computed numbers (see "Figure rendering" below).
- The executable code of exploratory or abandoned arms (hybrid M4 model, embedding-based semantic clustering, RAG/LLM case studies, the Hugging Face demo code). Their historical provenance remains in the included `semantic_microbiome/scripts/lit14_freeze_pipeline.py` and `semantic_microbiome/FROZEN_MICROBIOME_PIPELINE.json` (freeze record), and some included scripts (for example `CM/18_phase_x_incremental_value.py`, `CM/33_...py`) still carry superseded frozen-M3 or concordance-group sections; these sections and the superseded values they record (for example PRJNA842425 AUROC 0.55/0.567 under the earlier 32-event label) are not reported results.
- Per-sample read QC, host removal, MetaPhlAn/HUMAnN, assembly and contig-level resistome processing for the **Hospital A cohort itself** were run with scripts kept outside this project directory and are not part of this release (see "Known gaps").

## Paths are site-specific

All scripts use the hard-coded absolute paths of the analysis server (for example `/home/tcmhwd/microbiome/ICU_public/...`, `/data/tcmhwd/...`). The directory layout of this repository mirrors the original project root (`ICU_public/`), so relative imports work unchanged, but absolute paths must be edited to match a new site. Scripts were run as SLURM jobs; the wrappers for the clinical arm are in `clinical metadata/slurm/`, and the SBATCH headers of the sequencing scripts carry their original resources and conda environment names.

### Inputs each stage expects

| Stage | Expected inputs (not distributed) |
|---|---|
| Public-cohort download | `publicdata.csv` (EBI/SRA run table, `fastq_ftp` column), `AGI_public.csv` (CNCB/NGDC run table, `Download_path` column) |
| Per-sample processing | Paired FASTQ per run; reference databases under `ref/` (Bowtie2 GRCh38 index, MetaPhlAn `mpa_v30_CHOCOPhlAn_201901`, HUMAnN ChocoPhlAn v201901_v31 and UniRef90 v201901b, CARD 4.0.1, eggNOG 5.0.2, antiSMASH DB, GTDB R226, VFDB setB, CheckM2 DIAMOND DB); `{AGI,non_AGI,ICU}/sample_list.txt` |
| Feature merge (`merge/scripts/`) | Per-sample MetaPhlAn/HUMAnN outputs, cohort-level MAG results (`cohort_drep/`, `coverm_result/all_samples.coverm.tsv`, `EggNOG_result/`, `antiSMASH_result/`, `MAG_rgi_results/`, `VFDB_result/`, `gtdbtk_result/`), contig-level RGI and RPKM tables; Hospital A sample metadata `new_meta.csv`; PRJNA842425 `metadata.xlsx` (Stutz et al. 2022 supplementary data) |
| Cohort and views (`semantic_microbiome/scripts/0*.py`) | `metadata/microbiome.clinical.csv` (one row per linked patient, 430 rows: pseudonymised patient code, sequencing sample code, SOFA, outcome, sampling, admission and discharge dates, culture and antibiotic flags), `merge/output/*` |
| Literature panel (`lit01` to `lit08`, `lit14*`) | Internet access to NCBI E-utilities; environment variable `OPENAI_API_KEY` (required by `lit03`, `lit04`); environment variable `NCBI_EMAIL` (required by `lit01`, `lit02`) |
| Clinical model (`clinical metadata/scripts/01` to `15`, `10`, `33`) | `clinical metadata/ICU.csv` (institutional ICU admissions, both sites, site column `hospital`), `clinical metadata/mimic.csv` (MIMIC-IV extract, PhysioNet credentialed access) |
| Integration and sensitivity analyses | Outputs of all previous stages; `new_meta.csv` (ICU admission dates) for the landmark sensitivity analyses |

Environment variables introduced by the redactions (all required; each script stops with `SystemExit` and a clear message if one is missing or malformed; the check runs at the top of each script, before any project import, input read or output write): `NCBI_EMAIL` (non-empty), `OPENAI_API_KEY` (non-empty; replaces the original local key-file path), `ICU_DAY0_ATTESTED_IDS` (comma-separated list of exactly 18 unique, non-empty pseudonymised patient codes whose ICU-day-0 admission was confirmed by the data owner; used by `58_..._v4.py` and `lit_cc6_...py`, which additionally stop if any of the 18 is absent from the 302-patient index), `INFLUENTIAL_PATIENT_ID` (non-empty; the single patient whose data-driven logistic-regression out-of-fold probability is 1.0; used by `audit_outlier_sensitivity.py`, which additionally stops if the ID is not in the 302-patient index).

## Software environment

Versions stated in the Supplementary Methods (pinned in `environment.yml` for the statistical analyses):

- Python 3.10.19, scikit-learn 1.4.2, CatBoost 1.2.8, statsmodels 0.14.6, NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4, matplotlib 3.7.5
- R 4.4.3 with logistf 1.26.1 and survival 3.8.6; figures with ggplot2
- FastQC v0.11.9, Trimmomatic v0.39, Bowtie2 v2.4.4 (Hospital A reads); MetaPhlAn v3.0.14 (`mpa_v30_CHOCOPhlAn_201901`); HUMAnN v3.9 (ChocoPhlAn v201901_v31, UniRef90 v201901b); MEGAHIT v1.2.9; Prokka v1.15.6; RGI v6.0.4 with CARD v4.0.1 (Perfect and Strict hits); GTDB-Tk v2.6.1 with GTDB release 226 (confirmed by the job log of the cohort run: `GTDB-Tk v2.6.1`, `reference data version r226`; the reference data sit in a directory named `gtdbtk-2.5.2/db`, which is why `scripts/14_gtdb_re.sh` and the freeze record `FROZEN_MICROBIOME_PIPELINE.json` say 2.5.2; the freeze record's tool version is wrong and is left verbatim); MetaBAT2, MaxBin2, CONCOCT, DAS Tool, CheckM2, dRep, BWA-MEM, CoverM, eggNOG-mapper v2, antiSMASH, VFDB

Additional packages used by the clinical-model pipeline in the same Python environment (from `clinical_ai/reports/software_versions.txt` and the environment's package metadata, not stated in the manuscript): PyCaret 3.3.2, Optuna 4.6.0, PyTorch 2.5.1, pytorch-tabnet 4.1.0, LightGBM 4.6.0, XGBoost 2.0.3, joblib 1.3.2, openai 2.9.0.

Bioinformatics tool versions found in the server's conda environments at release time (not stated in the manuscript; environments may have been updated after the runs): CheckM2 1.1.0, CoverM 0.7.0, dRep 3.6.2, MetaBAT2 2.18, MaxBin2 2.2.7, CONCOCT 1.1.0, DAS Tool 1.1.7, eggNOG-mapper 2.1.13, antiSMASH 7.1.0, BWA 0.7.19, SAMtools 1.22.1, fastp 1.3.3 (used for the external cohorts, `TRIM_MODE=fastp`).

## Pipeline order

Run each stage after the previous one has finished. Within the clinical arm, numbering follows the original project.

1. **Public-cohort download**: `setup_ICU_public_download.sh` → `scripts/make_sample_lists.sh` → `scripts/submit_downloads.sh` (`download_ICU_public_array.sh`) → `scripts/check_ICU_public_downloads.sh`.
2. **Per-sample processing**: `scripts/1_trim_hostremove_public.sh` (and `1b_AGI_bowtie2_only_from_trimmed.sh`) → `2_humann3.sh` (public cohorts) / `2_humann3_ICU.sh` (Hospital A) → `3_hit.sh` (MEGAHIT) → `5_bam.sh` (`fix_CRR556748_bam.sh` for one AGI run) → `6_bin_re.sh` → `7_checkm2_re.sh`; contig-level resistome `19_prokka_contig.sh` → `20_rgi_contig.sh` → `21_rpkm_contig.sh`.
3. **Cohort MAG catalogue**: `8_mag_drep_re.sh` → `9_run_prokka_re.sh` → `10_run_eggnog_re.sh`, `11_run_antismash_re.sh`, `12_run_rgi_re.sh`, `13_run_vfdb_re.sh`, `14_gtdb_re.sh` → CoverM: `15a_make_index.sh` → `15b_bwa_map.sh` (`15b_retry.sh` for failed array tasks) → `15c_coverm_batch.sh` → `15c_merge_coverm.py`.
4. **Feature merge**: `merge/scripts/submit_merge_pipeline.sh` (01 metadata → 02 MetaPhlAn → 03 HUMAnN → 04a resistome adapter → 04 MAG summary). The submit script also chains steps 05 (ArcA/BUT ratio) and 06 (R-ready tables), which are not used by this manuscript and are not included; remove those two lines before running it.
5. **Cohort and structured views**: `semantic_microbiome/scripts/02_harmonize_sample_ids.py` → `03_define_discovery_cohort.py` → `09_build_structured_views.py` (`common.py` is the shared configuration module).
6. **Literature-derived panel**: `lit01_search_pubmed.py` → `lit02_fetch_articles.py` → `lit03_build_rag_corpus.py` → `lit04_extract_evidence.py` → `lit05_normalize_features.py` → `lit06_build_literature_library.py` → `lit07_freeze_literature_library.py` (writes the frozen library included here) → `lit08_build_literature_core_extended.py` (16 marker values) → `lit14_freeze_pipeline.py` (frozen M1 model used for external application) → `lit14_m1_refit_v4.py` (final M1 refit with the convergence fix; numerically identical predictions).
7. **Clinical model**: `clinical metadata/scripts/01_inspect_and_preprocess.py` → `02_missingness_analysis.py` → `03_ml_screening.py` → `04_tune_ml.py` → `05_tune_dl.py` → `06_evaluate_models.py` (full-feature models and conventional scores) → `08b_preprocess_reduced_by_site.py` → `12_tune_ml_reduced.py` → `13_tune_dl_reduced.py` → `14_aggregate_reduced_model_performance.py` → `15_select_best_reduced_model.py` (selects the Hospital A-developed 29-variable CatBoost) → `10_apply_reduced_model_to_microbiome_cohort.py` → `33_phase_x_gangnam_reference_and_groups.py` (Fig. S1A table). Shared helpers are in `clinical metadata/scripts/utils/`.
8. **Integration and incremental value**: `18_phase_x_incremental_value.py` (single-split out-of-fold M1 score, AUROC 0.592, and clinical stacking reference) → `45_phase_x_c1_c2_nested_cv_corrected_v4.py` (array over preprocessing variants, then `--aggregate`; slurm wrappers `45_..._v4.sh`, `45_..._v4_aggregate.sh`) → `61_phase_x_microbiome_over_sofa_nested_v4.py` → `55_..._v4.py`, `56_phase_x_panelA_discrimination_ci.py`, `58_..._v4.py`, `59_..._v4.py`, `60_..._v4.py`, `42_..._v4.py` → `clinical metadata/log/audit_v4_review_20261002/audit_outlier_sensitivity.py`. Table 1: `34_table1_baseline_characteristics.py`.
9. **External benchmarking and robustness**: `lit15_public_external_validation.py` → `lit26_covid_external_severity_concordance.py` → `lit_cc1_prjna842425_incremental.py` → `lit_cc6_round1_sensitivities.py`; `lit_LOCO_validation_sensitivity.py` → `lit_LOCO_AGI_M1_refit.py`; `lit_cc2_cross_cohort_biology.py`; `lit_cc4_a2_training_size.py`; `lit_cc5_a3_severity_antibiotics.py`; `lit53_excluded67_scoring.py` (verifies its own SHA-256 against `EXCLUDED67_PRESPECIFICATION.md` before running; the released copy is redacted, see "Redactions", so the check fails as shipped); `merge/downstream_analysis/scripts/12_sample_type_confounding.py`.
10. **MDRO and VRE (exploratory)**: `49_phase_x_simple_pathobiont_flag.py`, `51_phase_x_mdro_acquisition_prediction.py`, `52_phase_x_mdro_metagenome_concordance.py`, `53_phase_x_vre_sensitivity_landmark_firth.py`, `54_phase_x_vre_collection_day_adjusted.py`, `62_phase_x_vre_vana_vanb_split.py` (Firth regression through `utils/run_firth_logistf.R`).
11. **Figures that compute reported numbers**: `semantic_microbiome/R_fig/scripts/15b_paper1_figures_final_v4.R` (Figs. 1 to 4), `21_lean_figS4_external_transportability_v2.R` (Fig. S4), both sourcing `00_setup_theme.R`.

## Manuscript item to script map

Script paths are relative to the repository root. `CM/` = `clinical metadata/scripts/`, `SM/` = `semantic_microbiome/scripts/`, `RF/` = `semantic_microbiome/R_fig/scripts/`.

| Manuscript item | Content | Script(s) |
|---|---|---|
| Table 1 | Baseline characteristics, survivors vs deaths (N = 302) | `CM/34_table1_baseline_characteristics.py` (cohort from `SM/03_define_discovery_cohort.py`) |
| Table 2 | ΔAUROC vs clinical model, vs SOFA, vs SOFA + clinical model; paired bootstrap | Clinical-model rows: `CM/45_phase_x_c1_c2_nested_cv_corrected_v4.py` (variants REF, V3, V4_STRICT; bootstrap in the aggregate step). SOFA and SOFA + clinical rows: `CM/61_phase_x_microbiome_over_sofa_nested_v4.py`. Post hoc Holm across the three clinical-model contrasts (0.440, 0.135, 0.230): `analysis_extras/holm_table2.py` (written for this release) |
| Table 3 | External benchmarking of the frozen score; PRJNA842425 increments vs SOFA and APACHE II | AGI AUROC: `SM/lit15_public_external_validation.py`; PRJNA842425 severity scores and frozen M1 probabilities: `SM/lit26_covid_external_severity_concordance.py`; increments and 20-fold-assignment sensitivity: `SM/lit_cc1_prjna842425_incremental.py`; 27-event AUROC CI: `SM/lit_cc6_round1_sensitivities.py` |
| Fig. 1 | Study design and cohort flow | Counts from `CM/01_inspect_and_preprocess.py`, `CM/08b_preprocess_reduced_by_site.py`, `merge/scripts/01_build_metadata.py`, `SM/03_define_discovery_cohort.py`; drawn by `RF/15b_paper1_figures_final_v4.R` |
| Fig. 2A | AUROC of SOFA, clinical model, literature-derived score | `CM/56_phase_x_panelA_discrimination_ci.py` (reads `CM/18_phase_x_incremental_value.py` OOF output) |
| Fig. 2B | Paired decision curves vs clinical model | `CM/55_phase_x_decision_curve_nested_paired_v4.py`; Holm on selected thresholds also in `RF/15b_paper1_figures_final_v4.R` |
| Fig. 2C | Brier scores | `CM/59_phase_x_calibration_metrics_v4.py` |
| Fig. 3A | ΔAUROC forest plot | `CM/45_..._v4.py`, `CM/61_..._v4.py` |
| Fig. 3B | PRJNA842425 increments | `SM/lit_cc1_prjna842425_incremental.py` |
| Fig. 4 | Cause-specific Cox, time-split, Fine-Gray | `CM/42_phase_x_time_to_event_competing_risk_v4.py` |
| Table S1 | Decision curves vs clinical references | `CM/55_..._v4.py` (clinical-model referenced, Holm), `CM/61_..._v4.py` (SOFA referenced, Holm) |
| Table S2 | Calibration and Brier | `CM/59_phase_x_calibration_metrics_v4.py` |
| Table S3 | SOFA-adjusted ORs, LRT, NRI, IDI | `CM/61_phase_x_microbiome_over_sofa_nested_v4.py` |
| Table S4 | Influence sensitivity (one extreme OOF probability) | `clinical metadata/log/audit_v4_review_20261002/audit_outlier_sensitivity.py` (reads outputs of `CM/45_..._v4.py`, `CM/61_..._v4.py`, `CM/42_..._v4.py`) |
| Table S5 | Alternative sampling landmarks; excluded-67 analysis | `CM/58_phase_x_landmark_anchor_sensitivity_v4.py`; `SM/lit53_excluded67_scoring.py` (pre-specified in `EXCLUDED67_PRESPECIFICATION.md`) |
| Table S6 | Specimen type; severity and antibiotic confounding | A: `merge/downstream_analysis/scripts/12_sample_type_confounding.py`; B, C: `SM/lit_cc5_a3_severity_antibiotics.py` |
| Table S7 | External cohort characteristics and performance | `SM/lit15_...py`, `SM/lit26_...py`, `SM/lit_cc1_...py`, `SM/lit_cc6_...py` |
| Table S8 | Pathobiont rules; cross-cohort marker reproducibility | Rules (Holm within family): `CM/49_phase_x_simple_pathobiont_flag.py`; marker effects, effect-vector Spearman correlations, Enterococcus ≥30% replication: `SM/lit_cc2_cross_cohort_biology.py` |
| Table S9 | MDRO grid, metagenome-culture agreement, VRE sensitivity | A: `CM/51_phase_x_mdro_acquisition_prediction.py`; B: `CM/52_phase_x_mdro_metagenome_concordance.py`; C: `CM/53_...firth.py` (principal Firth OR), `CM/54_phase_x_vre_collection_day_adjusted.py` (sampling day), `CM/62_phase_x_vre_vana_vanb_split.py` (vanA only) |
| Fig. S1 | Clinical models across sites; reduced-feature candidates | A: `CM/01`, `03`, `04`, `05`, `06`, `33_phase_x_gangnam_reference_and_groups.py`; B: `CM/08b`, `12`, `13`, `14` (selection `15`) |
| Fig. S2 | Calibration curves | `CM/59_phase_x_calibration_metrics_v4.py` |
| Fig. S3 | A: repeated nested CV; B: learning curves | A: `CM/60_phase_x_repeated_nested_cv_v4.py`; B: `SM/lit_cc4_a2_training_size.py` |
| Fig. S4 | External decision curve; marker heat map; Enterococcus threshold | `SM/lit_cc1_...py` (A), `SM/lit_cc2_...py` (B, C); `RF/21_lean_figS4_external_transportability_v2.R` (binomial tests) |
| Fig. S5 | MDRO associations and concurrent agreement | `CM/51_...py`, `CM/52_...py`, `CM/53_...py` |
| Results: clinical model | AUROC 0.818 (Hospital B), 0.780 (MIMIC-IV); CatBoost selected | `CM/12_tune_ml_reduced.py`, `CM/14_...py`, `CM/15_select_best_reduced_model.py`; applied without retraining by `CM/10_apply_reduced_model_to_microbiome_cohort.py` |
| Results: excluded-67 | AUROC 0.605, SOFA-adjusted OR 1.82, ΔAUROC +0.019 | `SM/lit53_excluded67_scoring.py` |
| Results: LOCO | Prevotella removed; AGI AUROC 0.618 → 0.614; ρ 0.994 | `SM/lit_LOCO_validation_sensitivity.py`, `SM/lit_LOCO_AGI_M1_refit.py` |
| Results: Holm | Holm-adjusted DCA P (Table S1), MDRO family of 14 (Table S9), pathobiont rules (Table S8) | `CM/55_..._v4.py`, `CM/61_..._v4.py`, `CM/51_...py`, `CM/49_...py`. Table 2 three-contrast Holm: `analysis_extras/holm_table2.py` |
| Results: repeated CV | Median ΔAUROC +0.007, +0.007, +0.009 over 20 partitions | `CM/60_phase_x_repeated_nested_cv_v4.py` |
| Results: directional concordance | 11/16 and 10/16 markers; one-sided binomial P 0.105 and 0.227; ρ 0.59 and 0.39 | Counts and ρ: `SM/lit_cc2_...py`; binomial P: `RF/21_lean_figS4_external_transportability_v2.R` |
| Results: antibiotic adjustment | OR 1.47 vs 1.46 | `SM/lit_cc5_a3_severity_antibiotics.py` |
| Supplementary Methods: literature search | 58 queries, 674 records, 670 extracted | `SM/lit01`, `lit02`, `lit04`; counts recorded in `literature_prior/final/FROZEN_LIBRARY_MANIFEST.json` |
| Supplementary Methods: MAG catalogue | 18,501 bins → 10,843 representatives | `scripts/7_checkm2_re.sh`, `scripts/8_mag_drep_re.sh` |
| Supplementary Methods: sequencing depth | Median 36.2 M read pairs after QC (IQR, range) | `semantic_microbiome/reports/paper1/supp_methods_pass_2026-10-30/work/qc_read_depth_from_bwa_logs.py` (reads the BWA mapping logs of `scripts/15b_bwa_map.sh`) |

### Figure rendering

Figures S1, S2, S3 and S5 were rendered by R scripts that only read the CSV outputs of the scripts above and compute no reported number (`R_fig/scripts/02_regenerate_supplementary_figures.R`, `16_supplementary_TS9_TS11_figures_v3.R`, `20_lean_figS3_resampling_robustness_v4.R`, `22_lean_figS5_mdro_vre.R`). They are not included.

## Known gaps

- Hospital A per-sample read QC (FastQC, Trimmomatic, Bowtie2), MetaPhlAn/HUMAnN inputs to `2_humann3_ICU.sh`, and the contig-level RGI and RPKM tables for Hospital A were produced by scripts outside this project directory.
- Mortality among the 128 patients excluded for later sampling ("57 of 128 (44.5%)", Results, study population): no producing script was found in the project (it corresponds to a direct count of deaths among the `collection_time == 1` patients of `metadata/microbiome.clinical.csv`).
- `semantic_microbiome/R_fig/scripts/15b_paper1_figures_final_v4.R` reads `semantic_microbiome/reports/methodological_revision_v2/primary_results_table.csv`, a curated source-of-truth table maintained by hand during the project's audits (its `source` column names the script output behind each row); no script writes it.
- The superseded producers included above themselves read outputs of still earlier scripts that are not included (for example the single-split bootstrap of script 21 and the SOFA bootstrap of script 41, read by original 48). These feed only reference comparisons, not reported numbers.
- `lit_cc5_a3_severity_antibiotics.py` reads the M1 out-of-fold file written by the original version of script 61 (`phase_x_sofa_nested_oof_probabilities.csv`; that script is now included); its M1 column is numerically identical to the `_v4` file written by `61_..._v4.py`.
- The released copy of `lit53_excluded67_scoring.py` is redacted (admission-date bounds coarsened to month), so its self-integrity check against the pin in `EXCLUDED67_PRESPECIFICATION.md` fails as shipped. The pinned value is the SHA-256 of the original, unredacted script (see "Redactions").

## Data availability

- Hospital A host-filtered shotgun metagenomes: NCBI SRA BioProject **PRJNA1566100**.
- AGI cohort: **PRJCA011331** (Genome Sequence Archive **CRA007908** and **CRA008809**).
- COVID-19 ICU cohort: **PRJNA842425**.
- MIMIC-IV: available to credentialed users through PhysioNet.
- Institutional clinical data: available from the corresponding author on reasonable request, subject to IRB approval.
- Frozen models: Hugging Face Space [`tcmhwd/evidence-grounded-icu-risk`](https://huggingface.co/spaces/tcmhwd/evidence-grounded-icu-risk).
- Analysis code: this repository, release **v1.0.1** (supersedes v1.0.0, which was withdrawn; v1.0.1 has a fresh history).

## Redactions

Every file is listed in `MANIFEST.sha256` with the SHA-256 of its original source file, the SHA-256 of the released copy, a redaction flag and the reason. Redactions remove a personal e-mail address, patient and sample identifiers in comments and log strings, one single-patient narrative, day-level admission dates and an individual ward-stay range for the excluded-67 subgroup (coarsened to month), record-level example values and unreported whole-file counts in `CM/utils/config.py` comments, a local credential-file path, and two hard-coded patient-ID lists plus one single patient ID (replaced by environment variables). Comment and string redactions keep line numbers. The scripts that read the environment variables gain a few lines of validation only (required lookup, exact count and membership checks; see "Inputs each stage expects"); no analysis logic changed, and the comparison bounds of the excluded-67 date check in `lit53` were widened to whole months.

`semantic_microbiome/scripts/lit53_excluded67_scoring.py` and `semantic_microbiome/reports/paper1/EXCLUDED67_PRESPECIFICATION.md` are redacted copies. The SHA-256 pinned in section 7 of the prespecification, `e1157b9b28f8e507948050fb49cb8fb4471315c8023a20286902b32ea537c238`, is the hash of the ORIGINAL `lit53` as frozen before the analysis ran, not of the released copy; the original prespecification's SHA-256 is `f84ec60088b6a568046875d9e4e9b9dcfa36dc699a2b5d5cf7c98406dd1ff4da`. Both are recorded in `MANIFEST.sha256`.

## Citation

See `CITATION.cff`. A DOI will be added on publication.

## License

MIT (see `LICENSE`) for the code. Quoted article excerpts in the literature CSVs are not covered by the MIT grant; see `NOTICE`.
