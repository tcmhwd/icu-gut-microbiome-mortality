#!/usr/bin/env python3
"""03_ml_screening.py

PyCaret algorithm-family screening (Section 27/28). Screening ONLY — these
scores are not treated as final model performance; formal tuning happens in
04_tune_ml.py outside the PyCaret black box.

Screening is done separately per development dataset (Experiment A: ICU;
Experiment B: MIMIC), 5-fold stratified CV, sorted by AUROC, with AUPRC/F1/
recall/precision/accuracy also retained. No PCA / automatic feature
selection / SMOTE / automatic outlier removal / polynomial features
(PyCaret defaults already off; not enabled here).

Missing-data note: features are pre-processed with the Universal pipeline
(utils.preprocessing_pipeline.UniversalPreprocessor, scale=True) fit on the
FULL landmark cohort of the dataset being screened (not per-CV-fold). This is
a documented, screening-only simplification -- it never crosses between ICU
and MIMIC, and formal tuning (04_tune_ml.py) refits the Universal pipeline
strictly within each CV training fold, so this simplification never affects
a reported final performance number.

Run on cpu_partition/node2 via slurm/02_ml_screening.sh; CPU-only, can take
a while for ~11 algorithms x 5-fold CV x ~64k/63k rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, io_utils, model_registry, preprocessing_pipeline, reporting  # noqa: E402

log = io_utils.setup_logging("03_ml_screening")

FEATURE_COLS = preprocessing_pipeline.ALL_CLEAN_COLS
NON_FEATURE_COLS = ["sample_id", "op_value", "landmark_basis", config.TARGET_COL] + \
    [f"severity__{c}" for c in config.SEVERITY_SCORE_COMPARATORS]


def load_dataset(name: str) -> pd.DataFrame:
    path = os.path.join(config.DIR_PREPROCESSING, f"model_ready_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run 01_inspect_and_preprocess.py first")
    return pd.read_parquet(path)


def screen_dataset(name: str, n_jobs: int, sample_n: int | None = None, fold: int | None = None) -> pd.DataFrame:
    from pycaret.classification import compare_models, pull, setup

    fold = fold or config.N_CV_FOLDS
    df = load_dataset(name)
    if sample_n is not None and sample_n < len(df):
        # SMOKE-TEST ONLY: stratified subsample so both classes remain present.
        df = df.groupby(config.TARGET_COL, group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(sample_n // 2, 20)), random_state=config.RANDOM_SEED)
        )
        log.warning("dataset=%s: SMOKE-TEST subsample active (N=%d) -- not a real screening run", name, len(df))
    X_raw = df[FEATURE_COLS]
    y = df[config.TARGET_COL]

    prep = preprocessing_pipeline.UniversalPreprocessor(scale=True)
    X_processed = prep.fit_transform(X_raw)
    pycaret_data = X_processed.copy()
    pycaret_data[config.TARGET_COL] = y.values

    log.info("dataset=%s: setting up PyCaret classification env (N=%d, features=%d)",
              name, len(pycaret_data), X_processed.shape[1])
    setup(
        data=pycaret_data, target=config.TARGET_COL, session_id=config.RANDOM_SEED,
        train_size=0.9, fold=fold, fold_strategy="stratifiedkfold",
        normalize=False,  # already scaled by UniversalPreprocessor
        preprocess=True,  # no-op here: no missing values remain after UniversalPreprocessor
        fix_imbalance=False, remove_outliers=False, polynomial_features=False,
        remove_multicollinearity=False, transformation=False,
        html=False, verbose=False, n_jobs=n_jobs, use_gpu=False,
    )

    include = [model_registry.REGISTRY[m]["pycaret_id"] for m in model_registry.SCREENING_INCLUDE]
    log.info("dataset=%s: compare_models(include=%s)", name, include)
    try:
        compare_models(include=include, fold=fold, sort="AUC",
                        errors="ignore", verbose=False, n_select=len(include))
    except Exception as e:
        log.error("dataset=%s: compare_models raised: %s", name, e)
        reporting.append_failure_rows([{
            "stage": "03_ml_screening", "model_name": "ALL", "development_dataset": name,
            "error_type": type(e).__name__, "error_message": str(e)[:500],
            "resolution": "compare_models() call itself failed; see traceback in log",
        }])
        traceback.print_exc()

    results = pull()
    results = results.reset_index().rename(columns={"index": "model_id"})
    results.insert(0, "dataset", name)

    fitted_ids = set(results["model_id"].tolist())
    missing_ids = set(include) - fitted_ids
    failure_rows = []
    for mid in missing_ids:
        failure_rows.append({
            "stage": "03_ml_screening", "model_name": mid, "development_dataset": name,
            "error_type": "not_in_compare_models_results",
            "error_message": "model missing from PyCaret compare_models() output "
                              "(errors='ignore' suppressed the individual exception)",
            "resolution": "excluded from screening table for this dataset; verify "
                           "dependency/data compatibility manually if this model family is of interest",
        })
    reporting.append_failure_rows(failure_rows)
    if missing_ids:
        log.warning("dataset=%s: %d model(s) failed/absent from screening: %s", name, len(missing_ids), missing_ids)

    return results


def select_candidates(results_icu: pd.DataFrame, results_mimic: pd.DataFrame, n_extra: int = 2):
    icu_auc = results_icu.set_index("model_id")["AUC"] if len(results_icu) else pd.Series(dtype=float)
    mimic_auc = results_mimic.set_index("model_id")["AUC"] if len(results_mimic) else pd.Series(dtype=float)
    both = pd.DataFrame({"ICU_AUC": icu_auc, "MIMIC_AUC": mimic_auc})
    both["mean_AUC"] = both.mean(axis=1, skipna=True)
    both["n_datasets_present"] = both[["ICU_AUC", "MIMIC_AUC"]].notna().sum(axis=1)
    both = both.sort_values(["n_datasets_present", "mean_AUC"], ascending=[False, False])

    selected = ["lr"]  # Section 28: always retained as interpretable reference
    used_family_groups = {model_registry.REGISTRY["lr"]["family_group"]}
    rationale = {"lr": "Retained unconditionally as the interpretable clinical reference model (Section 28), "
                        "regardless of screening rank."}

    for mid, row in both.iterrows():
        if len(selected) >= 1 + n_extra:
            break
        if mid not in model_registry.REGISTRY or mid == "lr":
            continue
        fam = model_registry.REGISTRY[mid]["family_group"]
        if fam in used_family_groups:
            continue
        selected.append(mid)
        used_family_groups.add(fam)
        rationale[mid] = (
            f"Selected for formal Optuna tuning: mean screening AUROC={row['mean_AUC']:.4f} "
            f"(ICU={row.get('ICU_AUC', float('nan')):.4f}, MIMIC={row.get('MIMIC_AUC', float('nan')):.4f}), "
            f"present in {int(row['n_datasets_present'])}/2 screening runs, "
            f"family_group='{fam}' (not yet represented among selected candidates)."
        )

    # If diversity constraint left us short, fill remaining slots by pure rank
    if len(selected) < 1 + n_extra:
        for mid, row in both.iterrows():
            if len(selected) >= 1 + n_extra:
                break
            if mid in selected or mid not in model_registry.REGISTRY:
                continue
            selected.append(mid)
            rationale[mid] = (
                f"Selected to fill remaining candidate slot by rank (diversity constraint relaxed): "
                f"mean screening AUROC={row['mean_AUC']:.4f}."
            )

    return selected, rationale, both


def write_selection_doc(selected, rationale, ranking_table, results_icu, results_mimic):
    lines = ["# ML Candidate Selection", "",
              "Selection method: automated, rank+diversity heuristic over the PyCaret screening "
              "tables below (`pycaret_ICU_results.csv`, `pycaret_MIMIC_results.csv`). Penalized "
              "logistic regression is always retained as the interpretable clinical reference "
              "(Section 28). This is a documented DEFAULT selection; override with "
              "`04_tune_ml.py --models <id1> <id2> ...` if a different set is preferred after "
              "reviewing the tables.", "",
              "## Selected models", ""]
    for mid in selected:
        disp = model_registry.REGISTRY[mid]["display"]
        lines.append(f"- **{mid}** ({disp}, family_group={model_registry.REGISTRY[mid]['family_group']})")
        lines.append(f"  - {rationale[mid]}")
    lines += ["", "## Full mean-AUROC ranking (both screening runs)", "", "```",
              ranking_table.reset_index().rename(columns={"index": "model_id"}).to_string(index=False),
              "```", ""]
    if model_registry.SCREENING_EXCLUDED_WITH_REASON:
        lines += ["## Algorithms excluded from the screening zoo up front", ""]
        for mid, reason in model_registry.SCREENING_EXCLUDED_WITH_REASON.items():
            lines.append(f"- `{mid}`: {reason}")
    path = io_utils.save_text("\n".join(lines) + "\n",
                               os.path.join(config.DIR_ML_SCREENING, "ml_candidate_selection.md"))
    log.info("Wrote %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-extra", type=int, default=2, help="ML candidates in addition to lr (1 or 2 -> total 2-3)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--smoke-test-sample-n", type=int, default=None,
                     help="DEV/CI ONLY: subsample each dataset to ~N rows and use --smoke-test-fold folds. "
                          "Never use for a real screening run.")
    ap.add_argument("--smoke-test-fold", type=int, default=2)
    args = ap.parse_args()

    config.ensure_output_dirs()
    n_jobs = args.n_jobs or config.get_n_jobs()
    log.info("Using n_jobs=%d", n_jobs)

    fold = args.smoke_test_fold if args.smoke_test_sample_n else None

    results_icu = screen_dataset("ICU", n_jobs, sample_n=args.smoke_test_sample_n, fold=fold)
    io_utils.save_csv(results_icu, os.path.join(config.DIR_ML_SCREENING, "pycaret_ICU_results.csv"))

    results_mimic = screen_dataset("MIMIC", n_jobs, sample_n=args.smoke_test_sample_n, fold=fold)
    io_utils.save_csv(results_mimic, os.path.join(config.DIR_ML_SCREENING, "pycaret_MIMIC_results.csv"))

    selected, rationale, ranking = select_candidates(results_icu, results_mimic, n_extra=args.n_extra)
    write_selection_doc(selected, rationale, ranking, results_icu, results_mimic)

    sel_path = os.path.join(config.DIR_ML_SCREENING, "selected_ml_models.json")
    with open(sel_path, "w") as fh:
        json.dump({"selected_models": selected, "rationale": rationale}, fh, indent=2)
    log.info("Selected ML candidates for Optuna tuning: %s (see %s)", selected, sel_path)

    log.info("03_ml_screening.py complete.")


if __name__ == "__main__":
    main()
