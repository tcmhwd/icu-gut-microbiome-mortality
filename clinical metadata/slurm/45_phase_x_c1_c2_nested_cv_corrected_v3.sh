#!/bin/bash
#SBATCH --job-name=clinAI_45v3_foldlocal
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v3_%A_%a.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v3_%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition
#SBATCH --array=0-5

# Fold-local M3 preprocessing correction (owner-authorised 2026-10-02).
# One array task per variant of 45_phase_x_c1_c2_nested_cv_corrected_v3.py.
# Resources vs the original 45 wrapper (4 cpu / 8G / 30 min): LR max_iter
# raised 3000 -> 30000 (-> 100000 escalation) with saga on p~900 > n~240, so
# runtime can grow ~10-30x; 10 cpus = LogisticRegressionCV's 9 (3 folds x 3
# l1_ratios) parallel paths + 1. The script honours SLURM_CPUS_PER_TASK via
# config.get_n_jobs(). V3 loads the unfiltered CoverM/KO matrices (~60 MB) -> 24G.
# Aggregation: 45_phase_x_c1_c2_nested_cv_corrected_v3_aggregate.sh (afterok).

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

VARIANTS=(REF V0 V1 V2 V3 V2S)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
cd "$BASE/scripts"
echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}; variant: $V ==="
python3 -u 45_phase_x_c1_c2_nested_cv_corrected_v3.py --variant "$V"
echo "variant $V complete."
