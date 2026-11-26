#!/bin/bash
#SBATCH --job-name=clinAI_45_nestedcv_fix
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_nested_cv_corrected_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_nested_cv_corrected_%j.err"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu_partition

# Codex-review action item (C)-6 (2026-09-01): fixes a confirmed CV-nesting
# leakage bug in 18_phase_x_incremental_value.py's headline CV_AUROC_5fold
# for C1/C2a/C2b (two StratifiedKFold objects with identical
# n_splits/shuffle/random_state/y producing identical fold partitions across
# the stage-1 OOF microbiome-prob generation and the stage-2 meta-model
# evaluation). Does NOT modify 18 in place -- runs the new, nested-CV
# corrected sibling script only. See that script's own docstring for the
# full bug/fix description.
#
# Light job: 90 base-model refits (elastic-net LR / shallow CatBoost) + 15
# meta LR fits total across C1/C2a/C2b -- well under 10 minutes.
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/45_phase_x_c1_c2_nested_cv_corrected.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE/scripts"

echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset} ==="

python3 45_phase_x_c1_c2_nested_cv_corrected.py

echo "45_phase_x_c1_c2_nested_cv_corrected.sh complete."
