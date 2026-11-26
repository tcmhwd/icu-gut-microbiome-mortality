#!/bin/bash
#SBATCH --job-name=clinAI_58_landmark_v4
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/58_landmark_anchor_sensitivity_v4_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/58_landmark_anchor_sensitivity_v4_%j.err"
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_partition

# Fold-local M3 correction (2026-10-02) of 58_phase_x_landmark_anchor_sensitivity.py:
# re-scores already-persisted OOF columns (no model fitting) on ICU-anchored
# subsets of the same 302 patients, now reading the fold-local-M3-corrected
# C2a/C2b nested OOF columns (45_v3). Same small-scale bootstrap workload as
# the original script (6 models x 3 cohorts x 1000 resamples on N<=302) --
# single-threaded, BLAS pinned to 1, resource spec copied from the
# comparably-shaped 59_v3 wrapper with a modest margin.
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/58_phase_x_landmark_anchor_sensitivity_v4.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE/scripts"

echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}; OMP/OPENBLAS/MKL_NUM_THREADS: $OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS/$MKL_NUM_THREADS ==="

python3 58_phase_x_landmark_anchor_sensitivity_v4.py

echo "58_phase_x_landmark_anchor_sensitivity_v4.sh complete."
