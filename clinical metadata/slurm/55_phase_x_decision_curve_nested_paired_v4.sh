#!/bin/bash
#SBATCH --job-name=clinAI_55_dca_v4
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/55_decision_curve_nested_paired_v4_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/55_decision_curve_nested_paired_v4_%j.err"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_partition

# Fold-local M3 correction (2026-10-02, owner-authorised) of
# 55_phase_x_decision_curve_nested_paired.py: repoints the paired decision-curve
# bootstrap at 45_phase_x_c1_c2_nested_cv_corrected_v4.py's fold-local PRIMARY
# (V3) nested OOF predictions (phase_x_c1_c2_nested_oof_probabilities_v3_V3.csv)
# for C0/C1/C2a/C2b. No model fitting happens in this script -- it only reads an
# already-persisted OOF prediction file and computes paired bootstrap net
# benefit -- so this is a cheap, single-threaded, single-process rerun, sized
# the same as the comparably-shaped script 59 v3 wrapper (2000-replicate
# bootstrap over N=302, 25 thresholds x 3 models = 75 cells): single-threaded,
# BLAS pinned to 1, no multiprocessing. 30 minutes is a generous margin over
# that scale (script 59's analogous bootstrap, more models, finished in
# 2-3 minutes).
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/55_phase_x_decision_curve_nested_paired_v4.sh"

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

python3 55_phase_x_decision_curve_nested_paired_v4.py

echo "55_phase_x_decision_curve_nested_paired_v4.sh complete."
