#!/bin/bash
#SBATCH --job-name=clinAI_33_gangnam_ref
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/33_gangnam_reference_and_groups_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/33_gangnam_reference_and_groups_%j.err"
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_partition

# Rerun after fixing a hardcoded stale "N=2000" bootstrap-count label in
# Part B's figure (figures/phase_x_severity_score_comparison_302.svg) to
# read the actual N_BOOT=config.N_BOOTSTRAP=1000 (the .md report's prose
# was already corrected 2026-09-04; the figure's x-axis label was a literal
# string that never referenced the variable, so it stayed stale). Reusing
# this wrapper also regenerates Part A (phase_x_gangnam_reference_comparison,
# ~23k-row Gangnam ICU.csv bootstrap, N_BOOT=1000 resamples across 4 scores)
# and Part C (phase_x_concordance_group_characteristics, descriptive stats on
# N=302) -- both unaffected by the fix, rerun only because they share this
# script file; neither is expensive (Part A: ~4 scores x 1000 resamples of
# roc_auc_score on ~23k rows; Part C: plain descriptive stats on N=302).
# --cpus-per-task=2 / --mem=4G / --time=00:20:00: same reasoning as this
# project's other phase_x wrapper scripts (slurm/59_phase_x_calibration_
# metrics.sh) -- single-threaded script, generous margin over the expected
# <2 minute wall time.
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/33_phase_x_gangnam_reference_and_groups.sh"

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

echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset} ==="

python3 33_phase_x_gangnam_reference_and_groups.py

echo "33_phase_x_gangnam_reference_and_groups.sh complete."
