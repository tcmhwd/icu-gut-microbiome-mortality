#!/bin/bash
#SBATCH --job-name=clinAI_59_calibration_v4
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/59_calibration_metrics_v4_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/59_calibration_metrics_v4_%j.err"
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_partition

# V3 rerun (2026-10-02, owner-authorised correction) of
# 59_phase_x_calibration_metrics.py, repointed at the fold-local-corrected
# nested OOF file (clinical_ai/predictions/
# phase_x_c1_c2_nested_oof_probabilities_v3_V3.csv, from script
# 45_phase_x_c1_c2_nested_cv_corrected_v4.py) for C0/C1/C2a/C2b; M1-alone
# still reads the OLD script-18 file, unchanged, because M1 was not touched
# by the V3 correction. See that script's own docstring and
# semantic_microbiome/reports/paper1/FINAL_LEAN_SUBMISSION_2026-10-02_v3/QC/
# M3_FOLD_LOCAL_METHOD_AUDIT_v3.md for what changed and why. Resource spec
# copied verbatim from the original wrapper
# (slurm/59_phase_x_calibration_metrics.sh): single-threaded, BLAS pinned
# to 1, no multiprocessing -- measured ~2-3 minute wall time there applies
# unchanged here (same N=302, same bootstrap counts, same 5 models).
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/59_phase_x_calibration_metrics_v4.sh"

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

python3 59_phase_x_calibration_metrics_v4.py

echo "59_phase_x_calibration_metrics_v4.sh complete."
