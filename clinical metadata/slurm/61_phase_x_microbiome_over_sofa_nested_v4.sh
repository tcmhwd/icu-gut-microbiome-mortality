#!/bin/bash
#SBATCH --job-name=clinAI_61v4_sofa
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/61_v4_sofa_nested_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/61_v4_sofa_nested_%j.err"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition

# v4 (2026-10-02, "final technical closure pass"): runs
# 61_phase_x_microbiome_over_sofa_nested_v4.py -- M1 now on the same
# convergence ladder as every other LR fit (was pinned at max_iter=3000 in
# v3), and M3-LR moved from V3 to V4_STRICT (prevalence filter AND PCA both
# tuning-local). M3-CatBoost unchanged. Same resource profile and BLAS
# thread-limiting rationale as the v3 wrapper (unchanged by this pass --
# see that wrapper's comments for the full reproduction-gate/threading
# history); V4_STRICT's GridSearchCV(72 CV paths) cost is comparable to
# V3's own M3-LR fit, observed ~18 min for the analogous V2S variant inside
# the 45_v4 array job.
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

python3 -u 61_phase_x_microbiome_over_sofa_nested_v4.py

echo "61_phase_x_microbiome_over_sofa_nested_v4.sh complete."
