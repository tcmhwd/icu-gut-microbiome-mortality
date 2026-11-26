#!/bin/bash
#SBATCH --job-name=clinAI_60v4_repeated_nestedcv
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/60_v4_repeated_nested_cv_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/60_v4_repeated_nested_cv_%j.err"
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition
# v5 Fig S3A regeneration: 60_phase_x_repeated_nested_cv_v4.py, swaps the M3-LR arm to the
# V4_STRICT (fully tuning-local) specification. Timing test (job 641772, seed 2026 only,
# single-threaded) measured 2716s (~45min) and reproduced the published V4_STRICT anchor
# (C0=0.7528, C1=0.7610, C2a=0.7677, C2b=0.7711) exactly. 20 parallel single-threaded workers
# (one per seed, via this script's own multiprocessing.Pool) bound total wall time by one
# seed's own serial cost, not by the seed count -- so ~45-60 min expected for all 20.
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
python3 60_phase_x_repeated_nested_cv_v4.py
echo "60_phase_x_repeated_nested_cv_v4.sh complete."
