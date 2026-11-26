#!/bin/bash
#SBATCH --job-name=clinAI_02_ml_screening
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/02_ml_screening_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/02_ml_screening_%j.err"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

# Step 2/5 -- PyCaret algorithm-family screening (11 algorithms x 5-fold CV,
# separately on the ICU and MIMIC landmark cohorts). CPU-only. Requires
# slurm/01_preprocess.sh to have completed (reads model_ready_*.parquet).
#
# Usage: sbatch "slurm/02_ml_screening.sh"
# Optional overrides: sbatch "slurm/02_ml_screening.sh" --n-extra 1

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u   # the 'gpt' env's conda activate.d hooks (MKL) are not `set -u`-safe
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

python3 "$BASE/scripts/03_ml_screening.py" "$@"
