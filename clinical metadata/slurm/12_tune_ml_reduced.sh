#!/bin/bash
#SBATCH --job-name=clinAI_12_reduced_ml
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/12_reduced_ml_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/12_reduced_ml_%j.err"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

# Reduced-feature (29-column) ML training -- lr/catboost/et x 3 sites
# (Gangnam/Sinchon/MIMIC). Partition/node and --models/--sites are supplied
# by the caller (see the paired sbatch invocations that distribute the 3
# algorithms across node1/node2). Usage:
#   sbatch --partition=cpu_partition --nodelist=node2 \
#     "slurm/12_tune_ml_reduced.sh" --models lr --sites Gangnam Sinchon MIMIC

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

# 08b_preprocess_reduced_by_site.py already run once manually (2026-08-14) --
# NOT re-run here to avoid a multi-job race on the shared parquet files
# (several of these jobs run concurrently on node1/node2). Re-run it by hand
# first if the reduced feature set (utils/reduced_features.py) ever changes.
python3 "$BASE/scripts/12_tune_ml_reduced.py" "$@"
