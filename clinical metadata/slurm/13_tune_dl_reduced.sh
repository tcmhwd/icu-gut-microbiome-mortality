#!/bin/bash
#SBATCH --job-name=clinAI_13_reduced_dl
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/13_reduced_dl_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/13_reduced_dl_%j.err"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

# Reduced-feature (29-column) DL training -- MLP/FTTransformer/TabNet x 3
# sites (Gangnam/Sinchon/MIMIC). CPU-only by design (user decision
# 2026-08-14: GPU is occupied by other work; node1/node2 are used here only
# for their free CPU capacity). Partition/node/--families/--sites/
# --trials-only are supplied by the caller -- see the paired sbatch
# invocations that spread FTTransformer's 3 (primary + 2 helper) processes
# across node1/node2, mirroring Stage 1's log/04_helper_FT_*.out pattern.
#
# Usage:
#   sbatch --partition=cpu_partition --nodelist=node2 --job-name=clinAI_13_MLP \
#     "slurm/13_tune_dl_reduced.sh" --families MLP --sites Gangnam Sinchon MIMIC
#   sbatch --partition=gpu_partition --nodelist=node1 --job-name=clinAI_13_FT_helper1 \
#     "slurm/13_tune_dl_reduced.sh" --families FTTransformer --sites Gangnam Sinchon MIMIC --trials-only

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

python3 "$BASE/scripts/13_tune_dl_reduced.py" "$@"
