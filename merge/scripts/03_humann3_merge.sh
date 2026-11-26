#!/bin/bash
#SBATCH --job-name=merge_03_humann3
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/merge_03_humann3_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/merge_03_humann3_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

set -euo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate humann3

python3 /home/tcmhwd/microbiome/ICU_public/merge/scripts/03_humann3_merge.py --workers "${SLURM_CPUS_PER_TASK:-32}"
