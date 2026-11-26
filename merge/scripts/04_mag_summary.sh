#!/bin/bash
#SBATCH --job-name=merge_04_mag_summary
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/merge_04_mag_summary_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/merge_04_mag_summary_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=128G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

set -euo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate base

python3 /home/tcmhwd/microbiome/ICU_public/merge/scripts/04_mag_summary.py --workers "${SLURM_CPUS_PER_TASK:-60}"
