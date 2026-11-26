#!/bin/bash
#SBATCH --job-name=merge_02_metaphlan3
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/merge_02_metaphlan3_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/merge_02_metaphlan3_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

set -euo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate base

python3 /home/tcmhwd/microbiome/ICU_public/merge/scripts/02_metaphlan3_merge.py
