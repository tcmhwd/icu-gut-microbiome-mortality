#!/bin/bash
#SBATCH --job-name=merge_04a_resistome
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/merge_04a_resistome_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/merge_04a_resistome_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

set -euo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate base

python3 /home/tcmhwd/microbiome/ICU_public/merge/scripts/04a_resistome_adapter.py
