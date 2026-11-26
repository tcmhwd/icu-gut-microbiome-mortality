#!/bin/bash
#SBATCH --job-name=clinAI_62_vanAvanB
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/62_vana_vanb_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/62_vana_vanb_%j.err"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_partition
# PR-2 (review-triggered, pre-registered). Python from conda base; Firth (logistf) from the downstream_r env's R,
# because the compute nodes' system R is older than the login node's.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RSCRIPT_BIN=/home/tcmhwd/.local/share/mamba/envs/downstream_r/bin/Rscript
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate base
cd "/home/tcmhwd/microbiome/ICU_public/clinical metadata/scripts"
python3 -u 62_phase_x_vre_vana_vanb_split.py
