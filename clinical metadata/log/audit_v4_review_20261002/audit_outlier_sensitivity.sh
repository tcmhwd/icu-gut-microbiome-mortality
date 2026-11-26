#!/bin/bash
#SBATCH --job-name=audit_v4_outlier
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/audit_v4_review_20261002/audit_outlier_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/audit_v4_review_20261002/audit_outlier_%j.err"
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_partition
# Review-only sensitivity check (no project files written). 2 CPUs / 8 GB:
# ~30 small Logit/PHReg fits on 302 rows plus reading 11 filtered TSVs.
set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u; conda activate gpt; set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/audit_v4_review_20261002"
python3 audit_outlier_sensitivity.py
echo "audit_outlier_sensitivity done"
