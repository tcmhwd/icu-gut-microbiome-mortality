#!/bin/bash
#SBATCH --job-name=clinAI_45v3_agg
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v3_aggregate_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v3_aggregate_%j.err"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_partition

# Aggregation step for 45 v3: anchors vs published outputs, script-46 paired
# bootstrap (2000), script-56 AUROC bootstrap (1000), drop-in OOF files, report.
# Submit with --dependency=afterok:<array job id>.
set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u
cd "/home/tcmhwd/microbiome/ICU_public/clinical metadata/scripts"
python3 -u 45_phase_x_c1_c2_nested_cv_corrected_v3.py --aggregate
echo "aggregate complete."
