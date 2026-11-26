#!/bin/bash
#SBATCH --job-name=clinAI_01_preprocess
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/01_preprocess_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/01_preprocess_%j.err"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

# Step 1/5 -- dataset inspection/harmonization/landmark cohort/QC (01) +
# missingness reports A-D (02). Pure pandas, fast (~1 min for both scripts
# on this data). No downstream step can run before this one -- it produces
# clinical_ai/preprocessing/model_ready_{ICU,MIMIC}.parquet, which every
# later script reads.
#
# Usage: sbatch "slurm/01_preprocess.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u   # the 'gpt' env's conda activate.d hooks (MKL) are not `set -u`-safe
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

echo "=== 01_inspect_and_preprocess.py ==="
python3 "$BASE/scripts/01_inspect_and_preprocess.py"

echo "=== 02_missingness_analysis.py ==="
python3 "$BASE/scripts/02_missingness_analysis.py"

echo "=== done ==="
