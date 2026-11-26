#!/bin/bash
#SBATCH --job-name=clinAI_05_evaluate
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/05_evaluate_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/05_evaluate_%j.err"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

# Step 5/5 -- conventional severity-score comparison, bidirectional
# transportability summary, ML-vs-DL summary, figures, reproducibility
# audit, and the final Stage-1 summary report. CPU-only, fast. Can be run
# after slurm/03_tune_ml.sh and/or slurm/04_tune_dl.sh have produced at
# least some rows in clinical_ai/performance/model_performance_comparison.csv
# (it runs even with partial results and notes what's missing).
#
# Usage: sbatch "slurm/05_evaluate.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u   # the 'gpt' env's conda activate.d hooks (MKL) are not `set -u`-safe
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

python3 "$BASE/scripts/06_evaluate_models.py"

echo ""
echo "=== Stage 1 summary ==="
cat "$BASE/clinical_ai/reports/clinical_AI_stage1_summary.md"
