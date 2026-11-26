#!/bin/bash
#SBATCH --job-name=clinAI_03_tune_ml
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/03_tune_ml_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/03_tune_ml_%j.err"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

# Step 3/5 -- Optuna hyperparameter tuning of the ML candidates selected by
# 02_ml_screening.sh (100 trials/model/direction by default; 2 directions x
# up to 3 models x up to 2 pipeline variants for boosting models). CPU-only.
# Requires slurm/01_preprocess.sh (and normally slurm/02_ml_screening.sh, for
# clinical_ai/ml_screening/selected_ml_models.json) to have completed.
#
# Studies are resumable: re-submitting this script continues any Optuna
# study that has not yet reached --n-trials (sqlite storage under
# clinical_ai/optuna/ML/), so a walltime-limited job can be safely re-run.
#
# Usage: sbatch "slurm/03_tune_ml.sh"
# Common overrides:
#   sbatch "slurm/03_tune_ml.sh" --models lr catboost et
#   sbatch "slurm/03_tune_ml.sh" --n-trials 50
#   sbatch "slurm/03_tune_ml.sh" --skip-native-variant

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u   # the 'gpt' env's conda activate.d hooks (MKL) are not `set -u`-safe
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

python3 "$BASE/scripts/04_tune_ml.py" "$@"
