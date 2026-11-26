#!/bin/bash
#SBATCH --job-name=clinAI_45v4_m1fix_strictlr
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v4_%A_%a.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/45_v4_%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition
#SBATCH --array=0-6

# M1 convergence fix (REF now on the same 30000->100000 ladder as M3) +
# V4_STRICT (fully tuning-local prevalence filter + PCA for the data-driven
# LR model), owner-authorised 2026-10-02 "final technical closure pass".
# One array task per variant of 45_phase_x_c1_c2_nested_cv_corrected_v4.py.
# Same resource profile as the v3 wrapper (REF's workload is light -- 16
# static features, no PCA -- but the full ladder may still need headroom;
# V4_STRICT's GridSearchCV(72 CV paths) is comparable to V2S's).
# Aggregation: 45_phase_x_c1_c2_nested_cv_corrected_v4_aggregate.sh (afterok).

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

VARIANTS=(REF V0 V1 V2 V3 V2S V4_STRICT)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
cd "$BASE/scripts"
echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}; variant: $V ==="
python3 -u 45_phase_x_c1_c2_nested_cv_corrected_v4.py --variant "$V"
echo "variant $V complete."
