#!/bin/bash
#SBATCH --job-name=clinAI_42_survival_v4
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/42_survival_v4_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/42_survival_v4_%j.err"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition

# V3 rerun (2026-10-02, owner-authorised correction) of
# 42_phase_x_time_to_event_competing_risk.py, repointed at the fold-local-
# corrected M3 OOF probabilities (clinical_ai/predictions/
# phase_x_m3_single_level_oof_probabilities_v3_V3.csv, from script
# 45_phase_x_c1_c2_nested_cv_corrected_v4.py). See that script's own
# docstring and
# semantic_microbiome/reports/paper1/FINAL_LEAN_SUBMISSION_2026-10-02_v3/QC/
# M3_FOLD_LOCAL_METHOD_AUDIT_v3.md for what changed and why. Resource spec
# copied verbatim from the original wrapper
# (slurm/41_42_phase_x_sofa_and_survival.sh): the Fine-Gray bootstrap
# (7 covariate sets x config.N_BOOTSTRAP=1000 Cox refits) is the only heavy
# part, parallelised over config.get_n_jobs() (honours SLURM_CPUS_PER_TASK).
# Bootstrap resamples are drawn once in the parent from a seeded RNG, so the
# result is identical regardless of worker count.
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/42_phase_x_time_to_event_competing_risk_v4.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE/scripts"

echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset} ==="
echo "=== 42_phase_x_time_to_event_competing_risk_v4.py ==="
python3 42_phase_x_time_to_event_competing_risk_v4.py

echo "42_phase_x_time_to_event_competing_risk_v4.sh complete."
