#!/bin/bash
#SBATCH --job-name=coverm_batch
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/coverm_batch_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/coverm_batch_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu_partition
#SBATCH --nodelist=node1
# Submit with: sbatch --array=1-9 15c_coverm_batch.sh

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate cover

ICU_DIR="/home/tcmhwd/microbiome/ICU_public"
BAM_DIR="${ICU_DIR}/coverm_bam"
OUT_DIR="${ICU_DIR}/coverm_result/batches"
BATCH_SIZE=100

mkdir -p "${OUT_DIR}"

# 전체 BAM 파일 목록 (정렬된 순서)
mapfile -t ALL_BAMS < <(ls "${BAM_DIR}"/*.bam | sort)
TOTAL=${#ALL_BAMS[@]}

START=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE ))
END=$(( START + BATCH_SIZE - 1 ))
[[ ${END} -ge ${TOTAL} ]] && END=$(( TOTAL - 1 ))

BATCH_BAMS=("${ALL_BAMS[@]:${START}:${BATCH_SIZE}}")
COUNT=${#BATCH_BAMS[@]}

echo "[INFO] Batch ${SLURM_ARRAY_TASK_ID}: samples ${START}-${END} (${COUNT} BAMs)"

OUT_TSV="${OUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}.coverm.tsv"

if [[ -s "${OUT_TSV}" ]]; then
  echo "[SKIP] Already exists: ${OUT_TSV}"
  exit 0
fi

coverm genome \
  --bam-files "${BATCH_BAMS[@]}" \
  --separator "~" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --methods relative_abundance mean covered_bases \
  --output-format dense \
  --output-file "${OUT_TSV}"

echo "[DONE] Batch ${SLURM_ARRAY_TASK_ID} → ${OUT_TSV}"
