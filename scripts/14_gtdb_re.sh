#!/bin/bash
#SBATCH --job-name=gtdbtk_only
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/gtdbtk_only_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/gtdbtk_only_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate gtdb

WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

MAG_DIR="${ICU_DIR}/cohort_drep/dereplicated_genomes"
GTDBTK_OUT="${ICU_DIR}/gtdbtk_result"

# 실제 확인된 DB 경로
GTDBTK_DB="/data/tcmhwd/miniconda3/envs/gtdbtk/share/gtdbtk-2.5.2/db"

mkdir -p "${GTDBTK_OUT}"

if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" && -w "${TMPDIR}" ]]; then
  SCRATCH_DIR="${TMPDIR}/gtdbtk_${SLURM_JOB_ID}"
else
  SCRATCH_DIR="/tmp/${USER}/gtdbtk_${SLURM_JOB_ID}"
fi
mkdir -p "${SCRATCH_DIR}"

cleanup() {
  rm -rf "${SCRATCH_DIR}" || true
}
trap cleanup EXIT

export GTDBTK_DATA_PATH="${GTDBTK_DB}"

if [[ ! -d "${MAG_DIR}" ]]; then
  echo "[ERROR] MAG directory not found: ${MAG_DIR}"
  exit 1
fi

MAG_COUNT=$(find "${MAG_DIR}" -maxdepth 1 -type f -name "*.fa" | wc -l)
if [[ "${MAG_COUNT}" -eq 0 ]]; then
  echo "[ERROR] No .fa genomes found in ${MAG_DIR}"
  exit 1
fi

if [[ ! -d "${GTDBTK_DB}" ]]; then
  echo "[ERROR] GTDB-Tk database directory not found: ${GTDBTK_DB}"
  exit 1
fi

echo "[INFO] MAG_DIR=${MAG_DIR}"
echo "[INFO] GTDBTK_OUT=${GTDBTK_OUT}"
echo "[INFO] GTDBTK_DATA_PATH=${GTDBTK_DATA_PATH}"
echo "[INFO] SCRATCH_DIR=${SCRATCH_DIR}"
echo "[INFO] CPUS=${SLURM_CPUS_PER_TASK}"
echo "[INFO] MAG_COUNT=${MAG_COUNT}"

echo "[INFO] GTDB-Tk classify_wf 시작"

gtdbtk classify_wf \
    --genome_dir "${MAG_DIR}" \
    --out_dir "${GTDBTK_OUT}" \
    --scratch_dir "${SCRATCH_DIR}" \
    --skip_ani_screen \
    --cpus "${SLURM_CPUS_PER_TASK}" \
    -x fa

echo "[DONE] GTDB-Tk 분류 완료: ${GTDBTK_OUT}"
