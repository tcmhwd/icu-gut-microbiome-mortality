#!/usr/bin/env bash
#SBATCH --job-name=ICU_public_download
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/download_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/download_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"

if [[ -z "${LIST:-}" ]]; then
  echo "[ERROR] LIST is not set."
  exit 1
fi

if [[ -z "${OUTDIR:-}" ]]; then
  echo "[ERROR] OUTDIR is not set."
  exit 1
fi

mkdir -p "${OUTDIR}" "${BASE}/logs"

index="${SLURM_ARRAY_TASK_ID:-1}"
target_url="$(sed -n "${index}p" "${LIST}" | tr -d '\r')"

if [[ -z "${target_url}" ]]; then
  echo "[ERROR] Empty URL at line ${index} in ${LIST}"
  exit 1
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[INFO] Array task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "[INFO] URL list: ${LIST}"
echo "[INFO] Output directory: ${OUTDIR}"
echo "[INFO] Target URL: ${target_url}"

cd "${OUTDIR}"

wget \
  -c \
  --tries=20 \
  --waitretry=30 \
  --timeout=120 \
  --retry-connrefused \
  --content-disposition \
  "${target_url}"

echo "[OK] Finished: ${target_url}"
echo "[INFO] Date: $(date)"
