#!/bin/bash
#SBATCH --job-name=rgi_analysis
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/rgi_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/rgi_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate rgi

WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

PROKKA_DIR="${ICU_DIR}/Prokka_result"
RGI_OUT_DIR="${ICU_DIR}/MAG_rgi_results"
DREP_OUT="${ICU_DIR}/cohort_drep"
MAG_LIST="${DREP_OUT}/mag_list.txt"

CARD_DIR="${WD}/ref"
CARD_JSON="${CARD_DIR}/card.json"

THREADS="${SLURM_CPUS_PER_TASK:-16}"
ALIGN="DIAMOND"

mkdir -p "${RGI_OUT_DIR}"

echo "[RGI] threads=${THREADS}"
echo "[RGI] aligner=${ALIGN}"
echo "[RGI] PROKKA_DIR=${PROKKA_DIR}"
echo "[RGI] RGI_OUT_DIR=${RGI_OUT_DIR}"
echo "[RGI] CARD_DIR=${CARD_DIR}"
echo "[RGI] CARD_JSON=${CARD_JSON}"

if [[ ! -s "${CARD_JSON}" ]]; then
  echo "[ERROR] CARD JSON not found: ${CARD_JSON}" >&2
  exit 1
fi

cd "${CARD_DIR}"

if [[ ! -s "${CARD_DIR}/localDB/card.json" ]]; then
  echo "[INFO] localDB not found. Loading CARD into ${CARD_DIR}/localDB ..."
  rgi clean --local || true
  rgi load --card_json "${CARD_JSON}" --local
else
  echo "[INFO] Existing localDB found: ${CARD_DIR}/localDB/card.json"
fi

rgi database --version --local

RUN_LIST="${RGI_OUT_DIR}/rgi_todo.list"
: > "${RUN_LIST}"

if [[ -s "${MAG_LIST}" ]]; then
  while IFS= read -r fa; do
    [[ -z "${fa}" ]] && continue
    mag_name="$(basename "${fa}" .fa)"
    faa="${PROKKA_DIR}/${mag_name}/${mag_name}.faa"
    if [[ -s "${faa}" ]]; then
      echo "${faa}" >> "${RUN_LIST}"
    else
      echo "[WARN] Missing Prokka FAA for ${mag_name}: ${faa}" >&2
    fi
  done < "${MAG_LIST}"
else
  echo "[WARN] MAG_LIST not found. Scanning all Prokka .faa files."
  find "${PROKKA_DIR}" -mindepth 2 -maxdepth 2 -type f -name "*.faa" | sort > "${RUN_LIST}"
fi

if [[ ! -s "${RUN_LIST}" ]]; then
  echo "[ERROR] No .faa files found for RGI." >&2
  exit 1
fi

TOTAL=$(wc -l < "${RUN_LIST}")
echo "[RGI] total targets=${TOTAL}"

DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS= read -r FAA; do
  [[ -z "${FAA}" ]] && continue

  SAMPLE="$(basename "${FAA}" .faa)"
  OUT_PREFIX="${RGI_OUT_DIR}/${SAMPLE}_rgi"

  if [[ ! -s "${FAA}" ]]; then
    echo "[WARN] Missing FAA: ${FAA}" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  if [[ -s "${OUT_PREFIX}.txt" || -s "${OUT_PREFIX}.json" ]]; then
    echo "[SKIP] ${SAMPLE} (already exists)"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "[RUN ] ${SAMPLE}"
  echo "       input : ${FAA}"
  echo "       out   : ${OUT_PREFIX}"

  cd "${CARD_DIR}"

  if rgi main \
    --input_sequence "${FAA}" \
    --output_file "${OUT_PREFIX}" \
    --input_type protein \
    --clean \
    --num_threads "${THREADS}" \
    --alignment_tool "${ALIGN}" \
    --local
  then
    echo "[DONE] ${SAMPLE}"
    DONE_COUNT=$((DONE_COUNT + 1))
  else
    echo "[FAIL] ${SAMPLE}" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

done < "${RUN_LIST}"

echo "[SUMMARY] done=${DONE_COUNT}, skipped=${SKIP_COUNT}, failed=${FAIL_COUNT}"
echo "[INFO] MAG protein-based RGI analysis finished."
