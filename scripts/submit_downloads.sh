#!/usr/bin/env bash
set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"
SCRIPT="${BASE}/scripts/download_ICU_public_array.sh"

NON_AGI_LIST="${BASE}/non_AGI/non_AGI_fastq_urls.txt"
AGI_LIST="${BASE}/AGI/AGI_fastq_urls_https.txt"

NON_AGI_OUT="${BASE}/non_AGI/fastq"
AGI_OUT="${BASE}/AGI/fastq"

MAX_JOBS="${1:-10}"

N_NON_AGI=$(wc -l < "${NON_AGI_LIST}" | tr -d ' ')
N_AGI=$(wc -l < "${AGI_LIST}" | tr -d ' ')

echo "[INFO] non-AGI URLs: ${N_NON_AGI}"
echo "[INFO] AGI URLs: ${N_AGI}"
echo "[INFO] Max concurrent downloads per array: ${MAX_JOBS}"

if [[ "${N_NON_AGI}" -gt 0 ]]; then
  sbatch \
    --array=1-${N_NON_AGI}%${MAX_JOBS} \
    --export=ALL,LIST=${NON_AGI_LIST},OUTDIR=${NON_AGI_OUT} \
    "${SCRIPT}"
fi

if [[ "${N_AGI}" -gt 0 ]]; then
  sbatch \
    --array=1-${N_AGI}%${MAX_JOBS} \
    --export=ALL,LIST=${AGI_LIST},OUTDIR=${AGI_OUT} \
    "${SCRIPT}"
fi
