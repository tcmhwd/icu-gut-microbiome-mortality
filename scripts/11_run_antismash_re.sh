#!/bin/bash
#SBATCH --job-name=antismash_parallel
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/antismash_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/antismash_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda environment
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate antismash

export BLAST_USAGE_REPORT=false

ANTISMASH_BIN="$(command -v antismash || true)"
if [[ -z "${ANTISMASH_BIN}" ]]; then
  echo "[ERROR] antismash not found after conda activate antismash" >&2
  exit 1
fi

########################
# 2. paths
########################
WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

PROKKA_DIR="${ICU_DIR}/Prokka_result"
ANTISMASH_DIR="${ICU_DIR}/antiSMASH_result"
DREP_OUT="${ICU_DIR}/cohort_drep"
MAG_LIST="${DREP_OUT}/mag_list.txt"

mkdir -p "${ANTISMASH_DIR}"

########################
# 3. parallel settings
########################
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-32}"

# 60 CPU / 60G 기준 권장값
PARALLEL_JOBS=3
CPUS_PER_JOB=$(( TOTAL_CPUS / PARALLEL_JOBS ))

if [[ "${CPUS_PER_JOB}" -lt 1 ]]; then
  CPUS_PER_JOB=1
fi

########################
# 4. scratch / status / locks
########################
if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" && -w "${TMPDIR}" ]]; then
  SCRATCH_BASE="${TMPDIR}/antismash_${SLURM_JOB_ID}"
else
  SCRATCH_BASE="/tmp/${USER}/antismash_${SLURM_JOB_ID}"
fi
mkdir -p "${SCRATCH_BASE}"

STATUS_DIR="${ANTISMASH_DIR}/.status_${SLURM_JOB_ID}"
LOCK_DIR="${ANTISMASH_DIR}/.locks"

mkdir -p "${STATUS_DIR}"
mkdir -p "${LOCK_DIR}"

DONE_LIST="${STATUS_DIR}/done.list"
SKIP_LIST="${STATUS_DIR}/skip.list"
FAIL_LIST="${STATUS_DIR}/fail.list"

: > "${DONE_LIST}"
: > "${SKIP_LIST}"
: > "${FAIL_LIST}"

echo "[INFO] ANTISMASH_BIN=${ANTISMASH_BIN}"
echo "[INFO] TOTAL_CPUS=${TOTAL_CPUS}"
echo "[INFO] PARALLEL_JOBS=${PARALLEL_JOBS}"
echo "[INFO] CPUS_PER_JOB=${CPUS_PER_JOB}"
echo "[INFO] PROKKA_DIR=${PROKKA_DIR}"
echo "[INFO] ANTISMASH_DIR=${ANTISMASH_DIR}"
echo "[INFO] MAG_LIST=${MAG_LIST}"
echo "[INFO] SCRATCH_BASE=${SCRATCH_BASE}"
echo "[INFO] STATUS_DIR=${STATUS_DIR}"
echo "[INFO] LOCK_DIR=${LOCK_DIR}"

########################
# 5. build full todo list
########################
# 항상 전체 MAG 기준으로 todo list 생성
TODO_LIST="${ANTISMASH_DIR}/antismash_todo_all.list"
: > "${TODO_LIST}"

if [[ -s "${MAG_LIST}" ]]; then
  while IFS= read -r fa; do
    [[ -z "${fa}" ]] && continue
    mag_name="$(basename "${fa}" .fa)"
    fna="${PROKKA_DIR}/${mag_name}/${mag_name}.fna"
    if [[ -s "${fna}" ]]; then
      echo "${fna}" >> "${TODO_LIST}"
    else
      echo "[WARN] Missing Prokka .fna for ${mag_name}: ${fna}" >&2
    fi
  done < "${MAG_LIST}"
else
  echo "[WARN] MAG_LIST not found; scanning all Prokka .fna files."
  find "${PROKKA_DIR}" -mindepth 2 -maxdepth 2 -type f -name "*.fna" | sort >> "${TODO_LIST}"
fi

sort -u "${TODO_LIST}" -o "${TODO_LIST}"

if [[ ! -s "${TODO_LIST}" ]]; then
  echo "[ERROR] No input .fna files found for antiSMASH." >&2
  exit 1
fi

TOTAL_ALL=$(wc -l < "${TODO_LIST}")
echo "[INFO] total MAGs in todo list = ${TOTAL_ALL}"

########################
# 6. snapshot of completed/remaining
########################
COMPLETED_SNAPSHOT=0

while IFS= read -r fasta_file; do
  [[ -z "${fasta_file}" ]] && continue
  bin_name="$(basename "${fasta_file}" .fna)"
  out_dir="${ANTISMASH_DIR}/${bin_name}"

  if [[ -s "${out_dir}/index.html" || -s "${out_dir}/antismash.results.json" || -s "${out_dir}/regions.js" ]]; then
    COMPLETED_SNAPSHOT=$((COMPLETED_SNAPSHOT + 1))
  fi
done < "${TODO_LIST}"

REMAINING_SNAPSHOT=$((TOTAL_ALL - COMPLETED_SNAPSHOT))

echo "[INFO] already completed = ${COMPLETED_SNAPSHOT}"
echo "[INFO] remaining snapshot = ${REMAINING_SNAPSHOT}"

if [[ "${REMAINING_SNAPSHOT}" -eq 0 ]]; then
  echo "[DONE] Nothing left to run."
  exit 0
fi

########################
# 7. worker function
########################
run_one() {
  local fasta_file="$1"
  [[ -z "${fasta_file}" ]] && return 0

  local bin_name
  local out_dir
  local tmp_dir
  local lock_path

  bin_name="$(basename "${fasta_file}" .fna)"
  out_dir="${ANTISMASH_DIR}/${bin_name}"
  tmp_dir="${SCRATCH_BASE}/${bin_name}"
  lock_path="${LOCK_DIR}/${bin_name}.lock"

  if [[ ! -s "${fasta_file}" ]]; then
    echo "[WARN] Missing input fasta: ${fasta_file}" >&2
    echo "${bin_name}" >> "${FAIL_LIST}"
    return 0
  fi

  # 완료된 샘플은 자동 skip
  if [[ -s "${out_dir}/index.html" || -s "${out_dir}/antismash.results.json" || -s "${out_dir}/regions.js" ]]; then
    echo "[SKIP] ${bin_name} already completed"
    echo "${bin_name}" >> "${SKIP_LIST}"
    return 0
  fi

  # 중복 실행 방지
  if ! mkdir "${lock_path}" 2>/dev/null; then
    echo "[SKIP] ${bin_name} locked"
    echo "${bin_name}" >> "${SKIP_LIST}"
    return 0
  fi

  mkdir -p "${tmp_dir}"

  echo "[RUN ] ${bin_name}"
  echo "       input : ${fasta_file}"
  echo "       out   : ${out_dir}"
  echo "       cpus  : ${CPUS_PER_JOB}"

  if TMPDIR="${tmp_dir}" "${ANTISMASH_BIN}" \
      --cpus "${CPUS_PER_JOB}" \
      --genefinding-tool prodigal \
      --output-dir "${out_dir}" \
      "${fasta_file}"
  then
    echo "[DONE] ${bin_name}"
    echo "${bin_name}" >> "${DONE_LIST}"
  else
    echo "[FAIL] ${bin_name}" >&2
    echo "${bin_name}" >> "${FAIL_LIST}"
  fi

  rm -rf "${tmp_dir}" || true
  rm -rf "${lock_path}" || true
}

########################
# 8. parallel execution
########################
FAIL=0

while IFS= read -r fasta_file; do
  [[ -z "${fasta_file}" ]] && continue

  run_one "${fasta_file}" &

  while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
    sleep 5
  done
done < "${TODO_LIST}"

########################
# 9. wait for all jobs
########################
for job in $(jobs -rp); do
  if ! wait "${job}"; then
    FAIL=1
  fi
done

########################
# 10. summary
########################
count_lines() {
  local f="$1"
  if [[ -s "${f}" ]]; then
    wc -l < "${f}"
  else
    echo 0
  fi
}

DONE_COUNT=$(count_lines "${DONE_LIST}")
SKIP_COUNT=$(count_lines "${SKIP_LIST}")
FAIL_COUNT=$(count_lines "${FAIL_LIST}")

echo "[SUMMARY] done=${DONE_COUNT}, skipped=${SKIP_COUNT}, failed=${FAIL_COUNT}"

########################
# 11. result summary table
########################
if command -v jq >/dev/null 2>&1; then
  summary_tsv="${ANTISMASH_DIR}/antismash_summary.tsv"
  echo -e "sample\tbgc_count\tbgc_types" > "${summary_tsv}"

  find "${ANTISMASH_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name ".status_*" ! -name ".locks" | sort | while read -r d; do
    sample="$(basename "${d}")"
    json="${d}/antismash.results.json"

    if [[ -s "${json}" ]]; then
      count=$(jq '[.records[].regions[]] | length' "${json}" 2>/dev/null || echo 0)
      types=$(jq -r '[.records[].regions[].type] | flatten | unique | join(",")' "${json}" 2>/dev/null || echo "")
      echo -e "${sample}\t${count}\t${types}" >> "${summary_tsv}"
    else
      if ls "${d}"/region*.gbk >/dev/null 2>&1; then
        count=$(ls "${d}"/region*.gbk | wc -l | awk '{print $1}')
      else
        count=0
      fi
      echo -e "${sample}\t${count}\t" >> "${summary_tsv}"
    fi
  done

  echo "[INFO] summary written: ${summary_tsv}"
else
  echo "[INFO] jq not found; skipping summary aggregation."
fi

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[ERROR] One or more antiSMASH jobs failed."
  exit 1
fi

echo "[DONE] Cohort-wide antiSMASH parallel analysis completed."
