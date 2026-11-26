#!/bin/bash
#SBATCH --job-name=eggnog_batch
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/eggnog_batch_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/eggnog_batch_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=80G
#SBATCH --time=999:00:00
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda 환경 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate eggnog

########################
# 2. 경로 설정
########################
WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"
REF_DIR="${WD}/ref/eggnog_db"

prokka_dir="${ICU_DIR}/Prokka_result"
eggnog_dir="${ICU_DIR}/EggNOG_result"
todo_list="${ICU_DIR}/eggnog_todo.list"

mkdir -p "${eggnog_dir}"

########################
# 3. DB 설정
########################
find_db() {
  local name="$1"
  find "${REF_DIR}" -type f -name "${name}" | sort | head -n 1
}

dmnd_db="$(find_db "eggnog_proteins.dmnd")"
eggnog_db="$(find_db "eggnog.db")"
taxa_db="$(find_db "eggnog.taxa.db")"

[[ -n "${dmnd_db}" && -s "${dmnd_db}" ]] || { echo "[FATAL] eggnog_proteins.dmnd not found under ${REF_DIR}" >&2; exit 2; }
[[ -n "${eggnog_db}" && -s "${eggnog_db}" ]] || { echo "[FATAL] eggnog.db not found under ${REF_DIR}" >&2; exit 2; }
[[ -n "${taxa_db}" && -s "${taxa_db}" ]] || { echo "[FATAL] eggnog.taxa.db not found under ${REF_DIR}" >&2; exit 2; }

export EGGNOG_DATA_DIR="${REF_DIR}"

########################
# 4. 자원 설정
########################
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-64}"

if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" != "0" ]]; then
  TOTAL_MEM_MB="${SLURM_MEM_PER_NODE}"
else
  TOTAL_MEM_MB=$((128 * 1024))
fi

# 샘플당 자원
CPU_PER_JOB=16
MEM_PER_JOB_MB=$((10 * 1024))

MAX_JOBS_CPU=$(( TOTAL_CPUS / CPU_PER_JOB ))
MAX_JOBS_MEM=$(( TOTAL_MEM_MB / MEM_PER_JOB_MB ))

(( MAX_JOBS_CPU < 1 )) && MAX_JOBS_CPU=1
(( MAX_JOBS_MEM < 1 )) && MAX_JOBS_MEM=1

if (( MAX_JOBS_CPU < MAX_JOBS_MEM )); then
  MAX_JOBS="${MAX_JOBS_CPU}"
else
  MAX_JOBS="${MAX_JOBS_MEM}"
fi

(( MAX_JOBS < 1 )) && MAX_JOBS=1

########################
# 5. 안정성 설정
########################
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LC_ALL=C

if [[ -d /tmp && -w /tmp ]]; then
  TMPROOT="/tmp/${USER}/eggnog_${SLURM_JOB_ID:-manual}"
else
  TMPROOT="${ICU_DIR}/tmp/eggnog_${SLURM_JOB_ID:-manual}"
fi
mkdir -p "${TMPROOT}"

cleanup() {
  rm -rf "${TMPROOT}"
}
trap cleanup EXIT

########################
# 6. 완료 여부 확인
########################
is_done() {
  local s="$1"
  local ann_txt="${eggnog_dir}/${s}_eggnog.emapper.annotations"
  local ann_gz="${ann_txt}.gz"

  if [[ -s "${ann_txt}" ]] && grep -m1 -qv '^#' "${ann_txt}"; then
    return 0
  fi
  if [[ -s "${ann_gz}" ]] && zgrep -m1 -qv '^#' "${ann_gz}"; then
    return 0
  fi
  return 1
}

########################
# 7. todo_list 자동 생성
########################
prepare_todo_list() {
  local total=0
  local done_count=0
  local pending=0
  local faa sample parent

  : > "${todo_list}"

  while IFS= read -r faa; do
    ((total+=1))
    sample="$(basename "${faa}" .faa)"
    parent="$(basename "$(dirname "${faa}")")"

    if [[ "${sample}" != "${parent}" ]]; then
      echo "[WARN] sample directory and faa prefix mismatch: ${faa}"
      continue
    fi

    if is_done "${sample}"; then
      ((done_count+=1))
      continue
    fi

    echo "${sample}" >> "${todo_list}"
    ((pending+=1))
  done < <(find "${prokka_dir}" -mindepth 2 -maxdepth 2 -type f -name "*.faa" | sort)

  echo "[INFO] Total FAA found    : ${total}"
  echo "[INFO] Already completed : ${done_count}"
  echo "[INFO] Pending samples   : ${pending}"
  echo "[INFO] todo_list         : ${todo_list}"

  if (( total == 0 )); then
    echo "[FATAL] No .faa files found under ${prokka_dir}" >&2
    exit 2
  fi

  if (( pending == 0 )); then
    echo "[DONE] Nothing to do. All samples already completed."
    exit 0
  fi
}

########################
# 8. eggNOG 단일 샘플 실행
########################
run_emapper() {
  local sample="$1"
  local faa="${prokka_dir}/${sample}/${sample}.faa"
  local out_prefix="${sample}_eggnog"
  local tmp_sample_root="${TMPROOT}/${sample}"
  local tmp_try
  local try
  local ok=0

  if [[ ! -s "${faa}" ]]; then
    echo "[FAIL] ${sample}: FAA not found (${faa})" >&2
    return 1
  fi

  if is_done "${sample}"; then
    echo "[SKIP] ${sample} already done"
    return 0
  fi

  mkdir -p "${tmp_sample_root}"

  for try in 1 2 3; do
    tmp_try="${tmp_sample_root}/try_${try}"
    rm -rf "${tmp_try}"
    mkdir -p "${tmp_try}"

    echo "[RUN] ${sample} try ${try}/3 cpu=${CPU_PER_JOB}"

    emapper.py \
      -i "${faa}" \
      -o "${out_prefix}" \
      --output_dir "${eggnog_dir}" \
      --data_dir "${REF_DIR}" \
      --cpu "${CPU_PER_JOB}" \
      --itype proteins \
      -m diamond \
      --dmnd_db "${dmnd_db}" \
      --dmnd_iterate no \
      --temp_dir "${tmp_try}" \
      --override

    if is_done "${sample}"; then
      echo "[OK] ${sample}"
      ok=1
      rm -rf "${tmp_sample_root}" || true
      break
    fi

    echo "[WARN] ${sample} not confirmed; retrying..."
    sleep $((15 * try))
  done

  if (( ok == 0 )); then
    echo "[FAIL] ${sample} after retries" >&2
    return 1
  fi

  return 0
}

########################
# 9. 정보 출력
########################
echo "[INFO] CONDA_PREFIX   = ${CONDA_PREFIX}"
echo "[INFO] REF_DIR        = ${REF_DIR}"
echo "[INFO] prokka_dir     = ${prokka_dir}"
echo "[INFO] eggnog_dir     = ${eggnog_dir}"
echo "[INFO] dmnd_db        = ${dmnd_db}"
echo "[INFO] eggnog_db      = ${eggnog_db}"
echo "[INFO] taxa_db        = ${taxa_db}"
echo "[INFO] TOTAL_CPUS     = ${TOTAL_CPUS}"
echo "[INFO] TOTAL_MEM_MB   = ${TOTAL_MEM_MB}"
echo "[INFO] CPU_PER_JOB    = ${CPU_PER_JOB}"
echo "[INFO] MEM_PER_JOB_MB = ${MEM_PER_JOB_MB}"
echo "[INFO] MAX_JOBS       = ${MAX_JOBS}"
echo "[INFO] TMPROOT        = ${TMPROOT}"

########################
# 10. todo_list 생성
########################
prepare_todo_list

########################
# 11. 병렬 실행 (GNU parallel)
########################
export -f run_emapper is_done
export prokka_dir eggnog_dir REF_DIR dmnd_db CPU_PER_JOB TMPROOT

grep -v '^$' "${todo_list}" | \
  parallel --jobs "${MAX_JOBS}" --halt soon,fail=1 --line-buffer run_emapper {}

echo "[DONE] All eggNOG annotations completed successfully."
