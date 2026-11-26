#!/bin/bash
#SBATCH --job-name=prokka_batch
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/prokka_batch_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/prokka_batch_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda 환경 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate prokka_env

########################
# 2. 경로/환경 설정
########################
WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

DREP_OUT="${ICU_DIR}/cohort_drep"
DEREP_DIR="${DREP_OUT}/dereplicated_genomes"
MAG_LIST="${DREP_OUT}/mag_list.txt"
PROKKA_OUT="${ICU_DIR}/Prokka_result"

TOTAL_THREADS="${SLURM_CPUS_PER_TASK:-64}"
THREADS_PER_JOB=8
MAX_JOBS=$(( TOTAL_THREADS / THREADS_PER_JOB ))

mkdir -p "${PROKKA_OUT}"

echo "[INFO] CONDA_PREFIX      = ${CONDA_PREFIX}"
echo "[INFO] DREP_OUT          = ${DREP_OUT}"
echo "[INFO] DEREP_DIR         = ${DEREP_DIR}"
echo "[INFO] MAG_LIST          = ${MAG_LIST}"
echo "[INFO] PROKKA_OUT        = ${PROKKA_OUT}"
echo "[INFO] TOTAL_THREADS     = ${TOTAL_THREADS}"
echo "[INFO] THREADS_PER_JOB   = ${THREADS_PER_JOB}"
echo "[INFO] MAX_JOBS          = ${MAX_JOBS}"

########################
# 3. MAG 리스트 확인
########################
if [[ ! -s "${MAG_LIST}" ]]; then
  compgen -G "${DEREP_DIR}/*.fa" > /dev/null || {
    echo "[ERROR] No dereplicated MAGs found in ${DEREP_DIR}"
    exit 1
  }
  ls "${DEREP_DIR}"/*.fa | sort > "${MAG_LIST}"
  echo "[WARN] ${MAG_LIST} was missing and has been created automatically."
fi

MAG_COUNT=$(wc -l < "${MAG_LIST}")
if [[ "${MAG_COUNT}" -eq 0 ]]; then
  echo "[ERROR] MAG_LIST is empty: ${MAG_LIST}"
  exit 1
fi

echo "[INFO] Number of MAGs to annotate: ${MAG_COUNT}"

########################
# 4. Prokka 실행 함수
########################
run_prokka() {
  local bin="$1"
  local bin_name
  local outdir

  bin_name="$(basename "${bin}" .fa)"
  outdir="${PROKKA_OUT}/${bin_name}"

  mkdir -p "${outdir}"

  # 이미 완료된 경우 건너뛰기
  if [[ -s "${outdir}/${bin_name}.gff" && -s "${outdir}/${bin_name}.faa" ]]; then
    echo "[SKIP] Prokka already completed for ${bin_name}"
    return 0
  fi

  echo "[START] Prokka for ${bin_name}"

  prokka \
    --outdir "${outdir}" \
    --prefix "${bin_name}" \
    --cpus "${THREADS_PER_JOB}" \
    --force \
    --compliant \
    --centre X \
    --metagenome \
    "${bin}"

  echo "[DONE] Prokka finished for ${bin_name}"
}

export -f run_prokka
export PROKKA_OUT THREADS_PER_JOB

########################
# 5. 병렬 실행 (GNU parallel)
########################
grep -v '^$' "${MAG_LIST}" | \
  parallel --jobs "${MAX_JOBS}" --halt soon,fail=1 --line-buffer \
    'if [[ ! -f "{}" ]]; then echo "[WARN] Missing MAG file: {}"; exit 0; fi; run_prokka "{}"'

echo "[DONE] All Prokka annotations completed successfully."
