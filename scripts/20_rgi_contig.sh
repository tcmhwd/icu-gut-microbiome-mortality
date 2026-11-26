#!/bin/bash
#SBATCH --job-name=rgi_contig
#SBATCH --output=/home/tcmhwd/microbiome/log/rgi_contig_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/log/rgi_contig_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition


set -euo pipefail

export PATH="/data/tcmhwd/miniconda3/envs/rgi/bin:$PATH"

# ── 경로 설정 ──────────────────────────────────────────────────
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"

SAMPLE_LIST="${WD}/${project_id}/sample_list.txt"
PROKKA_DIR="${WD}/${project_id}/Prokka_contig_result"
OUT_DIR="${WD}/${project_id}/RGI_contig_result"
CARD_DIR="${WD}/ref"
CARD_JSON="${CARD_DIR}/card.json"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
ALIGN="DIAMOND"

mkdir -p "${OUT_DIR}"

# ── 샘플명 추출 ────────────────────────────────────────────────
sample=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}" | tr -d '\r')

if [[ -z "${sample}" ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
    exit 1
fi

FAA="${PROKKA_DIR}/${sample}/${sample}.faa"
SAMPLE_OUTDIR="${OUT_DIR}/${sample}"
OUT_PREFIX="${SAMPLE_OUTDIR}/${sample}_rgi"

echo "[INFO] sample=${sample}"
echo "[INFO] faa=${FAA}"
echo "[INFO] out_prefix=${OUT_PREFIX}"
echo "[INFO] threads=${THREADS}"

# ── 입력 파일 확인 ─────────────────────────────────────────────
if [[ ! -s "${CARD_JSON}" ]]; then
    echo "[ERROR] CARD JSON 없음: ${CARD_JSON}"
    exit 1
fi

if [[ ! -s "${FAA}" ]]; then
    echo "[ERROR] Prokka .faa 없음: ${FAA}"
    echo "[ERROR] 19_prokka_contig.sh 결과를 먼저 확인하세요."
    exit 1
fi

# ── 완료 여부 확인 ─────────────────────────────────────────────
if [[ -s "${OUT_PREFIX}.txt" ]]; then
    echo "[SKIP] RGI 이미 완료됨: ${sample}"
    exit 0
fi

mkdir -p "${SAMPLE_OUTDIR}"

# ── CARD localDB 초기화 (동시 실행 대비 lock 사용) ─────────────
cd "${CARD_DIR}"

LOCK_FILE="${CARD_DIR}/.rgi_card_load.lock"

init_card_localdb() {
    echo "[INFO] CARD localDB 초기화 중..."
    rgi clean --local || true
    rgi load --card_json "${CARD_JSON}" --local
}

# localDB 존재 여부를 폭넓게 확인
if [[ ! -s "${CARD_DIR}/localDB/card.json" ]]; then
    echo "[INFO] localDB/card.json 없음. local DB 준비 확인 시작."

    if command -v flock >/dev/null 2>&1; then
        exec 200>"${LOCK_FILE}"
        flock 200

        if [[ ! -s "${CARD_DIR}/localDB/card.json" ]]; then
            init_card_localdb
        else
            echo "[INFO] 다른 task가 이미 CARD localDB를 준비했습니다."
        fi

        flock -u 200
    else
        echo "[WARN] flock 사용 불가. lock 없이 CARD localDB 초기화 시도."
        init_card_localdb
    fi
fi

if [[ ! -s "${CARD_DIR}/localDB/card.json" ]]; then
    echo "[ERROR] CARD localDB 초기화 실패: ${CARD_DIR}/localDB/card.json"
    exit 1
fi

# ── RGI 실행 (protein 모드) ───────────────────────────────────
cd "${CARD_DIR}"

rgi main \
    --input_sequence "${FAA}" \
    --output_file "${OUT_PREFIX}" \
    --input_type protein \
    --clean \
    --num_threads "${THREADS}" \
    --alignment_tool "${ALIGN}" \
    --local

# ── 산출물 확인 ────────────────────────────────────────────────
if [[ -s "${OUT_PREFIX}.txt" ]]; then
    hit_n=$(tail -n +2 "${OUT_PREFIX}.txt" | wc -l || true)
    echo "[INFO] RGI hit rows=${hit_n}"
    echo "[DONE] RGI contig analysis completed: ${sample}"
else
    echo "[ERROR] RGI output missing: ${OUT_PREFIX}.txt"
    exit 1
fi
