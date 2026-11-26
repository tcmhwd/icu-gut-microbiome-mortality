#!/bin/bash
#SBATCH --job-name=prokka_contig
#SBATCH --output=/home/tcmhwd/microbiome/log/prokka_contig_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/log/prokka_contig_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition


set -euo pipefail

export PATH="/data/tcmhwd/miniconda3/envs/prokka_env/bin:$PATH"

# locale 고정: tbl2asn / tbl2asn-forever 날짜 처리 문제 방지
export LC_ALL=C
export LANG=C
export LC_TIME=C
unset LANGUAGE || true

# ── 경로 설정 ──────────────────────────────────────────────────
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"

SAMPLE_LIST="${WD}/${project_id}/sample_list.txt"
MEGAHIT_DIR="${WD}/${project_id}/MEGAHIT_result"
OUT_DIR="${WD}/${project_id}/Prokka_contig_result"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUT_DIR}"

# ── 샘플명 추출 ────────────────────────────────────────────────
sample=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}" | tr -d '\r')

if [[ -z "${sample}" ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
    exit 1
fi

CONTIG="${MEGAHIT_DIR}/${sample}/final.contigs.fa"
OUTDIR="${OUT_DIR}/${sample}"

# locus_tag prefix 생성
# sample=PM_NNN -> PMNNN
LOCUSTAG=$(echo "${sample}" | tr -cd '[:alnum:]' | tr '[:lower:]' '[:upper:]')

# locus_tag는 너무 짧으면 보강
if [[ ${#LOCUSTAG} -lt 3 ]]; then
    LOCUSTAG="SMP${SLURM_ARRAY_TASK_ID}"
fi

echo "[INFO] sample=${sample}"
echo "[INFO] contig=${CONTIG}"
echo "[INFO] outdir=${OUTDIR}"
echo "[INFO] threads=${THREADS}"
echo "[INFO] locustag=${LOCUSTAG}"

# ── 입력 파일 확인 ─────────────────────────────────────────────
if [[ ! -s "${CONTIG}" ]]; then
    echo "[ERROR] Contig 파일 없음: ${CONTIG}"
    exit 1
fi

# ── 완료 여부 확인 (RGI/RPKM 목적 기준) ───────────────────────
if [[ -s "${OUTDIR}/${sample}.gff" && -s "${OUTDIR}/${sample}.faa" ]]; then
    echo "[SKIP] Prokka 결과(.gff/.faa) 이미 존재: ${sample}"
    exit 0
fi

mkdir -p "${OUTDIR}"

# ── 기존 부분 출력 정리 (선택적) ───────────────────────────────
# 이전 실패 실행의 부분 산출물이 섞이지 않도록 핵심 파일만 제거
rm -f "${OUTDIR}/${sample}.gff" \
      "${OUTDIR}/${sample}.faa" \
      "${OUTDIR}/${sample}.ffn" \
      "${OUTDIR}/${sample}.fna" \
      "${OUTDIR}/${sample}.tbl" \
      "${OUTDIR}/${sample}.tsv" \
      "${OUTDIR}/${sample}.txt" \
      "${OUTDIR}/${sample}.gbk" \
      "${OUTDIR}/${sample}.gbf" \
      "${OUTDIR}/${sample}.sqn" \
      "${OUTDIR}/${sample}.err" || true

# ── Prokka 실행 ────────────────────────────────────────────────
# --noanno      : annotation search 생략
# --metagenome  : metagenome contig에 적합
# --locustag    : sample 기반 ID 추적성 향상
# --prefix      : 출력 파일명은 sample 유지
# --force       : 동일 outdir 재실행 허용
set +e
prokka \
    --outdir "${OUTDIR}" \
    --prefix "${sample}" \
    --locustag "${LOCUSTAG}" \
    --cpus "${THREADS}" \
    --force \
    --noanno \
    --metagenome \
    "${CONTIG}"
prokka_rc=$?
set -e

echo "[INFO] prokka exit code=${prokka_rc}"

# ── downstream 기준 성공 여부 판정 ────────────────────────────
if [[ -s "${OUTDIR}/${sample}.gff" && -s "${OUTDIR}/${sample}.faa" ]]; then
    cds_n=$(grep -c $'\tCDS\t' "${OUTDIR}/${sample}.gff" || true)
    faa_n=$(grep -c '^>' "${OUTDIR}/${sample}.faa" || true)

    echo "[INFO] gff CDS count=${cds_n}"
    echo "[INFO] faa sequence count=${faa_n}"

    if [[ "${prokka_rc}" -ne 0 ]]; then
        echo "[WARN] Prokka returned non-zero exit, but .gff and .faa were created."
        echo "[WARN] This is usually due to GenBank/tbl2asn post-processing, not the core CDS/protein outputs."
        echo "[WARN] Proceeding because this pipeline uses Prokka primarily for RGI and RPKM."
    fi

    echo "[DONE] Prokka contig annotation usable for downstream analysis: ${sample}"
    exit 0
else
    echo "[ERROR] Prokka failed to generate required outputs for downstream analysis."
    echo "[ERROR] Missing one or both of:"
    echo "        ${OUTDIR}/${sample}.gff"
    echo "        ${OUTDIR}/${sample}.faa"
    exit 1
fi
