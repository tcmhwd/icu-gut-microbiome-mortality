#!/bin/bash
#SBATCH --job-name=checkm2_filter
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/checkm2_filter_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/log/checkm2_filter_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu_partition

set -euo pipefail

########################
# 1. conda 환경 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate checkm2

########################
# 2. 기본 경로 설정
########################
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/non_AGI"

# 필요시 finished 샘플 리스트로 바꿔서 사용
sample_list="${WD}/${project_id}/sample_list.txt"

CHECKM2_OUT="${WD}/${project_id}/CheckM2_result"
BINNING_OUT="${WD}/${project_id}/Binning_results"

# CheckM2 DB는 디렉토리가 아니라 .dmnd 파일 경로
CHECKM2_DB="${WD}/ref/CheckM2_database/uniref100.KO.1.dmnd"

mkdir -p "${CHECKM2_OUT}"

threads="${SLURM_CPUS_PER_TASK:-8}"

########################
# 3. 현재 샘플 이름 추출
########################
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')

if [ -z "${sample_name}" ]; then
    echo "ERROR: 샘플 이름이 비어 있음 (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"
    exit 1
fi

echo "Processing sample: ${sample_name}"

########################
# 4. DAS Tool bin 경로 설정
########################
bins_dir="${BINNING_OUT}/${sample_name}/DAS_Tool/${sample_name}_DAS_Tool_output_DASTool_bins"

if [ ! -d "${bins_dir}" ]; then
    echo "ERROR: bin 디렉토리 없음: ${bins_dir}"
    exit 1
fi

bin_count=$(find "${bins_dir}" -maxdepth 1 -type f -name "*.fa" | wc -l)

if [ "${bin_count}" -eq 0 ]; then
    echo "ERROR: bins_dir 안에 .fa bin 파일이 없습니다: ${bins_dir}"
    exit 1
fi

if [ ! -f "${CHECKM2_DB}" ]; then
    echo "ERROR: CheckM2 DB 파일 없음: ${CHECKM2_DB}"
    exit 1
fi

########################
# 5. CheckM2 실행
########################
sample_out_dir="${CHECKM2_OUT}/${sample_name}"
filtered_bins="${sample_out_dir}/filtered_bins"

rm -rf "${sample_out_dir}"
mkdir -p "${sample_out_dir}"

echo "CheckM2 실행 시작: ${sample_name}"
echo "  bins_dir : ${bins_dir}"
echo "  out_dir  : ${sample_out_dir}"
echo "  db       : ${CHECKM2_DB}"
echo "  threads  : ${threads}"

checkm2 predict \
    --input "${bins_dir}" \
    --output-directory "${sample_out_dir}" \
    -x fa \
    --threads "${threads}" \
    --force \
    --database_path "${CHECKM2_DB}"

########################
# 6. CheckM2 결과 필터링
########################
checkm2_results="${sample_out_dir}/quality_report.tsv"

if [ ! -s "${checkm2_results}" ]; then
    echo "ERROR: CheckM2 결과 파일이 없거나 비어 있음: ${checkm2_results}"
    exit 1
fi

# CheckM2 실행 후에 생성해야 함
mkdir -p "${filtered_bins}"

echo "CheckM2 필터링 시작: completeness >= 50, contamination < 10"

awk -F'\t' '
NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "Name") name_col=i
        else if ($i == "Completeness") comp_col=i
        else if ($i == "Contamination") cont_col=i
    }
    next
}
name_col && comp_col && cont_col && $comp_col+0 >= 50 && $cont_col+0 < 10 {
    print $name_col
}
' "${checkm2_results}" | while read -r bin; do
    src="${bins_dir}/${bin}.fa"
    if [ -f "${src}" ]; then
        cp "${src}" "${filtered_bins}/"
    else
        echo "WARN: bin fasta not found for ${bin}: ${src}" >&2
    fi
done

filtered_count=$(find "${filtered_bins}" -maxdepth 1 -type f -name "*.fa" | wc -l)

echo "CheckM2 완료: ${sample_name}"
echo "Filtered bins: ${filtered_count}"
