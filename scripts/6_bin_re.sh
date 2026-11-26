#!/bin/bash
#SBATCH --job-name=multi_binning
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/binning_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/log/binning_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda 환경 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate binning

export RSCRIPT="${CONDA_PREFIX}/bin/Rscript"
export PATH="$(dirname "${RSCRIPT}"):${PATH}"

########################
# 2. 경로 설정
########################
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"

megahit_dir="${WD}/${project_id}/MEGAHIT_result"
bam_dir="${WD}/${project_id}/BAM_files"
output_dir="${WD}/${project_id}/Binning_results"
sample_list="${WD}/${project_id}/sample_list.txt"

mkdir -p "${output_dir}"

threads="${SLURM_CPUS_PER_TASK:-8}"

########################
# 3. 샘플 이름 추출
########################
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')

if [ -z "${sample_name}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플명이 없습니다."
    exit 1
fi

echo "Binning 시작: ${sample_name}"

########################
# 4. 샘플별 디렉토리
########################
sample_output_dir="${output_dir}/${sample_name}"
metabat_out="${sample_output_dir}/MetaBAT2"
maxbin_out="${sample_output_dir}/MaxBin2"
concoct_out="${sample_output_dir}/CONCOCT"
dastool_out="${sample_output_dir}/DAS_Tool"

mkdir -p "${metabat_out}" "${maxbin_out}" "${concoct_out}" "${dastool_out}"

########################
# 5. 입력 파일
########################
contig_file="${megahit_dir}/${sample_name}/final.contigs.fa"
sample_bam="${bam_dir}/${sample_name}/${sample_name}_sorted.bam"
sample_bai="${sample_bam}.bai"

if [ ! -f "${contig_file}" ]; then
    echo "ERROR: Contig 파일 없음: ${contig_file}"
    exit 1
fi

if [ ! -f "${sample_bam}" ]; then
    echo "ERROR: BAM 파일 없음: ${sample_bam}"
    exit 1
fi

if [ ! -f "${sample_bai}" ]; then
    echo "ERROR: BAM index 파일 없음: ${sample_bai}"
    exit 1
fi

########################
# 6. coverage 계산 (MetaBAT2 / MaxBin2용)
########################
coverage_file="${sample_output_dir}/${sample_name}_abundance.txt"

if [ ! -s "${coverage_file}" ]; then
    echo "coverage 계산 시작"
    jgi_summarize_bam_contig_depths --outputDepth "${coverage_file}" "${sample_bam}"
fi

if [ ! -s "${coverage_file}" ]; then
    echo "ERROR: coverage 파일 생성 실패: ${coverage_file}"
    exit 1
fi

########################
# 7. MetaBAT2 실행
########################
echo "MetaBAT2 실행 시작"
metabat2 \
    -i "${contig_file}" \
    -a "${coverage_file}" \
    -o "${metabat_out}/${sample_name}_bin" \
    -t "${threads}"

########################
# 8. MaxBin2 실행
########################
echo "MaxBin2 실행 시작"
run_MaxBin.pl \
    -contig "${contig_file}" \
    -out "${maxbin_out}/${sample_name}_bin" \
    -abund "${coverage_file}" \
    -thread "${threads}"

########################
# 9. CONCOCT 정식 파이프라인
########################
echo "CONCOCT 실행 시작"

cutup_fa="${concoct_out}/${sample_name}_contigs_10K.fa"
cutup_bed="${concoct_out}/${sample_name}_contigs_10K.bed"
concoct_cov="${concoct_out}/${sample_name}_coverage_table.tsv"
concoct_csv="${concoct_out}/clustering_gt1000.csv"
concoct_merged_csv="${concoct_out}/${sample_name}_concoct_clustering_merged.csv"
concoct_for_dastool="${concoct_out}/${sample_name}_concoct_for_dastool.tsv"

cut_up_fasta.py "${contig_file}" \
    -c 10000 -o 0 --merge_last \
    -b "${cutup_bed}" \
    > "${cutup_fa}"

concoct_coverage_table.py "${cutup_bed}" "${sample_bam}" > "${concoct_cov}"

concoct \
    --composition_file "${cutup_fa}" \
    --coverage_file "${concoct_cov}" \
    -b "${concoct_out}/" \
    --threads "${threads}"

if [ ! -s "${concoct_csv}" ]; then
    echo "ERROR: CONCOCT clustering 결과 파일 없음: ${concoct_csv}"
    exit 1
fi

merge_cutup_clustering.py "${concoct_csv}" > "${concoct_merged_csv}"

if [ ! -s "${concoct_merged_csv}" ]; then
    echo "ERROR: CONCOCT merged clustering 결과 없음: ${concoct_merged_csv}"
    exit 1
fi

awk -F',' 'NR>1 {
    gsub(/"/,"",$1);
    gsub(/"/,"",$2);
    print $1"\tCONCOCT_"$2
}' "${concoct_merged_csv}" > "${concoct_for_dastool}"

if [ ! -s "${concoct_for_dastool}" ]; then
    echo "ERROR: DAS Tool용 CONCOCT TSV 생성 실패: ${concoct_for_dastool}"
    exit 1
fi

########################
# 10. DAS Tool용 summary 파일 생성
########################

# 10-1. MetaBAT2 -> DAS Tool 입력용 2열 파일
metabat_dastool="${metabat_out}/${sample_name}_metabat_for_dastool.tsv"
: > "${metabat_dastool}"

shopt -s nullglob
for bin_fasta in "${metabat_out}/${sample_name}_bin."*.fa; do
    [ -e "$bin_fasta" ] || continue
    bin_name=$(basename "${bin_fasta}" .fa | sed "s/${sample_name}_bin\.//")
    grep "^>" "${bin_fasta}" | sed 's/^>//' | awk '{print $1}' | while read -r contig; do
        echo -e "${contig}\t${bin_name}" >> "${metabat_dastool}"
    done
done
shopt -u nullglob

if [ ! -s "${metabat_dastool}" ]; then
    echo "ERROR: MetaBAT2 DAS Tool 입력 파일 생성 실패: ${metabat_dastool}"
    exit 1
fi

# 10-2. MaxBin2 -> DAS Tool 입력용 2열 파일
maxbin_summary="${maxbin_out}/${sample_name}_bin_formatted_for_DASTool.tsv"
: > "${maxbin_summary}"

shopt -s nullglob
for bin_fasta in "${maxbin_out}/${sample_name}_bin."*.fasta; do
    [ -e "$bin_fasta" ] || continue
    bin_name=$(basename "${bin_fasta}" .fasta | sed 's/.*bin\.//')
    grep "^>" "${bin_fasta}" | sed 's/^>//' | awk '{print $1}' | while read -r contig; do
        echo -e "${contig}\t${bin_name}" >> "${maxbin_summary}"
    done
done
shopt -u nullglob

if [ ! -s "${maxbin_summary}" ]; then
    echo "ERROR: MaxBin2 DAS Tool 입력 파일 생성 실패: ${maxbin_summary}"
    exit 1
fi

########################
# 11. 입력 파일 열 수 점검
########################
for f in "${metabat_dastool}" "${maxbin_summary}" "${concoct_for_dastool}"; do
    nf=$(awk -F'\t' '{print NF}' "$f" | sort -nu | tr '\n' ' ')
    echo "CHECK ${f} -> columns: ${nf}"
done

########################
# 12. DAS Tool 실행
########################
echo "DAS Tool 실행 시작"

DAS_Tool \
    -i "${metabat_dastool},${maxbin_summary},${concoct_for_dastool}" \
    -l "MetaBAT2,MaxBin2,CONCOCT" \
    -c "${contig_file}" \
    -o "${dastool_out}/${sample_name}_DAS_Tool_output" \
    --write_bins \
    --threads "${threads}"

echo "DAS Tool binning 통합 완료: ${sample_name}"
