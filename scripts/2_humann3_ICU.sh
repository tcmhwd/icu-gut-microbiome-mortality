#!/bin/bash
#SBATCH --job-name=humann3_ICU
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/humann3_ICU_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/humann3_ICU_%A_%a.err
#SBATCH --time=999:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --partition=cpu_partition
#SBATCH --ntasks=1

# 사용법:
#   sbatch --array=1-549%20 scripts/2_humann3_ICU.sh

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate humann3

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

########################
# 1. 경로 설정
########################
WD="/home/tcmhwd/microbiome"

sample_list="${WD}/ICU_public/ICU/sample_list.txt"
host_removed_dir="/data/tcmhwd/read/trimmed"

humann_out_dir="${WD}/ICU_public/ICU/humann3"
metaphlan_out_dir="${WD}/ICU_public/ICU/metaphlan3"
# SLURM_TMPDIR가 있으면 노드 로컬 디스크 사용 (I/O 빠름), 없으면 home 사용
tmp_root="${SLURM_TMPDIR:-${WD}/ICU_public/ICU/humann3_tmp}"
logs_dir="${WD}/ICU_public/logs"

mkdir -p "${humann_out_dir}" "${metaphlan_out_dir}" "${tmp_root}" "${logs_dir}"

########################
# 2. DB 경로 설정
########################
REF_DIR="${WD}/ref"

METAPHLAN_DB="${REF_DIR}/metaphlan3_db"
METAPHLAN_INDEX="mpa_v30_CHOCOPhlAn_201901"
export METAPHLAN_BOWTIE2_DB="${METAPHLAN_DB}"

CHOCOPHLAN_DB="${REF_DIR}/humann3_databases/chocophlan"
UNIREF_DB="${REF_DIR}/humann3_databases/uniref"

########################
# 3. 샘플 이름 추출
########################
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')

if [ -z "${sample_name}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
  exit 1
fi

echo "HUMAnN3 분석 시작 - 샘플: ${sample_name}"

########################
# 4. 완료 여부 확인 (resume)
########################
final_marker="${humann_out_dir}/${sample_name}_pathabundance.tsv"
if [[ -s "${final_marker}" ]]; then
  echo "[SKIP] 이미 완료: ${sample_name}"
  exit 0
fi

########################
# 5. host-removed FASTQ 확인
# 내 ICU 데이터 naming: {sample}_host_removed.fastq.1.gz / .2.gz
########################
r1="${host_removed_dir}/${sample_name}_host_removed.fastq.1.gz"
r2="${host_removed_dir}/${sample_name}_host_removed.fastq.2.gz"

if [[ ! -f "${r1}" || ! -f "${r2}" ]]; then
  echo "[ERROR] host-removed FASTQ 없음: ${sample_name}"
  echo "  확인 경로: ${r1}"
  exit 1
fi

echo "사용 파일: R1=${r1}, R2=${r2}"

########################
# 6. downsample + merge (5M reads/mate, public 파이프라인과 동일)
########################
threads="${SLURM_CPUS_PER_TASK:-8}"
MAX_READS_PER_MATE=5000000

sample_tmp="${tmp_root}/${sample_name}"
mkdir -p "${sample_tmp}"

sub_r1="${sample_tmp}/${sample_name}_sub.1.fastq"
sub_r2="${sample_tmp}/${sample_name}_sub.2.fastq"
merged_fastq="${sample_tmp}/${sample_name}.fastq"

if [ ! -f "${merged_fastq}" ]; then
  echo "FASTQ downsample + 병합 시작"
  seqtk sample -s100 "${r1}" "${MAX_READS_PER_MATE}" > "${sub_r1}"
  seqtk sample -s100 "${r2}" "${MAX_READS_PER_MATE}" > "${sub_r2}"
  cat "${sub_r1}" "${sub_r2}" > "${merged_fastq}"
else
  echo "기존 merged FASTQ 사용: ${merged_fastq}"
fi

head -n 1 "${merged_fastq}" | grep -q "^@" || { echo "[ERROR] merged FASTQ 형식 이상"; exit 1; }

########################
# 7. MetaPhlAn3 실행
########################
metaphlan_profile="${metaphlan_out_dir}/${sample_name}_profile.tsv"
metaphlan_bowtie2out="${metaphlan_out_dir}/${sample_name}.bowtie2.bz2"

if [ ! -f "${metaphlan_profile}" ]; then
  echo "MetaPhlAn3 실행: ${sample_name}"
  metaphlan "${merged_fastq}" \
    --input_type fastq \
    --bowtie2db "${METAPHLAN_DB}" \
    --index "${METAPHLAN_INDEX}" \
    --nproc "${threads}" \
    --bowtie2out "${metaphlan_bowtie2out}" \
    -o "${metaphlan_profile}"
else
  echo "기존 MetaPhlAn profile 사용: ${metaphlan_profile}"
fi

[ -s "${metaphlan_profile}" ] || { echo "[ERROR] MetaPhlAn profile 비어있음"; exit 1; }

########################
# 8. HUMAnN3 실행
########################
echo "HUMAnN3 실행: ${sample_name}"

humann \
  --input "${merged_fastq}" \
  --input-format fastq \
  --output "${humann_out_dir}" \
  --threads "${threads}" \
  --nucleotide-database "${CHOCOPHLAN_DB}" \
  --protein-database "${UNIREF_DB}" \
  --taxonomic-profile "${metaphlan_profile}" \
  --diamond-options "--block-size 0.5 --index-chunks 8" \
  --resume \
  --remove-temp-output

[ $? -eq 0 ] || { echo "[ERROR] HUMAnN3 실패: ${sample_name}"; exit 1; }

echo "HUMAnN3 완료: ${sample_name}"

########################
# 9. 임시 파일 정리
########################
rm -f "${sub_r1}" "${sub_r2}" "${merged_fastq}"
rmdir "${sample_tmp}" 2>/dev/null || true

echo "전체 완료: ${sample_name}"
