#!/bin/bash
#SBATCH --job-name=humann3
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/humann3_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/log/humann3_%A_%a.err
#SBATCH --time=999:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --partition=cpu_partition
#SBATCH --ntasks=1

set -euo pipefail

########################
# 0. conda env 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate humann3

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

########################
# 1. 기본 경로 설정
########################
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"

sample_list="${WD}/${project_id}/sample_list.txt"
host_removed_dir="${WD}/${project_id}/host_removed"

humann_out_dir="${WD}/${project_id}/humann3"
metaphlan_out_dir="${WD}/${project_id}/metaphlan3"
tmp_root="${WD}/${project_id}/humann3_tmp"
log_dir="${WD}/log"

mkdir -p "${humann_out_dir}" "${metaphlan_out_dir}" "${tmp_root}" "${log_dir}"

########################
# 2. DB 경로 설정
########################
REF_DIR="${WD}/ref"

# MetaPhlAn 3 DB
METAPHLAN_DB="${REF_DIR}/metaphlan3_db"
METAPHLAN_INDEX="mpa_v30_CHOCOPhlAn_201901"
export METAPHLAN_BOWTIE2_DB="${METAPHLAN_DB}"

# HUMAnN DB
CHOCOPHLAN_DB="${REF_DIR}/humann3_databases/chocophlan"
UNIREF_DB="${REF_DIR}/humann3_databases/uniref"

########################
# 3. 환경 체크
########################
echo "=== ENV CHECK ==="
echo "CONDA_PREFIX = ${CONDA_PREFIX}"
echo -n "humann:      "; which humann || echo "NOT FOUND"
humann --version 2>/dev/null || true
echo -n "metaphlan:   "; which metaphlan || echo "NOT FOUND"
metaphlan --version 2>/dev/null || true
echo -n "bowtie2:     "; which bowtie2 || echo "NOT FOUND"
bowtie2 --version 2>/dev/null || true
echo -n "diamond:     "; which diamond || echo "NOT FOUND"
diamond version 2>/dev/null || true
echo -n "seqtk:       "; which seqtk || echo "NOT FOUND"
echo "METAPHLAN_DB    = ${METAPHLAN_DB}"
echo "METAPHLAN_INDEX = ${METAPHLAN_INDEX}"
echo "CHOCOPHLAN_DB   = ${CHOCOPHLAN_DB}"
echo "UNIREF_DB       = ${UNIREF_DB}"
echo "================="

########################
# 4. 샘플 이름 가져오기
########################
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')

if [ -z "${sample_name}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
  exit 1
fi

echo "HUMAnN3 분석 시작 - 샘플: ${sample_name}"

########################
# 5. host-removed FASTQ 자동 인식
########################
# 패턴 A: sample_host_removed_1.fastq.gz / _2.fastq.gz
r1_a="${host_removed_dir}/${sample_name}_host_removed_R1.fastq.gz"
r2_a="${host_removed_dir}/${sample_name}_host_removed_R2.fastq.gz"

# 패턴 B: sample_host_removed.fastq.1.gz / .2.gz
r1_b="${host_removed_dir}/${sample_name}_host_removed.fastq.1.gz"
r2_b="${host_removed_dir}/${sample_name}_host_removed.fastq.2.gz"

if [[ -f "${r1_a}" && -f "${r2_a}" ]]; then
  r1="${r1_a}"
  r2="${r2_a}"
elif [[ -f "${r1_b}" && -f "${r2_b}" ]]; then
  r1="${r1_b}"
  r2="${r2_b}"
else
  echo "host-removed 파일을 찾을 수 없습니다."
  echo "확인한 경로:"
  echo "  ${r1_a}"
  echo "  ${r2_a}"
  echo "  ${r1_b}"
  echo "  ${r2_b}"
  exit 1
fi

echo "사용 파일:"
echo "  R1: ${r1}"
echo "  R2: ${r2}"

########################
# 6. downsample + merge
########################
threads="${SLURM_CPUS_PER_TASK:-8}"
MAX_READS_PER_MATE=5000000

sample_tmp="${tmp_root}/${sample_name}"
mkdir -p "${sample_tmp}"

sub_r1="${sample_tmp}/${sample_name}_sub.1.fastq"
sub_r2="${sample_tmp}/${sample_name}_sub.2.fastq"
merged_fastq="${sample_tmp}/${sample_name}.fastq"

if ! command -v seqtk >/dev/null 2>&1; then
  echo "seqtk를 찾을 수 없습니다. humann3 환경에 seqtk를 설치해 주세요."
  exit 1
fi

if [ ! -f "${merged_fastq}" ]; then
  echo "FASTQ downsample + 병합 시작"

  echo "R1 downsample (${MAX_READS_PER_MATE} reads)"
  seqtk sample -s100 "${r1}" "${MAX_READS_PER_MATE}" > "${sub_r1}"

  echo "R2 downsample (${MAX_READS_PER_MATE} reads)"
  seqtk sample -s100 "${r2}" "${MAX_READS_PER_MATE}" > "${sub_r2}"

  echo "R1/R2 병합"
  cat "${sub_r1}" "${sub_r2}" > "${merged_fastq}"
else
  echo "기존 merged FASTQ 사용: ${merged_fastq}"
fi

if head -n 1 "${merged_fastq}" | grep -q "^@"; then
  echo "merged FASTQ 형식 확인 완료"
else
  echo "merged FASTQ 첫 줄이 FASTQ 형식이 아닙니다: ${merged_fastq}"
  exit 1
fi

########################
# 7. MetaPhlAn3 실행
########################
metaphlan_profile="${metaphlan_out_dir}/${sample_name}_profile.tsv"
metaphlan_bowtie2out="${metaphlan_out_dir}/${sample_name}.bowtie2.bz2"

if [ ! -f "${metaphlan_profile}" ]; then
  echo "MetaPhlAn3 실행 시작: ${sample_name}"

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

if [ ! -s "${metaphlan_profile}" ]; then
  echo "MetaPhlAn profile 파일이 생성되지 않았거나 비어 있습니다: ${metaphlan_profile}"
  exit 1
fi

########################
# 8. HUMAnN3 실행
########################
echo "HUMAnN3 실행:"
echo "  sample   : ${sample_name}"
echo "  input    : ${merged_fastq}"
echo "  profile  : ${metaphlan_profile}"
echo "  output   : ${humann_out_dir}"
echo "  threads  : ${threads}"

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

humann_exit=$?

if [ ${humann_exit} -ne 0 ]; then
  echo "HUMAnN3 실패: ${sample_name} (exit code: ${humann_exit})"
  exit 1
fi

echo "HUMAnN3 완료: ${sample_name}"

########################
# 9. HUMAnN 출력 파일 이름 정리
########################
# 입력 파일명이 ${sample_name}.fastq 이므로 보통 basename은 sample_name 으로 생성됨.
# 환경에 따라 sample_name_Abundance 등 변형될 가능성은 낮지만, 필요시 아래에서 확인 가능.
ls -lh "${humann_out_dir}" | grep "${sample_name}" || true

########################
# 10. 임시 파일 정리
########################
rm -f "${sub_r1}" "${sub_r2}" "${merged_fastq}"

echo "전체 완료: ${sample_name}"
