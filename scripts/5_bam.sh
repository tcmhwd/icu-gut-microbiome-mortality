#!/bin/bash
#SBATCH --job-name=bwa_mapping
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/bwa_mapping_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/log/bwa_mapping_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate bam

# 경로 설정
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"
megahit_dir="${WD}/${project_id}/MEGAHIT_result"
bam_dir="${WD}/${project_id}/BAM_files"
fastq_dir="${WD}/${project_id}/host_removed"
sample_list="${WD}/${project_id}/sample_list.txt"

mkdir -p "${bam_dir}"

# 샘플 이름 추출
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')

if [ -z "${sample_name}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
  exit 1
fi

# 결과 디렉토리 생성
sample_bam_dir="${bam_dir}/${sample_name}"
mkdir -p "${sample_bam_dir}"

# 입력 파일 설정
contig_file="${megahit_dir}/${sample_name}/final.contigs.fa"

# host-removed FASTQ 파일명 자동 인식
fastq_1_a="${fastq_dir}/${sample_name}_host_removed_R1.fastq.gz"
fastq_2_a="${fastq_dir}/${sample_name}_host_removed_R2.fastq.gz"

fastq_1_b="${fastq_dir}/${sample_name}_host_removed.fastq.1.gz"
fastq_2_b="${fastq_dir}/${sample_name}_host_removed.fastq.2.gz"

if [[ -f "${fastq_1_a}" && -f "${fastq_2_a}" ]]; then
  fastq_1="${fastq_1_a}"
  fastq_2="${fastq_2_a}"
elif [[ -f "${fastq_1_b}" && -f "${fastq_2_b}" ]]; then
  fastq_1="${fastq_1_b}"
  fastq_2="${fastq_2_b}"
else
  echo "host-removed FASTQ 파일을 찾을 수 없습니다."
  echo "확인한 경로:"
  echo "  ${fastq_1_a}"
  echo "  ${fastq_2_a}"
  echo "  ${fastq_1_b}"
  echo "  ${fastq_2_b}"
  exit 1
fi

# 파일 존재 여부 확인
if [[ ! -f "${contig_file}" ]]; then
  echo "contig 파일이 없습니다: ${contig_file}"
  exit 1
fi

threads="${SLURM_CPUS_PER_TASK:-8}"
sorted_bam="${sample_bam_dir}/${sample_name}_sorted.bam"

echo "샘플: ${sample_name}"
echo "contig: ${contig_file}"
echo "R1: ${fastq_1}"
echo "R2: ${fastq_2}"
echo "threads: ${threads}"

# BWA 인덱싱 (존재하지 않을 때만)
if [[ ! -f "${contig_file}.bwt" && ! -f "${contig_file}.0123" ]]; then
  echo "BWA 인덱싱 시작: ${contig_file}"
  bwa index "${contig_file}"
fi

# bwa mem -> samtools sort 바로 연결
echo "BWA MEM + BAM sort 시작: ${sample_name}"
bwa mem -t "${threads}" "${contig_file}" "${fastq_1}" "${fastq_2}" \
  | samtools sort -@ "${threads}" -m 4G -o "${sorted_bam}" -

# BAM index
echo "BAM index 생성: ${sample_name}"
samtools index -@ "${threads}" "${sorted_bam}"

# 간단한 통계
echo "BAM flagstat 생성: ${sample_name}"
samtools flagstat -@ "${threads}" "${sorted_bam}" > "${sample_bam_dir}/${sample_name}_flagstat.txt"

echo "BAM 파일 생성 완료: ${sample_name}"
