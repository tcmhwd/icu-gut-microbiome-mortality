#!/bin/bash
#SBATCH --job-name=megahit_array
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/megahit_%A_%a.txt
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate assemble

# 작업 디렉토리 설정
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/non_AGI"
fastq_dir="${WD}/${project_id}/host_removed"
output_dir="${WD}/${project_id}/MEGAHIT_result"
sample_list="${WD}/${project_id}/sample_list.txt"

mkdir -p "${output_dir}"

# 샘플 이름 추출
sample_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$sample_list" | tr -d '\r')

# 입력 파일 경로
file1="${fastq_dir}/${sample_name}_host_removed_R1.fastq.gz"
file2="${fastq_dir}/${sample_name}_host_removed_R2.fastq.gz"

# 파일 존재 여부 확인
if [[ ! -f "$file1" || ! -f "$file2" ]]; then
  echo "입력 파일 없음: $file1 또는 $file2"
  exit 1
fi

# MEGAHIT 결과 디렉토리
sample_output="${output_dir}/${sample_name}"

threads="${SLURM_CPUS_PER_TASK:-8}"

# 이미 결과 디렉토리가 있으면 중단
if [[ -d "$sample_output" ]]; then
  echo "이미 존재하는 결과 디렉토리: $sample_output"
  echo "기존 결과를 삭제하거나 다른 이름을 사용하세요."
  exit 1
fi

# MEGAHIT 실행
megahit \
  -1 "$file1" \
  -2 "$file2" \
  -o "$sample_output" \
  --num-cpu-threads "$threads"

echo "MEGAHIT 조립 완료: ${sample_name}"
