#!/bin/bash
#SBATCH --job-name=bwa_map
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/bwa_map_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/bwa_map_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate bam

WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

REF="${ICU_DIR}/cohort_drep/all_mags.fa"
SAMPLE_LIST="${ICU_DIR}/combined_sample_list.txt"
BAM_DIR="${ICU_DIR}/coverm_bam"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${BAM_DIR}"

if [[ ! -f "${REF}.bwt" ]]; then
  echo "[ERROR] BWA index not found: ${REF}.bwt"
  echo "[ERROR] Run 15a_make_index.sh first."
  exit 1
fi

sample=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}" | tr -d '\r' | xargs)
if [[ -z "${sample}" ]]; then
  echo "[ERROR] No sample for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

OUT_BAM="${BAM_DIR}/${sample}.bam"

if [[ -s "${OUT_BAM}" && -f "${OUT_BAM}.bai" ]]; then
  echo "[SKIP] BAM already exists: ${sample}"
  exit 0
fi

# FASTQ 경로 결정
prefix="${sample:0:3}"
case "${prefix}" in
  SRR|ERR)
    R1="${ICU_DIR}/non_AGI/host_removed/${sample}_host_removed_R1.fastq.gz"
    R2="${ICU_DIR}/non_AGI/host_removed/${sample}_host_removed_R2.fastq.gz"
    ;;
  CRR)
    R1="${ICU_DIR}/AGI/host_removed/${sample}_host_removed_R1.fastq.gz"
    R2="${ICU_DIR}/AGI/host_removed/${sample}_host_removed_R2.fastq.gz"
    ;;
  A08|ICU)
    R1="/data/tcmhwd/read/trimmed/${sample}_host_removed.fastq.1.gz"
    R2="/data/tcmhwd/read/trimmed/${sample}_host_removed.fastq.2.gz"
    ;;
  *)
    echo "[ERROR] Unknown sample prefix '${prefix}' for sample ${sample}"
    exit 1
    ;;
esac

if [[ ! -f "${R1}" || ! -f "${R2}" ]]; then
  echo "[ERROR] FASTQ pair not found for ${sample}"
  echo "  R1: ${R1}"
  echo "  R2: ${R2}"
  exit 1
fi

echo "[INFO] sample=${sample}"
echo "[INFO] R1=${R1}"
echo "[INFO] R2=${R2}"
echo "[INFO] threads=${THREADS}"

bwa mem -t "${THREADS}" "${REF}" "${R1}" "${R2}" \
  | samtools sort -@ 4 -m 4G -o "${OUT_BAM}"

samtools index "${OUT_BAM}"

echo "[DONE] BAM ready: ${OUT_BAM}"
