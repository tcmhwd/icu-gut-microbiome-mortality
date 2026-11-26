#!/usr/bin/env bash
#SBATCH --job-name=AGI_bowtie2_only
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/AGI_bowtie2_only_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/AGI_bowtie2_only_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate trim

BASE="/home/tcmhwd/microbiome/ICU_public"
WD="/home/tcmhwd/microbiome"

sample_list="${BASE}/AGI/sample_list.txt"
trim_dir="${BASE}/AGI/trimmed"
host_removed_dir="${BASE}/AGI/host_removed"
qc_dir="${BASE}/AGI/qc"

mkdir -p "${host_removed_dir}" "${qc_dir}"

host_ref="${WD}/ref/Homo_sapiens_GRCh38_index"
threads="${SLURM_CPUS_PER_TASK:-8}"

sample_name="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')"

if [[ -z "${sample_name}" ]]; then
  echo "[ERROR] No sample found at line ${SLURM_ARRAY_TASK_ID} in ${sample_list}"
  exit 1
fi

trimmed1="${trim_dir}/${sample_name}_R1.trimmed.fastq.gz"
trimmed2="${trim_dir}/${sample_name}_R2.trimmed.fastq.gz"

out1="${host_removed_dir}/${sample_name}_host_removed_R1.fastq.gz"
out2="${host_removed_dir}/${sample_name}_host_removed_R2.fastq.gz"

echo "[INFO] Bowtie2-only host removal from existing trimmed FASTQ"
echo "[INFO] Date: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Sample: ${sample_name}"
echo "[INFO] Threads: ${threads}"
echo "[INFO] Trimmed R1: ${trimmed1}"
echo "[INFO] Trimmed R2: ${trimmed2}"

if [[ ! -s "${trimmed1}" || ! -s "${trimmed2}" ]]; then
  echo "[ERROR] Trimmed FASTQ files are missing or empty."
  echo "  ${trimmed1}"
  echo "  ${trimmed2}"
  exit 1
fi

echo "[INFO] Checking gzip integrity of trimmed FASTQ files..."

if ! gzip -t "${trimmed1}"; then
  echo "[ERROR] Corrupt or incomplete trimmed R1: ${trimmed1}"
  echo "[ERROR] Do not use this file for Bowtie2. Re-run fastp/trimming for this sample."
  exit 2
fi

if ! gzip -t "${trimmed2}"; then
  echo "[ERROR] Corrupt or incomplete trimmed R2: ${trimmed2}"
  echo "[ERROR] Do not use this file for Bowtie2. Re-run fastp/trimming for this sample."
  exit 2
fi

echo "[INFO] gzip integrity check passed."

if [[ -s "${out1}" && -s "${out2}" ]]; then
  echo "[INFO] Existing host-removed files found. Skipping Bowtie2."
  echo "  ${out1}"
  echo "  ${out2}"
  exit 0
fi

echo "[INFO] Removing partial previous host-removed outputs, if any."
rm -f "${out1}" "${out2}"

echo "[INFO] Running Bowtie2 host removal: ${sample_name}"

bowtie2 \
  -x "${host_ref}" \
  -1 "${trimmed1}" \
  -2 "${trimmed2}" \
  --threads "${threads}" \
  --very-sensitive \
  --un-conc-gz "${host_removed_dir}/${sample_name}_host_removed_R%.fastq.gz" \
  -S /dev/null \
  2> "${qc_dir}/${sample_name}.bowtie2_only.bowtie2.log"

if [[ ! -s "${out1}" || ! -s "${out2}" ]]; then
  echo "[ERROR] Bowtie2 host-removed outputs were not created properly."
  echo "  ${out1}"
  echo "  ${out2}"
  exit 1
fi

echo "[OK] Bowtie2-only host removal completed: ${sample_name}"
echo "[OK] Output R1: ${out1}"
echo "[OK] Output R2: ${out2}"
echo "[INFO] Date: $(date)"
