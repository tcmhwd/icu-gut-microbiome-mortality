#!/usr/bin/env bash
#SBATCH --job-name=trim_host_public
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/trim_host_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/trim_host_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate trim

BASE="/home/tcmhwd/microbiome/ICU_public"
WD="/home/tcmhwd/microbiome"

DATASET="${DATASET:-}"
TRIM_MODE="${TRIM_MODE:-fastp}"

if [[ -z "${DATASET}" ]]; then
  echo "[ERROR] DATASET is not set. Use DATASET=non_AGI or DATASET=AGI."
  exit 1
fi

if [[ "${DATASET}" != "non_AGI" && "${DATASET}" != "AGI" ]]; then
  echo "[ERROR] DATASET must be non_AGI or AGI."
  exit 1
fi

input_dir="${BASE}/${DATASET}/fastq"
sample_list="${BASE}/${DATASET}/sample_list.txt"

trim_dir="${BASE}/${DATASET}/trimmed"
host_removed_dir="${BASE}/${DATASET}/host_removed"
qc_dir="${BASE}/${DATASET}/qc"

mkdir -p "${trim_dir}" "${host_removed_dir}" "${qc_dir}" "${BASE}/logs"

host_ref="${WD}/ref/Homo_sapiens_GRCh38_index"

trimmomatic_path="${WD}/ref/trimmomatic-0.39.jar"
adapters="${WD}/ref/adapters/TruSeq3-PE-2.fa"

threads="${SLURM_CPUS_PER_TASK:-4}"
java_memory="-Xmx16g"

sample_name="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${sample_list}" | tr -d '\r')"

if [[ -z "${sample_name}" ]]; then
  echo "[ERROR] Empty sample name at line ${SLURM_ARRAY_TASK_ID} in ${sample_list}"
  exit 1
fi

find_pair() {
  local sample="$1"
  local r1=""
  local r2=""

  local r1_candidates=(
    "${input_dir}/${sample}_1.fastq.gz"
    "${input_dir}/${sample}_1.fq.gz"
    "${input_dir}/${sample}_R1.fastq.gz"
    "${input_dir}/${sample}_R1_001.fastq.gz"
    "${input_dir}/${sample}_f1.fq.gz"
  )

  local r2_candidates=(
    "${input_dir}/${sample}_2.fastq.gz"
    "${input_dir}/${sample}_2.fq.gz"
    "${input_dir}/${sample}_R2.fastq.gz"
    "${input_dir}/${sample}_R2_001.fastq.gz"
    "${input_dir}/${sample}_r2.fq.gz"
  )

  for i in "${!r1_candidates[@]}"; do
    if [[ -f "${r1_candidates[$i]}" && -f "${r2_candidates[$i]}" ]]; then
      r1="${r1_candidates[$i]}"
      r2="${r2_candidates[$i]}"
      break
    fi
  done

  if [[ -z "${r1}" || -z "${r2}" ]]; then
    echo "[ERROR] Paired FASTQ files not found for sample: ${sample}" >&2
    echo "[ERROR] Input directory: ${input_dir}" >&2
    exit 1
  fi

  echo "${r1} ${r2}"
}

read -r file1 file2 < <(find_pair "${sample_name}")

echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Trim mode: ${TRIM_MODE}"
echo "[INFO] Sample: ${sample_name}"
echo "[INFO] R1: ${file1}"
echo "[INFO] R2: ${file2}"
echo "[INFO] Threads: ${threads}"

trimmed1="${trim_dir}/${sample_name}_R1.trimmed.fastq.gz"
trimmed2="${trim_dir}/${sample_name}_R2.trimmed.fastq.gz"

if [[ "${TRIM_MODE}" == "trimmomatic" ]]; then

  if [[ ! -f "${trimmomatic_path}" ]]; then
    echo "[ERROR] Trimmomatic jar not found: ${trimmomatic_path}"
    exit 1
  fi

  if [[ ! -f "${adapters}" ]]; then
    echo "[ERROR] Adapter file not found: ${adapters}"
    exit 1
  fi

  echo "[INFO] Running Trimmomatic: ${sample_name}"

  java ${java_memory} -jar "${trimmomatic_path}" PE \
    -threads "${threads}" \
    -phred33 \
    "${file1}" "${file2}" \
    "${trimmed1}" /dev/null \
    "${trimmed2}" /dev/null \
    ILLUMINACLIP:"${adapters}":2:30:10 \
    LEADING:3 \
    TRAILING:3 \
    SLIDINGWINDOW:4:15 \
    MINLEN:60 \
    2> "${qc_dir}/${sample_name}.trimmomatic.log"

elif [[ "${TRIM_MODE}" == "fastp" ]]; then

  if ! command -v fastp >/dev/null 2>&1; then
    echo "[ERROR] fastp is not available in the current conda environment."
    exit 1
  fi

  echo "[INFO] Running fastp with paired-end adapter auto-detection: ${sample_name}"

  extra_args=()
  if [[ -n "${FASTP_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${FASTP_EXTRA_ARGS}"
  fi

  fastp \
    -i "${file1}" \
    -I "${file2}" \
    -o "${trimmed1}" \
    -O "${trimmed2}" \
    --detect_adapter_for_pe \
    --qualified_quality_phred 20 \
    --length_required 60 \
    --thread "${threads}" \
    --html "${qc_dir}/${sample_name}.fastp.html" \
    --json "${qc_dir}/${sample_name}.fastp.json" \
    "${extra_args[@]}"

else
  echo "[ERROR] Unsupported TRIM_MODE: ${TRIM_MODE}"
  echo "[ERROR] Use TRIM_MODE=trimmomatic or TRIM_MODE=fastp."
  exit 1
fi

if [[ ! -s "${trimmed1}" || ! -s "${trimmed2}" ]]; then
  echo "[ERROR] Trimmed FASTQ files were not created properly for ${sample_name}"
  exit 1
fi

echo "[INFO] Running Bowtie2 host removal: ${sample_name}"

bowtie2 \
  -x "${host_ref}" \
  -1 "${trimmed1}" \
  -2 "${trimmed2}" \
  --threads "${threads}" \
  --very-sensitive \
  --un-conc-gz "${host_removed_dir}/${sample_name}_host_removed_R%.fastq.gz" \
  -S /dev/null \
  2> "${qc_dir}/${sample_name}.bowtie2.log"

host_removed1="${host_removed_dir}/${sample_name}_host_removed_R1.fastq.gz"
host_removed2="${host_removed_dir}/${sample_name}_host_removed_R2.fastq.gz"

if [[ ! -s "${host_removed1}" || ! -s "${host_removed2}" ]]; then
  echo "[ERROR] Host-removed FASTQ files were not created properly for ${sample_name}"
  exit 1
fi

if [[ "${REMOVE_TRIMMED:-0}" == "1" ]]; then
  rm -f "${trimmed1}" "${trimmed2}"
fi

echo "[OK] Completed: ${sample_name}"
echo "[OK] Host-removed R1: ${host_removed1}"
echo "[OK] Host-removed R2: ${host_removed2}"
