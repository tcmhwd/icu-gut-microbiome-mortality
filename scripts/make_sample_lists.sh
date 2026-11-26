#!/usr/bin/env bash
set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"

make_list() {
  input_dir="$1"
  output_list="$2"

  if [[ ! -d "${input_dir}" ]]; then
    echo "[ERROR] Input directory not found: ${input_dir}"
    exit 1
  fi

  find "${input_dir}" -maxdepth 1 -type f \( \
      -name "*_1.fastq.gz" -o \
      -name "*_1.fq.gz" -o \
      -name "*_R1.fastq.gz" -o \
      -name "*_R1_001.fastq.gz" -o \
      -name "*_f1.fq.gz" \
    \) -printf "%f\n" \
    | sed -E 's/(_R1_001\.fastq\.gz|_R1\.fastq\.gz|_1\.fastq\.gz|_1\.fq\.gz|_f1\.fq\.gz)$//' \
    | sort -u > "${output_list}"

  echo "[OK] ${output_list}: $(wc -l < "${output_list}") samples"
}

make_list "${BASE}/non_AGI/fastq" "${BASE}/non_AGI/sample_list.txt"
make_list "${BASE}/AGI/fastq" "${BASE}/AGI/sample_list.txt"
