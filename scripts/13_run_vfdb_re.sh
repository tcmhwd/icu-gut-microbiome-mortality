#!/bin/bash
#SBATCH --job-name=vfdb_blast
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/vfdb_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/vfdb_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda environment
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate vfdb

########################
# 2. paths
########################
WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

REF_DIR="${WD}/ref"
VFDB_DB_DIR="${REF_DIR}/VFDB"
VFDB_DB_NAME="VFDB_setB_nt"
VFDB_DB_PREFIX="${VFDB_DB_DIR}/${VFDB_DB_NAME}"

# candidate FASTA names
VFDB_FASTA1="${VFDB_DB_PREFIX}.fasta"
VFDB_FASTA2="${VFDB_DB_PREFIX}.fa"
VFDB_FASTA3="${VFDB_DB_PREFIX}.fna"
VFDB_FASTA4="${VFDB_DB_PREFIX}.fas"

PROKKA_DIR="${ICU_DIR}/Prokka_result"
OUT_DIR="${ICU_DIR}/VFDB_result"
DREP_OUT="${ICU_DIR}/cohort_drep"
MAG_LIST="${DREP_OUT}/mag_list.txt"

mkdir -p "${OUT_DIR}"

########################
# 3. cutoffs
########################
PID_CUTOFF=80
EVALUE_CUTOFF="1e-5"
BITSCORE_CUTOFF=50
ALN_LEN_CUTOFF=50

########################
# 4. common settings
########################
THREADS="${SLURM_CPUS_PER_TASK:-16}"

OUTFMT_FIELDS="qseqid sseqid pident length evalue bitscore qlen slen qstart qend sstart send staxids stitle"
OUTFMT_STR="6 ${OUTFMT_FIELDS}"
HEADER=$(echo "${OUTFMT_FIELDS}" | tr ' ' '\t')

echo "[VFDB] threads=${THREADS}"
echo "[VFDB] DB_DIR=${VFDB_DB_DIR}"
echo "[VFDB] DB_PREFIX=${VFDB_DB_PREFIX}"
echo "[VFDB] PROKKA_DIR=${PROKKA_DIR}"
echo "[VFDB] OUT_DIR=${OUT_DIR}"
echo "[VFDB] Cutoffs: pident>=${PID_CUTOFF}, bitscore>=${BITSCORE_CUTOFF}, evalue<=${EVALUE_CUTOFF}, length>=${ALN_LEN_CUTOFF}"

########################
# 5. locate/build VFDB BLAST database
########################
DB_READY=0

if [[ -f "${VFDB_DB_PREFIX}.nin" || -f "${VFDB_DB_PREFIX}.00.nin" ]]; then
  DB_READY=1
  echo "[VFDB] Existing BLAST nucleotide DB detected."
fi

if [[ "${DB_READY}" -eq 0 ]]; then
  VFDB_FASTA=""

  for f in "${VFDB_FASTA1}" "${VFDB_FASTA2}" "${VFDB_FASTA3}" "${VFDB_FASTA4}"; do
    if [[ -s "${f}" ]]; then
      VFDB_FASTA="${f}"
      break
    fi
  done

  if [[ -n "${VFDB_FASTA}" ]]; then
    echo "[VFDB] BLAST DB not found. Building database from ${VFDB_FASTA}"
    makeblastdb \
      -in "${VFDB_FASTA}" \
      -dbtype nucl \
      -out "${VFDB_DB_PREFIX}" \
      -parse_seqids
    DB_READY=1
  fi
fi

if [[ "${DB_READY}" -eq 0 ]]; then
  echo "[ERROR] VFDB blastn database not found: ${VFDB_DB_PREFIX}" >&2
  echo "[ERROR] Also could not find source FASTA among:" >&2
  echo "        ${VFDB_FASTA1}" >&2
  echo "        ${VFDB_FASTA2}" >&2
  echo "        ${VFDB_FASTA3}" >&2
  echo "        ${VFDB_FASTA4}" >&2
  echo "[ERROR] Please place VFDB nucleotide FASTA there or build DB manually with makeblastdb." >&2
  exit 1
fi

########################
# 6. prepare input list
########################
RUN_LIST="${OUT_DIR}/vfdb_todo.list"
: > "${RUN_LIST}"

if [[ -s "${MAG_LIST}" ]]; then
  while IFS= read -r fa; do
    [[ -z "${fa}" ]] && continue
    mag_name="$(basename "${fa}" .fa)"
    ffn="${PROKKA_DIR}/${mag_name}/${mag_name}.ffn"
    if [[ -s "${ffn}" ]]; then
      echo "${ffn}" >> "${RUN_LIST}"
    else
      echo "[WARN] Missing Prokka FFN for ${mag_name}: ${ffn}" >&2
    fi
  done < "${MAG_LIST}"
else
  echo "[WARN] MAG_LIST not found; scanning all Prokka .ffn files."
  find "${PROKKA_DIR}" -mindepth 2 -maxdepth 2 -type f -name "*.ffn" | sort > "${RUN_LIST}"
fi

if [[ ! -s "${RUN_LIST}" ]]; then
  echo "[ERROR] No .ffn files found for VFDB analysis." >&2
  exit 1
fi

TOTAL=$(wc -l < "${RUN_LIST}")
echo "[VFDB] total targets=${TOTAL}"

########################
# 7. run BLASTn against VFDB
########################
DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS= read -r ffn_file; do
  [[ -z "${ffn_file}" ]] && continue

  bin_name="$(basename "${ffn_file}" .ffn)"
  raw_out="${OUT_DIR}/${bin_name}_vfdb_raw.tsv"
  filt_out="${OUT_DIR}/${bin_name}_vfdb_filtered.tsv"
  best_out="${OUT_DIR}/${bin_name}_vfdb_besthit.tsv"

  if [[ ! -s "${ffn_file}" ]]; then
    echo "[WARN] Missing input file: ${ffn_file}" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  if [[ -s "${filt_out}" && -s "${best_out}" ]]; then
    echo "[SKIP] ${bin_name}: filtered and besthit results already exist"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "[RUN ] ${bin_name}"
  echo "       input : ${ffn_file}"

  if blastn \
    -query "${ffn_file}" \
    -db "${VFDB_DB_PREFIX}" \
    -out "${raw_out}" \
    -outfmt "${OUTFMT_STR}" \
    -evalue "${EVALUE_CUTOFF}" \
    -perc_identity "${PID_CUTOFF}" \
    -num_threads "${THREADS}" \
    -max_hsps 1
  then
    {
      echo -e "${HEADER}"
      awk -F'\t' -v OFS='\t' -v min_len="${ALN_LEN_CUTOFF}" -v min_bit="${BITSCORE_CUTOFF}" '
        $4+0 >= min_len && $6+0 >= min_bit { print }
      ' "${raw_out}"
    } > "${filt_out}"

    {
      echo -e "${HEADER}"
      awk -F'\t' -v OFS='\t' '
        NR==1 { next }
        {
          key=$1
          if (!(key in best) || $6+0 > best_bit[key] || ($6+0 == best_bit[key] && $4+0 > best_len[key])) {
            best[key]=$0
            best_bit[key]=$6+0
            best_len[key]=$4+0
          }
        }
        END {
          for (k in best) print best[k]
        }
      ' "${filt_out}" | sort -k1,1
    } > "${best_out}"

    echo "[DONE] ${bin_name}"
    DONE_COUNT=$((DONE_COUNT + 1))
  else
    echo "[FAIL] ${bin_name}" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done < "${RUN_LIST}"

echo "[SUMMARY] done=${DONE_COUNT}, skipped=${SKIP_COUNT}, failed=${FAIL_COUNT}"
echo "[INFO] VFDB BLASTn analysis completed"
