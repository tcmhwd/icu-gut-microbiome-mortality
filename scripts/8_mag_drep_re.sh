#!/bin/bash
#SBATCH --job-name=drep_cohort
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/drep_cohort_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/drep_cohort_%j.err
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=125G
#SBATCH --partition=cpu_partition

set -euo pipefail

########################
# 1. conda 환경 활성화
########################
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate drep

########################
# 2. 경로 설정
########################
WD="/home/tcmhwd/microbiome"
ICU_DIR="${WD}/ICU_public"

COHORT_MAG_DIR="${ICU_DIR}/Cohort_MAGs"
DREP_OUT="${ICU_DIR}/cohort_drep"

THREADS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${COHORT_MAG_DIR}" "${DREP_OUT}"

########################
# 3. 이전 결과 정리
########################
rm -rf "${COHORT_MAG_DIR}"
mkdir -p "${COHORT_MAG_DIR}"

rm -rf "${DREP_OUT}"
mkdir -p "${DREP_OUT}"

########################
# 4. CheckM2 filtered bins 수집 함수
########################
GENOME_LIST="${DREP_OUT}/genome_list.txt"
GENOME_INFO="${DREP_OUT}/genomeInfo.csv"

: > "${GENOME_LIST}"
echo "genome,completeness,contamination" > "${GENOME_INFO}"

COLLECTED=0

# collect_mags <sample_list_file> <checkm2_base_dir>
#   sample_list_file : 한 줄에 sample 이름 하나 (탭/공백 포함 시 첫 필드만 사용)
#   checkm2_base_dir : {base_dir}/{sample_name}/filtered_bins/ 구조 가정
collect_mags() {
    local sample_list="$1"
    local checkm2_base="$2"

    while IFS=$'\n' read -r raw; do
        sample_name="$(printf "%s" "$raw" | awk '{print $1}' | tr -d '\r')"
        [[ -z "${sample_name}" ]] && continue

        local fb_dir="${checkm2_base}/${sample_name}/filtered_bins"
        local qr="${checkm2_base}/${sample_name}/quality_report.tsv"

        if [[ ! -d "${fb_dir}" ]]; then
            echo "[WARN] filtered_bins not found: ${fb_dir}" >&2
            continue
        fi

        if [[ ! -s "${qr}" ]]; then
            echo "[WARN] quality_report.tsv not found or empty: ${qr}" >&2
            continue
        fi

        shopt -s nullglob
        for bin in "${fb_dir}"/*.fa; do
            local base stem dest_base dest
            base="$(basename "${bin}")"
            stem="${base%.fa}"
            dest_base="${sample_name}_${base}"
            dest="${COHORT_MAG_DIR}/${dest_base}"

            # 파일명 충돌 시 md5 해시로 구분
            if [[ -e "${dest}" ]]; then
                local hash
                hash="$(md5sum "${bin}" | awk '{print $1}')"
                dest_base="${sample_name}_${hash}_${base}"
                dest="${COHORT_MAG_DIR}/${dest_base}"
            fi

            cp -f "${bin}" "${dest}"
            echo "${dest}" >> "${GENOME_LIST}"

            local comp cont
            comp="$(awk -F'\t' -v t="${stem}" 'NR>1 && $1==t {print $2; exit}' "${qr}")"
            cont="$(awk -F'\t' -v t="${stem}" 'NR>1 && $1==t {print $3; exit}' "${qr}")"

            if [[ -z "${comp}" || -z "${cont}" ]]; then
                echo "[WARN] Quality info missing for ${stem} in ${qr}" >&2
            else
                echo "${dest_base},${comp},${cont}" >> "${GENOME_INFO}"
            fi

            ((COLLECTED++)) || true
        done
        shopt -u nullglob
    done < "${sample_list}"
}

########################
# 5. 각 소스에서 MAG 수집
########################
echo "[INFO] === Source 1: ICU_public non_AGI (SRR*) ==="
collect_mags \
    "${ICU_DIR}/non_AGI/sample_list.txt" \
    "${ICU_DIR}/non_AGI/CheckM2_result"

echo "[INFO] === Source 2: ICU_public AGI (CRR*) ==="
collect_mags \
    "${ICU_DIR}/AGI/sample_list.txt" \
    "${ICU_DIR}/AGI/CheckM2_result"

echo "[INFO] === Source 3: /data/tcmhwd/MAG (ICU_sample_list.txt A column) ==="
# ICU_sample_list.txt의 첫 번째 컬럼(A column)만 사용
awk '{print $1}' "${ICU_DIR}/ICU_sample_list.txt" > /tmp/icu_mag_samples_$$.txt
collect_mags \
    /tmp/icu_mag_samples_$$.txt \
    "/data/tcmhwd/MAG/CheckM2_result"
rm -f /tmp/icu_mag_samples_$$.txt

########################
# 6. 수집 결과 확인
########################
if [[ "${COLLECTED}" -eq 0 ]]; then
    echo "[ERROR] No MAGs collected into ${COHORT_MAG_DIR}. Abort."
    exit 1
fi

echo "[INFO] Total collected MAGs: ${COLLECTED}"
echo "[INFO] Genome list : ${GENOME_LIST}"
echo "[INFO] Genome info : ${GENOME_INFO}"

########################
# 7. dRep 실행
########################
echo "[INFO] Running dRep dereplicate..."

dRep dereplicate "${DREP_OUT}" \
    -g "${GENOME_LIST}" \
    --genomeInfo "${GENOME_INFO}" \
    -p "${THREADS}" \
    --S_algorithm fastANI \
    --P_ani 0.90 \
    --S_ani 0.99 \
    -l 50000 \
    -comp 50 \
    -con 10

echo "[INFO] dRep finished."

########################
# 8. dereplicated MAG 리스트 생성
########################
DEREP_DIR="${DREP_OUT}/dereplicated_genomes"
MAG_LIST="${DREP_OUT}/mag_list.txt"

if compgen -G "${DEREP_DIR}/*.fa" > /dev/null; then
    ls "${DEREP_DIR}"/*.fa | sort > "${MAG_LIST}"
    N=$(wc -l < "${MAG_LIST}")
    echo "[INFO] mag_list.txt created: ${MAG_LIST} (N=${N})"
else
    echo "[WARN] No dereplicated genomes found in ${DEREP_DIR}. mag_list.txt not created."
fi

echo "[DONE] dRep 중복 제거 완료. 결과: ${DREP_OUT}"
