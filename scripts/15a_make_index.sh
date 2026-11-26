#!/bin/bash
#SBATCH --job-name=bwa_index
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/bwa_index_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/bwa_index_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --partition=gpu_partition
#SBATCH --nodelist=node1

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate bam

MAG_DIR="/home/tcmhwd/microbiome/ICU_public/cohort_drep/dereplicated_genomes"
REF="/home/tcmhwd/microbiome/ICU_public/cohort_drep/all_mags.fa"

# contig 이름을 {MAG명}~{원래contig명} 으로 prefixing해서 중복 방지
# CoverM --separator "~" 와 대응
echo "[INFO] Concatenating with MAG name prefix → ${REF}"
rm -f "${REF}"
for fa in "${MAG_DIR}"/*.fa; do
  mag=$(basename "${fa}" .fa)
  awk -v mag="${mag}" '/^>/{print ">" mag "~" substr($0,2); next} {print}' "${fa}"
done > "${REF}"
echo "[INFO] Reference size: $(du -sh ${REF} | cut -f1)"

# 불완전한 인덱스 파일 제거 후 재생성
echo "[INFO] Removing any partial index files..."
rm -f "${REF}.amb" "${REF}.ann" "${REF}.bwt" "${REF}.pac" "${REF}.sa"

echo "[INFO] Building BWA index..."
bwa index "${REF}"

echo "[DONE] BWA index ready: ${REF}.bwt"
