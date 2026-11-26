#!/bin/bash
#SBATCH --job-name=fix_bam_CRR556748
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/log/fix_bam_CRR556748.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/log/fix_bam_CRR556748.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition

set -euo pipefail

source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
conda activate bam

BAM="/home/tcmhwd/microbiome/ICU_public/AGI/BAM_files/CRR556748/CRR556748_sorted.bam"

echo "BAM index 생성: CRR556748"
samtools index -@ "${SLURM_CPUS_PER_TASK}" "${BAM}"

echo "BAM flagstat 생성: CRR556748"
samtools flagstat -@ "${SLURM_CPUS_PER_TASK}" "${BAM}" > "$(dirname ${BAM})/CRR556748_flagstat.txt"

echo "완료: CRR556748"
