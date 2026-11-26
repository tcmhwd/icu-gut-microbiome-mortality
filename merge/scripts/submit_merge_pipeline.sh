#!/bin/bash
# submit_merge_pipeline.sh — merge/ 파이프라인 01~06을 SLURM 의존성 체인으로 제출
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/home/tcmhwd/microbiome/ICU_public/merge/output"
mkdir -p "${OUTPUT_DIR}" /home/tcmhwd/microbiome/ICU_public/logs

# 이전 실행이 남긴 스테이징 디렉토리를 정리한다. 남아있으면 04a/03이 오래된
# 심링크·중간 파일을 재사용해 잘못된 결과를 낼 수 있다.
rm -rf "${OUTPUT_DIR}/_humann3_staging" "${OUTPUT_DIR}/_icu_resistome_adapted"

jid01=$(sbatch --parsable "${SCRIPT_DIR}/01_build_metadata.sh")
jid02=$(sbatch --parsable --dependency=afterok:${jid01} "${SCRIPT_DIR}/02_metaphlan3_merge.sh")
jid03=$(sbatch --parsable --dependency=afterok:${jid01} "${SCRIPT_DIR}/03_humann3_merge.sh")
jid04a=$(sbatch --parsable --dependency=afterok:${jid01} "${SCRIPT_DIR}/04a_resistome_adapter.sh")
jid04=$(sbatch --parsable --dependency=afterok:${jid01}:${jid04a} "${SCRIPT_DIR}/04_mag_summary.sh")
jid05=$(sbatch --parsable --dependency=afterok:${jid04} "${SCRIPT_DIR}/05_arcA_but_ratio.sh")
jid06=$(sbatch --parsable --dependency=afterok:${jid02}:${jid03}:${jid04}:${jid05} "${SCRIPT_DIR}/06_merge_for_R.sh")

echo "제출된 작업:"
echo "  01_build_metadata:     ${jid01}"
echo "  02_metaphlan3_merge:   ${jid02}"
echo "  03_humann3_merge:      ${jid03}"
echo "  04a_resistome_adapter: ${jid04a}"
echo "  04_mag_summary:        ${jid04}"
echo "  05_arcA_but_ratio:     ${jid05}"
echo "  06_merge_for_R:        ${jid06}"
echo ""
echo "상태 확인: squeue -u tcmhwd"
echo "최종 산출물: /home/tcmhwd/microbiome/ICU_public/merge/output/R_ready/"
