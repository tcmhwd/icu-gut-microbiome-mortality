#!/bin/bash
#SBATCH --job-name=rpkm_contig
#SBATCH --output=/home/tcmhwd/microbiome/log/rpkm_contig_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/log/rpkm_contig_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu_partition

# 개요:
#   RGI로 탐지된 저항성 유전자에 대해 유전자별 RPKM을 계산합니다.
#
#   흐름:
#     RGI .txt   → 저항성 유전자 locus_tag 추출 (Perfect / Strict)
#     Prokka .gff → 해당 유전자의 contig + 좌표 조회
#     BAM        → 유전자 영역 mapped read 수 계산
#     RPKM = (reads × 1e9) / (gene_length × total_mapped_reads)
#
# 주의:
#   total_mapped_reads와 gene별 mapped_reads는 반드시 같은 filtering rule을 써야 함
#
# 출력:
#   pd/RPKM_rgi_result/{sample}/{sample}_rgi_rpkm.tsv

set -euo pipefail

export PATH="/data/tcmhwd/miniconda3/envs/bam/bin:$PATH"

# ── 경로 설정 ──────────────────────────────────────────────────
WD="/home/tcmhwd/microbiome"
project_id="ICU_public/AGI"

SAMPLE_LIST="${WD}/${project_id}/sample_list.txt"
RGI_DIR="${WD}/${project_id}/RGI_contig_result"
GFF_DIR="${WD}/${project_id}/Prokka_contig_result"
BAM_DIR="${WD}/${project_id}/BAM_files"
OUT_DIR="${WD}/${project_id}/RPKM_rgi_result"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

# exclude flags:
# 4    unmapped
# 256  secondary
# 2048 supplementary
# 합계 = 2308
#
# duplicate(1024)까지 제외하려면 3332(=2308+1024)로 바꾸면 됨.
# metagenome에서는 duplicate 처리 해석이 애매할 수 있어 기본은 2308 추천.
EXCLUDE_FLAGS=2308

mkdir -p "${OUT_DIR}"

# ── 샘플명 추출 ────────────────────────────────────────────────
sample=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}" | tr -d '\r')

if [[ -z "${sample}" ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} 에 해당하는 샘플이 없습니다."
    exit 1
fi

RGI_TXT="${RGI_DIR}/${sample}/${sample}_rgi.txt"
GFF="${GFF_DIR}/${sample}/${sample}.gff"
BAM="${BAM_DIR}/${sample}/${sample}_sorted.bam"
SAMPLE_OUT="${OUT_DIR}/${sample}"
OUT_TSV="${SAMPLE_OUT}/${sample}_rgi_rpkm.tsv"
TOTAL_MAPPED_TXT="${SAMPLE_OUT}/${sample}_total_mapped_reads.txt"

echo "[INFO] sample=${sample}"
echo "[INFO] rgi=${RGI_TXT}"
echo "[INFO] gff=${GFF}"
echo "[INFO] bam=${BAM}"
echo "[INFO] exclude_flags=${EXCLUDE_FLAGS}"

# ── 입력 파일 확인 ─────────────────────────────────────────────
for f in "${RGI_TXT}" "${GFF}" "${BAM}"; do
    if [[ ! -s "${f}" ]]; then
        echo "[ERROR] 파일 없음: ${f}"
        exit 1
    fi
done

# ── 완료 여부 확인 ─────────────────────────────────────────────
if [[ -s "${OUT_TSV}" ]]; then
    echo "[SKIP] RPKM 이미 완료됨: ${sample}"
    exit 0
fi

mkdir -p "${SAMPLE_OUT}"

# ── BAM 인덱스 확인 ────────────────────────────────────────────
if [[ ! -f "${BAM}.bai" ]]; then
    echo "[INFO] BAM 인덱스 생성 중..."
    samtools index -@ "${THREADS}" "${BAM}"
fi

# ── total mapped reads 계산 ────────────────────────────────────
# 분모와 분자의 정의를 맞추기 위해 flagstat 텍스트 파싱 대신 직접 계산
echo "[INFO] total mapped reads 계산 중..."
samtools view -@ "${THREADS}" -c -F "${EXCLUDE_FLAGS}" "${BAM}" > "${TOTAL_MAPPED_TXT}"

# ── RPKM 계산 (Python) ────────────────────────────────────────
python3 - \
    "${RGI_TXT}" \
    "${GFF}" \
    "${TOTAL_MAPPED_TXT}" \
    "${BAM}" \
    "${OUT_TSV}" \
    "${sample}" \
    "${EXCLUDE_FLAGS}" \
<<'PYEOF'
import sys
import re
import subprocess

rgi_path, gff_path, total_mapped_path, bam_path, out_path, sample_id, exclude_flags = sys.argv[1:]
exclude_flags = str(exclude_flags)

# ── 1. total mapped reads ──────────────────────────────────────
with open(total_mapped_path) as fh:
    text = fh.read().strip()
    total_mapped = int(text) if text else 0

if total_mapped == 0:
    print(f"[WARN] total_mapped=0 for {sample_id}. RPKM will be 0.")

# ── 2. RGI 파싱: Perfect / Strict 유전자만 ────────────────────
def find_col(idx_map, *aliases):
    for name in aliases:
        if name in idx_map:
            return idx_map[name]
    raise KeyError(f"Missing required column. Tried: {aliases}")

resistance_genes = {}
duplicate_orf_hits = 0

with open(rgi_path) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    idx = {col: i for i, col in enumerate(header)}

    i_cutoff = find_col(idx, "Cut_Off", "Cut Off")
    i_orf    = find_col(idx, "ORF_ID", "ORF ID")
    i_best   = find_col(idx, "Best_Hit_ARO", "Best Hit ARO")
    i_drug   = find_col(idx, "Drug Class", "Drug_Class")
    i_mech   = find_col(idx, "Resistance Mechanism", "Resistance_Mechanism")
    i_family = find_col(idx, "AMR Gene Family", "AMR_Gene_Family")

    for line in fh:
        cols = line.rstrip("\n").split("\t")
        if len(cols) < len(header):
            continue

        cut_off = cols[i_cutoff]
        if cut_off not in ("Perfect", "Strict"):
            continue

        locus_tag = cols[i_orf].split()[0].strip()
        if not locus_tag:
            continue

        # 같은 ORF_ID에 hit가 여러 개면 첫 번째 것 유지
        # 보통 RGI output은 상위 hit가 앞쪽에 오는 경우가 많아 이 편이 더 안전함
        if locus_tag in resistance_genes:
            duplicate_orf_hits += 1
            continue

        resistance_genes[locus_tag] = {
            "best_hit_aro":         cols[i_best],
            "drug_class":           cols[i_drug],
            "resistance_mechanism": cols[i_mech],
            "amr_gene_family":      cols[i_family],
            "cut_off":              cut_off,
        }

print(f"[INFO] RGI resistance genes (Perfect+Strict): {len(resistance_genes)}")
print(f"[INFO] duplicate ORF hits skipped: {duplicate_orf_hits}")

if not resistance_genes:
    print(f"[WARN] No Perfect/Strict resistance genes found in {rgi_path}")
    with open(out_path, "w") as out:
        out.write(
            "locus_tag\tcontig\tstart\tend\tgene_length\t"
            "mapped_reads\trpkm\tbest_hit_aro\tdrug_class\t"
            "resistance_mechanism\tamr_gene_family\tcut_off\n"
        )
    sys.exit(0)

# ── 3. Prokka GFF 파싱: CDS 좌표 추출 ────────────────────────
gene_coords = {}

with open(gff_path) as fh:
    for line in fh:
        if line.startswith("#"):
            continue

        cols = line.rstrip("\n").split("\t")
        if len(cols) < 9 or cols[2] != "CDS":
            continue

        attrs = cols[8]
        m = re.search(r"locus_tag=([^;]+)", attrs)
        if not m:
            m = re.search(r"ID=([^;]+)", attrs)
        if not m:
            continue

        locus_tag = m.group(1)
        if locus_tag in resistance_genes:
            gene_coords[locus_tag] = (cols[0], int(cols[3]), int(cols[4]))

print(f"[INFO] Resistance genes found in GFF: {len(gene_coords)}/{len(resistance_genes)}")

missing_in_gff = sorted(set(resistance_genes) - set(gene_coords))
if missing_in_gff:
    print(f"[WARN] {len(missing_in_gff)} resistance genes were not found in GFF.")

# ── 4. 유전자별 mapped read count ─────────────────────────────
def count_reads(bam, contig, start, end, exclude_flags):
    region = f"{contig}:{start}-{end}"
    result = subprocess.run(
        ["samtools", "view", "-c", "-F", exclude_flags, bam, region],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"samtools view failed for region {region}: {result.stderr.strip()}"
        )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0

# ── 5. RPKM 계산 및 출력 ──────────────────────────────────────
rows = []

for locus_tag, (contig, start, end) in gene_coords.items():
    gene_length = end - start + 1
    reads = count_reads(bam_path, contig, start, end, exclude_flags)
    rpkm = (reads * 1e9) / (gene_length * total_mapped) if total_mapped > 0 else 0.0
    info = resistance_genes[locus_tag]

    rows.append({
        "locus_tag":            locus_tag,
        "contig":               contig,
        "start":                start,
        "end":                  end,
        "gene_length":          gene_length,
        "mapped_reads":         reads,
        "rpkm":                 rpkm,
        "best_hit_aro":         info["best_hit_aro"],
        "drug_class":           info["drug_class"],
        "resistance_mechanism": info["resistance_mechanism"],
        "amr_gene_family":      info["amr_gene_family"],
        "cut_off":              info["cut_off"],
    })

rows.sort(key=lambda x: x["rpkm"], reverse=True)

out_cols = [
    "locus_tag", "contig", "start", "end", "gene_length",
    "mapped_reads", "rpkm", "best_hit_aro", "drug_class",
    "resistance_mechanism", "amr_gene_family", "cut_off"
]

with open(out_path, "w") as out:
    out.write("\t".join(out_cols) + "\n")
    for row in rows:
        out.write("\t".join(str(row[c]) for c in out_cols) + "\n")

print(f"[INFO] total_mapped_reads={total_mapped}")
print(f"[INFO] output rows={len(rows)}")
print(f"[INFO] output={out_path}")
PYEOF

echo "[DONE] RGI RPKM 계산 완료: ${sample} → ${OUT_TSV}"
