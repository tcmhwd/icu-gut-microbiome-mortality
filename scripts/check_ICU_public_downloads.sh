#!/usr/bin/env bash
#SBATCH --job-name=check_ICU_download
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/check_download_%j.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/check_download_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu_partition

set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"
CHECK_DIR="${BASE}/download_check"
mkdir -p "${CHECK_DIR}"

NON_AGI_URLS="${BASE}/non_AGI/non_AGI_fastq_urls.txt"
AGI_URLS="${BASE}/AGI/AGI_fastq_urls_https.txt"

NON_AGI_FASTQ="${BASE}/non_AGI/fastq"
AGI_FASTQ="${BASE}/AGI/fastq"

THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "=========================================="
echo "[INFO] ICU public FASTQ download check"
echo "[INFO] Date: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] BASE: ${BASE}"
echo "=========================================="

echo
echo "=== 1. URL count vs downloaded .gz count ==="

NON_AGI_EXPECTED=$(grep -cve '^[[:space:]]*$' "${NON_AGI_URLS}")
AGI_EXPECTED=$(grep -cve '^[[:space:]]*$' "${AGI_URLS}")

NON_AGI_DOWNLOADED=$(find "${NON_AGI_FASTQ}" -maxdepth 1 -type f -name "*.gz" | wc -l)
AGI_DOWNLOADED=$(find "${AGI_FASTQ}" -maxdepth 1 -type f -name "*.gz" | wc -l)

echo "[non-AGI] Expected URLs:      ${NON_AGI_EXPECTED}"
echo "[non-AGI] Downloaded .gz:     ${NON_AGI_DOWNLOADED}"
echo "[AGI]     Expected URLs:      ${AGI_EXPECTED}"
echo "[AGI]     Downloaded .gz:     ${AGI_DOWNLOADED}"

cat > "${CHECK_DIR}/download_count_summary.tsv" <<EOF
dataset	expected_urls	downloaded_gz
non_AGI	${NON_AGI_EXPECTED}	${NON_AGI_DOWNLOADED}
AGI	${AGI_EXPECTED}	${AGI_DOWNLOADED}
EOF

echo
echo "=== 2. Missing and zero-byte file check ==="

python3 - <<'PY'
from pathlib import Path
from urllib.parse import urlparse

BASE = Path("/home/tcmhwd/microbiome/ICU_public")
CHECK = BASE / "download_check"
CHECK.mkdir(exist_ok=True)

jobs = [
    {
        "name": "non_AGI",
        "url_file": BASE / "non_AGI" / "non_AGI_fastq_urls.txt",
        "fastq_dir": BASE / "non_AGI" / "fastq",
    },
    {
        "name": "AGI",
        "url_file": BASE / "AGI" / "AGI_fastq_urls_https.txt",
        "fastq_dir": BASE / "AGI" / "fastq",
    },
]

summary_lines = ["dataset\texpected_urls\tpresent_files\tmissing_files\tzero_byte_files"]

for job in jobs:
    name = job["name"]
    url_file = job["url_file"]
    fastq_dir = job["fastq_dir"]

    urls = [x.strip() for x in url_file.read_text().splitlines() if x.strip()]

    missing = []
    zero_byte = []
    present = []

    for url in urls:
        fname = Path(urlparse(url).path).name
        fpath = fastq_dir / fname

        if not fpath.exists():
            missing.append(url)
        elif fpath.stat().st_size == 0:
            zero_byte.append(url)
        else:
            present.append(url)

    (CHECK / f"{name}_missing_urls.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    (CHECK / f"{name}_zero_byte_urls.txt").write_text("\n".join(zero_byte) + ("\n" if zero_byte else ""))

    summary_lines.append(
        f"{name}\t{len(urls)}\t{len(present)}\t{len(missing)}\t{len(zero_byte)}"
    )

    print(f"[{name}]")
    print(f"  Expected URLs:   {len(urls)}")
    print(f"  Present files:   {len(present)}")
    print(f"  Missing files:   {len(missing)}")
    print(f"  Zero-byte files: {len(zero_byte)}")
    print(f"  Missing URL list:   {CHECK / f'{name}_missing_urls.txt'}")
    print(f"  Zero-byte URL list: {CHECK / f'{name}_zero_byte_urls.txt'}")
    print()

(CHECK / "missing_zero_byte_summary.tsv").write_text("\n".join(summary_lines) + "\n")
PY

echo
echo "=== 3. gzip integrity check ==="

echo "[INFO] Checking non-AGI gzip files with ${THREADS} parallel processes..."
find "${NON_AGI_FASTQ}" -maxdepth 1 -type f -name "*.gz" -print0 \
  | xargs -0 -n 1 -P "${THREADS}" sh -c '
      gzip -t "$1" 2>/dev/null || echo "$1"
    ' sh \
  > "${CHECK_DIR}/non_AGI_corrupt_gzip_files.txt"

echo "[INFO] Checking AGI gzip files with ${THREADS} parallel processes..."
find "${AGI_FASTQ}" -maxdepth 1 -type f -name "*.gz" -print0 \
  | xargs -0 -n 1 -P "${THREADS}" sh -c '
      gzip -t "$1" 2>/dev/null || echo "$1"
    ' sh \
  > "${CHECK_DIR}/AGI_corrupt_gzip_files.txt"

NON_AGI_CORRUPT=$(wc -l < "${CHECK_DIR}/non_AGI_corrupt_gzip_files.txt" | tr -d ' ')
AGI_CORRUPT=$(wc -l < "${CHECK_DIR}/AGI_corrupt_gzip_files.txt" | tr -d ' ')

echo "[non-AGI] Corrupt gzip files: ${NON_AGI_CORRUPT}"
echo "[AGI]     Corrupt gzip files: ${AGI_CORRUPT}"

cat > "${CHECK_DIR}/gzip_integrity_summary.tsv" <<EOF
dataset	corrupt_gzip_files
non_AGI	${NON_AGI_CORRUPT}
AGI	${AGI_CORRUPT}
EOF

echo
echo "=== 4. Create retry URL lists for missing, zero-byte, and corrupt files ==="

python3 - <<'PY'
from pathlib import Path
from urllib.parse import urlparse

BASE = Path("/home/tcmhwd/microbiome/ICU_public")
CHECK = BASE / "download_check"

jobs = [
    {
        "name": "non_AGI",
        "url_file": BASE / "non_AGI" / "non_AGI_fastq_urls.txt",
        "fastq_dir": BASE / "non_AGI" / "fastq",
        "corrupt_file": CHECK / "non_AGI_corrupt_gzip_files.txt",
    },
    {
        "name": "AGI",
        "url_file": BASE / "AGI" / "AGI_fastq_urls_https.txt",
        "fastq_dir": BASE / "AGI" / "fastq",
        "corrupt_file": CHECK / "AGI_corrupt_gzip_files.txt",
    },
]

for job in jobs:
    name = job["name"]
    url_file = job["url_file"]
    fastq_dir = job["fastq_dir"]
    corrupt_file = job["corrupt_file"]

    urls = [x.strip() for x in url_file.read_text().splitlines() if x.strip()]

    filename_to_url = {}
    for url in urls:
        fname = Path(urlparse(url).path).name
        filename_to_url[fname] = url

    retry_urls = []

    # Missing or zero-byte
    for url in urls:
        fname = Path(urlparse(url).path).name
        fpath = fastq_dir / fname
        if (not fpath.exists()) or fpath.stat().st_size == 0:
            retry_urls.append(url)

    # Corrupt gzip
    corrupt_paths = []
    if corrupt_file.exists():
        corrupt_paths = [x.strip() for x in corrupt_file.read_text().splitlines() if x.strip()]

    for p in corrupt_paths:
        fname = Path(p).name
        url = filename_to_url.get(fname)
        if url:
            retry_urls.append(url)

    retry_urls = sorted(dict.fromkeys(retry_urls))

    out = CHECK / f"{name}_retry_urls.txt"
    out.write_text("\n".join(retry_urls) + ("\n" if retry_urls else ""))

    print(f"[{name}] Retry URLs: {len(retry_urls)}")
    print(f"[{name}] Retry URL list: {out}")
PY

echo
echo "=== 5. Final summary ==="

NON_AGI_MISSING=$(wc -l < "${CHECK_DIR}/non_AGI_missing_urls.txt" | tr -d ' ')
AGI_MISSING=$(wc -l < "${CHECK_DIR}/AGI_missing_urls.txt" | tr -d ' ')

NON_AGI_ZERO=$(wc -l < "${CHECK_DIR}/non_AGI_zero_byte_urls.txt" | tr -d ' ')
AGI_ZERO=$(wc -l < "${CHECK_DIR}/AGI_zero_byte_urls.txt" | tr -d ' ')

NON_AGI_RETRY=$(wc -l < "${CHECK_DIR}/non_AGI_retry_urls.txt" | tr -d ' ')
AGI_RETRY=$(wc -l < "${CHECK_DIR}/AGI_retry_urls.txt" | tr -d ' ')

cat > "${CHECK_DIR}/final_download_check_summary.tsv" <<EOF
dataset	expected_urls	downloaded_gz	missing	zero_byte	corrupt_gzip	retry_urls
non_AGI	${NON_AGI_EXPECTED}	${NON_AGI_DOWNLOADED}	${NON_AGI_MISSING}	${NON_AGI_ZERO}	${NON_AGI_CORRUPT}	${NON_AGI_RETRY}
AGI	${AGI_EXPECTED}	${AGI_DOWNLOADED}	${AGI_MISSING}	${AGI_ZERO}	${AGI_CORRUPT}	${AGI_RETRY}
EOF

cat "${CHECK_DIR}/final_download_check_summary.tsv"

TOTAL_PROBLEMS=$((NON_AGI_MISSING + AGI_MISSING + NON_AGI_ZERO + AGI_ZERO + NON_AGI_CORRUPT + AGI_CORRUPT))

echo
echo "[INFO] Check output directory: ${CHECK_DIR}"
echo "[INFO] Date: $(date)"

if [[ "${TOTAL_PROBLEMS}" -eq 0 ]]; then
  echo "[OK] All expected FASTQ.gz files are present and passed gzip integrity check."
  exit 0
else
  echo "[WARNING] Some files are missing, zero-byte, or corrupt."
  echo "[WARNING] Retry lists were created:"
  echo "  ${CHECK_DIR}/non_AGI_retry_urls.txt"
  echo "  ${CHECK_DIR}/AGI_retry_urls.txt"
  exit 2
fi
