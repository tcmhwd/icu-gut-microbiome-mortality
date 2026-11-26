#!/usr/bin/env bash
set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"
PUBLIC_CSV="${1:-publicdata.csv}"
AGI_CSV="${2:-AGI_public.csv}"

SCRIPT_DIR="${BASE}/scripts"
META_DIR="${BASE}/metadata"
NON_AGI_DIR="${BASE}/non_AGI"
AGI_DIR="${BASE}/AGI"
LOG_DIR="${BASE}/logs"

mkdir -p "${SCRIPT_DIR}" "${META_DIR}" "${NON_AGI_DIR}/fastq" "${AGI_DIR}/fastq" "${LOG_DIR}"

if [[ ! -s "${PUBLIC_CSV}" ]]; then
  echo "[ERROR] Cannot find ${PUBLIC_CSV}"
  echo "Usage: bash setup_ICU_public_download.sh /path/to/publicdata.csv /path/to/AGI_public.csv"
  exit 1
fi

if [[ ! -s "${AGI_CSV}" ]]; then
  echo "[ERROR] Cannot find ${AGI_CSV}"
  echo "Usage: bash setup_ICU_public_download.sh /path/to/publicdata.csv /path/to/AGI_public.csv"
  exit 1
fi

cp -f "${PUBLIC_CSV}" "${META_DIR}/publicdata.csv"
cp -f "${AGI_CSV}" "${META_DIR}/AGI_public.csv"

python3 - <<'PY'
from pathlib import Path
import csv
import re

BASE = Path("/home/tcmhwd/microbiome/ICU_public")
META = BASE / "metadata"

public_csv = META / "publicdata.csv"
agi_csv = META / "AGI_public.csv"

non_agi_list = BASE / "non_AGI" / "non_AGI_fastq_urls.txt"
agi_ftp_list = BASE / "AGI" / "AGI_fastq_urls_ftp.txt"
agi_https_list = BASE / "AGI" / "AGI_fastq_urls_https.txt"

non_agi_urls = []

with public_csv.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    if "fastq_ftp" not in reader.fieldnames:
        raise SystemExit("[ERROR] publicdata.csv must contain a fastq_ftp column.")

    for row in reader:
        value = (row.get("fastq_ftp") or "").strip()
        if not value:
            continue

        for url in value.split(";"):
            url = url.strip()
            if not url:
                continue

            if not re.match(r"^[A-Za-z]+://", url):
                url = "https://" + url

            non_agi_urls.append(url)

non_agi_urls = sorted(dict.fromkeys(non_agi_urls))
non_agi_list.write_text("\n".join(non_agi_urls) + ("\n" if non_agi_urls else ""))

agi_ftp_urls = []

with agi_csv.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    if "Download_path" not in reader.fieldnames:
        raise SystemExit("[ERROR] AGI_public.csv must contain a Download_path column.")

    for row in reader:
        value = (row.get("Download_path") or "").strip()
        if not value:
            continue

        for url in value.split("|"):
            url = url.strip()
            if not url:
                continue

            agi_ftp_urls.append(url)

agi_ftp_urls = sorted(dict.fromkeys(agi_ftp_urls))
agi_ftp_list.write_text("\n".join(agi_ftp_urls) + ("\n" if agi_ftp_urls else ""))

agi_https_urls = []

for url in agi_ftp_urls:
    if url.startswith("ftp://download.big.ac.cn/"):
        url = url.replace("ftp://download.big.ac.cn/", "https://download.cncb.ac.cn/")
    agi_https_urls.append(url)

agi_https_urls = sorted(dict.fromkeys(agi_https_urls))
agi_https_list.write_text("\n".join(agi_https_urls) + ("\n" if agi_https_urls else ""))

print("[OK] URL list files were created")
print(f"non-AGI FASTQ URLs: {len(non_agi_urls)}")
print(f"AGI FASTQ URLs, FTP: {len(agi_ftp_urls)}")
print(f"AGI FASTQ URLs, HTTPS: {len(agi_https_urls)}")
PY

cat > "${SCRIPT_DIR}/download_ICU_public_array.sh" <<'SH'
#!/usr/bin/env bash
#SBATCH --job-name=ICU_public_download
#SBATCH --output=/home/tcmhwd/microbiome/ICU_public/logs/download_%A_%a.out
#SBATCH --error=/home/tcmhwd/microbiome/ICU_public/logs/download_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"

if [[ -z "${LIST:-}" ]]; then
  echo "[ERROR] LIST is not set."
  exit 1
fi

if [[ -z "${OUTDIR:-}" ]]; then
  echo "[ERROR] OUTDIR is not set."
  exit 1
fi

mkdir -p "${OUTDIR}" "${BASE}/logs"

index="${SLURM_ARRAY_TASK_ID:-1}"
target_url="$(sed -n "${index}p" "${LIST}" | tr -d '\r')"

if [[ -z "${target_url}" ]]; then
  echo "[ERROR] Empty URL at line ${index} in ${LIST}"
  exit 1
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[INFO] Array task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "[INFO] URL list: ${LIST}"
echo "[INFO] Output directory: ${OUTDIR}"
echo "[INFO] Target URL: ${target_url}"

cd "${OUTDIR}"

wget \
  -c \
  --tries=20 \
  --waitretry=30 \
  --timeout=120 \
  --retry-connrefused \
  --content-disposition \
  "${target_url}"

echo "[OK] Finished: ${target_url}"
echo "[INFO] Date: $(date)"
SH

chmod +x "${SCRIPT_DIR}/download_ICU_public_array.sh"

cat > "${SCRIPT_DIR}/submit_downloads.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

BASE="/home/tcmhwd/microbiome/ICU_public"
SCRIPT="${BASE}/scripts/download_ICU_public_array.sh"

NON_AGI_LIST="${BASE}/non_AGI/non_AGI_fastq_urls.txt"
AGI_LIST="${BASE}/AGI/AGI_fastq_urls_https.txt"

NON_AGI_OUT="${BASE}/non_AGI/fastq"
AGI_OUT="${BASE}/AGI/fastq"

MAX_JOBS="${1:-10}"

N_NON_AGI=$(wc -l < "${NON_AGI_LIST}" | tr -d ' ')
N_AGI=$(wc -l < "${AGI_LIST}" | tr -d ' ')

echo "[INFO] non-AGI URLs: ${N_NON_AGI}"
echo "[INFO] AGI URLs: ${N_AGI}"
echo "[INFO] Max concurrent downloads per array: ${MAX_JOBS}"

if [[ "${N_NON_AGI}" -gt 0 ]]; then
  sbatch \
    --array=1-${N_NON_AGI}%${MAX_JOBS} \
    --export=ALL,LIST=${NON_AGI_LIST},OUTDIR=${NON_AGI_OUT} \
    "${SCRIPT}"
fi

if [[ "${N_AGI}" -gt 0 ]]; then
  sbatch \
    --array=1-${N_AGI}%${MAX_JOBS} \
    --export=ALL,LIST=${AGI_LIST},OUTDIR=${AGI_OUT} \
    "${SCRIPT}"
fi
SH

chmod +x "${SCRIPT_DIR}/submit_downloads.sh"

echo
echo "[OK] Setup completed."
echo
echo "Folder structure:"
echo "  ${BASE}/non_AGI/fastq"
echo "  ${BASE}/AGI/fastq"
echo "  ${BASE}/logs"
echo
echo "URL counts:"
wc -l "${NON_AGI_DIR}/non_AGI_fastq_urls.txt" "${AGI_DIR}/AGI_fastq_urls_https.txt"
echo
echo "To submit downloads, run:"
echo "  bash ${SCRIPT_DIR}/submit_downloads.sh 10"
echo
echo "Use 5 instead of 10 if the server/network is unstable:"
echo "  bash ${SCRIPT_DIR}/submit_downloads.sh 5"
