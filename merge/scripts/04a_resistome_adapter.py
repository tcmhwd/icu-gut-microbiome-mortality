#!/usr/bin/env python3
"""04a_resistome_adapter.py — ICU 자체 코호트 contig-level RGI/RPKM을
AGI/non_AGI와 동일한 스키마로 변환 (/data/tcmhwd/read/resistome -> merge/output)."""
import logging
import os

import pandas as pd

ICU_PUBLIC = "/home/tcmhwd/microbiome/ICU_public"
RESISTOME_DIR = "/data/tcmhwd/read/resistome"
RGI_SRC_DIR = os.path.join(RESISTOME_DIR, "rgi_results")
RPKM_SRC_DIR = os.path.join(RESISTOME_DIR, "gene_rpkm_results")

OUT_BASE = os.path.join(ICU_PUBLIC, "merge", "output", "_icu_resistome_adapted")
RGI_OUT_DIR = os.path.join(OUT_BASE, "RGI_contig_result")
RPKM_OUT_DIR = os.path.join(OUT_BASE, "RPKM_rgi_result")

ICU_SAMPLE_LIST = os.path.join(ICU_PUBLIC, "ICU_sample_list.txt")


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def load_icu_samples() -> list:
    samples = []
    with open(ICU_SAMPLE_LIST, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0].strip():
                samples.append(parts[0].strip())
    return samples


def adapt_sample(sample: str) -> bool:
    rgi_src = os.path.join(RGI_SRC_DIR, sample, f"{sample}_rgi_output.txt")
    rpkm_src = os.path.join(RPKM_SRC_DIR, f"{sample}_gene_rpkm.tsv")
    if not os.path.exists(rgi_src) or not os.path.exists(rpkm_src):
        logging.warning(f"  {sample}: 원본 파일 없음 (rgi={os.path.exists(rgi_src)}, "
                         f"rpkm={os.path.exists(rpkm_src)})")
        return False

    rgi_out_dir = os.path.join(RGI_OUT_DIR, sample)
    os.makedirs(rgi_out_dir, exist_ok=True)
    rgi_link = os.path.join(rgi_out_dir, f"{sample}_rgi.txt")
    if os.path.islink(rgi_link) or os.path.exists(rgi_link):
        os.remove(rgi_link)
    os.symlink(rgi_src, rgi_link)

    rgi = pd.read_csv(rgi_src, sep="\t", low_memory=False)
    rpkm = pd.read_csv(rpkm_src, sep="\t", low_memory=False)

    if len(rgi) != len(rpkm):
        logging.error(f"  {sample}: 행 수 불일치 (rgi={len(rgi)}, rpkm={len(rpkm)}) - 건너뜀")
        return False

    # RGI Contigs may have digit suffixes (e.g., k141_15445_3 -> k141_15445).
    # Only remove suffix if the contig has 3+ parts (2+ underscores) and last part is all digits.
    def extract_contig_base(contig_str):
        contig_str = str(contig_str).strip()
        parts = contig_str.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return contig_str

    rgi_contig_base = rgi["Contig"].apply(extract_contig_base)
    rpkm_contig = rpkm["Contig"].astype(str).str.strip()

    if (rgi_contig_base.values != rpkm_contig.values).any():
        logging.error(f"  {sample}: Contig 순서 불일치 - 건너뜀")
        return False
    if (rgi["Start"].astype(str).values != rpkm["Start"].astype(str).values).any():
        logging.error(f"  {sample}: Start 좌표 불일치 - 건너뜀")
        return False

    combined = pd.DataFrame({
        "locus_tag": rgi["ORF_ID"].str.split().str[0],
        "contig": rpkm_contig,
        "start": rgi["Start"],
        "end": rgi["Stop"],
        "gene_length": rpkm["Length(bp)"],
        "mapped_reads": rpkm["Mapped_Reads"],
        "rpkm": rpkm["RPKM"],
        "best_hit_aro": rgi["Best_Hit_ARO"],
        "drug_class": rgi["Drug Class"],
        "resistance_mechanism": rgi["Resistance Mechanism"],
        "amr_gene_family": rgi["AMR Gene Family"],
        "cut_off": rgi["Cut_Off"],
    })

    rpkm_out_dir = os.path.join(RPKM_OUT_DIR, sample)
    os.makedirs(rpkm_out_dir, exist_ok=True)
    out_path = os.path.join(rpkm_out_dir, f"{sample}_rgi_rpkm.tsv")
    combined.to_csv(out_path, sep="\t", index=False)
    return True


def main():
    setup_logging()
    samples = load_icu_samples()
    logging.info(f"ICU 샘플 {len(samples)}개 어댑터 변환 시작")
    os.makedirs(RGI_OUT_DIR, exist_ok=True)
    os.makedirs(RPKM_OUT_DIR, exist_ok=True)

    n_ok = 0
    for sample in samples:
        if adapt_sample(sample):
            n_ok += 1

    logging.info(f"완료: {n_ok}/{len(samples)}개 샘플 변환 성공")
    logging.info(f"  RGI: {RGI_OUT_DIR}")
    logging.info(f"  RPKM: {RPKM_OUT_DIR}")


if __name__ == "__main__":
    main()
