#!/usr/bin/env python3
"""04_mag_summary.py — ICU_public MAG 기반 결과(CoverM/EggNOG/antiSMASH/RGI/VFDB/GTDB) 통합 집계.
scripts/17_summary_all.py를 ICU_public 구조(A0824G_WM/CRR/SRR 3개 명명, mortality 기반 그룹,
AGI+non_AGI+ICU-어댑터 3원 contig-RGI/RPKM)에 맞게 이식."""
import argparse
import glob
import logging
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ICU_PUBLIC = "/home/tcmhwd/microbiome/ICU_public"
MERGE_DIR = os.path.join(ICU_PUBLIC, "merge")
OUTPUT_DIR = os.path.join(MERGE_DIR, "output", "mag_summary")

COVERM_ALL_FILE = os.path.join(ICU_PUBLIC, "coverm_result", "all_samples.coverm.tsv")
EGGNOG_DIR = os.path.join(ICU_PUBLIC, "EggNOG_result")
ANTISMASH_DIR = os.path.join(ICU_PUBLIC, "antiSMASH_result")
RGI_DIR = os.path.join(ICU_PUBLIC, "MAG_rgi_results")
VFDB_DIR = os.path.join(ICU_PUBLIC, "VFDB_result")
GTDB_BAC_SUMMARY = os.path.join(ICU_PUBLIC, "gtdbtk_result", "gtdbtk.bac120.summary.tsv")
GTDB_AR_SUMMARY = os.path.join(ICU_PUBLIC, "gtdbtk_result", "gtdbtk.ar53.summary.tsv")

RESISTOME_ADAPTED = os.path.join(MERGE_DIR, "output", "_icu_resistome_adapted")
RGI_CONTIG_DIRS = [
    os.path.join(ICU_PUBLIC, "AGI", "RGI_contig_result"),
    os.path.join(ICU_PUBLIC, "non_AGI", "RGI_contig_result"),
    os.path.join(RESISTOME_ADAPTED, "RGI_contig_result"),
]
RPKM_RGI_DIRS = [
    os.path.join(ICU_PUBLIC, "AGI", "RPKM_rgi_result"),
    os.path.join(ICU_PUBLIC, "non_AGI", "RPKM_rgi_result"),
    os.path.join(RESISTOME_ADAPTED, "RPKM_rgi_result"),
]

SAMPLE_METADATA_TSV = os.path.join(MERGE_DIR, "output", "sample_metadata.tsv")

N_WORKERS = 60
RGI_VALID_CUTOFFS = {"Strict", "Perfect"}
VFDB_COLS = ["qseqid", "sseqid", "pident", "length", "evalue",
             "bitscore", "qlen", "slen", "qstart", "qend",
             "sstart", "send", "staxids", "stitle"]

# 실제 sample_metadata.tsv(887개 샘플) 검증 결과 ICU 코호트는 두 가지 명명 규칙이 섞여 있다:
# - A0824G/A0825G_WM 배치(299개, 리터럴 "A0824G"만 매치하면 A0825G 계열 187개 샘플이 누락됨)
# - ICU%04d 배치(250개, 브리프에는 언급되지 않았지만 EggNOG/RGI/VFDB/antiSMASH 원본 파일의
#   43.7%가 이 명명을 사용함 — 리터럴 "A0824G"만 있는 옛 정규식은 이 배치 전체를 조용히 누락시킴).
# 887개 sample_id 전체 및 EggNOG/RGI/VFDB 파일명과 교차검증하여 아래 패턴으로 일반화했다.
SAMPLE_ID_RE = re.compile(r'^(A\d+G\d+_WM|ICU\d+|CRR\d+|SRR\d+)')


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def sample_from_mag(mag_name: str):
    m = SAMPLE_ID_RE.match(mag_name)
    return m.group(1) if m else None


_SAMPLE_MORTALITY = None


def _load_sample_mortality() -> dict:
    global _SAMPLE_MORTALITY
    if _SAMPLE_MORTALITY is None:
        df = pd.read_csv(SAMPLE_METADATA_TSV, sep="\t")
        _SAMPLE_MORTALITY = dict(zip(df["sample_id"], df["mortality"]))
    return _SAMPLE_MORTALITY


def group_from_sample(sample_id: str) -> str:
    mortality = _load_sample_mortality().get(sample_id)
    if mortality == 1:
        return "Non_survivor"
    if mortality == 0:
        return "Survivor"
    return "Unknown"


def save(df: pd.DataFrame, subdir: str, filename: str, index: bool = True):
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    logging.info(f"  저장: {path}  [{df.shape[0]}행 x {df.shape[1]}열]")


def _merge_count_dicts(accumulator, sample, local_dict):
    for feat, cnt in local_dict.items():
        accumulator[feat][sample] += cnt


def _dicts_to_df(nested: dict) -> pd.DataFrame:
    df = pd.DataFrame(nested).T.fillna(0).astype(int)
    df.index.name = "feature"
    df.sort_index(inplace=True)
    return df


# ── Worker 함수 (multiprocessing.Pool용, 모두 모듈 최상위 정의) ──────────────

def _worker_eggnog(filepath: str):
    mag = os.path.basename(filepath).replace("_eggnog.emapper.annotations", "")
    sample = sample_from_mag(mag)
    if sample is None:
        return None
    cog_cnt, ko_cnt, path_cnt, mod_cnt = {}, {}, {}, {}
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 14:
                    continue
                cog, ko_raw, pw_raw, mod_raw = parts[6].strip(), parts[11].strip(), parts[12].strip(), parts[13].strip()
                if cog and cog != "-":
                    for c in cog:
                        cog_cnt[c] = cog_cnt.get(c, 0) + 1
                if ko_raw and ko_raw != "-":
                    for k in ko_raw.split(","):
                        k = k.strip().replace("ko:", "")
                        if k:
                            ko_cnt[k] = ko_cnt.get(k, 0) + 1
                if pw_raw and pw_raw != "-":
                    for p in pw_raw.split(","):
                        p = p.strip()
                        if p:
                            path_cnt[p] = path_cnt.get(p, 0) + 1
                if mod_raw and mod_raw != "-":
                    for m in mod_raw.split(","):
                        m = m.strip()
                        if m:
                            mod_cnt[m] = mod_cnt.get(m, 0) + 1
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, cog_cnt, ko_cnt, path_cnt, mod_cnt)


def _parse_rgi_table(rgi: pd.DataFrame):
    drug_cnt, mech_cnt, gene_cnt, aro_cnt = {}, {}, {}, {}
    if rgi.empty:
        return drug_cnt, mech_cnt, gene_cnt, aro_cnt
    if "Cut_Off" in rgi.columns:
        rgi = rgi[rgi["Cut_Off"].isin(RGI_VALID_CUTOFFS)]
    if rgi.empty:
        return drug_cnt, mech_cnt, gene_cnt, aro_cnt
    if "Drug Class" in rgi.columns:
        for val in rgi["Drug Class"].dropna().astype(str):
            for dc in val.split(";"):
                dc = dc.strip()
                if dc and dc != "nan":
                    drug_cnt[dc] = drug_cnt.get(dc, 0) + 1
    if "Resistance Mechanism" in rgi.columns:
        for val in rgi["Resistance Mechanism"].dropna().astype(str):
            val = val.strip()
            if val and val != "nan":
                mech_cnt[val] = mech_cnt.get(val, 0) + 1
    if "AMR Gene Family" in rgi.columns:
        for val in rgi["AMR Gene Family"].dropna().astype(str):
            for gf in val.split(";"):
                gf = gf.strip()
                if gf and gf != "nan":
                    gene_cnt[gf] = gene_cnt.get(gf, 0) + 1
    if "Best_Hit_ARO" in rgi.columns:
        for val in rgi["Best_Hit_ARO"].dropna().astype(str):
            val = val.strip()
            if val and val != "nan":
                aro_cnt[val] = aro_cnt.get(val, 0) + 1
    return drug_cnt, mech_cnt, gene_cnt, aro_cnt


def _worker_rgi(filepath: str):
    mag = os.path.basename(filepath).replace("_rgi.txt", "")
    sample = sample_from_mag(mag)
    if sample is None:
        return None
    try:
        rgi = pd.read_csv(filepath, sep="\t", low_memory=False)
        drug, mech, gene, aro = _parse_rgi_table(rgi)
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, drug, mech, gene, aro)


def _worker_rgi_contig(filepath: str):
    sample = os.path.basename(filepath).replace("_rgi.txt", "")
    try:
        rgi = pd.read_csv(filepath, sep="\t", low_memory=False)
        drug, mech, gene, aro = _parse_rgi_table(rgi)
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, drug, mech, gene, aro)


def _worker_rpkm_rgi(filepath: str):
    sample = os.path.basename(filepath).replace("_rgi_rpkm.tsv", "")
    drug_sum, mech_sum, gene_sum, aro_sum = {}, {}, {}, {}
    try:
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        if df.empty:
            return (sample, {}, {}, {}, {})
        if "cut_off" in df.columns:
            df = df[df["cut_off"].isin(RGI_VALID_CUTOFFS)]
        if df.empty:
            return (sample, {}, {}, {}, {})
        if "drug_class" in df.columns and "rpkm" in df.columns:
            for _, row in df[["drug_class", "rpkm"]].dropna().iterrows():
                parts = [p.strip() for p in str(row["drug_class"]).split(";") if p.strip() and p.strip() != "nan"]
                if parts:
                    share = float(row["rpkm"]) / len(parts)
                    for dc in parts:
                        drug_sum[dc] = drug_sum.get(dc, 0.0) + share
        if "resistance_mechanism" in df.columns and "rpkm" in df.columns:
            for _, row in df[["resistance_mechanism", "rpkm"]].dropna().iterrows():
                val = str(row["resistance_mechanism"]).strip()
                if val and val != "nan":
                    mech_sum[val] = mech_sum.get(val, 0.0) + float(row["rpkm"])
        if "amr_gene_family" in df.columns and "rpkm" in df.columns:
            for _, row in df[["amr_gene_family", "rpkm"]].dropna().iterrows():
                parts = [p.strip() for p in str(row["amr_gene_family"]).split(";") if p.strip() and p.strip() != "nan"]
                if parts:
                    share = float(row["rpkm"]) / len(parts)
                    for gf in parts:
                        gene_sum[gf] = gene_sum.get(gf, 0.0) + share
        if "best_hit_aro" in df.columns and "rpkm" in df.columns:
            for _, row in df[["best_hit_aro", "rpkm"]].dropna().iterrows():
                val = str(row["best_hit_aro"]).strip()
                if val and val != "nan":
                    aro_sum[val] = aro_sum.get(val, 0.0) + float(row["rpkm"])
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, drug_sum, mech_sum, gene_sum, aro_sum)


def _parse_vf_fields(stitle: str):
    m = re.search(r'\)\s+\(([^)]+)\)\s+\[([^\]]+)\]', stitle)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r'\[([^\]]+)\]', stitle)
    if m:
        return stitle.split("(")[0].strip()[:50], m.group(1).strip()
    return stitle[:50].strip(), "Unknown"


def _worker_antismash_json(filepath: str):
    """MAG 디렉토리의 <mag>.json(원본 antiSMASH 결과)에서 직접 BGC 영역을 집계.
    antismash_summary.tsv는 상위 파이프라인(11_run_antismash_re.sh)의 요약 단계가
    'antismash.results.json' 파일명과 'regions' 키를 기대하지만 실제 산출물은
    '<mag>.json' 파일명에 'areas' 키를 쓰기 때문에 bgc_count가 전체 10,843개 행에서
    항상 0으로 기록되는 버그가 있다 (region*.gbk fallback도 실제 파일명이
    'gnlX<contig>.region001.gbk'라 매치되지 않음). 원본 JSON은 정상이므로 여기서 직접 파싱한다."""
    mag = os.path.basename(filepath)[:-len(".json")]
    sample = sample_from_mag(mag)
    if sample is None:
        return None
    try:
        import json
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        areas = []
        for rec in data.get("records", []):
            areas.extend(rec.get("areas", []))
        bgc_count = len(areas)
        types = set()
        for area in areas:
            for prod in area.get("products", []):
                if prod:
                    types.add(str(prod).strip())
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, bgc_count, types)


def _worker_vfdb(filepath: str):
    basename = os.path.basename(filepath)
    mag = re.sub(r'_vfdb_(filtered|besthit|raw)\.tsv$', '', basename)
    sample = sample_from_mag(mag)
    if sample is None:
        return None
    cat_cnt, gene_cnt = {}, {}
    try:
        # 실제 파일에 컬럼명이 그대로 담긴 헤더 행이 있으므로 header=0으로 읽는다
        # (header=None + names=VFDB_COLS로 읽으면 헤더 행이 가짜 데이터 행으로 섞여
        # 모든 샘플에 "Unknown"/"stitle" 유령 레코드가 하나씩 생긴다).
        vf = pd.read_csv(filepath, sep="\t", header=0, low_memory=False)
        if list(vf.columns) != VFDB_COLS:
            vf.columns = VFDB_COLS[:len(vf.columns)]
        if vf.empty:
            return (sample, {}, {})
        for stitle in vf["stitle"].dropna().astype(str):
            gene, cat = _parse_vf_fields(stitle)
            cat_cnt[cat] = cat_cnt.get(cat, 0) + 1
            gene_cnt[gene] = gene_cnt.get(gene, 0) + 1
    except Exception as e:
        return ("__ERROR__", str(e), filepath)
    return (sample, cat_cnt, gene_cnt)


# ── 섹션 함수 ──────────────────────────────────────────────────────────────

def make_metadata():
    logging.info("=== 메타데이터 ===")
    if not os.path.exists(COVERM_ALL_FILE):
        logging.warning("CoverM 파일이 없어 메타데이터를 생성할 수 없습니다.")
        return
    header = pd.read_csv(COVERM_ALL_FILE, sep="\t", nrows=0)
    samples = [c.replace(" Relative Abundance (%)", "")
               for c in header.columns if c.endswith("Relative Abundance (%)")]
    rows = [{"sample_id": s, "group": group_from_sample(s)} for s in samples]
    meta = pd.DataFrame(rows)
    save(meta, "", "metadata.tsv", index=False)
    logging.info(f"  총 {len(meta)}개 샘플 (Non_survivor: {(meta.group == 'Non_survivor').sum()}, "
                 f"Survivor: {(meta.group == 'Survivor').sum()}, "
                 f"Unknown: {(meta.group == 'Unknown').sum()})")


def summarize_coverm():
    logging.info("=== CoverM 상대풍도/깊이 행렬 ===")
    if not os.path.exists(COVERM_ALL_FILE):
        logging.warning(f"CoverM 병합 파일 없음: {COVERM_ALL_FILE}")
        return
    df = pd.read_csv(COVERM_ALL_FILE, sep="\t", index_col=0)
    abund_cols = [c for c in df.columns if c.endswith("Relative Abundance (%)")]
    mean_cols = [c for c in df.columns if c.endswith("Mean")]

    abund = df[abund_cols].copy()
    abund.columns = [c.replace(" Relative Abundance (%)", "") for c in abund.columns]
    abund.index.name = "MAG"
    save(abund, "coverm", "coverm_abundance_matrix.tsv")

    mean = df[mean_cols].copy()
    mean.columns = [c.replace(" Mean", "") for c in mean.columns]
    mean.index.name = "MAG"
    save(mean, "coverm", "coverm_mean_depth_matrix.tsv")

    logging.info(f"  {abund.shape[0]}개 MAG(+unmapped) x {abund.shape[1]}개 샘플")


def summarize_eggnog():
    logging.info("=== EggNOG 기능 어노테이션 ===")
    files = sorted(glob.glob(os.path.join(EGGNOG_DIR, "*.emapper.annotations")))
    if not files:
        logging.warning("EggNOG 어노테이션 파일을 찾을 수 없습니다.")
        return
    cog_cnt = defaultdict(lambda: defaultdict(int))
    ko_cnt = defaultdict(lambda: defaultdict(int))
    path_cnt = defaultdict(lambda: defaultdict(int))
    mod_cnt = defaultdict(lambda: defaultdict(int))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_eggnog, files, chunksize=4), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"EggNOG 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, cog, ko, pw, mod = result
            _merge_count_dicts(cog_cnt, sample, cog)
            _merge_count_dicts(ko_cnt, sample, ko)
            _merge_count_dicts(path_cnt, sample, pw)
            _merge_count_dicts(mod_cnt, sample, mod)
            if i % 1000 == 0:
                logging.info(f"  EggNOG 진행: {i}/{len(files)}")
    for data, fname in [(cog_cnt, "eggnog_cog_per_sample.tsv"),
                         (ko_cnt, "eggnog_kegg_ko_per_sample.tsv"),
                         (path_cnt, "eggnog_kegg_pathway_per_sample.tsv"),
                         (mod_cnt, "eggnog_kegg_module_per_sample.tsv")]:
        if data:
            save(_dicts_to_df(data), "eggnog", fname)
    logging.info(f"  실패 파일: {errors}개")


def _summarize_antismash_from_json():
    """<mag>.json 원본 파일에서 직접 BGC를 집계 (antismash_summary.tsv가 신뢰할 수 없을 때 사용).
    실제 데이터 검증 결과 antismash_summary.tsv의 bgc_count는 전체 10,843행에서 100% 0으로 기록되어
    있었다 (17,883개의 실제 region*.gbk 파일이 존재하는데도). 원인은 11_run_antismash_re.sh의 요약
    집계 로직이 존재하지 않는 파일명('antismash.results.json')과 존재하지 않는 JSON 키('.regions')를
    찾기 때문이다(실제로는 '<mag>.json'과 '.areas'). 이 함수는 그 우회 경로로, 각 MAG의 원본 antiSMASH
    JSON을 직접 파싱해 정확한 bgc_count/bgc_types를 계산한다."""
    files = sorted(glob.glob(os.path.join(ANTISMASH_DIR, "*", "*.json")))
    files = [f for f in files if os.path.basename(f) != "antismash.results.json"]
    if not files:
        return False
    total_bgc = defaultdict(int)
    type_counts = defaultdict(lambda: defaultdict(int))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_antismash_json, files, chunksize=8), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"antiSMASH JSON 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, bgc_count, types = result
            total_bgc[sample] += bgc_count
            for t in types:
                type_counts[t][sample] += 1
            if i % 2000 == 0:
                logging.info(f"  antiSMASH JSON 진행: {i}/{len(files)}")
    if total_bgc:
        s = pd.Series(total_bgc, name="bgc_count").to_frame()
        s.index.name = "sample"
        save(s, "antismash", "antismash_bgc_total_per_sample.tsv")
    if type_counts:
        save(_dicts_to_df(type_counts), "antismash", "antismash_bgc_types_per_sample.tsv")
    logging.info(f"  MAG JSON {len(files)}개 직접 파싱, 실패: {errors}개, "
                 f"BGC 보유 샘플: {sum(1 for v in total_bgc.values() if v > 0)}개")
    return True


def summarize_antismash():
    logging.info("=== antiSMASH 생합성 유전자 클러스터 ===")
    if _summarize_antismash_from_json():
        return
    logging.warning("MAG 단위 antiSMASH JSON을 찾지 못해 antismash_summary.tsv로 대체합니다 "
                     "(주의: 이 요약 파일은 알려진 버그로 bgc_count가 0으로 기록될 수 있음).")
    summary_file = os.path.join(ANTISMASH_DIR, "antismash_summary.tsv")
    if not os.path.exists(summary_file):
        logging.warning(f"antiSMASH 요약 파일 없음: {summary_file}")
        return
    df = pd.read_csv(summary_file, sep="\t")
    mag_col = df.columns[0]
    total_bgc = defaultdict(int)
    type_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        sample = sample_from_mag(str(row[mag_col]))
        if sample is None:
            continue
        if "bgc_count" in df.columns:
            try:
                total_bgc[sample] += int(row["bgc_count"])
            except (ValueError, TypeError):
                pass
        if "bgc_types" in df.columns:
            types_str = str(row["bgc_types"])
            if types_str.lower() != "nan":
                for t in re.split(r"[,;|]", types_str):
                    t = t.strip()
                    if t:
                        type_counts[t][sample] += 1
    if total_bgc:
        s = pd.Series(total_bgc, name="bgc_count").to_frame()
        s.index.name = "sample"
        save(s, "antismash", "antismash_bgc_total_per_sample.tsv")
    if type_counts:
        save(_dicts_to_df(type_counts), "antismash", "antismash_bgc_types_per_sample.tsv")


def summarize_rgi():
    logging.info("=== RGI 항생제 내성 유전자 (MAG 기반) ===")
    files = sorted(glob.glob(os.path.join(RGI_DIR, "*_rgi.txt")))
    if not files:
        logging.warning("RGI 결과 파일을 찾을 수 없습니다.")
        return
    drug_cnt = defaultdict(lambda: defaultdict(int))
    mech_cnt = defaultdict(lambda: defaultdict(int))
    gene_cnt = defaultdict(lambda: defaultdict(int))
    aro_cnt = defaultdict(lambda: defaultdict(int))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_rgi, files, chunksize=8), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"RGI 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, drug, mech, gene, aro = result
            _merge_count_dicts(drug_cnt, sample, drug)
            _merge_count_dicts(mech_cnt, sample, mech)
            _merge_count_dicts(gene_cnt, sample, gene)
            _merge_count_dicts(aro_cnt, sample, aro)
            if i % 1000 == 0:
                logging.info(f"  RGI 진행: {i}/{len(files)}")
    for data, fname in [(drug_cnt, "rgi_drug_class_per_sample.tsv"),
                         (mech_cnt, "rgi_resistance_mechanism_per_sample.tsv"),
                         (gene_cnt, "rgi_amr_gene_family_per_sample.tsv"),
                         (aro_cnt, "rgi_aro_per_sample.tsv")]:
        if data:
            save(_dicts_to_df(data), "rgi", fname)
    logging.info(f"  실패 파일: {errors}개")


def summarize_rgi_contig():
    logging.info("=== RGI Contig 직접 분석 (카운트, AGI+non_AGI+ICU-어댑터) ===")
    files = []
    for d in RGI_CONTIG_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*", "*_rgi.txt"))))
    if not files:
        logging.warning("RGI contig 결과 파일을 찾을 수 없습니다.")
        return
    drug_cnt = defaultdict(lambda: defaultdict(int))
    mech_cnt = defaultdict(lambda: defaultdict(int))
    gene_cnt = defaultdict(lambda: defaultdict(int))
    aro_cnt = defaultdict(lambda: defaultdict(int))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_rgi_contig, files, chunksize=8), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"RGI contig 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, drug, mech, gene, aro = result
            _merge_count_dicts(drug_cnt, sample, drug)
            _merge_count_dicts(mech_cnt, sample, mech)
            _merge_count_dicts(gene_cnt, sample, gene)
            _merge_count_dicts(aro_cnt, sample, aro)
    for data, fname in [(drug_cnt, "rgi_contig_drug_class_per_sample.tsv"),
                         (mech_cnt, "rgi_contig_resistance_mechanism_per_sample.tsv"),
                         (gene_cnt, "rgi_contig_amr_gene_family_per_sample.tsv"),
                         (aro_cnt, "rgi_contig_aro_per_sample.tsv")]:
        if data:
            save(_dicts_to_df(data), "rgi_contig", fname)
    logging.info(f"  총 {len(files)}개 샘플, 실패: {errors}개")


def summarize_rpkm_rgi():
    logging.info("=== RPKM RGI (RPKM 가중 합산, AGI+non_AGI+ICU-어댑터) ===")
    files = []
    for d in RPKM_RGI_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*", "*_rgi_rpkm.tsv"))))
    if not files:
        logging.warning("RPKM RGI 결과 파일을 찾을 수 없습니다.")
        return
    drug_sum = defaultdict(lambda: defaultdict(float))
    mech_sum = defaultdict(lambda: defaultdict(float))
    gene_sum = defaultdict(lambda: defaultdict(float))
    aro_sum = defaultdict(lambda: defaultdict(float))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_rpkm_rgi, files, chunksize=8), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"RPKM RGI 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, drug, mech, gene, aro = result
            for feat, val in drug.items():
                drug_sum[feat][sample] += val
            for feat, val in mech.items():
                mech_sum[feat][sample] += val
            for feat, val in gene.items():
                gene_sum[feat][sample] += val
            for feat, val in aro.items():
                aro_sum[feat][sample] += val

    def _float_dicts_to_df(nested: dict) -> pd.DataFrame:
        df = pd.DataFrame(nested).T.fillna(0.0)
        df.index.name = "feature"
        df.sort_index(inplace=True)
        return df

    for data, fname in [(drug_sum, "rpkm_rgi_drug_class_per_sample.tsv"),
                         (mech_sum, "rpkm_rgi_resistance_mechanism_per_sample.tsv"),
                         (gene_sum, "rpkm_rgi_amr_gene_family_per_sample.tsv"),
                         (aro_sum, "rpkm_rgi_aro_per_sample.tsv")]:
        if data:
            save(_float_dicts_to_df(data), "rpkm_rgi", fname)
    logging.info(f"  총 {len(files)}개 샘플, 실패: {errors}개")


def summarize_vfdb():
    logging.info("=== VFDB 독성인자 ===")
    files = sorted(glob.glob(os.path.join(VFDB_DIR, "*_vfdb_filtered.tsv")))
    if not files:
        files = sorted(glob.glob(os.path.join(VFDB_DIR, "*_vfdb_besthit.tsv")))
    if not files:
        logging.warning("VFDB filtered/besthit 파일을 찾을 수 없습니다.")
        return
    cat_cnt = defaultdict(lambda: defaultdict(int))
    gene_cnt = defaultdict(lambda: defaultdict(int))
    errors = 0
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_vfdb, files, chunksize=8), 1):
            if result is None:
                continue
            if result[0] == "__ERROR__":
                logging.warning(f"VFDB 실패 {result[2]}: {result[1]}")
                errors += 1
                continue
            sample, cat, gene = result
            _merge_count_dicts(cat_cnt, sample, cat)
            _merge_count_dicts(gene_cnt, sample, gene)
    for data, fname in [(cat_cnt, "vfdb_category_per_sample.tsv"),
                         (gene_cnt, "vfdb_gene_per_sample.tsv")]:
        if data:
            save(_dicts_to_df(data), "vfdb", fname)
    logging.info(f"  실패 파일: {errors}개")


def summarize_gtdb():
    logging.info("=== GTDB-Tk 분류 + 종 수준 상대풍도 (coverm 섹션 이후 실행 필요) ===")
    tax_rows = []
    for path in (GTDB_BAC_SUMMARY, GTDB_AR_SUMMARY):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        prefix_map = {"d__": "domain", "p__": "phylum", "c__": "class",
                      "o__": "order", "f__": "family", "g__": "genus", "s__": "species"}
        for _, row in df.iterrows():
            levels = {v: "" for v in prefix_map.values()}
            for token in str(row.get("classification", "")).split(";"):
                token = token.strip()
                for prefix, level in prefix_map.items():
                    if token.startswith(prefix):
                        levels[level] = token[len(prefix):]
            tax_rows.append({"MAG": row["user_genome"], **levels})

    if not tax_rows:
        logging.warning("GTDB-Tk summary 파일을 찾을 수 없습니다.")
        return

    tax = pd.DataFrame(tax_rows).set_index("MAG")
    save(tax, "gtdb", "gtdb_mag_taxonomy.tsv")
    logging.info(f"  {len(tax)}개 MAG 분류")

    abund_path = os.path.join(OUTPUT_DIR, "coverm", "coverm_abundance_matrix.tsv")
    if not os.path.exists(abund_path):
        logging.warning("coverm_abundance_matrix.tsv 없음 - 종 수준 풍도 계산 건너뜀 "
                         "(coverm 섹션을 먼저 실행하세요)")
        return
    abund = pd.read_csv(abund_path, sep="\t", index_col=0)
    abund = abund[~abund.index.str.lower().str.contains("unmapped")]
    joined = abund.join(tax["species"], how="left")
    joined["species"] = joined["species"].fillna("Unclassified")
    species_abund = joined.groupby("species").sum(numeric_only=True)
    species_abund.index.name = "species"
    save(species_abund, "gtdb", "gtdb_species_abundance_matrix.tsv")
    logging.info(f"  {species_abund.shape[0]}개 종 x {species_abund.shape[1]}개 샘플")


SECTIONS = {
    "metadata": make_metadata,
    "coverm": summarize_coverm,
    "eggnog": summarize_eggnog,
    "antismash": summarize_antismash,
    "rgi": summarize_rgi,
    "rgi_contig": summarize_rgi_contig,
    "rpkm_rgi": summarize_rpkm_rgi,
    "vfdb": summarize_vfdb,
    "gtdb": summarize_gtdb,
}


def parse_args():
    p = argparse.ArgumentParser(description="ICU_public MAG 기반 결과 통합 요약")
    p.add_argument("--workers", type=int, default=60)
    p.add_argument("--sections", default="all",
                   help=f"실행할 섹션 (쉼표 구분). 사용 가능: {', '.join(SECTIONS)}")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    N_WORKERS = args.workers

    if args.sections.strip().lower() == "all":
        to_run = list(SECTIONS.keys())
    else:
        requested = {s.strip() for s in args.sections.split(",")}
        to_run = [s for s in SECTIONS.keys() if s in requested]
    invalid = [s for s in requested if s not in SECTIONS] if args.sections.strip().lower() != "all" else []
    if invalid:
        logging.error(f"알 수 없는 섹션: {invalid}. 유효: {list(SECTIONS.keys())}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"출력 디렉토리: {OUTPUT_DIR}")
    logging.info(f"병렬 프로세스: {N_WORKERS}개")
    logging.info(f"실행 섹션: {to_run}")

    for sec in to_run:
        SECTIONS[sec]()

    logging.info("완료")
