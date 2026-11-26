#!/usr/bin/env python3
"""03_humann3_merge.py — HUMAnN3 pathabundance/pathcoverage/genefamilies(KO 축약)를
ICU/AGI/non_AGI 3개 코호트에 걸쳐 병합. 공식 HUMAnN 유틸리티
(humann_join_tables/humann_regroup_table/humann_split_stratified_table)를 사용 —
커스텀 pandas 병합이 887샘플 x 20만행 genefamilies에서 확장성 문제(OOM/1시간+ 미완료)를
일으켜 교체됨. genefamilies는 raw UniRef90이 아니라 KO 단위로 축약 후 병합
(raw UniRef90은 R-ready 파이프라인 어디서도 소비되지 않고, 통계적으로도 887샘플
코호트에서 다루기엔 차원이 지나치게 큼)."""
import glob
import logging
import os
import shutil
import subprocess
import sys
import argparse
from multiprocessing import Pool

ICU_PUBLIC = "/home/tcmhwd/microbiome/ICU_public"
OUTPUT_DIR = os.path.join(ICU_PUBLIC, "merge", "output", "humann3")
STAGING_DIR = os.path.join(ICU_PUBLIC, "merge", "output", "_humann3_staging")

DATASETS = {
    "ICU": os.path.join(ICU_PUBLIC, "ICU", "humann3"),
    "AGI": os.path.join(ICU_PUBLIC, "AGI", "humann3"),
    "non_AGI": os.path.join(ICU_PUBLIC, "non_AGI", "humann3"),
}

FILE_TYPES = ["genefamilies", "pathabundance", "pathcoverage"]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def stage_symlinks():
    logging.info("=== 1/6: 심링크 스테이징 ===")
    os.makedirs(STAGING_DIR, exist_ok=True)
    n = 0
    for dataset, path in DATASETS.items():
        for ftype in FILE_TYPES:
            for f in sorted(glob.glob(os.path.join(path, f"*_{ftype}.tsv"))):
                link = os.path.join(STAGING_DIR, os.path.basename(f))
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(f, link)
                n += 1
    logging.info(f"  {n}개 파일 심링크 생성")


def _regroup_one(args):
    src, dst = args
    result = subprocess.run(
        ["humann_regroup_table", "-i", src, "-g", "uniref90_ko", "-o", dst],
        capture_output=True, text=True
    )
    return (src, result.returncode, result.stderr[-500:] if result.returncode != 0 else "")


def regroup_genefamilies_to_ko(workers: int):
    logging.info("=== 2/6: genefamilies -> KO 축약 (샘플별 병렬) ===")
    files = sorted(glob.glob(os.path.join(STAGING_DIR, "*_genefamilies.tsv")))
    jobs = []
    for f in files:
        sample = os.path.basename(f).replace("_genefamilies.tsv", "")
        dst = os.path.join(STAGING_DIR, f"{sample}_genefamilies_ko.tsv")
        jobs.append((f, dst))
    errors = 0
    with Pool(workers) as pool:
        for i, (src, rc, err) in enumerate(pool.imap_unordered(_regroup_one, jobs), 1):
            if rc != 0:
                logging.warning(f"  실패 {src}: {err}")
                errors += 1
            if i % 100 == 0:
                logging.info(f"  진행: {i}/{len(jobs)}")
    logging.info(f"  완료: {len(jobs) - errors}/{len(jobs)}개 성공 (실패 {errors}개)")
    if errors > 0:
        logging.error(f"  {errors}개 샘플 KO 축약 실패 — 이후 join에서 조용히 누락되지 않도록 중단합니다.")
        sys.exit(1)


def join_table(file_name_pattern: str, out_path: str):
    result = subprocess.run(
        ["humann_join_tables", "-i", STAGING_DIR, "-o", out_path,
         "--file_name", file_name_pattern],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logging.error(f"  humann_join_tables 실패({file_name_pattern}): {result.stderr}")
        sys.exit(1)
    logging.info(f"  저장: {out_path}")


def strip_sample_suffix(path: str):
    """HUMAnN join_tables가 붙이는 컬럼 접미사(_Abundance, _Abundance-RPKs, _Coverage)를
    제거해 다른 R-ready 파일들과 동일한 순수 sample_id 컬럼명으로 맞춘다."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", index_col=0)
    new_cols = []
    for c in df.columns:
        c2 = c
        for suffix in ("_Abundance-RPKs", "_Abundance", "_Coverage"):
            if c2.endswith(suffix):
                c2 = c2[: -len(suffix)]
                break
        new_cols.append(c2)
    df.columns = new_cols
    df.to_csv(path, sep="\t")


def split_stratified(raw_path: str, dest_basename: str):
    """humann_split_stratified_table로 stratified/unstratified 분리 후
    '{dest_basename}.unstratified.tsv' / '{dest_basename}.stratified.tsv'로 이름을 맞추고
    원본(raw_path)은 삭제한다. (dest_unstrat_path, dest_strat_path) 반환."""
    result = subprocess.run(
        ["humann_split_stratified_table", "-i", raw_path, "-o", OUTPUT_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logging.error(f"  humann_split_stratified_table 실패({raw_path}): {result.stderr}")
        sys.exit(1)

    base = os.path.basename(raw_path).replace(".tsv", "")
    unstrat_src = os.path.join(OUTPUT_DIR, f"{base}_unstratified.tsv")
    strat_src = os.path.join(OUTPUT_DIR, f"{base}_stratified.tsv")
    unstrat_dst = os.path.join(OUTPUT_DIR, f"{dest_basename}.unstratified.tsv")
    strat_dst = os.path.join(OUTPUT_DIR, f"{dest_basename}.stratified.tsv")
    shutil.move(unstrat_src, unstrat_dst)
    shutil.move(strat_src, strat_dst)
    os.remove(raw_path)
    logging.info(f"  저장: {unstrat_dst}")
    logging.info(f"  저장: {strat_dst}")
    return unstrat_dst, strat_dst


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stage_symlinks()
    regroup_genefamilies_to_ko(args.workers)

    logging.info("=== 3/6: pathabundance/pathcoverage/genefamilies(KO) 병합 ===")
    pathabund_raw = os.path.join(OUTPUT_DIR, "humann_pathabundance_wide_raw.tsv")
    pathcov_out = os.path.join(OUTPUT_DIR, "humann_pathcoverage_wide.tsv")
    ko_raw = os.path.join(OUTPUT_DIR, "humann_genefamilies_ko_wide_raw.tsv")

    join_table("pathabundance", pathabund_raw)
    join_table("pathcoverage", pathcov_out)
    join_table("genefamilies_ko", ko_raw)

    logging.info("=== 4/6: pathabundance stratified/unstratified 분리 ===")
    pathabund_unstrat, pathabund_strat = split_stratified(
        pathabund_raw, "humann_pathabundance_wide")

    logging.info("=== 5/6: 컬럼명 정리 (sample_id로 통일) ===")
    for path in (pathabund_unstrat, pathabund_strat, pathcov_out, ko_raw):
        strip_sample_suffix(path)

    logging.info("=== 6/6: genefamilies(KO) stratified/unstratified 분리 ===")
    # KO 축약만으로는 species-level stratification이 남아있어 23만 행 규모가 되어
    # 통계/AI에 바로 쓰기엔 여전히 크다 (한 샘플 raw UniRef90 89,777행 -> KO 축약
    # 13,730행이지만, 887개 샘플 union은 stratification 포함 시 다시 커짐). pathabundance와
    # 동일하게 한 번 더 분리해 순수 KO 단위(수천 행) unstratified 테이블을 만든다.
    split_stratified(ko_raw, "humann_genefamilies_ko_wide")

    logging.info("완료")


if __name__ == "__main__":
    main()
