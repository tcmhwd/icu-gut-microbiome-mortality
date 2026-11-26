#!/usr/bin/env python3
"""02_metaphlan3_merge.py — ICU/AGI/non_AGI MetaPhlAn3 프로파일 병합."""
import glob
import logging
import os

import pandas as pd

ICU_PUBLIC = "/home/tcmhwd/microbiome/ICU_public"
OUTPUT_DIR = os.path.join(ICU_PUBLIC, "merge", "output", "metaphlan3")

DATASETS = {
    "ICU": os.path.join(ICU_PUBLIC, "ICU", "metaphlan3"),
    "AGI": os.path.join(ICU_PUBLIC, "AGI", "metaphlan3"),
    "non_AGI": os.path.join(ICU_PUBLIC, "non_AGI", "metaphlan3"),
}


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def _read_metaphlan_profile(filepath: str) -> pd.DataFrame:
    """MetaPhlAn3 profile 파일을 읽는다. 진짜 헤더 줄(#clade_name...)도 '#'로
    시작하므로 comment='#'로 읽으면 헤더가 통째로 스킵되고 첫 데이터 행이
    헤더로 잘못 승격되는 문제를 피하기 위해 직접 파싱한다."""
    from io import StringIO
    lines = []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#clade_name"):
                lines.append(line[1:])  # 앞의 '#'만 제거해 진짜 헤더로 사용
            elif line.startswith("#"):
                continue  # 다른 주석 줄은 건너뜀
            else:
                lines.append(line)
    return pd.read_csv(StringIO("".join(lines)), sep="\t")


def load_all_profiles() -> list:
    series_list = []
    for dataset, path in DATASETS.items():
        files = sorted(glob.glob(os.path.join(path, "*_profile.tsv")))
        logging.info(f"  {dataset}: {len(files)}개 프로파일")
        for f in files:
            sample = os.path.basename(f).replace("_profile.tsv", "")
            try:
                tmp = _read_metaphlan_profile(f)
                if tmp.empty:
                    continue
                clade_col = tmp.columns[0]
                abund_col = next(
                    (c for c in tmp.columns if "relative_abundance" in c.lower()),
                    tmp.columns[2] if len(tmp.columns) > 2 else tmp.columns[1]
                )
                s = tmp.set_index(clade_col)[abund_col]
                s.name = sample
                series_list.append(s)
            except Exception as e:
                logging.warning(f"  읽기 실패 {f}: {e}")
    return series_list


def save(df: pd.DataFrame, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, sep="\t")
    logging.info(f"  저장: {path}  [{df.shape[0]}행 x {df.shape[1]}열]")


def main():
    setup_logging()
    series_list = load_all_profiles()
    if not series_list:
        logging.error("MetaPhlAn3 프로파일을 하나도 읽지 못했습니다.")
        return

    merged = pd.concat(series_list, axis=1).fillna(0)
    merged.index.name = "clade_name"
    save(merged, "metaphlan3_merged.tsv")

    for level, prefix in [("phylum", "p__"), ("class", "c__"), ("order", "o__"),
                           ("family", "f__"), ("genus", "g__"), ("species", "s__")]:
        sub = merged[merged.index.map(lambda x: x.split("|")[-1].startswith(prefix))]
        if not sub.empty:
            save(sub, f"metaphlan3_{level}_merged.tsv")

    logging.info(f"총 {merged.shape[1]}개 샘플, {merged.shape[0]}개 clade")


if __name__ == "__main__":
    main()
