#!/usr/bin/env python3
"""01_build_metadata.py — ICU/AGI/non_AGI 샘플 메타데이터 통합 (mortality 라벨 포함)."""
import logging
import os
import re

import pandas as pd

ICU_PUBLIC = "/home/tcmhwd/microbiome/ICU_public"
OUTPUT_DIR = os.path.join(ICU_PUBLIC, "merge", "output")

ICU_SAMPLE_LIST = os.path.join(ICU_PUBLIC, "ICU_sample_list.txt")
NEW_META_CSV = os.path.join(ICU_PUBLIC, "new_meta.csv")
AGI_PUBLIC_CSV = os.path.join(ICU_PUBLIC, "AGI_public.csv")
PUBLICDATA_CSV = os.path.join(ICU_PUBLIC, "publicdata.csv")
METADATA_XLSX = os.path.join(ICU_PUBLIC, "metadata.xlsx")


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def build_icu_metadata() -> pd.DataFrame:
    """ICU_sample_list.txt x new_meta.csv -> sample_id, patient_id, mortality, sample_type, age, sex."""
    logging.info("ICU 메타데이터 구축 중...")
    meta = pd.read_csv(NEW_META_CSV, encoding="utf-8-sig", dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    by_kcdc = {row["KCDC_no"].strip(): row for _, row in meta.iterrows()
               if pd.notna(row.get("KCDC_no"))}
    by_no = {row["No"].strip(): row for _, row in meta.iterrows()
             if pd.notna(row.get("No"))}

    rows = []
    n_missing = 0
    with open(ICU_SAMPLE_LIST, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sample_id, _clinical_id = parts[0].strip(), parts[1].strip()
            kcdc_key = sample_id.replace("_WM", "")
            row = None
            if kcdc_key in by_kcdc:
                row = by_kcdc[kcdc_key]
            elif sample_id in by_no:
                row = by_no[sample_id]
            elif kcdc_key in by_no:
                row = by_no[kcdc_key]
            if row is None:
                n_missing += 1
                logging.warning(f"  ICU 샘플 메타데이터 매칭 실패: {sample_id}")
                continue
            mort_raw = str(row.get("in.hospital.mortality", "")).strip()
            mortality = int(mort_raw) if mort_raw in ("0", "1") else None
            chart_no = str(row.get("환자번호(차트번호)", "")).strip()
            rows.append({
                "sample_id": sample_id,
                "dataset": "ICU",
                "subcohort": "ICU_own",
                "patient_id": f"ICU_{chart_no}",
                "mortality": mortality,
                "sample_type": str(row.get("검체종류 (stool/swab)", "")).strip(),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "timepoint": "",
            })
    logging.info(f"  ICU: {len(rows)}개 샘플 매칭, {n_missing}개 실패")
    return pd.DataFrame(rows)


def build_agi_metadata() -> pd.DataFrame:
    """AGI_public.csv Title 말미의 S<n>(생존)/D<n>(사망) 코드 파싱."""
    logging.info("AGI 메타데이터 구축 중...")
    df = pd.read_csv(AGI_PUBLIC_CSV, dtype=str)
    pattern = re.compile(r'([SD]\d+)\s*$')
    rows = []
    n_unparsed = 0
    for _, row in df.iterrows():
        title = str(row.get("Title", "")).strip()
        m = pattern.search(title)
        if not m:
            n_unparsed += 1
            logging.warning(f"  AGI Title 파싱 실패: Run={row['Run']} Title={title!r}")
            continue
        code_full = m.group(1)
        mortality = 1 if code_full[0] == "D" else 0
        rows.append({
            "sample_id": row["Run"].strip(),
            "dataset": "AGI",
            "subcohort": "AGI_BGISEQ",
            "patient_id": f"AGI_{code_full}",
            "mortality": mortality,
            "sample_type": "stool",
            "age": "",
            "sex": "",
            "timepoint": "",
        })
    logging.info(f"  AGI: {len(rows)}개 샘플 파싱, {n_unparsed}개 실패")
    return pd.DataFrame(rows)


def build_non_agi_metadata() -> pd.DataFrame:
    """publicdata.csv study_accession별 분기: 전원사망 2개 코호트 + mortality 컬럼 보유 1개 코호트."""
    logging.info("non_AGI 메타데이터 구축 중...")
    df = pd.read_csv(PUBLICDATA_CSV, dtype=str)
    rows = []

    death_pattern = re.compile(r'(?:ITO_)?(No\d+)(?:_(AM|PA))?(?:_RS)?', re.IGNORECASE)
    for study in ("PRJNA1100796", "PRJNA1102991"):
        sub = df[df["study_accession"] == study]
        for _, row in sub.iterrows():
            alias = str(row.get("sample_alias", ""))
            m = death_pattern.search(alias)
            patient_num = m.group(1) if m else alias
            timepoint = (m.group(2) or "") if m else ""
            rows.append({
                "sample_id": row["run_accession"].strip(),
                "dataset": "non_AGI",
                "subcohort": study,
                "patient_id": f"{study}_{patient_num}",
                "mortality": 1,
                "sample_type": "",
                "age": "",
                "sex": "",
                "timepoint": timepoint,
            })

    xl = pd.read_excel(METADATA_XLSX, sheet_name="PRJNA842425")
    mort_by_patient = xl.set_index("patient_ID")["mortality"].to_dict()
    sub = df[df["study_accession"] == "PRJNA842425"]
    n_unmatched = 0
    for _, row in sub.iterrows():
        alias = str(row.get("sample_alias", "")).strip()
        mortality = mort_by_patient.get(alias)
        if mortality is None:
            n_unmatched += 1
            logging.warning(f"  PRJNA842425 mortality 매칭 실패: {alias}")
            continue
        rows.append({
            "sample_id": row["run_accession"].strip(),
            "dataset": "non_AGI",
            "subcohort": "PRJNA842425",
            "patient_id": f"PRJNA842425_{alias}",
            "mortality": int(mortality),
            "sample_type": "stool",
            "age": "",
            "sex": "",
            "timepoint": "",
        })
    logging.info(f"  non_AGI: {len(rows)}개 샘플, PRJNA842425 미매칭 {n_unmatched}개")
    return pd.DataFrame(rows)


def combine_and_finalize(icu: pd.DataFrame, agi: pd.DataFrame,
                          non_agi: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([icu, agi, non_agi], ignore_index=True)
    all_df["n_samples_for_patient"] = all_df.groupby("patient_id")["sample_id"].transform("count")
    all_df = all_df.sort_values(["patient_id", "sample_id"]).reset_index(drop=True)
    all_df["is_first_sample_for_patient"] = ~all_df.duplicated(subset="patient_id", keep="first")
    return all_df


def main():
    setup_logging()
    icu = build_icu_metadata()
    agi = build_agi_metadata()
    non_agi = build_non_agi_metadata()
    combined = combine_and_finalize(icu, agi, non_agi)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "sample_metadata.tsv")
    combined.to_csv(out_path, sep="\t", index=False)

    logging.info("=" * 50)
    logging.info(f"총 샘플: {len(combined)}")
    for ds in ("ICU", "AGI", "non_AGI"):
        sub = combined[combined["dataset"] == ds]
        logging.info(f"  {ds}: {len(sub)}개 "
                     f"(mortality=1: {(sub['mortality'] == 1).sum()}, "
                     f"mortality=0: {(sub['mortality'] == 0).sum()}, "
                     f"결측: {sub['mortality'].isna().sum()})")
    logging.info(f"고유 환자 수: {combined['patient_id'].nunique()}")
    n_repeat = combined.loc[combined["n_samples_for_patient"] > 1, "patient_id"].nunique()
    logging.info(f"반복 채취 환자 수(n>1): {n_repeat}")
    logging.info(f"저장: {out_path}")


if __name__ == "__main__":
    main()
