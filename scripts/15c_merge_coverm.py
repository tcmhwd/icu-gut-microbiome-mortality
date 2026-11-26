#!/usr/bin/env python3
"""Merge per-batch coverm TSVs into a single matrix."""
import sys
import glob
import pandas as pd

ICU_DIR = "/home/tcmhwd/microbiome/ICU_public"
BATCH_DIR = f"{ICU_DIR}/coverm_result/batches"
OUT_TSV = f"{ICU_DIR}/coverm_result/all_samples.coverm.tsv"

batch_files = sorted(glob.glob(f"{BATCH_DIR}/batch_*.coverm.tsv"))
if not batch_files:
    sys.exit(f"[ERROR] No batch files found in {BATCH_DIR}")

print(f"[INFO] Merging {len(batch_files)} batch files...")

dfs = []
for f in batch_files:
    df = pd.read_csv(f, sep="\t", index_col=0)
    dfs.append(df)
    print(f"  {f}: {df.shape[1]} columns")

merged = pd.concat(dfs, axis=1)
print(f"[INFO] Merged shape: {merged.shape}")

merged.to_csv(OUT_TSV, sep="\t")
print(f"[DONE] → {OUT_TSV}")
