"""io_utils.py — raw data loading and small I/O helpers shared across scripts."""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from . import config


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def load_raw(dataset: str) -> pd.DataFrame:
    """Load ICU.csv or mimic.csv as-is (all columns as string, no coercion).

    Never writes to or mutates the source files. Returns the frame in its
    original row order with an added 'dataset' column ('ICU' / 'MIMIC').
    """
    if dataset not in config.DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {config.DATASETS}")

    if dataset == "ICU":
        df = pd.read_csv(config.ICU_CSV, dtype=str, low_memory=False)
    else:
        df = pd.read_csv(config.MIMIC_CSV, dtype=str, low_memory=False, encoding="utf-8-sig")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.insert(0, "dataset", dataset)
    return df


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path
