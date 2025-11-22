"""Utilities to load raw data, clean, and persist processed market data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV or Parquet file into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform lightweight cleaning: drop duplicates and remove zero prices."""
    df = df.drop_duplicates()
    if {"close", "open", "high", "low"}.issubset(df.columns):
        price_cols = ["close", "open", "high", "low"]
        for col in price_cols:
            df.loc[df[col] == 0, col] = pd.NA
    return df


def persist_processed(df: pd.DataFrame, dst: str | Path) -> Path:
    """Write processed data to parquet in the processed directory."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    return dst


def load_and_clean(src: str | Path, dst: Optional[str | Path] = None) -> Path:
    df = load_raw(src)
    df = clean(df)
    if dst is None:
        src_path = Path(src)
        dst = src_path.with_suffix(".parquet")
    return persist_processed(df, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw market data into processed parquet")
    parser.add_argument("src", help="Path to raw CSV or Parquet file")
    parser.add_argument("dst", nargs="?", help="Destination processed parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dst = load_and_clean(args.src, args.dst)
    print(f"Processed data written to {dst}")


if __name__ == "__main__":
    main()
