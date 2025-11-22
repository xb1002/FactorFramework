"""Utilities to load raw data, clean, and persist processed market data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """加载原始 CSV 或 Parquet 文件。
    
    Args:
        path: 文件路径
        
    Returns:
        加载的 DataFrame
        
    Raises:
        ValueError: 当文件类型不支持时
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """执行轻量级数据清洗。
    
    清洗操作包括：
    1. 删除重复行
    2. 将价格为 0 的值替换为 NA（如果存在 close/open/high/low 列）
    
    Args:
        df: 待清洗的 DataFrame
        
    Returns:
        清洗后的 DataFrame
    """
    df = df.drop_duplicates()
    if {"close", "open", "high", "low"}.issubset(df.columns):
        price_cols = ["close", "open", "high", "low"]
        for col in price_cols:
            df.loc[df[col] == 0, col] = pd.NA
    return df


def persist_processed(df: pd.DataFrame, dst: str | Path) -> Path:
    """将处理后的数据写入 Parquet 文件。
    
    自动创建目标目录（如果不存在）。
    
    Args:
        df: 待保存的 DataFrame
        dst: 目标文件路径
        
    Returns:
        保存文件的 Path 对象
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    return dst


def load_and_clean(src: str | Path, dst: Optional[str | Path] = None) -> Path:
    """加载、清洗并保存数据的便捷函数。
    
    执行完整的数据处理流程：加载 -> 清洗 -> 保存。
    
    Args:
        src: 源文件路径
        dst: 目标文件路径，None 时使用源文件名并改为 .parquet 后缀
        
    Returns:
        保存文件的 Path 对象
    """
    df = load_raw(src)
    df = clean(df)
    if dst is None:
        src_path = Path(src)
        dst = src_path.with_suffix(".parquet")
    return persist_processed(df, dst)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        包含解析后参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(description="Clean raw market data into processed parquet")
    parser.add_argument("src", help="Path to raw CSV or Parquet file")
    parser.add_argument("dst", nargs="?", help="Destination processed parquet path")
    return parser.parse_args()


def main() -> None:
    """命令行入口函数。
    
    从命令行接收参数，执行数据加载、清洗和保存流程。
    """
    args = parse_args()
    dst = load_and_clean(args.src, args.dst)
    print(f"Processed data written to {dst}")


if __name__ == "__main__":
    main()
