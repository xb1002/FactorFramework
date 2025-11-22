"""Data source abstractions and implementations for market data."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class MarketDataSource(ABC):
    """市场数据源抽象基类。

    所有数据源实现都应该继承此类，并实现 load 方法。
    返回的 DataFrame 必须满足以下要求：
    - 使用 MultiIndex (date, code) 作为索引
    - 索引按升序排序
    - 至少包含 OHLCV (开高低收成交量) 字段
    
    Attributes:
        无特定属性要求
    """

    @abstractmethod
    def load(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fields: Optional[Iterable[str]] = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """加载指定范围和字段的市场数据。
        
        Args:
            start: 起始日期，None 表示从数据最早日期开始
            end: 结束日期，None 表示到数据最晚日期
            fields: 需要的字段列表，None 表示加载所有字段
            freq: 数据频率，默认 "1d" 表示日频
        
        Returns:
            包含市场数据的 DataFrame，索引为 MultiIndex(date, code)
        """


class LocalParquetSource(MarketDataSource):
    """本地 Parquet/CSV 文件数据源实现。

    提供以下功能：
    - 读取本地 Parquet 或 CSV 格式的市场数据
    - 自动执行数据规范化（日期时间索引、MultiIndex、排序）
    - 支持基于 LRU 缓存的查询结果缓存
    
    Attributes:
        path: 数据文件路径
        cache: 是否启用缓存机制
    """

    def __init__(self, path: str | Path, cache: bool = True) -> None:
        """初始化本地数据源。
        
        Args:
            path: 数据文件路径（支持 .parquet 和 .csv 格式）
            cache: 是否启用 LRU 缓存，默认 True
        """
        self.path = Path(path)
        self.cache = cache

    def _read(self) -> pd.DataFrame:
        """读取文件内容。
        
        根据文件后缀名选择相应的读取方法。
        
        Returns:
            原始 DataFrame
            
        Raises:
            ValueError: 当文件类型不支持时
        """
        if self.path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.path)
        elif self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported file type: {self.path.suffix}")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """规范化数据格式。
        
        执行以下操作：
        1. 验证必需的 date 和 code 列是否存在
        2. 将 date 列转换为 datetime 类型
        3. 设置 (date, code) 为 MultiIndex
        4. 对索引进行排序
        
        Args:
            df: 原始 DataFrame
            
        Returns:
            规范化后的 DataFrame
            
        Raises:
            ValueError: 当缺少必需列时
        """
        if "date" not in df.columns or "code" not in df.columns:
            raise ValueError("Input data must contain 'date' and 'code' columns")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(["date", "code"], inplace=True)
        df.sort_index(inplace=True)
        return df

    def _filter_range(
        self, df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """根据日期范围过滤数据。
        
        Args:
            df: 待过滤的 DataFrame
            start: 起始日期
            end: 结束日期
            
        Returns:
            过滤后的 DataFrame
        """
        if start is not None:
            df = df[df.index.get_level_values(0) >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index.get_level_values(0) <= pd.to_datetime(end)]
        return df

    def _filter_fields(self, df: pd.DataFrame, fields: Optional[Iterable[str]]) -> pd.DataFrame:
        """选择指定的字段。
        
        Args:
            df: 待过滤的 DataFrame
            fields: 需要的字段列表，None 表示返回所有字段
            
        Returns:
            仅包含指定字段的 DataFrame
            
        Raises:
            KeyError: 当请求的字段不存在时
        """
        if fields is None:
            return df
        missing = set(fields) - set(df.columns)
        if missing:
            raise KeyError(f"Requested fields missing in data: {missing}")
        return df[list(fields)]

    def _cache_key(
        self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], fields: Optional[Iterable[str]], freq: str
    ) -> str:
        """生成缓存键。
        
        基于查询参数生成唯一的 MD5 哈希值作为缓存键。
        
        Args:
            start: 起始日期
            end: 结束日期
            fields: 字段列表
            freq: 数据频率
            
        Returns:
            缓存键（MD5 哈希字符串）
        """
        fields_tuple = tuple(sorted(fields)) if fields is not None else None
        raw = json.dumps({"start": str(start), "end": str(end), "fields": fields_tuple, "freq": freq})
        return hashlib.md5(raw.encode()).hexdigest()

    def load(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fields: Optional[Iterable[str]] = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        if freq != "1d":
            raise ValueError("LocalParquetSource currently supports only daily frequency")

        if self.cache:
            return self._cached_load(start, end, None if fields is None else tuple(fields), freq)
        return self._load_impl(start, end, fields, freq)

    @lru_cache(maxsize=16)
    def _cached_load(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        fields: Optional[tuple[str, ...]],
        freq: str,
    ) -> pd.DataFrame:
        return self._load_impl(start, end, fields, freq)

    def _load_impl(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        fields: Optional[Iterable[str]],
        freq: str,
    ) -> pd.DataFrame:
        df = self._read()
        df = self._normalize(df)
        df = self._filter_range(df, start, end)
        df = self._filter_fields(df, fields)
        return df
