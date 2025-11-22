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
    """Abstract market data source.

    Implementations should return a DataFrame with a MultiIndex of
    ``(date, code)`` sorted ascending, containing at least OHLCV fields.
    """

    @abstractmethod
    def load(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fields: Optional[Iterable[str]] = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Load market data for the given range and fields."""


class LocalParquetSource(MarketDataSource):
    """Simple local parquet/csv data source.

    It performs basic normalization (datetime index, MultiIndex, sorting) and
    supports optional caching via LRU on the load arguments.
    """

    def __init__(self, path: str | Path, cache: bool = True) -> None:
        self.path = Path(path)
        self.cache = cache

    def _read(self) -> pd.DataFrame:
        if self.path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.path)
        elif self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported file type: {self.path.suffix}")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if start is not None:
            df = df[df.index.get_level_values(0) >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index.get_level_values(0) <= pd.to_datetime(end)]
        return df

    def _filter_fields(self, df: pd.DataFrame, fields: Optional[Iterable[str]]) -> pd.DataFrame:
        if fields is None:
            return df
        missing = set(fields) - set(df.columns)
        if missing:
            raise KeyError(f"Requested fields missing in data: {missing}")
        return df[list(fields)]

    def _cache_key(
        self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], fields: Optional[Iterable[str]], freq: str
    ) -> str:
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
