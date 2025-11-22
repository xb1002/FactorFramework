"""Standardization pipeline for factor values."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
import pandas as pd


class PreprocessStrategy(ABC):
    @abstractmethod
    def apply(self, series: pd.Series) -> pd.Series:
        ...


class WinsorizeStrategy(PreprocessStrategy):
    def __init__(self, q: float = 0.01) -> None:
        self.q = q

    def apply(self, series: pd.Series) -> pd.Series:
        grouped_by_date = series.groupby(level="date")
        lower = grouped_by_date.transform(lambda x: x.quantile(self.q))
        upper = grouped_by_date.transform(lambda x: x.quantile(1 - self.q))
        return series.clip(lower=lower, upper=upper)


class ZScoreStrategy(PreprocessStrategy):
    def apply(self, series: pd.Series) -> pd.Series:
        grouped_by_date = series.groupby(level="date")
        mean = grouped_by_date.transform("mean")
        std = grouped_by_date.transform("std")
        return (series - mean) / std.replace(0, np.nan)


class Standardizer:
    """Apply a sequence of preprocessing strategies."""

    def __init__(self, strategies: Iterable[PreprocessStrategy]) -> None:
        self.strategies = list(strategies)

    def apply(self, series: pd.Series) -> pd.Series:
        result = series
        for strategy in self.strategies:
            result = strategy.apply(result)
        return result

    @classmethod
    def from_config(cls, config: dict) -> "Standardizer":
        strategies: List[PreprocessStrategy] = []
        if config.get("winsorize_q") is not None:
            strategies.append(WinsorizeStrategy(config["winsorize_q"]))
        if config.get("zscore", True):
            strategies.append(ZScoreStrategy())
        return cls(strategies)
