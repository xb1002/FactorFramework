"""Universe definitions and filters for tradable instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Universe(ABC):
    """Abstract universe filter."""

    @abstractmethod
    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """Return a boolean mask aligned with the input index."""


class DefaultUniverse(Universe):
    """Default universe: drop NaNs and suspend samples with zero volume/amount."""

    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(data, pd.Series):
            base = data
        else:
            base = data.iloc[:, 0]
        mask = ~base.isna()
        if isinstance(data, pd.DataFrame):
            for col in ("volume", "amount"):
                if col in data.columns:
                    mask &= data[col] > 0
        return mask


class CompositeUniverse(Universe):
    """Combine multiple universe filters with logical AND."""

    def __init__(self, *universes: Universe) -> None:
        self.universes = universes

    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        mask = None
        for uni in self.universes:
            current = uni.mask(data)
            mask = current if mask is None else (mask & current)
        return mask if mask is not None else pd.Series(True, index=data.index)
