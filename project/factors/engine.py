"""FactorEngine orchestrates factor computation with data loading and preprocessing."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from project.data_manager.datasource import MarketDataSource
from project.data_manager.universe import Universe, DefaultUniverse
from project.factors.registry import get
from project.factors.standardizer import Standardizer


class FactorEngine:
    def __init__(
        self,
        source: MarketDataSource,
        standardizer: Standardizer,
        default_universe: Optional[Universe] = None,
    ) -> None:
        self.source = source
        self.standardizer = standardizer
        self.default_universe = default_universe or DefaultUniverse()

    def compute(
        self,
        factor_name: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        universe: Optional[Universe] = None,
    ) -> pd.Series:
        spec = get(factor_name)
        df = self.source.load(start=start, end=end, fields=spec.required_fields)
        raw_factor = spec.func(df)
        if not isinstance(raw_factor, pd.Series):
            raise TypeError("Factor function must return a pandas Series")
        raw_factor = raw_factor.rename(factor_name)
        aligned = raw_factor.reindex(df.index)
        standardized = self.standardizer.apply(aligned)
        universe_filter = universe or self.default_universe
        mask = universe_filter.mask(df.join(standardized.rename("factor")))
        return standardized[mask]
