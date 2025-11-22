"""Factor evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class HorizonMetrics:
    rank_ic_mean: float
    icir: float
    turnover_adjusted: float


@dataclass
class FactorReport:
    factor_name: str
    metrics: Dict[int, HorizonMetrics]
    best_horizon: Optional[int] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "metrics": {h: m.__dict__ for h, m in self.metrics.items()},
            "best_horizon": self.best_horizon,
            "extra": self.extra,
        }


class FactorEvaluator:
    def __init__(self, horizons: Iterable[int]) -> None:
        self.horizons = list(horizons)

    def evaluate(
        self,
        factor: pd.Series,
        fwd_returns: Dict[int, pd.Series],
        universe_mask: Optional[pd.Series] = None,
    ) -> FactorReport:
        metrics: Dict[int, HorizonMetrics] = {}
        for h in self.horizons:
            fwd = fwd_returns[h]
            aligned_factor, aligned_fwd = self._align(factor, fwd, universe_mask)
            ic_series = aligned_factor.groupby(level="date").corr(aligned_fwd, method="spearman")
            rank_ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = rank_ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan
            turnover_adj = self._turnover(aligned_factor)
            metrics[h] = HorizonMetrics(rank_ic_mean, icir, turnover_adj)
        best_horizon = self._best_horizon(metrics)
        return FactorReport(factor.name, metrics, best_horizon)

    def _align(
        self, factor: pd.Series, fwd: pd.Series, universe_mask: Optional[pd.Series]
    ) -> tuple[pd.Series, pd.Series]:
        joined = pd.concat([factor, fwd], axis=1, keys=["factor", "fwd"]).dropna()
        if universe_mask is not None:
            joined = joined[universe_mask.reindex(joined.index).fillna(False)]
        return joined["factor"], joined["fwd"]

    def _turnover(self, factor: pd.Series) -> float:
        sorted_codes = factor.groupby(level="date").apply(
            lambda x: x.sort_values(ascending=False).index.get_level_values(1)
        )
        turnovers = []
        previous = None
        for codes in sorted_codes:
            if previous is not None:
                overlap = len(set(previous) & set(codes))
                total = len(set(previous) | set(codes))
                turnovers.append(1 - overlap / total if total else np.nan)
            previous = codes
        return float(np.nanmean(turnovers)) if turnovers else np.nan

    def _best_horizon(self, metrics: Dict[int, HorizonMetrics]) -> Optional[int]:
        if not metrics:
            return None
        return max(metrics.items(), key=lambda kv: kv[1].rank_ic_mean)[0]
