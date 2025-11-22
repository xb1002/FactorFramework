"""Factor evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class HorizonMetrics:
    """单个时间窗口的评价指标。
    
    Attributes:
        rank_ic_mean: 平均秩相关系数（Rank IC）
        icir: 信息比率（IC / IC 标准差）
        turnover_adjusted: 调整后的换手率
    """
    rank_ic_mean: float
    icir: float
    turnover_adjusted: float


@dataclass
class FactorReport:
    """因子评价报告。
    
    包含因子在多个时间窗口上的评价指标。
    
    Attributes:
        factor_name: 因子名称
        metrics: 各时间窗口的评价指标字典
        best_horizon: 最佳时间窗口（按 IC 选择）
        extra: 额外信息字典
    """
    factor_name: str
    metrics: Dict[int, HorizonMetrics]
    best_horizon: Optional[int] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典格式。
        
        Returns:
            包含所有信息的字典
        """
        return {
            "factor_name": self.factor_name,
            "metrics": {h: m.__dict__ for h, m in self.metrics.items()},
            "best_horizon": self.best_horizon,
            "extra": self.extra,
        }


class FactorEvaluator:
    """因子评价器。
    
    计算因子的多期预测能力和换手率等指标。
    
    Attributes:
        horizons: 评价的时间窗口列表
    """
    
    def __init__(self, horizons: Iterable[int]) -> None:
        """初始化评价器。
        
        Args:
            horizons: 时间窗口列表
        """
        self.horizons = list(horizons)

    def evaluate(
        self,
        factor: pd.Series,
        fwd_returns: Dict[int, pd.Series],
        universe_mask: Optional[pd.Series] = None,
    ) -> FactorReport:
        """评价因子表现。
        
        对每个时间窗口计算：
        1. 平均 Rank IC（因子与未来收益的 Spearman 相关系数）
        2. ICIR（信息比率）
        3. 调整后换手率
        
        Args:
            factor: 因子值 Series
            fwd_returns: 前瞻收益率字典（由 forward_return.build 生成）
            universe_mask: 可交易标的掩码（可选）
            
        Returns:
            包含所有指标的 FactorReport 对象
        """
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
        """对齐因子值和前瞻收益，并应用筛选。
        
        Args:
            factor: 因子值
            fwd: 前瞻收益
            universe_mask: 筛选掩码
            
        Returns:
            对齐并筛选后的 (factor, fwd) 元组
        """
        joined = pd.concat([factor, fwd], axis=1, keys=["factor", "fwd"]).dropna()
        if universe_mask is not None:
            joined = joined[universe_mask.reindex(joined.index).fillna(False)]
        return joined["factor"], joined["fwd"]

    def _turnover(self, factor: pd.Series) -> float:
        """计算因子的平均换手率。
        
        基于每个日期因子排序后的标的集合变化计算。
        
        Args:
            factor: 因子值 Series
            
        Returns:
            平均换手率（0-1 之间）
        """
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
        """选择最佳时间窗口。
        
        按 Rank IC 均值选择表现最好的窗口。
        
        Args:
            metrics: 各窗口的指标字典
            
        Returns:
            最佳窗口，无指标时返回 None
        """
        if not metrics:
            return None
        return max(metrics.items(), key=lambda kv: kv[1].rank_ic_mean)[0]
