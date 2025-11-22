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
        print(f"   评价 {len(self.horizons)} 个时间窗口: {self.horizons}")
        
        for i, h in enumerate(self.horizons, 1):
            print(f"   [{i}/{len(self.horizons)}] 窗口 {h} 天...", end=" ", flush=True)
            fwd = fwd_returns[h]
            aligned_factor, aligned_fwd = self._align(factor, fwd, universe_mask)
            
            # 计算 IC
            ic_series = aligned_factor.groupby(level="date").corr(aligned_fwd, method="spearman")
            rank_ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = rank_ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan
            
            # 计算换手率（简化版，更快）
            print("计算换手率...", end=" ", flush=True)
            turnover_adj = self._turnover_fast(aligned_factor) / h
            
            metrics[h] = HorizonMetrics(rank_ic_mean, icir, turnover_adj)
            print(f"✓ IC={rank_ic_mean:.4f}, ICIR={icir:.4f}")
            
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
        """计算因子的平均换手率（原始版本，较慢）。
        
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
    
    def _turnover_fast(self, factor: pd.Series) -> float:
        """计算因子的平均换手率（优化版本，更快）。
        
        使用向量化操作计算换手率，适合大数据集。
        采样前20%的股票计算换手率。
        
        Args:
            factor: 因子值 Series
            
        Returns:
            平均换手率（0-1 之间）
        """
        # 按日期分组
        grouped = factor.groupby(level="date")
        dates = sorted(factor.index.get_level_values(0).unique())
        
        if len(dates) < 2:
            return np.nan
        
        turnovers = []
        previous_top = None
        
        # 只取前20%作为换手率计算（大幅加速）
        for date in dates:
            date_data = grouped.get_group(date).sort_values(ascending=False)
            n_top = max(1, int(len(date_data) * 0.2))
            current_top = set(date_data.head(n_top).index.get_level_values(1))
            
            if previous_top is not None:
                overlap = len(previous_top & current_top)
                turnover = 1 - overlap / n_top if n_top > 0 else 0
                turnovers.append(turnover)
            
            previous_top = current_top
        
        return float(np.mean(turnovers)) if turnovers else np.nan

    def _best_horizon(self, metrics: Dict[int, HorizonMetrics]) -> Optional[int]:
        """选择最佳时间窗口。
        
        按 Rank IC 绝对值均值选择表现最好的窗口。
        负 IC 的因子同样有价值（可以反向使用）。
        
        Args:
            metrics: 各窗口的指标字典
            
        Returns:
            最佳窗口，无指标时返回 None
        """
        if not metrics:
            return None
        return max(metrics.items(), key=lambda kv: abs(kv[1].rank_ic_mean))[0]
