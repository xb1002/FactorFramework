"""Factor evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from numba import jit, prange


@jit(nopython=True, cache=True)
def _rank_data_numba(x):
    """使用 numba 优化的排序函数
    
    Args:
        x: 输入数组
        
    Returns:
        排序后的秩数组
    """
    n = len(x)
    sorter = np.argsort(x)
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    ranks = inv.astype(np.float64) + 1
    return ranks


@jit(nopython=True, cache=True)
def _spearman_numba(x, y):
    """使用 numba 优化的 Spearman 相关系数计算
    
    Args:
        x: 因子值数组
        y: 收益率数组
        
    Returns:
        Spearman 相关系数
    """
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    
    # 计算秩
    rank_x = _rank_data_numba(x)
    rank_y = _rank_data_numba(y)
    
    # 计算 Pearson 相关系数（秩的相关性即 Spearman）
    mean_x = np.mean(rank_x)
    mean_y = np.mean(rank_y)
    
    cov = np.sum((rank_x - mean_x) * (rank_y - mean_y))
    std_x = np.sqrt(np.sum((rank_x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((rank_y - mean_y) ** 2))
    
    if std_x == 0 or std_y == 0:
        return np.nan
    
    return cov / (std_x * std_y)


@jit(nopython=True, cache=True, parallel=True)
def _compute_ic_batch(factor_arr, fwd_arr, date_starts, date_ends):
    """并行计算每个日期的 IC
    
    Args:
        factor_arr: 因子值数组
        fwd_arr: 前瞻收益数组
        date_starts: 每个日期的起始索引数组
        date_ends: 每个日期的结束索引数组
        
    Returns:
        每个日期的 IC 数组
    """
    n_dates = len(date_starts)
    ic_array = np.empty(n_dates, dtype=np.float64)
    
    for i in prange(n_dates):
        start = date_starts[i]
        end = date_ends[i]
        if end - start < 2:
            ic_array[i] = np.nan
        else:
            ic_array[i] = _spearman_numba(
                factor_arr[start:end],
                fwd_arr[start:end]
            )
    
    return ic_array


@jit(nopython=True, cache=True)
def _compute_turnover_numba(top_codes_matrix, n_top):
    """使用 numba 优化的换手率计算
    
    Args:
        top_codes_matrix: shape (n_dates, n_top) 每日前 N 名股票代码（整数编码）
        n_top: 前 N 名数量
        
    Returns:
        平均换手率
    """
    n_dates = top_codes_matrix.shape[0]
    if n_dates < 2:
        return np.nan
    
    turnovers = np.empty(n_dates - 1, dtype=np.float64)
    
    for i in range(n_dates - 1):
        # 使用排序后的二路归并算法计算交集
        current = np.sort(top_codes_matrix[i])
        next_day = np.sort(top_codes_matrix[i + 1])
        
        # 双指针计算交集大小
        overlap = 0
        j = 0
        k = 0
        
        while j < len(current) and k < len(next_day):
            if current[j] == -1:  # 填充值
                break
            if next_day[k] == -1:
                break
                
            if current[j] == next_day[k]:
                overlap += 1
                j += 1
                k += 1
            elif current[j] < next_day[k]:
                j += 1
            else:
                k += 1
        
        turnover = 1.0 - overlap / n_top if n_top > 0 else 0.0
        turnovers[i] = turnover
    
    return np.mean(turnovers)


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
            
            # 计算 IC（使用 numba 优化）
            ic_array = self._compute_ic_optimized(aligned_factor, aligned_fwd)
            rank_ic_mean = np.nanmean(ic_array)
            ic_std = np.nanstd(ic_array)
            icir = rank_ic_mean / ic_std if ic_std and not np.isnan(ic_std) else np.nan
            
            # 计算换手率（简化版，更快）
            print("计算换手率...", end=" ", flush=True)
            turnover_adj = self._turnover_fast(aligned_factor) / h
            
            metrics[h] = HorizonMetrics(rank_ic_mean, icir, turnover_adj)
            print(f"✓ IC={rank_ic_mean:.4f}, ICIR={icir:.4f}, 换手={turnover_adj:.4f}")
            
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

    def _compute_ic_optimized(self, factor: pd.Series, fwd: pd.Series) -> np.ndarray:
        """使用 numba 优化的 IC 计算
        
        Args:
            factor: 因子值 Series
            fwd: 前瞻收益 Series
            
        Returns:
            每个日期的 IC 数组
        """
        # 准备数据：按日期和代码排序
        df = pd.DataFrame({"factor": factor, "fwd": fwd})
        df = df.sort_index(level=["date", "code"])
        
        # 获取唯一日期
        dates = df.index.get_level_values("date").unique()
        n_dates = len(dates)
        
        # 构建每个日期的起始和结束索引
        date_starts = np.empty(n_dates, dtype=np.int64)
        date_ends = np.empty(n_dates, dtype=np.int64)
        
        pos = 0
        for i, date in enumerate(dates):
            # 计算该日期的数据量
            count = (df.index.get_level_values("date") == date).sum()
            date_starts[i] = pos
            date_ends[i] = pos + count
            pos += count
        
        # 调用 numba 优化的批量 IC 计算
        ic_array = _compute_ic_batch(
            df["factor"].values,
            df["fwd"].values,
            date_starts,
            date_ends
        )
        
        return ic_array

    def _turnover(self, factor: pd.Series) -> float:
        """计算因子的平均换手率（原始版本，较慢）。
        
        基于每个日期因子排序后的标的集合变化计算。
        
        Args:
            factor: 因子值 Series
            
        Returns:
            平均换手率（0-1 之间）
        """
        sorted_codes = factor.groupby(level="date").apply(
            lambda x: x.sort_values(ascending=False).index.get_level_values("code")
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
        """计算因子的平均换手率（numba 优化版本）。
        
        使用 numba JIT 加速换手率计算。
        采样前20%的股票计算换手率。
        
        Args:
            factor: 因子值 Series
            
        Returns:
            平均换手率（0-1 之间）
        """
        # 按日期分组
        dates = sorted(factor.index.get_level_values("date").unique())
        
        if len(dates) < 2:
            return np.nan
        
        # 为每个日期找出前 20% 的股票
        grouped = factor.groupby(level="date")
        n_top = None
        
        # 收集每日前 N 名股票代码（转为整数编码）
        all_codes = factor.index.get_level_values("code").unique()
        code_to_int = {code: i for i, code in enumerate(all_codes)}
        
        top_codes_list = []
        for date in dates:
            date_data = grouped.get_group(date).sort_values(ascending=False)
            if n_top is None:
                n_top = max(1, int(len(date_data) * 0.2))
            
            top_codes = date_data.head(n_top).index.get_level_values("code")
            top_codes_int = np.array([code_to_int[code] for code in top_codes], dtype=np.int64)
            
            # 填充到固定长度
            if len(top_codes_int) < n_top:
                top_codes_int = np.pad(top_codes_int, (0, n_top - len(top_codes_int)), constant_values=-1)
            
            top_codes_list.append(top_codes_int)
        
        top_codes_matrix = np.array(top_codes_list, dtype=np.int64)
        
        # 调用 numba 优化的函数
        return float(_compute_turnover_numba(top_codes_matrix, n_top))

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
