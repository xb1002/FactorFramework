"""Standardization pipeline for factor values."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
import pandas as pd


class PreprocessStrategy(ABC):
    """预处理策略抽象基类。
    
    所有预处理策略都应继承此类并实现 apply 方法。
    """
    
    @abstractmethod
    def apply(self, series: pd.Series) -> pd.Series:
        """应用预处理策略。
        
        Args:
            series: 输入的因子值 Series
            
        Returns:
            处理后的 Series
        """
        ...


class WinsorizeStrategy(PreprocessStrategy):
    """缩尾处理策略。
    
    将极端值截断到指定分位数水平。
    
    Attributes:
        q: 缩尾分位数，默认 0.01 表示截断到 1% 和 99% 分位数
    """
    
    def __init__(self, q: float = 0.01) -> None:
        """初始化缩尾策略。
        
        Args:
            q: 缩尾分位数，范围 (0, 0.5)
        """
        self.q = q

    def apply(self, series: pd.Series) -> pd.Series:
        """对每个日期截面应用缩尾处理。
        
        Args:
            series: 输入 Series
            
        Returns:
            缩尾后的 Series
        """
        grouped_by_date = series.groupby(level="date")
        lower = grouped_by_date.transform(lambda x: x.quantile(self.q))
        upper = grouped_by_date.transform(lambda x: x.quantile(1 - self.q))
        return series.clip(lower=lower, upper=upper)


class ZScoreStrategy(PreprocessStrategy):
    """Z-Score 标准化策略。
    
    将因子值在每个日期截面上标准化为均值 0、标准差 1。
    """
    
    def apply(self, series: pd.Series) -> pd.Series:
        """对每个日期截面应用 Z-Score 标准化。
        
        Args:
            series: 输入 Series
            
        Returns:
            标准化后的 Series
        """
        grouped_by_date = series.groupby(level="date")
        mean = grouped_by_date.transform("mean")
        std = grouped_by_date.transform("std")
        return (series - mean) / std.replace(0, np.nan)


class Standardizer:
    """因子标准化器。
    
    按顺序应用一系列预处理策略到因子值上。
    
    Attributes:
        strategies: 预处理策略列表
    """

    def __init__(self, strategies: Iterable[PreprocessStrategy]) -> None:
        """初始化标准化器。
        
        Args:
            strategies: 预处理策略的可迭代对象
        """
        self.strategies = list(strategies)

    def apply(self, series: pd.Series) -> pd.Series:
        """按顺序应用所有策略。
        
        Args:
            series: 输入因子值 Series
            
        Returns:
            处理后的 Series
        """
        result = series
        for strategy in self.strategies:
            result = strategy.apply(result)
        return result

    @classmethod
    def from_config(cls, config: dict) -> "Standardizer":
        """从配置字典创建标准化器。
        
        支持的配置项：
        - winsorize_q: 缩尾分位数
        - zscore: 是否应用 Z-Score 标准化
        
        Args:
            config: 配置字典
            
        Returns:
            配置好的 Standardizer 实例
        """
        strategies: List[PreprocessStrategy] = []
        if config.get("winsorize_q") is not None:
            strategies.append(WinsorizeStrategy(config["winsorize_q"]))
        if config.get("zscore", True):
            strategies.append(ZScoreStrategy())
        return cls(strategies)
