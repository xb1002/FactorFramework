"""FactorEngine orchestrates factor computation with data loading and preprocessing."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from project.data_manager.datasource import MarketDataSource
from project.data_manager.universe import Universe, DefaultUniverse
from project.factors.registry import get
from project.factors.standardizer import Standardizer


class FactorEngine:
    """因子计算引擎。
    
    协调数据加载、因子计算、标准化和筛选的完整流程。
    
    Attributes:
        source: 市场数据源
        standardizer: 因子标准化器
        default_universe: 默认的可交易标的筛选器
    """
    
    def __init__(
        self,
        source: MarketDataSource,
        standardizer: Standardizer,
        default_universe: Optional[Universe] = None,
    ) -> None:
        """初始化因子引擎。
        
        Args:
            source: 市场数据源实例
            standardizer: 标准化器实例
            default_universe: 默认筛选器，None 时使用 DefaultUniverse
        """
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
        """计算因子值。
        
        完整流程：
        1. 从注册表获取因子规范
        2. 加载所需的市场数据
        3. 执行因子计算函数
        4. 应用标准化处理
        5. 应用筛选器过滤
        
        Args:
            factor_name: 已注册的因子名称
            start: 起始日期
            end: 结束日期
            universe: 筛选器，None 时使用默认筛选器
            
        Returns:
            计算并处理后的因子值 Series
            
        Raises:
            TypeError: 当因子函数返回值不是 Series 时
        """
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
