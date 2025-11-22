"""Universe definitions and filters for tradable instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Universe(ABC):
    """可交易标的筛选器抽象基类。
    
    用于定义哪些标的在特定时间点可以交易。
    所有具体的 Universe 实现都应继承此类并实现 mask 方法。
    """

    @abstractmethod
    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """生成布尔掩码以筛选可交易标的。
        
        Args:
            data: 输入的 Series 或 DataFrame
            
        Returns:
            与输入索引对齐的布尔 Series，True 表示可交易
        """


class DefaultUniverse(Universe):
    """默认的可交易标的筛选器。
    
    筛选规则：
    1. 排除 NaN 值
    2. 排除成交量或成交额为 0 的样本（如果数据中包含这些字段）
    """

    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """生成默认筛选掩码。
        
        Args:
            data: 输入数据
            
        Returns:
            布尔掩码 Series
        """
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
    """组合多个筛选器的复合筛选器。
    
    使用逻辑 AND 运算组合多个 Universe 筛选器，
    只有同时满足所有筛选条件的标的才会被选中。
    
    Attributes:
        universes: 要组合的 Universe 筛选器列表
    """

    def __init__(self, *universes: Universe) -> None:
        """初始化复合筛选器。
        
        Args:
            *universes: 可变数量的 Universe 筛选器
        """
        self.universes = universes

    def mask(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """生成组合筛选掩码。
        
        对所有子筛选器的结果执行逻辑 AND 运算。
        
        Args:
            data: 输入数据
            
        Returns:
            组合后的布尔掩码 Series
        """
        mask = None
        for uni in self.universes:
            current = uni.mask(data)
            mask = current if mask is None else (mask & current)
        return mask if mask is not None else pd.Series(True, index=data.index)
