"""Forward return label construction."""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def build(df: pd.DataFrame, horizons: Iterable[int], price_col: str = "close") -> Dict[int, pd.Series]:
    """构建多期前瞻收益率。
    
    对每个时间窗口，计算未来 h 期的收益率。
    
    Args:
        df: 包含价格数据的 DataFrame（MultiIndex: date, code）
        horizons: 时间窗口列表（单位：天）
        price_col: 价格列名，默认 "close"
        
    Returns:
        字典，键为时间窗口，值为对应的前瞻收益率 Series
        
    Raises:
        KeyError: 当价格列不存在时
        
    Example:
        fwd_returns = build(df, [1, 5, 10])
        # 返回 {1: Series, 5: Series, 10: Series}
    """
    if price_col not in df.columns:
        raise KeyError(f"Price column {price_col} missing in input DataFrame")
    price = df[price_col]
    returns: Dict[int, pd.Series] = {}
    for h in horizons:
        future = price.groupby(level=1).shift(-h)
        ret = future / price - 1
        ret.name = f"fwd_ret_{h}"
        returns[h] = ret
    return returns
