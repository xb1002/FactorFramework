"""Forward return label construction."""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def build(df: pd.DataFrame, horizons: Iterable[int], price_col: str = "close") -> Dict[int, pd.Series]:
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
