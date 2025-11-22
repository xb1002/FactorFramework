"""
因子定义文件

在这个文件中定义所有需要计算的因子。
每个因子都是一个使用 @register_factor 装饰器注册的函数。

因子函数规范：
1. 输入：pd.DataFrame，包含 MultiIndex (date, code) 和 OHLCV 字段
2. 输出：pd.Series，包含 MultiIndex (date, code)
3. 使用 @register_factor 装饰器注册，指定：
   - name: 因子名称（可选，默认使用函数名）
   - required_fields: 所需字段列表
   - horizons: 评价窗口列表
   - version: 版本号（必需，用于因子库管理）
   - preprocess: 预处理配置（可选）

示例：
    @register_factor(
        name="momentum_20d",
        required_fields=["close"],
        horizons=[1, 5, 10],
        version="v1"
    )
    def my_momentum(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(20)
"""

import pandas as pd
from project.factors.registry import register_factor


# ============================================================================
# 动量因子（Momentum Factors）
# ============================================================================

@register_factor(
    name="momentum_20d",
    required_fields=["close"],
    horizons=[1, 5, 10, 20],
    version="v1"
)
def momentum_20d(df: pd.DataFrame) -> pd.Series:
    """20日动量因子
    
    计算过去20个交易日的累计收益率。
    理论：过去表现好的股票短期内可能继续上涨（动量效应）。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        20日累计收益率 Series
    """
    return df["close"].pct_change(20)


@register_factor(
    name="momentum_60d",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def momentum_60d(df: pd.DataFrame) -> pd.Series:
    """60日动量因子
    
    计算过去60个交易日的累计收益率（约3个月）。
    理论：中期动量效应，适合捕捉趋势性行情。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        60日累计收益率 Series
    """
    return df["close"].pct_change(60)


# ============================================================================
# 反转因子（Reversal Factors）
# ============================================================================

@register_factor(
    name="reversal_5d",
    required_fields=["close"],
    horizons=[1, 5],
    version="v1"
)
def reversal_5d(df: pd.DataFrame) -> pd.Series:
    """5日反转因子
    
    计算过去5个交易日的累计收益率，并取负值。
    理论：短期内涨幅过大的股票可能回调（反转效应）。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        5日累计收益率的负值 Series
    """
    return -df["close"].pct_change(5)


# ============================================================================
# 波动率因子（Volatility Factors）
# ============================================================================

@register_factor(
    name="volatility_20d",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20日波动率因子
    
    计算过去20个交易日收益率的标准差（滚动窗口）。
    理论：低波动率股票可能更稳定，高波动率可能蕴含机会或风险。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        20日收益率标准差 Series
    """
    returns = df["close"].pct_change()
    return returns.groupby(level="code").rolling(window=20).std().droplevel(0)


@register_factor(
    name="volatility_60d",
    required_fields=["close"],
    horizons=[10, 20],
    version="v1"
)
def volatility_60d(df: pd.DataFrame) -> pd.Series:
    """60日波动率因子
    
    计算过去60个交易日收益率的标准差（约3个月）。
    理论：中期波动率指标，用于风险评估。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        60日收益率标准差 Series
    """
    returns = df["close"].pct_change()
    return returns.groupby(level="code").rolling(window=60).std().droplevel(0)


# ============================================================================
# 成交量因子（Volume Factors）
# ============================================================================

@register_factor(
    name="volume_ratio_20d",
    required_fields=["volume"],
    horizons=[1, 5, 10],
    version="v1"
)
def volume_ratio_20d(df: pd.DataFrame) -> pd.Series:
    """20日成交量比率因子
    
    计算当日成交量与过去20日平均成交量的比率。
    理论：成交量放大可能预示价格变动，量价配合是重要信号。
    
    Args:
        df: 包含 volume 列的 DataFrame
        
    Returns:
        成交量比率 Series
    """
    vol_ma20 = df["volume"].groupby(level="code").rolling(window=20).mean().droplevel(0)
    return df["volume"] / vol_ma20


@register_factor(
    name="turnover_rate",
    required_fields=["volume", "amount"],
    horizons=[1, 5, 10],
    version="v1"
)
def turnover_rate(df: pd.DataFrame) -> pd.Series:
    """换手率因子
    
    计算成交量与流通市值的比率（简化版）。
    理论：高换手率可能表示活跃度高，但也可能过度交易。
    
    Args:
        df: 包含 volume 和 amount 列的 DataFrame
        
    Returns:
        换手率代理指标 Series
    """
    # 简化计算：使用成交量/成交额比率作为代理
    return df["volume"] / (df["amount"] + 1e-6)


# ============================================================================
# 价格因子（Price Factors）
# ============================================================================

@register_factor(
    name="price_to_ma20",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def price_to_ma20(df: pd.DataFrame) -> pd.Series:
    """价格相对20日均线因子
    
    计算当前价格与20日移动平均线的比率。
    理论：突破均线可能是趋势信号，偏离过大可能回归。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        价格/均线比率 Series
    """
    ma20 = df["close"].groupby(level="code").rolling(window=20).mean().droplevel(0)
    return df["close"] / ma20


@register_factor(
    name="high_low_spread",
    required_fields=["high", "low", "close"],
    horizons=[1, 5],
    version="v1"
)
def high_low_spread(df: pd.DataFrame) -> pd.Series:
    """日内振幅因子
    
    计算日内最高价和最低价的价差占收盘价的比例。
    理论：振幅大可能表示波动性高或市场不确定性。
    
    Args:
        df: 包含 high, low, close 列的 DataFrame
        
    Returns:
        日内振幅比率 Series
    """
    return (df["high"] - df["low"]) / df["close"]


# ============================================================================
# 自定义因子示例
# ============================================================================

# 取消下面的注释来添加你自己的因子：
# 
# @register_factor(
#     name="my_custom_factor",
#     required_fields=["close", "volume"],
#     horizons=[1, 5, 10],
#     version="v1"
# )
# def my_custom_factor(df: pd.DataFrame) -> pd.Series:
#     """你的因子描述"""
#     # 在这里实现你的因子逻辑
#     # 必须返回 pd.Series with MultiIndex (date, code)
#     return df["close"] * df["volume"]


if __name__ == "__main__":
    # 查看已注册的因子
    from project.factors.registry import list_all
    
    print("已注册的因子：")
    print("-" * 60)
    for name in list_all():
        from project.factors.registry import get
        spec = get(name)
        print(f"因子名称: {spec.name}")
        print(f"  版本: {spec.version}")
        print(f"  所需字段: {spec.required_fields}")
        print(f"  评价窗口: {spec.horizons}")
        print(f"  函数: {spec.func.__name__}")
        print()
