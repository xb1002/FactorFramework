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
    return df["close"].groupby(level="code").pct_change(20)


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
    return df["close"].groupby(level="code").pct_change(60)


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
    return -df["close"].groupby(level="code").pct_change(5)


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
# 动量/反转类因子（扩展）
# ============================================================================

@register_factor(
    name="residual_momentum",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def residual_momentum(df: pd.DataFrame) -> pd.Series:
    """残差动量因子
    
    计算股票收益与市场收益的残差动量。
    理论：剔除市场因素后的个股特异动量，更稳健的动量信号。
    计算：过去20日收益率 - 市场平均收益率
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        残差动量 Series
    """
    returns = df["close"].groupby(level="code").pct_change(20)
    # 计算每日市场平均收益率（简化版，实际应使用市值加权）
    market_returns = returns.groupby(level="date").mean()
    # 计算残差
    residual = returns.groupby(level="date", group_keys=False).apply(
        lambda x: x - market_returns.loc[x.name]
    )
    return residual


@register_factor(
    name="momentum_acceleration",
    required_fields=["close"],
    horizons=[5, 10],
    version="v1"
)
def momentum_acceleration(df: pd.DataFrame) -> pd.Series:
    """动量加速度因子
    
    计算动量的变化率（二阶导数）。
    理论：动量加速可能预示趋势加强，减速可能预示反转。
    计算：近期动量（10日） - 远期动量（20日）
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        动量加速度 Series
    """
    mom_10 = df["close"].groupby(level="code").pct_change(10)
    mom_20 = df["close"].groupby(level="code").pct_change(20)
    return mom_10 - mom_20


@register_factor(
    name="reversal_momentum_combo",
    required_fields=["close"],
    horizons=[1, 5, 10],
    version="v1"
)
def reversal_momentum_combo(df: pd.DataFrame) -> pd.Series:
    """反转-动量组合因子
    
    结合短期反转和中期动量的信号。
    理论：短期反转 + 中期动量 = 更强的预测能力
    计算：-5日收益率 + 60日收益率
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        组合因子 Series
    """
    reversal = -df["close"].groupby(level="code").pct_change(5)
    momentum = df["close"].groupby(level="code").pct_change(60)
    return reversal * 0.3 + momentum * 0.7  # 权重组合


# ============================================================================
# 波动/极端风险类因子
# ============================================================================

@register_factor(
    name="downside_volatility",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def downside_volatility(df: pd.DataFrame) -> pd.Series:
    """下行波动率因子
    
    只计算负收益的标准差，衡量下跌风险。
    理论：下行波动率比总波动率更能反映真实风险。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        20日下行波动率 Series
    """
    returns = df["close"].groupby(level="code").pct_change()
    # 只保留负收益
    downside_returns = returns.where(returns < 0, 0)
    return downside_returns.groupby(level="code").rolling(window=20).std().droplevel(0)


@register_factor(
    name="skewness_20d",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def skewness_20d(df: pd.DataFrame) -> pd.Series:
    """20日收益率偏度因子
    
    计算收益率分布的偏度（三阶矩）。
    理论：负偏表示左尾风险（极端下跌），正偏表示右尾机会。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        20日偏度 Series
    """
    returns = df["close"].groupby(level="code").pct_change()
    return returns.groupby(level="code").rolling(window=20).skew().droplevel(0)


@register_factor(
    name="kurtosis_20d",
    required_fields=["close"],
    horizons=[5, 10, 20],
    version="v1"
)
def kurtosis_20d(df: pd.DataFrame) -> pd.Series:
    """20日收益率峰度因子
    
    计算收益率分布的峰度（四阶矩）。
    理论：高峰度表示极端值出现概率高（尾部风险）。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        20日峰度 Series
    """
    returns = df["close"].groupby(level="code").pct_change()
    return returns.groupby(level="code").rolling(window=20).kurt().droplevel(0)

# ============================================================================
# 区间/形态/突破类因子
# ============================================================================

@register_factor(
    name="breakout_20d",
    required_fields=["close", "high"],
    horizons=[1, 5, 10],
    version="v1"
)
def breakout_20d(df: pd.DataFrame) -> pd.Series:
    """20日突破因子
    
    判断当前价格是否突破过去20日高点。
    理论：突破新高可能是趋势延续信号（Donchian通道）。
    
    Args:
        df: 包含 close, high 列的 DataFrame
        
    Returns:
        突破幅度 Series（0表示未突破，>0表示突破程度）
    """
    high_20 = df["high"].groupby(level="code").shift(1).rolling(window=20).max().droplevel(0)
    # 突破幅度 = (当前价 - 20日最高) / 20日最高
    return (df["close"] - high_20) / high_20


@register_factor(
    name="support_resistance",
    required_fields=["close", "low", "high"],
    horizons=[1, 5, 10],
    version="v1"
)
def support_resistance(df: pd.DataFrame) -> pd.Series:
    """支撑阻力位因子
    
    计算当前价格相对于20日高低区间的位置。
    理论：价格在区间上方可能继续上涨，下方可能继续下跌。
    
    Args:
        df: 包含 close, low, high 列的 DataFrame
        
    Returns:
        区间位置指标（0-1之间）Series
    """
    low_20 = df["low"].groupby(level="code").rolling(window=20).min().droplevel(0)
    high_20 = df["high"].groupby(level="code").rolling(window=20).max().droplevel(0)
    # 归一化位置 = (当前价 - 最低) / (最高 - 最低)
    return (df["close"] - low_20) / (high_20 - low_20 + 1e-6)


@register_factor(
    name="pattern_flag",
    required_fields=["close", "high", "low"],
    horizons=[1, 5],
    version="v1"
)
def pattern_flag(df: pd.DataFrame) -> pd.Series:
    """旗形整理因子
    
    识别旗形整理形态：先强势上涨，后窄幅震荡。
    理论：旗形整理后可能继续原趋势。
    计算：前10日涨幅大 且 近5日振幅收窄
    
    Args:
        df: 包含 close, high, low 列的 DataFrame
        
    Returns:
        旗形信号强度 Series
    """
    # 前期上涨幅度
    prior_gain = df["close"].groupby(level="code").pct_change(10)
    # 近期振幅
    recent_range = (df["high"] - df["low"]) / df["close"]
    recent_range_avg = recent_range.groupby(level="code").rolling(window=5).mean().droplevel(0)
    # 旗形信号 = 前期涨幅 * (1 / 近期振幅)，振幅越小信号越强
    return prior_gain / (recent_range_avg + 1e-6)


@register_factor(
    name="ma_crossover",
    required_fields=["close"],
    horizons=[1, 5, 10],
    version="v1"
)
def ma_crossover(df: pd.DataFrame) -> pd.Series:
    """均线交叉因子
    
    计算短期均线(5日)与长期均线(20日)的距离。
    理论：金叉（短期上穿长期）看涨，死叉看跌。
    
    Args:
        df: 包含 close 列的 DataFrame
        
    Returns:
        均线距离比率 Series
    """
    ma5 = df["close"].groupby(level="code").rolling(window=5).mean().droplevel(0)
    ma20 = df["close"].groupby(level="code").rolling(window=20).mean().droplevel(0)
    return (ma5 - ma20) / ma20


# ============================================================================
# 价量协同类因子
# ============================================================================

@register_factor(
    name="price_volume_correlation",
    required_fields=["close", "volume"],
    horizons=[5, 10, 20],
    version="v1"
)
def price_volume_correlation(df: pd.DataFrame) -> pd.Series:
    """量价相关性因子
    
    计算价格变动与成交量的相关系数（20日滚动）。
    理论：量价齐升是健康上涨，量价背离可能反转。
    
    Args:
        df: 包含 close, volume 列的 DataFrame
        
    Returns:
        20日量价相关系数 Series
    """
    price_chg = df["close"].groupby(level="code").pct_change()
    vol_chg = df["volume"].groupby(level="code").pct_change()
    
    def rolling_corr(group):
        return group["price"].rolling(window=20).corr(group["volume"])
    
    temp_df = pd.DataFrame({"price": price_chg, "volume": vol_chg})
    return temp_df.groupby(level="code", group_keys=False).apply(rolling_corr)


@register_factor(
    name="volume_price_trend",
    required_fields=["close", "volume"],
    horizons=[1, 5, 10],
    version="v1"
)
def volume_price_trend(df: pd.DataFrame) -> pd.Series:
    """量价趋势因子（VPT）
    
    计算累计的量价指标：成交量 * 价格变动百分比。
    理论：成交量加权的价格动量，更可靠的趋势信号。
    
    Args:
        df: 包含 close, volume 列的 DataFrame
        
    Returns:
        VPT指标 Series
    """
    price_chg = df["close"].groupby(level="code").pct_change()
    vpt = (price_chg * df["volume"]).groupby(level="code").cumsum()
    return vpt


@register_factor(
    name="obv_normalized",
    required_fields=["close", "volume"],
    horizons=[5, 10, 20],
    version="v1"
)
def obv_normalized(df: pd.DataFrame) -> pd.Series:
    """归一化能量潮指标（OBV）
    
    根据价格涨跌累计成交量，然后标准化。
    理论：OBV领先价格，可预测趋势变化。
    
    Args:
        df: 包含 close, volume 列的 DataFrame
        
    Returns:
        归一化OBV Series
    """
    price_chg = df["close"].groupby(level="code").diff()
    # 价格上涨计为正成交量，下跌为负
    signed_volume = df["volume"] * price_chg.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = signed_volume.groupby(level="code").cumsum()
    # 标准化：除以20日移动平均
    obv_ma20 = obv.groupby(level="code").rolling(window=20).mean().droplevel(0)
    return obv / (obv_ma20.abs() + 1e-6)


@register_factor(
    name="money_flow_index",
    required_fields=["close", "high", "low", "volume"],
    horizons=[5, 10],
    version="v1"
)
def money_flow_index(df: pd.DataFrame) -> pd.Series:
    """资金流量指数（MFI）
    
    类似RSI，但考虑成交量，衡量资金流入流出。
    理论：高MFI可能超买，低MFI可能超卖。
    
    Args:
        df: 包含 close, high, low, volume 列的 DataFrame
        
    Returns:
        MFI指标 Series
    """
    # 典型价格
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    # 资金流量
    money_flow = typical_price * df["volume"]
    
    # 价格变动方向
    price_diff = typical_price.groupby(level="code").diff()
    
    def calc_mfi(group):
        if len(group) < 14:
            return pd.Series(index=group.index, dtype=float)
        
        pos_flow = (group * (price_diff.loc[group.index] > 0)).rolling(14).sum()
        neg_flow = (group * (price_diff.loc[group.index] < 0)).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + pos_flow / (neg_flow.abs() + 1e-6)))
        return mfi
    
    return money_flow.groupby(level="code", group_keys=False).apply(calc_mfi)


# ============================================================================
# 流动性/交易活跃度类因子
# ============================================================================

@register_factor(
    name="amihud_illiquidity",
    required_fields=["close", "volume", "amount"],
    horizons=[5, 10, 20],
    version="v1"
)
def amihud_illiquidity(df: pd.DataFrame) -> pd.Series:
    """Amihud非流动性指标
    
    计算单位成交额导致的价格变动（20日平均）。
    理论：非流动性越高，买卖价差越大，需要更高补偿。
    计算：|收益率| / 成交额
    
    Args:
        df: 包含 close, volume, amount 列的 DataFrame
        
    Returns:
        Amihud非流动性指标 Series
    """
    returns = df["close"].groupby(level="code").pct_change().abs()
    # 非流动性 = |收益率| / 成交额
    illiq = returns / (df["amount"] + 1e-6)
    # 取20日平均
    return illiq.groupby(level="code").rolling(window=20).mean().droplevel(0)


@register_factor(
    name="turnover_volatility",
    required_fields=["volume", "amount"],
    horizons=[5, 10, 20],
    version="v1"
)
def turnover_volatility(df: pd.DataFrame) -> pd.Series:
    """换手率波动性因子
    
    计算换手率的标准差（20日）。
    理论：换手率稳定的股票流动性更可预测。
    
    Args:
        df: 包含 volume, amount 列的 DataFrame
        
    Returns:
        换手率波动性 Series
    """
    # 简化换手率
    turnover = df["volume"] / (df["amount"] + 1e-6)
    return turnover.groupby(level="code").rolling(window=20).std().droplevel(0)


@register_factor(
    name="volume_trend",
    required_fields=["volume"],
    horizons=[1, 5, 10],
    version="v1"
)
def volume_trend(df: pd.DataFrame) -> pd.Series:
    """成交量趋势因子
    
    计算成交量的移动平均斜率（线性回归）。
    理论：成交量递增可能预示趋势加强，递减可能衰竭。
    
    Args:
        df: 包含 volume 列的 DataFrame
        
    Returns:
        成交量趋势斜率 Series
    """
    def calc_slope(series):
        if len(series) < 2:
            return 0
        x = range(len(series))
        y = series.values
        # 简单线性回归斜率
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        return numerator / (denominator + 1e-6)
    
    return df["volume"].groupby(level="code").rolling(window=20).apply(
        calc_slope, raw=False
    ).droplevel(0)


@register_factor(
    name="bid_ask_spread_proxy",
    required_fields=["high", "low", "volume"],
    horizons=[1, 5, 10],
    version="v1"
)
def bid_ask_spread_proxy(df: pd.DataFrame) -> pd.Series:
    """买卖价差代理因子
    
    使用日内振幅和成交量估算买卖价差。
    理论：价差大表示流动性差，交易成本高。
    计算：(最高-最低) / 成交量^(1/3)
    
    Args:
        df: 包含 high, low, volume 列的 DataFrame
        
    Returns:
        价差代理指标 Series
    """
    range_hl = df["high"] - df["low"]
    vol_adj = df["volume"] ** (1/3) + 1e-6
    spread_proxy = range_hl / vol_adj
    # 标准化：除以20日均值
    spread_ma20 = spread_proxy.groupby(level="code").rolling(window=20).mean().droplevel(0)
    return spread_proxy / (spread_ma20 + 1e-6)


@register_factor(
    name="zero_volume_days",
    required_fields=["volume"],
    horizons=[5, 10],
    version="v1"
)
def zero_volume_days(df: pd.DataFrame) -> pd.Series:
    """零成交日比例因子
    
    计算过去20日中零成交日的比例。
    理论：零成交日多表示流动性极差。
    
    Args:
        df: 包含 volume 列的 DataFrame
        
    Returns:
        零成交日比例 Series
    """
    zero_vol = (df["volume"] == 0).astype(int)
    return zero_vol.groupby(level="code").rolling(window=20).mean().droplevel(0)


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
