# 因子库API - 快速上手指南

## 1. 初始化API

```python
from factor_api import FactorAPI

# 创建API实例
api = FactorAPI(factor_store_path="factor_store")
```

## 2. 获取所有因子名称

```python
# 方法1: 只获取因子名称
factors = api.list_factors()
print(factors)
# ['amihud_illiquidity', 'downside_volatility', 'high_low_spread', 'ma_crossover']

# 方法2: 获取因子及其版本信息
factor_versions = api.list_factors(include_versions=True)
print(factor_versions)
# {'amihud_illiquidity': ['v1'], 'downside_volatility': ['v1'], ...}
```

## 3. 获取因子元数据

```python
# 获取因子的详细信息
info = api.get_factor_info("ma_crossover")

print(f"版本: {info['version']}")          # v1
print(f"所需字段: {info['required_fields']}")  # ['close']
print(f"评价窗口: {info['horizons']}")       # [1, 5, 10]
print(f"表达式: {info['expr']}")            # Python代码
```

## 4. 准备数据

```python
import pandas as pd
import numpy as np

# 方法1: 使用真实数据
market_data = pd.read_parquet("data/processed/market.parquet")

# 方法2: 使用模拟数据
dates = pd.date_range("2024-01-01", "2024-01-30", freq="D")
codes = ["000001.SZ", "000002.SZ", "600000.SH"]

index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

mock_data = pd.DataFrame({
    "open": 100 + np.random.randn(len(index)).cumsum(),
    "high": 102 + np.random.randn(len(index)).cumsum(),
    "low": 98 + np.random.randn(len(index)).cumsum(),
    "close": 100 + np.random.randn(len(index)).cumsum(),
    "volume": np.random.randint(1000000, 10000000, len(index)),
    "amount": np.random.randint(100000000, 1000000000, len(index)),
}, index=index)
```

## 5. 计算单个因子

```python
# 计算因子值
factor_values = api.compute_factor("ma_crossover", market_data)

# 查看结果
print(f"因子值数量: {len(factor_values)}")
print(f"非空值数量: {factor_values.notna().sum()}")
print(f"均值: {factor_values.mean():.6f}")
print(f"标准差: {factor_values.std():.6f}")
print(factor_values.head())
```

## 6. 批量计算多个因子

```python
# 选择要计算的因子
selected_factors = ["ma_crossover", "momentum_20d", "volatility_20d"]

# 批量计算（跳过失败的因子）
results = api.compute_factors(
    factor_names=selected_factors,
    data=market_data,
    skip_errors=True
)

# 合并为DataFrame
df_factors = pd.DataFrame(results)
print(df_factors.head())

# 查看相关系数
print(df_factors.corr())
```

## 7. 获取因子评价报告

```python
# 获取评价报告
report = api.get_factor_report("ma_crossover")

print(f"平均IC: {report.get('rank_ic_mean', 'N/A')}")
print(f"IC标准差: {report.get('rank_ic_std', 'N/A')}")
print(f"ICIR: {report.get('icir', 'N/A')}")
print(f"最佳窗口: {report.get('best_horizon', 'N/A')}")

# 查看各窗口IC
if 'ic_by_horizon' in report:
    for horizon, ic in report['ic_by_horizon'].items():
        print(f"{horizon}日IC: {ic:.4f}")
```

## 8. 获取因子表达式代码

```python
# 获取因子的Python代码
expression = api.get_factor_expression("ma_crossover")
print(expression)
```

## 9. 便捷函数

```python
from factor_api import list_all_factors, compute_single_factor

# 快速获取所有因子
factors = list_all_factors()

# 快速计算单个因子
result = compute_single_factor("ma_crossover", market_data)
```

## 10. 命令行使用

```bash
# 列出所有因子
python factor_api.py list

# 查看因子信息
python factor_api.py info --factor ma_crossover

# 计算因子并保存
python factor_api.py compute \
    --factor ma_crossover \
    --data data/processed/market.parquet \
    --output results/ma_crossover.parquet
```

## 完整示例

```python
from factor_api import FactorAPI
import pandas as pd

# 1. 初始化
api = FactorAPI()

# 2. 查看可用因子
all_factors = api.list_factors()
print(f"共有 {len(all_factors)} 个因子: {all_factors}")

# 3. 选择感兴趣的因子并查看信息
factor_name = "ma_crossover"
info = api.get_factor_info(factor_name)
print(f"\n因子 '{factor_name}' 信息:")
print(f"  所需字段: {info['required_fields']}")
print(f"  评价窗口: {info['horizons']}")

# 4. 加载数据
market_data = pd.read_parquet("data/processed/market.parquet")

# 5. 计算因子
factor_values = api.compute_factor(factor_name, market_data)
print(f"\n计算结果:")
print(f"  总数: {len(factor_values)}")
print(f"  非空: {factor_values.notna().sum()}")
print(f"  均值: {factor_values.mean():.6f}")

# 6. 批量计算多个因子
selected = all_factors[:3]  # 选择前3个因子
results = api.compute_factors(selected, market_data, skip_errors=True)

# 7. 保存结果
df_results = pd.DataFrame(results)
df_results.to_parquet("factor_results.parquet")
print(f"\n✓ 已保存 {len(results)} 个因子到 factor_results.parquet")
```

## 错误处理

```python
# 处理因子不存在的情况
try:
    result = api.compute_factor("nonexistent_factor", market_data)
except FileNotFoundError as e:
    print(f"因子不存在: {e}")
    print(f"可用因子: {api.list_factors()}")

# 处理数据字段缺失
try:
    result = api.compute_factor("ma_crossover", incomplete_data)
except ValueError as e:
    print(f"数据字段不完整: {e}")

# 批量计算时跳过错误
results = api.compute_factors(
    all_factors,
    market_data,
    skip_errors=True  # 跳过失败的因子
)
```

## 注意事项

1. **数据格式**: 输入数据必须是带 `MultiIndex (date, code)` 的 DataFrame
2. **字段要求**: 数据需包含因子所需的字段（通过 `get_factor_info()` 查看）
3. **版本管理**: 不指定版本时自动使用最新版本
4. **性能**: 计算大量数据时可能耗时较长，建议先在小数据集测试

## 更多信息

- 详细文档: `FACTOR_API_README.md`
- 示例代码: `example_factor_api.py`
- 测试脚本: `test_factor_api.py`
