# 因子库API使用说明

## 概述

因子库API（`factor_api.py`）提供了一个统一的接口，允许外部程序调用已入库的因子表达式计算因子值。

## 主要功能

1. **获取因子列表** - 查看所有已入库的因子名称
2. **获取因子信息** - 查看因子的元数据（所需字段、评价窗口等）
3. **计算因子值** - 使用入库的表达式对外部数据计算因子
4. **批量计算** - 一次性计算多个因子
5. **获取评价报告** - 查看因子的IC、ICIR等评价指标

## 快速开始

### 1. 基础使用

```python
from factor_api import FactorAPI

# 初始化API
api = FactorAPI(factor_store_path="factor_store")

# 获取所有因子名称
factors = api.list_factors()
print(f"可用因子: {factors}")
```

### 2. 获取因子信息

```python
# 获取因子元数据
info = api.get_factor_info("ma_crossover")
print(f"所需字段: {info['required_fields']}")
print(f"评价窗口: {info['horizons']}")
print(f"版本: {info['version']}")
```

### 3. 计算单个因子

```python
import pandas as pd

# 加载数据（需要包含因子所需字段）
market_data = pd.read_parquet("data/processed/market.parquet")

# 计算因子值
factor_values = api.compute_factor("ma_crossover", market_data)

print(f"因子值: {len(factor_values)} 个")
print(factor_values.head())
```

### 4. 批量计算多个因子

```python
# 批量计算
results = api.compute_factors(
    factor_names=["ma_crossover", "momentum_20d", "volatility_20d"],
    data=market_data,
    skip_errors=True  # 跳过计算失败的因子
)

# 合并为DataFrame
df_factors = pd.DataFrame(results)
print(df_factors.head())
```

### 5. 获取评价报告

```python
# 获取因子评价报告
report = api.get_factor_report("ma_crossover")

print(f"平均IC: {report['rank_ic_mean']}")
print(f"ICIR: {report['icir']}")
print(f"最佳窗口: {report['best_horizon']}")
```

## API接口详解

### FactorAPI类

#### `__init__(factor_store_path="factor_store")`
初始化因子库API。

**参数：**
- `factor_store_path`: 因子库存储路径

---

#### `list_factors(include_versions=False)`
获取所有已入库的因子名称。

**参数：**
- `include_versions`: 是否包含版本信息
  - `False` (默认): 返回因子名称列表
  - `True`: 返回字典，键为因子名，值为版本列表

**返回：**
- `List[str]` 或 `Dict[str, List[str]]`

**示例：**
```python
# 获取因子名称列表
factors = api.list_factors()
# ['ma_crossover', 'momentum_20d', ...]

# 获取因子及版本
factor_versions = api.list_factors(include_versions=True)
# {'ma_crossover': ['v1'], 'momentum_20d': ['v1', 'v2'], ...}
```

---

#### `get_factor_info(factor_name, version=None)`
获取因子的元数据信息。

**参数：**
- `factor_name`: 因子名称
- `version`: 版本号，`None` 时返回最新版本

**返回：**
- `Dict[str, Any]`: 因子元数据字典，包含：
  - `name`: 因子名称
  - `version`: 版本号
  - `required_fields`: 所需数据字段列表
  - `horizons`: 评价窗口列表
  - `expr`: 因子表达式代码
  - `code_hash`: 代码哈希值
  - `env`: 运行环境信息

**示例：**
```python
info = api.get_factor_info("ma_crossover")
print(info['required_fields'])  # ['close']
print(info['horizons'])  # [1, 5, 10]
```

---

#### `compute_factor(factor_name, data, version=None, validate_fields=True)`
计算单个因子的值。

**参数：**
- `factor_name`: 因子名称
- `data`: 输入数据 DataFrame，需包含因子所需字段
  - 索引应为 `MultiIndex (date, code)` 或 `DatetimeIndex`
- `version`: 版本号，`None` 时使用最新版本
- `validate_fields`: 是否验证输入数据包含所需字段

**返回：**
- `pd.Series`: 因子值 Series，索引与输入数据一致

**示例：**
```python
factor_values = api.compute_factor("ma_crossover", market_data)
print(factor_values.head())
```

---

#### `compute_factors(factor_names, data, version=None, validate_fields=True, skip_errors=False)`
批量计算多个因子的值。

**参数：**
- `factor_names`: 因子名称列表
- `data`: 输入数据 DataFrame
- `version`: 版本号，`None` 时使用最新版本
- `validate_fields`: 是否验证输入数据包含所需字段
- `skip_errors`: 是否跳过计算失败的因子
  - `False` (默认): 遇到错误时抛出异常
  - `True`: 跳过失败的因子，继续计算其他因子

**返回：**
- `Dict[str, pd.Series]`: 字典，键为因子名称，值为因子值 Series

**示例：**
```python
results = api.compute_factors(
    ["ma_crossover", "momentum_20d"],
    market_data,
    skip_errors=True
)

for name, values in results.items():
    print(f"{name}: {len(values)} 个值")
```

---

#### `get_factor_expression(factor_name, version=None)`
获取因子的表达式代码。

**参数：**
- `factor_name`: 因子名称
- `version`: 版本号，`None` 时返回最新版本

**返回：**
- `str`: 因子表达式代码字符串

**示例：**
```python
expr = api.get_factor_expression("ma_crossover")
print(expr)
```

---

#### `get_factor_report(factor_name, version=None)`
获取因子的评价报告。

**参数：**
- `factor_name`: 因子名称
- `version`: 版本号，`None` 时返回最新版本

**返回：**
- `Dict[str, Any]`: 评价报告字典，包含IC、ICIR、换手率等指标

**示例：**
```python
report = api.get_factor_report("ma_crossover")
print(f"IC: {report['rank_ic_mean']}")
print(f"ICIR: {report['icir']}")
```

---

## 便捷函数

除了 `FactorAPI` 类，还提供了一些便捷函数：

```python
from factor_api import list_all_factors, compute_single_factor

# 快速获取所有因子
factors = list_all_factors()

# 快速计算单个因子
result = compute_single_factor("ma_crossover", market_data)
```

---

## 命令行接口

也可以通过命令行使用因子库API：

### 列出所有因子
```bash
python factor_api.py list
```

### 查看因子信息
```bash
python factor_api.py info --factor ma_crossover
```

### 计算因子
```bash
python factor_api.py compute \
    --factor ma_crossover \
    --data data/processed/market.parquet \
    --output results/ma_crossover.parquet
```

---

## 数据格式要求

### 输入数据格式

计算因子时，输入的 DataFrame 需要满足：

1. **索引**：MultiIndex (date, code) 或 DatetimeIndex
2. **列**：包含因子所需的字段（如 open, high, low, close, volume, amount）

示例：
```
                          open    high     low   close      volume        amount
date       code                                                                   
2023-01-01 000001.SZ  10.50   10.80   10.30   10.60   1000000.0   1.060000e+07
           000002.SZ  20.30   20.50   20.10   20.40   2000000.0   4.080000e+07
           600000.SH  15.20   15.40   15.00   15.30   1500000.0   2.295000e+07
...
```

### 输出数据格式

因子值为 `pd.Series`，索引与输入数据一致：

```
date        code
2023-01-01  000001.SZ    0.025000
            000002.SZ   -0.010000
            600000.SH    0.015000
...
Name: ma_crossover, dtype: float64
```

---

## 使用示例

完整的使用示例请参考 `example_factor_api.py` 文件：

```bash
python example_factor_api.py
```

该脚本包含以下示例：
1. 基础使用
2. 计算单个因子
3. 批量计算多个因子
4. 获取因子评价报告
5. 获取因子表达式
6. 使用外部数据计算因子

---

## 注意事项

1. **因子库必须存在**：在使用API之前，需要先运行因子评价流程生成因子库：
   ```bash
   python main.py --start 2023-01-01 --end 2023-12-31
   ```

2. **数据字段验证**：默认会验证输入数据是否包含因子所需字段，可以通过 `validate_fields=False` 关闭验证。

3. **版本管理**：每个因子可以有多个版本，不指定版本时默认使用最新版本。

4. **错误处理**：在批量计算时，可以使用 `skip_errors=True` 跳过失败的因子，避免中断整个流程。

5. **性能考虑**：计算因子可能需要一定时间，建议先在小数据集上测试。

---

## 故障排除

### 问题1：找不到因子库
```
FileNotFoundError: 因子库路径不存在: factor_store
```

**解决方法**：先运行因子评价流程生成因子库：
```bash
python main.py
```

### 问题2：因子不存在
```
FileNotFoundError: 因子 'xxx' 不存在
```

**解决方法**：使用 `list_factors()` 查看可用因子列表。

### 问题3：缺少必需字段
```
ValueError: 输入数据缺少必需字段: {'volume'}
```

**解决方法**：确保输入数据包含因子所需的所有字段，或在 `factors.py` 中查看因子定义。

---

## 集成到项目中

### Python项目集成

```python
# 在你的项目中导入
from factor_api import FactorAPI

# 初始化
api = FactorAPI(factor_store_path="path/to/factor_store")

# 使用API计算因子
def calculate_factors(data):
    factors = ["ma_crossover", "momentum_20d"]
    results = api.compute_factors(factors, data, skip_errors=True)
    return pd.DataFrame(results)
```

### 作为微服务

可以将API封装为REST服务：

```python
from flask import Flask, request, jsonify
from factor_api import FactorAPI

app = Flask(__name__)
api = FactorAPI()

@app.route('/factors', methods=['GET'])
def list_factors():
    return jsonify(api.list_factors())

@app.route('/factor/<name>', methods=['POST'])
def compute_factor(name):
    data = request.get_json()
    # ... 处理数据并计算因子
    return jsonify(result)
```

---

## 更多信息

- 因子定义文件：`factors.py`
- 因子评价流程：`main.py`
- 因子库实现：`project/factors/library.py`
