# FactorFramework

一个轻量级的日频因子研究/入库脚手架，包含数据加载、因子注册与标准化、评价与入库、以及研究流程编排。

**✨ 新增功能**：自动跳过已评价的因子，避免重复计算，大幅提升研发效率！

## 快速开始

```bash
# 第一次运行：评价所有因子
python main.py

# 第二次运行：自动跳过已评价的因子，只处理新增的
python main.py

# 强制重新评价指定因子
python main.py --force momentum_20d volatility_20d

# 强制重新评价所有因子
python main.py --force-all
```

## 目录结构概览
```
project/
  data_manager/    # 数据源接口、本地加载、universe 过滤
  factors/         # 因子注册、标准化、引擎、因子库
  evaluation/      # 前瞻收益构建、评价指标、入库规则、评价历史
  research/        # 研究/入库流水线入口
  config.yaml      # 默认配置

factors.py         # 因子定义（40+个因子）
main.py            # 一键评价脚本
factor_api.py      # 因子库API接口
```

## 安装依赖
项目使用 pandas 生态：
```bash
pip install pandas numpy pyyaml
```

## 准备数据
1. 将日频行情写入 `project/config.yaml` 中的 `data.processed_path`（默认 `data/processed/market.parquet`）。
2. 数据必须包含字段 `code, date, open, high, low, close, volume, amount`，并能转换为 `MultiIndex(date, code)`。
3. 若有原始 CSV/Parquet，可使用 `project/data_manager/loaders.py` 中的工具函数/CLI 进行基础清洗与落盘。

## 注册因子
在任意模块中用装饰器注册纯函数因子：
```python
from project.factors.registry import register_factor
import pandas as pd

@register_factor(
    name="return_5d",
    required_fields=["close"],
    horizons=[1, 5, 10],
    preprocess={"winsorize_q": 0.01, "zscore": True},
    version="v1",
)
def return_5d(df: pd.DataFrame) -> pd.Series:
    return df["close"].groupby(level=1).pct_change(5)
```
> 因子函数必须是纯函数，不读取外部文件/网络，只使用传入的 DataFrame。

## 直接计算因子
```python
from project.data_manager.datasource import LocalParquetSource
from project.data_manager.universe import DefaultUniverse
from project.factors.engine import FactorEngine
from project.factors.standardizer import Standardizer

source = LocalParquetSource("data/processed/market.parquet")
standardizer = Standardizer.from_config({"winsorize_q": 0.01, "zscore": True})
engine = FactorEngine(source, standardizer, DefaultUniverse())

series = engine.compute("return_5d", start="2020-01-01", end="2020-12-31")
print(series.head())
```

## 评价与入库流水线

### 方式1：使用 main.py（推荐）

一键评价所有因子，自动跳过已评价的：

```bash
# 默认模式：评价并入库，自动跳过已评价的因子
python main.py

# 指定日期范围
python main.py --start 2022-01-01 --end 2023-12-31

# 仅处理指定因子
python main.py --factors momentum_20d volatility_20d

# 强制重新评价指定因子
python main.py --force momentum_20d

# 强制重新评价所有因子
python main.py --force-all

# 覆盖入库阈值
python main.py --min-ic 0.03 --min-icir 0.5
```

**核心特性**：
- ✅ 自动跳过已评价的因子（包括未通过入库的）
- ✅ 记录评价历史到 `factor_evaluation_history.json`
- ✅ 只有通过入库标准的因子才会保存到 `factor_store/`
- ✅ 支持强制重新评价指定因子或所有因子

**评价历史记录**：

系统会自动记录所有已评价的因子，下次运行时跳过已评价的，避免重复计算。历史记录包括：
- 评价状态（成功/失败）
- 是否通过入库标准
- 关键指标（IC、ICIR、换手率）
- 评价时间和日期范围

查看历史记录：
```python
from project.evaluation.history import EvaluationHistory
history = EvaluationHistory()
history.print_summary()  # 打印摘要
passed = history.get_passed_factors()  # 获取通过入库的因子
failed = history.get_failed_factors()  # 获取未通过的因子
```

清空历史记录：
```bash
rm factor_evaluation_history.json  # 删除历史文件
python main.py --force-all          # 或使用 --force-all 重新评价所有因子
```

### 方式2：使用 research/run_pipeline.py

单个因子评价：

```bash
# 仅评价，输出 report JSON
python -m project.research.run_pipeline return_5d --start 2020-01-01 --end 2020-12-31 --mode evaluate

# 评价后满足阈值则入库（输出到 config.library.root）
python -m project.research.run_pipeline return_5d --start 2020-01-01 --end 2020-12-31 --mode admit
```

**数据流程**：
1. 因子计算
2. 构建多期前瞻收益
3. 评价（IC/ICIR/换手）
4. 判断入库标准（`config.admission`）
5. 保存到因子库（如果通过）

**因子库结构**：
```
factor_store/<factor_name>/
  values/<version>.parquet       # 因子值
  meta/<version>.json            # 元数据
  reports/<version>.json         # 评价报告
  expressions/<version>.py       # 因子表达式
```

## 因子库API

使用 `factor_api.py` 调用已入库的因子：

```python
from factor_api import FactorAPI
import pandas as pd

# 初始化API
api = FactorAPI(factor_store_path="factor_store")

# 获取所有因子
factors = api.list_factors()
print(f"可用因子: {factors}")

# 获取因子信息
info = api.get_factor_info("momentum_20d")
print(f"所需字段: {info['required_fields']}")

# 准备数据（MultiIndex: date, code）
market_data = pd.read_parquet("data/processed/market.parquet")

# 计算因子
factor_values = api.compute_factor("momentum_20d", market_data)

# 批量计算多个因子
results = api.compute_factors(["momentum_20d", "volatility_20d"], market_data)

# 获取评价报告
report = api.get_factor_report("momentum_20d")
print(f"IC均值: {report['best_metrics']['rank_ic_mean']}")
```

## 自定义配置
修改 `project/config.yaml` 可以调整：
- 数据路径、默认 horizons
- 标准化参数（winsorize、zscore）
- 入库阈值（IC、ICIR、换手、相关性）
- 因子库根目录

## 进一步扩展
- 新数据源：实现 `MarketDataSource.load` 接口
- 新标准化步骤：继承 `PreprocessStrategy`
- 新评价指标/入库规则：继承 `FactorEvaluator` 中的计算或实现新的 `AdmissionRule`
- 自定义 universe：实现 `Universe.mask`
