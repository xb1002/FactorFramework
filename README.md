# FactorFramework

一个轻量级的日频因子研究/入库脚手架，包含数据加载、因子注册与标准化、评价与入库、以及研究流程编排。下面给出使用说明与快速上手示例。

## 目录结构概览
```
project/
  data_manager/    # 数据源接口、本地加载、universe 过滤
  factors/         # 因子注册、标准化、引擎、因子库
  evaluation/      # 前瞻收益构建、评价指标、入库规则
  research/        # 研究/入库流水线入口
  config.yaml      # 默认配置
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
使用 `research/run_pipeline.py` 封装的一键流程：
```bash
# 仅评价，输出 report JSON
python -m project.research.run_pipeline return_5d --start 2020-01-01 --end 2020-12-31 --mode evaluate

# 评价后满足阈值则入库（输出到 config.library.root）
python -m project.research.run_pipeline return_5d --start 2020-01-01 --end 2020-12-31 --mode admit
```
- 流程：因子计算 → 构建多期前瞻收益 → 评价（IC/ICIR/换手）→ 按 `config.admission` 阈值判断 → 选定模式保存到因子库。
- 因子库路径结构：`values/<factor>/<version>.parquet`、`meta/<factor>/<version>.json`、`reports/<factor>/<version>.json`、`expressions/<factor>/<version>.py`。

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
