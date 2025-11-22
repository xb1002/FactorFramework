"""
因子全流程自动化脚本

一键执行完整的因子研究流程：
1. 加载配置和数据
2. 导入因子定义（自动注册）
3. 批量计算因子
4. 评价因子表现
5. 判断是否入库
6. 保存到因子库

用法：
    # 默认模式：仅评价，不入库
    python main.py
    
    # 评价并入库通过的因子
    python main.py --mode admit
    
    # 指定日期范围
    python main.py --start 2022-01-01 --end 2023-12-31
    
    # 仅处理指定因子
    python main.py --factors momentum_20d volatility_20d
    
    # 覆盖入库阈值
    python main.py --mode admit --min-ic 0.03 --min-icir 0.5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# 导入所有必需的模块
from project.data_manager.datasource import LocalParquetSource
from project.data_manager.universe import DefaultUniverse
from project.evaluation.admission import (
    CorrelationRule,
    FactorAdmissionStandard,
    ICThresholdRule,
    TurnoverRule,
)
from project.evaluation.evaluator import FactorEvaluator, FactorReport
from project.evaluation.forward_return import build as build_forward_returns
from project.factors.engine import FactorEngine
from project.factors.library import FactorLibrary
from project.factors.registry import get, list_all
from project.factors.standardizer import Standardizer


class FactorPipeline:
    """因子全流程管道"""
    
    def __init__(self, config: dict):
        """初始化管道
        
        Args:
            config: 配置字典
        """
        self.config = config
        self._init_components()
        
    def _init_components(self):
        """初始化所有组件"""
        print("🔧 初始化组件...")
        
        # 数据源
        data_path = self.config["data"]["processed_path"]
        self.source = LocalParquetSource(
            data_path, 
            cache=self.config["engine"].get("cache", True)
        )
        print(f"   数据源: {data_path}")
        
        # 标准化器
        self.standardizer = Standardizer.from_config(self.config["standardizer"])
        print(f"   标准化: winsorize={self.config['standardizer']['winsorize_q']}, "
              f"zscore={self.config['standardizer']['zscore']}")
        
        # Universe 筛选器
        self.universe = DefaultUniverse()
        print(f"   Universe: DefaultUniverse")
        
        # 因子引擎
        self.engine = FactorEngine(
            source=self.source,
            standardizer=self.standardizer,
            default_universe=self.universe
        )
        print(f"   引擎: FactorEngine")
        
        # 评价器
        horizons = self.config["engine"]["default_horizons"]
        self.evaluator = FactorEvaluator(horizons=horizons)
        print(f"   评价器: horizons={horizons}")
        
        # 入库标准
        admission_cfg = self.config["admission"]
        self.admission = FactorAdmissionStandard([
            ICThresholdRule(
                min_ic=admission_cfg["min_ic"],
                min_icir=admission_cfg["min_icir"]
            ),
            TurnoverRule(max_turnover_adj=admission_cfg["max_turnover_adj"]),
            CorrelationRule(max_abs_corr=admission_cfg["max_abs_corr"])
        ])
        print(f"   入库标准: IC>={admission_cfg['min_ic']}, "
              f"ICIR>={admission_cfg['min_icir']}, "
              f"换手<={admission_cfg['max_turnover_adj']}")
        
        # 因子库
        library_root = self.config["library"]["root"]
        self.library = FactorLibrary(root=library_root)
        print(f"   因子库: {library_root}")
        print()
        
    def run(
        self,
        start: str,
        end: str,
        mode: str = "admit",
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """运行因子全流程
        
        Args:
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
            mode: 运行模式 ("evaluate" 或 "admit")
            factor_names: 指定因子列表（None 表示全部）
            
        Returns:
            结果汇总 DataFrame
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        print("=" * 80)
        print(f"📊 因子研究全流程")
        print("=" * 80)
        print(f"日期范围: {start} 至 {end}")
        print(f"运行模式: {mode.upper()}")
        print()
        
        # 导入因子定义（触发注册）
        print("📥 导入因子定义...")
        try:
            import factors  # noqa: F401
            print("   ✓ 因子定义已导入")
        except ImportError as e:
            print(f"   ✗ 无法导入 factors.py: {e}")
            sys.exit(1)
        
        # 获取要处理的因子列表
        all_factors = list_all()
        if factor_names:
            # 验证指定的因子是否存在
            invalid = [f for f in factor_names if f not in all_factors]
            if invalid:
                print(f"   ✗ 未找到因子: {invalid}")
                sys.exit(1)
            factors_to_process = factor_names
        else:
            factors_to_process = all_factors
        
        print(f"   检测到 {len(all_factors)} 个因子，将处理 {len(factors_to_process)} 个")
        print()
        
        # 收集所有因子需要的 horizons
        all_horizons = set()
        for factor_name in factors_to_process:
            spec = get(factor_name)
            all_horizons.update(spec.horizons)
        all_horizons = sorted(all_horizons)
        
        # 准备前瞻收益（一次性加载）
        print("🔄 准备前瞻收益...")
        try:
            market_data = self.source.load(
                start=start_ts,
                end=end_ts,
                fields=["close"]
            )
            print(f"   数据行数: {len(market_data):,}")
            print(f"   日期范围: {market_data.index.get_level_values('date').min()} 至 "
                  f"{market_data.index.get_level_values('date').max()}")
            
            fwd_returns = build_forward_returns(
                df=market_data,
                horizons=all_horizons,
                price_col="close"
            )
            print(f"   ✓ 已构建 {len(all_horizons)} 个窗口的前瞻收益: {all_horizons}")
            print()
        except Exception as e:
            print(f"   ✗ 构建前瞻收益失败: {e}")
            sys.exit(1)
        
        # Universe 掩码
        universe_mask = self.universe.mask(market_data)
        
        # 批量处理因子
        results = []
        for i, factor_name in enumerate(factors_to_process, 1):
            print("-" * 80)
            print(f"[{i}/{len(factors_to_process)}] 处理因子: {factor_name}")
            print("-" * 80)
            
            result = self._process_single_factor(
                factor_name=factor_name,
                start=start_ts,
                end=end_ts,
                fwd_returns=fwd_returns,
                universe_mask=universe_mask,
                mode=mode
            )
            results.append(result)
            print()
        
        # 生成汇总报告
        summary_df = self._generate_summary(results)
        self._print_summary(summary_df, mode)
        
        return summary_df
    
    def _process_single_factor(
        self,
        factor_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fwd_returns: Dict[int, pd.Series],
        universe_mask: pd.Series,
        mode: str
    ) -> dict:
        """处理单个因子
        
        Args:
            factor_name: 因子名称
            start: 开始时间
            end: 结束时间
            fwd_returns: 前瞻收益字典
            universe_mask: Universe 掩码
            mode: 运行模式
            
        Returns:
            结果字典
        """
        result = {
            "factor_name": factor_name,
            "status": "pending",
            "error": None,
            "ic_mean": None,
            "icir": None,
            "turnover": None,
            "best_horizon": None,
            "passed": None,
            "reasons": []
        }
        
        try:
            # 获取因子规范
            spec = get(factor_name)
            
            # 1. 计算因子值
            print("   🧮 计算因子值...")
            factor_values = self.engine.compute(
                factor_name=factor_name,
                start=start,
                end=end,
                universe=None  # 使用默认 universe
            )
            print(f"      ✓ 因子值数量: {len(factor_values):,}")
            
            # 2. 评价因子（使用因子自己的 horizons）
            print("   🔍 评价因子表现...")
            factor_evaluator = FactorEvaluator(horizons=spec.horizons)
            report = factor_evaluator.evaluate(
                factor=factor_values,
                fwd_returns=fwd_returns,
                universe_mask=universe_mask
            )
            
            # 提取最佳窗口指标
            if report.best_horizon:
                best_metrics = report.metrics[report.best_horizon]
                result["ic_mean"] = best_metrics.rank_ic_mean
                result["icir"] = best_metrics.icir
                result["turnover"] = best_metrics.turnover_adjusted
                result["best_horizon"] = report.best_horizon
                
                print(f"      ✓ 最佳窗口: {report.best_horizon} 天")
                print(f"      ✓ IC={best_metrics.rank_ic_mean:.4f}, "
                      f"ICIR={best_metrics.icir:.4f}, "
                      f"换手={best_metrics.turnover_adjusted:.4f}")
            
            # 3. 判断是否通过入库标准
            print("   📋 判断入库资格...")
            passed, reasons = self.admission.judge(report)
            result["passed"] = passed
            result["reasons"] = reasons
            
            if passed:
                print(f"      ✅ 通过入库标准")
            else:
                print(f"      ❌ 未通过入库标准:")
                for reason in reasons:
                    print(f"         - {reason}")
            
            # 4. 如果模式是 admit 且通过，则保存到因子库
            if mode == "admit" and passed:
                print("   💾 保存到因子库...")
                spec = get(factor_name)
                self.library.save_factor(
                    spec=spec,
                    values=factor_values,
                    report=report.to_dict()
                )
                print(f"      ✓ 已保存到 {self.config['library']['root']}/{factor_name}/")
            
            result["status"] = "success"
            
        except Exception as e:
            print(f"   ✗ 处理失败: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    def _generate_summary(self, results: List[dict]) -> pd.DataFrame:
        """生成结果汇总表
        
        Args:
            results: 结果字典列表
            
        Returns:
            汇总 DataFrame
        """
        rows = []
        for r in results:
            rows.append({
                "因子名称": r["factor_name"],
                "状态": "✓" if r["status"] == "success" else "✗",
                "最佳窗口": r["best_horizon"] if r["best_horizon"] else "-",
                "IC均值": f"{r['ic_mean']:.4f}" if r["ic_mean"] is not None else "-",
                "ICIR": f"{r['icir']:.4f}" if r["icir"] is not None else "-",
                "换手率": f"{r['turnover']:.4f}" if r["turnover"] is not None else "-",
                "入库": "✅" if r["passed"] else "❌" if r["passed"] is False else "-",
                "错误": r["error"] if r["error"] else ""
            })
        
        return pd.DataFrame(rows)
    
    def _print_summary(self, summary_df: pd.DataFrame, mode: str):
        """打印汇总报告
        
        Args:
            summary_df: 汇总 DataFrame
            mode: 运行模式
        """
        print("=" * 80)
        print("📈 结果汇总")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print()
        
        # 统计
        total = len(summary_df)
        success = (summary_df["状态"] == "✓").sum()
        failed = total - success
        
        print(f"总计: {total} 个因子")
        print(f"  ✓ 成功: {success}")
        print(f"  ✗ 失败: {failed}")
        
        if mode == "admit":
            passed = (summary_df["入库"] == "✅").sum()
            rejected = (summary_df["入库"] == "❌").sum()
            print(f"  ✅ 已入库: {passed}")
            print(f"  ❌ 未入库: {rejected}")
        
        print("=" * 80)


def load_config(config_path: str) -> dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="因子全流程自动化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认模式：仅评价，不入库
  python main.py
  
  # 评价并入库通过的因子
  python main.py --mode admit
  
  # 指定日期范围
  python main.py --start 2022-01-01 --end 2023-12-31
  
  # 仅处理指定因子
  python main.py --factors momentum_20d volatility_20d
  
  # 覆盖入库阈值
  python main.py --mode admit --min-ic 0.03 --min-icir 0.5
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="开始日期 (YYYY-MM-DD)，默认: 2022-01-01"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="结束日期 (YYYY-MM-DD)，默认: 2023-12-31"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "admit"],
        default="admit",
        help="运行模式: evaluate=仅评价, admit=评价并入库，默认: admit"
    )
    
    parser.add_argument(
        "--factors",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的因子名称列表，默认: 全部因子"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="project/config.yaml",
        help="配置文件路径，默认: project/config.yaml"
    )
    
    parser.add_argument(
        "--min-ic",
        type=float,
        default=None,
        help="覆盖配置中的最小 IC 阈值"
    )
    
    parser.add_argument(
        "--min-icir",
        type=float,
        default=None,
        help="覆盖配置中的最小 ICIR 阈值"
    )
    
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="覆盖配置中的最大换手率阈值"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="factor_results",
        help="结果输出目录，默认: factor_results/"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    print("📋 加载配置...")
    try:
        config = load_config(args.config)
        print(f"   ✓ 配置文件: {args.config}")
    except Exception as e:
        print(f"   ✗ 无法加载配置文件: {e}")
        sys.exit(1)
    
    # 覆盖配置（如果指定了命令行参数）
    if args.min_ic is not None:
        config["admission"]["min_ic"] = args.min_ic
        print(f"   覆盖 min_ic: {args.min_ic}")
    
    if args.min_icir is not None:
        config["admission"]["min_icir"] = args.min_icir
        print(f"   覆盖 min_icir: {args.min_icir}")
    
    if args.max_turnover is not None:
        config["admission"]["max_turnover_adj"] = args.max_turnover
        print(f"   覆盖 max_turnover_adj: {args.max_turnover}")
    
    print()
    
    # 创建管道并运行
    try:
        pipeline = FactorPipeline(config)
        summary = pipeline.run(
            start=args.start,
            end=args.end,
            mode=args.mode,
            factor_names=args.factors
        )
        
        # 保存结果到文件
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"factor_results_{timestamp}.csv"
        summary.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 结果已保存到: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
