"""
因子库对外API接口

提供给外部调用的因子库接口，允许外部数据调用因子表达式生成因子值。

主要功能：
1. 获取所有已入库的因子名称列表
2. 根据因子名称获取因子元数据
3. 根据因子名称计算因子值（使用入库的表达式）
4. 批量计算多个因子

使用示例：
    >>> from factor_api import FactorAPI
    >>> import pandas as pd
    >>> 
    >>> # 初始化API
    >>> api = FactorAPI()
    >>> 
    >>> # 获取所有因子名称
    >>> factors = api.list_factors()
    >>> print(factors)
    >>> 
    >>> # 获取因子元数据
    >>> meta = api.get_factor_info("ma_crossover")
    >>> print(meta)
    >>> 
    >>> # 计算因子值
    >>> market_data = pd.read_parquet("data/processed/market.parquet")
    >>> factor_values = api.compute_factor("ma_crossover", market_data)
    >>> 
    >>> # 批量计算多个因子
    >>> results = api.compute_factors(["ma_crossover", "momentum_20d"], market_data)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd


class FactorAPI:
    """因子库对外API接口。
    
    提供统一的接口供外部调用已入库的因子表达式。
    
    Attributes:
        factor_store_path: 因子库存储路径
    """
    
    def __init__(self, factor_store_path: str | Path = "factor_store") -> None:
        """初始化因子库API。
        
        Args:
            factor_store_path: 因子库根目录路径，默认为 "factor_store"
        """
        self.factor_store_path = Path(factor_store_path)
        if not self.factor_store_path.exists():
            raise FileNotFoundError(
                f"因子库路径不存在: {self.factor_store_path}\n"
                f"请先运行因子评价流程生成因子库。"
            )
    
    def list_factors(self, include_versions: bool = False) -> List[str] | Dict[str, List[str]]:
        """获取所有已入库的因子名称列表。
        
        Args:
            include_versions: 是否包含版本信息
                - False (默认): 返回因子名称列表
                - True: 返回字典，键为因子名，值为版本列表
        
        Returns:
            因子名称列表或因子-版本字典
            
        Example:
            >>> api = FactorAPI()
            >>> # 获取因子名称
            >>> api.list_factors()
            ['ma_crossover', 'momentum_20d', 'volatility_20d']
            >>> 
            >>> # 获取因子及其版本
            >>> api.list_factors(include_versions=True)
            {'ma_crossover': ['v1'], 'momentum_20d': ['v1', 'v2']}
        """
        if not self.factor_store_path.exists():
            return {} if include_versions else []
        
        factor_dirs = [d for d in self.factor_store_path.iterdir() if d.is_dir()]
        
        if not include_versions:
            return sorted([d.name for d in factor_dirs])
        
        # 包含版本信息
        factor_versions = {}
        for factor_dir in factor_dirs:
            factor_name = factor_dir.name
            meta_dir = factor_dir / "meta"
            if meta_dir.exists():
                versions = sorted([p.stem for p in meta_dir.glob("*.json")])
                if versions:
                    factor_versions[factor_name] = versions
        
        return dict(sorted(factor_versions.items()))
    
    def get_factor_info(
        self, 
        factor_name: str, 
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取因子的元数据信息。
        
        Args:
            factor_name: 因子名称
            version: 版本号，None 时返回最新版本
        
        Returns:
            因子元数据字典，包含：
            - name: 因子名称
            - version: 版本号
            - required_fields: 所需数据字段
            - horizons: 评价窗口
            - expr: 因子表达式代码
            - code_hash: 代码哈希值
            - env: 运行环境信息
            
        Raises:
            FileNotFoundError: 当因子不存在时
            
        Example:
            >>> info = api.get_factor_info("ma_crossover")
            >>> print(info['required_fields'])
            ['close']
            >>> print(info['horizons'])
            [1, 5, 10]
        """
        factor_dir = self.factor_store_path / factor_name
        if not factor_dir.exists():
            raise FileNotFoundError(
                f"因子 '{factor_name}' 不存在。\n"
                f"可用因子: {', '.join(self.list_factors())}"
            )
        
        # 解析版本
        meta_dir = factor_dir / "meta"
        if not meta_dir.exists():
            raise FileNotFoundError(f"因子 '{factor_name}' 缺少元数据目录")
        
        if version is None:
            # 获取最新版本
            versions = sorted([p.stem for p in meta_dir.glob("*.json")])
            if not versions:
                raise FileNotFoundError(f"因子 '{factor_name}' 没有可用版本")
            version = versions[-1]
        
        # 读取元数据
        meta_path = meta_dir / f"{version}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"因子 '{factor_name}' 的版本 '{version}' 不存在。\n"
                f"可用版本: {', '.join(versions)}"
            )
        
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return metadata
    
    def compute_factor(
        self, 
        factor_name: str, 
        data: pd.DataFrame,
        version: Optional[str] = None,
        validate_fields: bool = True
    ) -> pd.Series:
        """计算单个因子的值。
        
        使用因子库中存储的表达式对输入数据计算因子值。
        
        Args:
            factor_name: 因子名称
            data: 输入数据 DataFrame，需包含因子所需字段
                  索引应为 MultiIndex (date, code) 或 DatetimeIndex
            version: 版本号，None 时使用最新版本
            validate_fields: 是否验证输入数据包含所需字段
        
        Returns:
            因子值 Series，索引与输入数据一致
            
        Raises:
            FileNotFoundError: 当因子不存在时
            ValueError: 当数据缺少必需字段时
            RuntimeError: 当因子计算失败时
            
        Example:
            >>> market_data = pd.read_parquet("data/processed/market.parquet")
            >>> factor_values = api.compute_factor("ma_crossover", market_data)
            >>> print(factor_values.head())
        """
        # 获取因子元数据
        metadata = self.get_factor_info(factor_name, version)
        
        # 验证必需字段
        if validate_fields:
            required_fields = metadata.get("required_fields", [])
            missing_fields = set(required_fields) - set(data.columns)
            if missing_fields:
                raise ValueError(
                    f"输入数据缺少必需字段: {missing_fields}\n"
                    f"因子 '{factor_name}' 需要字段: {required_fields}\n"
                    f"数据包含字段: {list(data.columns)}"
                )
        
        # 获取因子表达式
        factor_expr = metadata.get("expr")
        if not factor_expr:
            raise RuntimeError(f"因子 '{factor_name}' 缺少表达式代码")
        
        # 执行因子计算
        try:
            # 创建执行环境
            # 定义一个空的 register_factor 装饰器（只返回函数本身）
            def dummy_register_factor(**kwargs):
                def decorator(func):
                    return func
                return decorator
            
            exec_globals = {
                "pd": pd,
                "register_factor": dummy_register_factor,
                "__name__": "__main__",
            }
            
            # 执行因子代码（定义函数）
            exec(factor_expr, exec_globals)
            
            # 获取因子函数（使用 qualname）
            func_name = metadata.get("qualname", factor_name)
            if func_name not in exec_globals:
                raise RuntimeError(f"无法找到因子函数 '{func_name}'")
            
            factor_func = exec_globals[func_name]
            
            # 计算因子值
            result = factor_func(data)
            
            if not isinstance(result, pd.Series):
                raise RuntimeError(
                    f"因子函数返回类型错误: {type(result)}，应为 pd.Series"
                )
            
            return result
            
        except Exception as e:
            raise RuntimeError(
                f"计算因子 '{factor_name}' 时发生错误: {str(e)}"
            ) from e
    
    def compute_factors(
        self,
        factor_names: List[str],
        data: pd.DataFrame,
        version: Optional[str] = None,
        validate_fields: bool = True,
        skip_errors: bool = False
    ) -> Dict[str, pd.Series]:
        """批量计算多个因子的值。
        
        Args:
            factor_names: 因子名称列表
            data: 输入数据 DataFrame
            version: 版本号，None 时使用最新版本
            validate_fields: 是否验证输入数据包含所需字段
            skip_errors: 是否跳过计算失败的因子
                - False (默认): 遇到错误时抛出异常
                - True: 跳过失败的因子，继续计算其他因子
        
        Returns:
            字典，键为因子名称，值为因子值 Series
            
        Example:
            >>> results = api.compute_factors(
            ...     ["ma_crossover", "momentum_20d"], 
            ...     market_data
            ... )
            >>> for name, values in results.items():
            ...     print(f"{name}: {len(values)} 个值")
        """
        results = {}
        errors = {}
        
        for factor_name in factor_names:
            try:
                result = self.compute_factor(
                    factor_name, 
                    data, 
                    version=version,
                    validate_fields=validate_fields
                )
                results[factor_name] = result
                
            except Exception as e:
                if skip_errors:
                    errors[factor_name] = str(e)
                    warnings.warn(
                        f"计算因子 '{factor_name}' 失败: {str(e)}",
                        RuntimeWarning
                    )
                else:
                    raise
        
        if errors and skip_errors:
            print(f"\n警告: {len(errors)} 个因子计算失败:")
            for name, error in errors.items():
                print(f"  - {name}: {error}")
        
        return results
    
    def get_factor_expression(
        self,
        factor_name: str,
        version: Optional[str] = None
    ) -> str:
        """获取因子的表达式代码。
        
        Args:
            factor_name: 因子名称
            version: 版本号，None 时返回最新版本
        
        Returns:
            因子表达式代码字符串
            
        Example:
            >>> expr = api.get_factor_expression("ma_crossover")
            >>> print(expr)
        """
        metadata = self.get_factor_info(factor_name, version)
        return metadata.get("expr", "")
    
    def get_factor_report(
        self,
        factor_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取因子的评价报告。
        
        Args:
            factor_name: 因子名称
            version: 版本号，None 时返回最新版本
        
        Returns:
            评价报告字典，包含IC、ICIR、换手率等指标
            
        Raises:
            FileNotFoundError: 当报告不存在时
            
        Example:
            >>> report = api.get_factor_report("ma_crossover")
            >>> print(f"IC: {report['rank_ic_mean']}")
            >>> print(f"ICIR: {report['icir']}")
        """
        metadata = self.get_factor_info(factor_name, version)
        resolved_version = metadata["version"]
        
        factor_dir = self.factor_store_path / factor_name
        report_path = factor_dir / "reports" / f"{resolved_version}.json"
        
        if not report_path.exists():
            raise FileNotFoundError(
                f"因子 '{factor_name}' (版本 {resolved_version}) 没有评价报告"
            )
        
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)


# ============================================================================
# 便捷函数
# ============================================================================

def list_all_factors(factor_store_path: str | Path = "factor_store") -> List[str]:
    """获取所有因子名称（便捷函数）。
    
    Args:
        factor_store_path: 因子库路径
    
    Returns:
        因子名称列表
    """
    api = FactorAPI(factor_store_path)
    return api.list_factors()


def compute_single_factor(
    factor_name: str,
    data: pd.DataFrame,
    factor_store_path: str | Path = "factor_store",
    version: Optional[str] = None
) -> pd.Series:
    """计算单个因子（便捷函数）。
    
    Args:
        factor_name: 因子名称
        data: 输入数据
        factor_store_path: 因子库路径
        version: 版本号
    
    Returns:
        因子值 Series
    """
    api = FactorAPI(factor_store_path)
    return api.compute_factor(factor_name, data, version=version)


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="因子库API命令行接口")
    parser.add_argument(
        "command",
        choices=["list", "info", "compute"],
        help="命令: list(列出因子), info(查看信息), compute(计算因子)"
    )
    parser.add_argument(
        "--factor",
        type=str,
        help="因子名称（info和compute命令需要）"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="数据文件路径（compute命令需要）"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="因子版本号（可选）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径（compute命令可选）"
    )
    parser.add_argument(
        "--store",
        type=str,
        default="factor_store",
        help="因子库路径（默认: factor_store）"
    )
    
    args = parser.parse_args()
    
    try:
        api = FactorAPI(args.store)
        
        if args.command == "list":
            # 列出所有因子
            factors = api.list_factors(include_versions=True)
            print("\n已入库的因子:")
            print("=" * 60)
            for name, versions in factors.items():
                print(f"{name:30s} 版本: {', '.join(versions)}")
            print(f"\n总计: {len(factors)} 个因子")
            
        elif args.command == "info":
            # 查看因子信息
            if not args.factor:
                parser.error("info 命令需要 --factor 参数")
            
            info = api.get_factor_info(args.factor, args.version)
            print(f"\n因子信息: {args.factor}")
            print("=" * 60)
            print(f"版本: {info['version']}")
            print(f"所需字段: {', '.join(info['required_fields'])}")
            print(f"评价窗口: {info['horizons']}")
            print(f"\n表达式代码:")
            print("-" * 60)
            print(info['expr'])
            
        elif args.command == "compute":
            # 计算因子
            if not args.factor:
                parser.error("compute 命令需要 --factor 参数")
            if not args.data:
                parser.error("compute 命令需要 --data 参数")
            
            print(f"加载数据: {args.data}")
            data = pd.read_parquet(args.data)
            
            print(f"计算因子: {args.factor}")
            result = api.compute_factor(args.factor, data, args.version)
            
            print(f"\n计算完成:")
            print(f"  因子值数量: {len(result)}")
            print(f"  非空值数量: {result.notna().sum()}")
            print(f"  均值: {result.mean():.6f}")
            print(f"  标准差: {result.std():.6f}")
            
            if args.output:
                result.to_frame(name=args.factor).to_parquet(args.output)
                print(f"\n结果已保存到: {args.output}")
    
    except Exception as e:
        print(f"\n错误: {e}")
        exit(1)
