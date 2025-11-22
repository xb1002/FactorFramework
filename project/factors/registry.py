"""Factor registry and specification utilities."""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class FactorSpec:
    """因子规范数据类。
    
    存储因子的完整元数据信息，包括函数、依赖字段、预处理配置等。
    
    Attributes:
        name: 因子名称
        func: 因子计算函数，接受 DataFrame 返回 Series
        required_fields: 计算所需的数据字段列表
        horizons: 评价的时间窗口列表
        preprocess: 预处理配置字典
        expr_type: 表达式类型，默认 "python"
        expr: 因子表达式源代码
        code_hash: 源代码的 SHA256 哈希值
        module: 函数所在模块名
        qualname: 函数的限定名称
        git_commit: Git 提交哈希（如果有）
        env: 环境指纹信息（Python 版本、pandas 版本等）
        version: 因子版本号
    """
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    required_fields: Iterable[str]
    horizons: Iterable[int]
    preprocess: dict | None = None
    expr_type: str = "python"
    expr: str | None = None
    code_hash: str | None = None
    module: str | None = None
    qualname: str | None = None
    git_commit: str | None = None
    env: dict | None = None
    version: str | None = None

    def to_dict(self) -> dict:
        """将因子规范转换为字典格式。
        
        Returns:
            包含所有元数据的字典
        """
        return {
            "name": self.name,
            "required_fields": list(self.required_fields),
            "horizons": list(self.horizons),
            "preprocess": self.preprocess or {},
            "expr_type": self.expr_type,
            "expr": self.expr,
            "code_hash": self.code_hash,
            "module": self.module,
            "qualname": self.qualname,
            "git_commit": self.git_commit,
            "env": self.env,
            "version": self.version,
        }


_registry: Dict[str, FactorSpec] = {}


def _hash_source(source: str) -> str:
    """计算源代码的 SHA256 哈希值。
    
    Args:
        source: 源代码字符串
        
    Returns:
        SHA256 哈希值（十六进制字符串）
    """
    return hashlib.sha256(source.encode()).hexdigest()


def _env_fingerprint() -> dict:
    """生成环境指纹信息。
    
    Returns:
        包含 Python 和 pandas 版本的字典
    """
    return {"python": sys.version, "pandas": pd.__version__}


def register_factor(
    name: Optional[str] = None,
    required_fields: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    preprocess: Optional[dict] = None,
    version: Optional[str] = None,
):
    """注册因子函数的装饰器。
    
    使用此装饰器可以将因子函数注册到全局注册表中，
    同时自动提取函数源代码、计算哈希值等元数据。
    
    Args:
        name: 因子名称，None 时使用函数名
        required_fields: 计算所需字段列表
        horizons: 评价窗口列表
        preprocess: 预处理配置
        version: 版本号
        
    Returns:
        装饰器函数
        
    Raises:
        ValueError: 当因子名称已存在时
        
    Example:
        @register_factor(
            name="momentum",
            required_fields=["close"],
            horizons=[5, 10, 20]
        )
        def momentum(df):
            return df["close"].pct_change(20)
    """

    def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
        factor_name = name or func.__name__
        if factor_name in _registry:
            raise ValueError(f"Factor {factor_name} already registered")

        source = inspect.getsource(func)
        module = func.__module__
        qualname = func.__qualname__
        code_hash = _hash_source(source)
        spec = FactorSpec(
            name=factor_name,
            func=func,
            required_fields=required_fields or [],
            horizons=horizons or [],
            preprocess=preprocess or {},
            expr_type="python",
            expr=source,
            code_hash=code_hash,
            module=module,
            qualname=qualname,
            git_commit=os.getenv("GIT_COMMIT"),
            env=_env_fingerprint(),
            version=version,
        )
        _registry[factor_name] = spec
        return func

    return decorator


def get(name: str) -> FactorSpec:
    """获取已注册的因子规范。
    
    Args:
        name: 因子名称
        
    Returns:
        因子规范对象
        
    Raises:
        KeyError: 当因子不存在时
    """
    if name not in _registry:
        raise KeyError(f"Factor {name} not found")
    return _registry[name]


def list_all() -> List[str]:
    """列出所有已注册的因子名称。
    
    Returns:
        排序后的因子名称列表
    """
    return sorted(_registry.keys())


def exists(name: str) -> bool:
    """检查因子是否已注册。
    
    Args:
        name: 因子名称
        
    Returns:
        True 表示已注册，False 表示未注册
    """
    return name in _registry
