"""Persistent storage for factors, metadata, and reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from project.factors.registry import FactorSpec


class FactorLibrary:
    """因子库持久化存储。
    
    管理因子值、元数据、评价报告和表达式的文件系统存储。
    
    存储结构：
    - values/<factor>/<version>.parquet: 因子值
    - meta/<factor>/<version>.json: 元数据
    - reports/<factor>/<version>.json: 评价报告
    - expressions/<factor>/<version>.py: 表达式源代码
    
    Attributes:
        root: 因子库根目录路径
    """
    
    def __init__(self, root: str | Path = "factor_store") -> None:
        """初始化因子库。
        
        Args:
            root: 因子库根目录路径
        """
        self.root = Path(root)

    def _factor_dir(self, factor: str) -> Path:
        """获取因子目录路径。"""
        return self.root / factor

    def _value_path(self, factor: str, version: str) -> Path:
        """获取因子值文件路径。"""
        return self._factor_dir(factor) / "values" / f"{version}.parquet"

    def _meta_path(self, factor: str, version: str) -> Path:
        """获取元数据文件路径。"""
        return self._factor_dir(factor) / "meta" / f"{version}.json"

    def _report_path(self, factor: str, version: str) -> Path:
        """获取评价报告文件路径。"""
        return self._factor_dir(factor) / "reports" / f"{version}.json"

    def _expr_path(self, factor: str, version: str) -> Path:
        """获取表达式文件路径。"""
        return self._factor_dir(factor) / "expressions" / f"{version}.py"

    def _ensure_paths(self, path: Path) -> None:
        """确保目录存在，不存在则创建。"""
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_factor(self, spec: FactorSpec, values: pd.Series, report: dict | None = None) -> str:
        """保存因子到库中。
        
        保存因子值、元数据、评价报告和表达式到相应文件。
        
        Args:
            spec: 因子规范对象
            values: 因子值 Series
            report: 评价报告字典（可选）
            
        Returns:
            因子目录的字符串路径
            
        Raises:
            ValueError: 当 spec 缺少 version 时
            FileExistsError: 当版本已存在时
        """
        if spec.version is None:
            raise ValueError("FactorSpec must include a version for persistence")
        factor_dir = self._factor_dir(spec.name)
        value_path = self._value_path(spec.name, spec.version)
        if value_path.exists():
            raise FileExistsError(f"Version {spec.version} for factor {spec.name} already exists")

        self._ensure_paths(value_path)
        values.to_frame(name=spec.name).to_parquet(value_path)

        meta_path = self._meta_path(spec.name, spec.version)
        self._ensure_paths(meta_path)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)

        if report is not None:
            report_path = self._report_path(spec.name, spec.version)
            self._ensure_paths(report_path)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        expr_path = self._expr_path(spec.name, spec.version)
        self._ensure_paths(expr_path)
        if spec.expr:
            expr_path.write_text(spec.expr, encoding="utf-8")
        return str(factor_dir)

    def _resolve_version(self, factor: str, version: Optional[str]) -> str:
        """解析版本号。
        
        如果未指定版本，返回最新版本（按字母序最大）。
        
        Args:
            factor: 因子名称
            version: 版本号，None 时自动选择最新版本
            
        Returns:
            解析后的版本号
            
        Raises:
            FileNotFoundError: 当因子不存在或无可用版本时
        """
        factor_dir = self._factor_dir(factor)
        values_dir = factor_dir / "values"
        if version:
            return version
        if not values_dir.exists():
            raise FileNotFoundError(f"No stored versions found for factor {factor}")
        versions = sorted(p.stem for p in values_dir.glob("*.parquet"))
        if not versions:
            raise FileNotFoundError(f"No stored versions found for factor {factor}")
        return versions[-1]

    def load_values(self, factor: str, version: Optional[str] = None) -> pd.Series:
        """加载因子值。
        
        Args:
            factor: 因子名称
            version: 版本号，None 时加载最新版本
            
        Returns:
            因子值 Series
            
        Raises:
            FileNotFoundError: 当因子或版本不存在时
        """
        resolved_version = self._resolve_version(factor, version)
        path = self._value_path(factor, resolved_version)
        df = pd.read_parquet(path)
        return df.iloc[:, 0]

    def load_report(self, factor: str, version: Optional[str] = None) -> dict:
        """加载评价报告。
        
        Args:
            factor: 因子名称
            version: 版本号，None 时加载最新版本
            
        Returns:
            评价报告字典
            
        Raises:
            FileNotFoundError: 当报告文件不存在时
        """
        resolved_version = self._resolve_version(factor, version)
        path = self._report_path(factor, resolved_version)
        with path.open(encoding="utf-8") as f:
            return json.load(f)
