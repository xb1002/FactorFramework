"""Persistent storage for factors, metadata, and reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from project.factors.registry import FactorSpec


class FactorLibrary:
    def __init__(self, root: str | Path = "factor_store") -> None:
        self.root = Path(root)

    def _factor_dir(self, factor: str) -> Path:
        return self.root / factor

    def _value_path(self, factor: str, version: str) -> Path:
        return self._factor_dir(factor) / "values" / f"{version}.parquet"

    def _meta_path(self, factor: str, version: str) -> Path:
        return self._factor_dir(factor) / "meta" / f"{version}.json"

    def _report_path(self, factor: str, version: str) -> Path:
        return self._factor_dir(factor) / "reports" / f"{version}.json"

    def _expr_path(self, factor: str, version: str) -> Path:
        return self._factor_dir(factor) / "expressions" / f"{version}.py"

    def _ensure_paths(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_factor(self, spec: FactorSpec, values: pd.Series, report: dict | None = None) -> str:
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
        with meta_path.open("w") as f:
            json.dump(spec.to_dict(), f, indent=2)

        if report is not None:
            report_path = self._report_path(spec.name, spec.version)
            self._ensure_paths(report_path)
            with report_path.open("w") as f:
                json.dump(report, f, indent=2)

        expr_path = self._expr_path(spec.name, spec.version)
        self._ensure_paths(expr_path)
        if spec.expr:
            expr_path.write_text(spec.expr)
        return str(factor_dir)

    def _resolve_version(self, factor: str, version: Optional[str]) -> str:
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
        resolved_version = self._resolve_version(factor, version)
        path = self._value_path(factor, resolved_version)
        df = pd.read_parquet(path)
        return df.iloc[:, 0]

    def load_report(self, factor: str, version: Optional[str] = None) -> dict:
        resolved_version = self._resolve_version(factor, version)
        path = self._report_path(factor, resolved_version)
        with path.open() as f:
            return json.load(f)
