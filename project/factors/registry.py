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
    return hashlib.sha256(source.encode()).hexdigest()


def _env_fingerprint() -> dict:
    return {"python": sys.version, "pandas": pd.__version__}


def register_factor(
    name: Optional[str] = None,
    required_fields: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    preprocess: Optional[dict] = None,
    version: Optional[str] = None,
):
    """Decorator to register a factor function with metadata."""

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
    if name not in _registry:
        raise KeyError(f"Factor {name} not found")
    return _registry[name]


def list_all() -> List[str]:
    return sorted(_registry.keys())


def exists(name: str) -> bool:
    return name in _registry
