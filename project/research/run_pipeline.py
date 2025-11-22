"""Research orchestration for evaluating and admitting factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import yaml

from project.data_manager.datasource import LocalParquetSource
from project.data_manager.universe import DefaultUniverse
from project.evaluation.admission import (
    CorrelationRule,
    FactorAdmissionStandard,
    ICThresholdRule,
    TurnoverRule,
)
from project.evaluation.evaluator import FactorEvaluator
from project.evaluation.forward_return import build as build_forward_returns
from project.factors.engine import FactorEngine
from project.factors.library import FactorLibrary
from project.factors.registry import get
from project.factors.standardizer import Standardizer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_and_maybe_admit(
    factor_name: str,
    start: Optional[str],
    end: Optional[str],
    mode: str,
    config_path: str | Path = "project/config.yaml",
) -> None:
    config = load_config(config_path)
    source = LocalParquetSource(config["data"]["processed_path"])
    standardizer = Standardizer.from_config(config.get("standardizer", {}))
    engine = FactorEngine(source, standardizer, DefaultUniverse())
    spec = get(factor_name)

    factor_values = engine.compute(factor_name, start=start, end=end)
    df = source.load(start=start, end=end, fields=spec.required_fields)
    fwd = build_forward_returns(df, spec.horizons, price_col="close")
    evaluator = FactorEvaluator(spec.horizons)
    report = evaluator.evaluate(factor_values, fwd)

    if mode == "evaluate":
        print(json.dumps(report.to_dict(), indent=2))
        return

    admission_cfg = config.get("admission", {})
    rules = [
        ICThresholdRule(admission_cfg.get("min_ic", 0), admission_cfg.get("min_icir", 0)),
        TurnoverRule(admission_cfg.get("max_turnover_adj", 1.0)),
        CorrelationRule(admission_cfg.get("max_abs_corr", 0.85)),
    ]
    admission = FactorAdmissionStandard(rules)
    passed, reasons = admission.judge(report)

    if not passed:
        print("Admission failed:", reasons)
        return

    if mode == "admit":
        library = FactorLibrary(config.get("library", {}).get("root", "factor_store"))
        library.save_factor(spec, factor_values, report.to_dict())
        print(f"Factor {spec.name} saved to library")
    else:
        print("Admission passed (mode does not save)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor evaluation and optional admission")
    parser.add_argument("factor", help="Registered factor name")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--mode", choices=["evaluate", "admit", "batch"], default="evaluate")
    parser.add_argument("--config", default="project/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_and_maybe_admit(args.factor, args.start, args.end, args.mode, args.config)


if __name__ == "__main__":
    main()
