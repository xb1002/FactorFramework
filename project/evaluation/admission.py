"""Admission rules for factor library entrance."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from project.evaluation.evaluator import FactorReport


class AdmissionRule(ABC):
    @abstractmethod
    def check(self, report: FactorReport) -> tuple[bool, str]:
        ...


class ICThresholdRule(AdmissionRule):
    def __init__(self, min_ic: float, min_icir: float) -> None:
        self.min_ic = min_ic
        self.min_icir = min_icir

    def check(self, report: FactorReport) -> tuple[bool, str]:
        for h, metrics in report.metrics.items():
            if metrics.rank_ic_mean >= self.min_ic and metrics.icir >= self.min_icir:
                return True, f"horizon {h} passed IC thresholds"
        return False, "No horizon meets IC thresholds"


class TurnoverRule(AdmissionRule):
    def __init__(self, max_turnover_adj: float) -> None:
        self.max_turnover_adj = max_turnover_adj

    def check(self, report: FactorReport) -> tuple[bool, str]:
        for h, metrics in report.metrics.items():
            if metrics.turnover_adjusted <= self.max_turnover_adj:
                return True, f"horizon {h} passes turnover threshold"
        return False, "Turnover too high for all horizons"


class CorrelationRule(AdmissionRule):
    def __init__(self, max_abs_corr: float = 0.85) -> None:
        self.max_abs_corr = max_abs_corr

    def check(self, report: FactorReport) -> tuple[bool, str]:
        # Placeholder for future correlation checks against library factors
        return True, "Correlation check skipped (not implemented)"


class FactorAdmissionStandard:
    def __init__(self, rules: Iterable[AdmissionRule]) -> None:
        self.rules = list(rules)

    def judge(self, report: FactorReport) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        for rule in self.rules:
            passed, msg = rule.check(report)
            if not passed:
                reasons.append(msg)
        return len(reasons) == 0, reasons
