"""Admission rules for factor library entrance."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from project.evaluation.evaluator import FactorReport


class AdmissionRule(ABC):
    """入库规则抽象基类。
    
    所有入库规则都应继承此类并实现 check 方法。
    """
    
    @abstractmethod
    def check(self, report: FactorReport) -> tuple[bool, str]:
        """检查因子是否满足规则。
        
        Args:
            report: 因子评价报告
            
        Returns:
            (是否通过, 说明信息) 元组
        """
        ...


class ICThresholdRule(AdmissionRule):
    """IC 阈值规则。
    
    要求至少一个时间窗口同时满足最小 IC 和 ICIR 要求。
    
    Attributes:
        min_ic: 最小平均 Rank IC
        min_icir: 最小信息比率
    """
    
    def __init__(self, min_ic: float, min_icir: float) -> None:
        """初始化 IC 阈值规则。
        
        Args:
            min_ic: 最小 IC 要求
            min_icir: 最小 ICIR 要求
        """
        self.min_ic = min_ic
        self.min_icir = min_icir

    def check(self, report: FactorReport) -> tuple[bool, str]:
        """检查是否满足 IC 阈值。
        
        Args:
            report: 因子评价报告
            
        Returns:
            (是否通过, 说明) 元组
        """
        for h, metrics in report.metrics.items():
            if metrics.rank_ic_mean >= self.min_ic and metrics.icir >= self.min_icir:
                return True, f"horizon {h} passed IC thresholds"
        return False, "No horizon meets IC thresholds"


class TurnoverRule(AdmissionRule):
    """换手率规则。
    
    要求至少一个时间窗口的换手率不超过阈值。
    
    Attributes:
        max_turnover_adj: 最大允许换手率
    """
    
    def __init__(self, max_turnover_adj: float) -> None:
        """初始化换手率规则。
        
        Args:
            max_turnover_adj: 最大换手率阈值
        """
        self.max_turnover_adj = max_turnover_adj

    def check(self, report: FactorReport) -> tuple[bool, str]:
        """检查是否满足换手率要求。
        
        Args:
            report: 因子评价报告
            
        Returns:
            (是否通过, 说明) 元组
        """
        for h, metrics in report.metrics.items():
            if metrics.turnover_adjusted <= self.max_turnover_adj:
                return True, f"horizon {h} passes turnover threshold"
        return False, "Turnover too high for all horizons"


class CorrelationRule(AdmissionRule):
    """相关性规则。
    
    检查因子与已有库中因子的相关性（当前为占位实现）。
    
    Attributes:
        max_abs_corr: 最大允许绝对相关系数
    """
    
    def __init__(self, max_abs_corr: float = 0.85) -> None:
        """初始化相关性规则。
        
        Args:
            max_abs_corr: 最大绝对相关系数阈值
        """
        self.max_abs_corr = max_abs_corr

    def check(self, report: FactorReport) -> tuple[bool, str]:
        """检查相关性（占位实现）。
        
        Args:
            report: 因子评价报告
            
        Returns:
            (是否通过, 说明) 元组
            
        Note:
            当前为占位实现，始终返回通过
        """
        # Placeholder for future correlation checks against library factors
        return True, "Correlation check skipped (not implemented)"


class FactorAdmissionStandard:
    """因子入库标准。
    
    组合多个入库规则，要求全部通过才允许入库。
    
    Attributes:
        rules: 入库规则列表
    """
    
    def __init__(self, rules: Iterable[AdmissionRule]) -> None:
        """初始化入库标准。
        
        Args:
            rules: 入库规则的可迭代对象
        """
        self.rules = list(rules)

    def judge(self, report: FactorReport) -> tuple[bool, List[str]]:
        """判断因子是否可入库。
        
        检查所有规则，收集未通过规则的原因。
        
        Args:
            report: 因子评价报告
            
        Returns:
            (是否全部通过, 未通过原因列表) 元组
        """
        reasons: List[str] = []
        for rule in self.rules:
            passed, msg = rule.check(report)
            if not passed:
                reasons.append(msg)
        return len(reasons) == 0, reasons
