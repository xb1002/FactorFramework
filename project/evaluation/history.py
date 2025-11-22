"""因子评价历史记录管理。

记录所有已评价的因子，无论是否通过入库标准。
用于跳过已评价的因子，避免重复计算。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class EvaluationHistory:
    """因子评价历史记录。
    
    记录结构：
    {
        "factor_name": {
            "last_evaluated": "2024-01-01 12:00:00",
            "status": "success/failed",
            "passed": true/false,
            "ic_mean": 0.05,
            "icir": 0.8,
            "best_horizon": 20,
            "date_range": "2022-01-01 to 2023-12-31",
            "error": null
        }
    }
    
    Attributes:
        history_file: 历史记录文件路径
        records: 历史记录字典
    """
    
    def __init__(self, history_file: str | Path = "factor_evaluation_history.json") -> None:
        """初始化评价历史记录。
        
        Args:
            history_file: 历史记录文件路径
        """
        self.history_file = Path(history_file)
        self.records = self._load()
    
    def _load(self) -> dict:
        """从文件加载历史记录。
        
        Returns:
            历史记录字典
        """
        if not self.history_file.exists():
            return {}
        
        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # 如果文件损坏，返回空字典
            return {}
    
    def _save(self) -> None:
        """保存历史记录到文件。"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
    
    def is_evaluated(self, factor_name: str) -> bool:
        """检查因子是否已评价过。
        
        Args:
            factor_name: 因子名称
            
        Returns:
            True 如果已评价，False 否则
        """
        return factor_name in self.records and self.records[factor_name].get("status") == "success"
    
    def record_evaluation(
        self,
        factor_name: str,
        status: str,
        date_range: str,
        passed: Optional[bool] = None,
        ic_mean: Optional[float] = None,
        icir: Optional[float] = None,
        best_horizon: Optional[int] = None,
        turnover: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """记录一次因子评价。
        
        Args:
            factor_name: 因子名称
            status: 评价状态 ("success" 或 "failed")
            date_range: 评价日期范围
            passed: 是否通过入库标准
            ic_mean: IC均值
            icir: ICIR值
            best_horizon: 最佳持有期
            turnover: 换手率
            error: 错误信息（如果失败）
        """
        self.records[factor_name] = {
            "last_evaluated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "passed": passed,
            "ic_mean": ic_mean,
            "icir": icir,
            "best_horizon": best_horizon,
            "turnover": turnover,
            "date_range": date_range,
            "error": error,
        }
        self._save()
    
    def get_record(self, factor_name: str) -> Optional[dict]:
        """获取因子的评价记录。
        
        Args:
            factor_name: 因子名称
            
        Returns:
            评价记录字典，如果不存在返回 None
        """
        return self.records.get(factor_name)
    
    def remove_record(self, factor_name: str) -> bool:
        """删除因子的评价记录。
        
        用于强制重新评价某个因子。
        
        Args:
            factor_name: 因子名称
            
        Returns:
            True 如果删除成功，False 如果记录不存在
        """
        if factor_name in self.records:
            del self.records[factor_name]
            self._save()
            return True
        return False
    
    def clear_all(self) -> None:
        """清空所有评价记录。
        
        用于强制重新评价所有因子。
        """
        self.records = {}
        self._save()
    
    def get_all_evaluated(self) -> list[str]:
        """获取所有已评价的因子名称列表。
        
        Returns:
            因子名称列表
        """
        return [name for name, record in self.records.items() 
                if record.get("status") == "success"]
    
    def get_passed_factors(self) -> list[str]:
        """获取所有通过入库标准的因子名称列表。
        
        Returns:
            因子名称列表
        """
        return [name for name, record in self.records.items() 
                if record.get("status") == "success" and record.get("passed")]
    
    def get_failed_factors(self) -> list[str]:
        """获取所有未通过入库标准的因子名称列表。
        
        Returns:
            因子名称列表
        """
        return [name for name, record in self.records.items() 
                if record.get("status") == "success" and not record.get("passed")]
    
    def print_summary(self) -> None:
        """打印评价历史摘要。"""
        if not self.records:
            print("📝 评价历史: 无记录")
            return
        
        total = len(self.records)
        success = sum(1 for r in self.records.values() if r.get("status") == "success")
        failed = total - success
        passed = sum(1 for r in self.records.values() 
                    if r.get("status") == "success" and r.get("passed"))
        rejected = sum(1 for r in self.records.values() 
                      if r.get("status") == "success" and not r.get("passed"))
        
        print(f"📝 评价历史摘要:")
        print(f"   总计: {total} 个因子")
        print(f"   ✓ 评价成功: {success}")
        print(f"   ✗ 评价失败: {failed}")
        print(f"   ✅ 通过入库: {passed}")
        print(f"   ❌ 未通过入库: {rejected}")
