"""
Token统计收集器

用于收集和统计LLM调用的token使用情况
"""

from typing import Dict, List, Any
from collections import defaultdict


class TokenTracker:
    """Token统计收集器"""
    
    def __init__(self):
        """初始化统计收集器"""
        self.stats = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "call_details": []  # 记录每次调用的详细信息
        })
    
    def record_llm_call(
        self, 
        call_type: str, 
        usage: Dict[str, int], 
        provider: str = "unknown",
        context: str = None
    ):
        """
        记录一次LLM调用
        
        Args:
            call_type: 调用类型（如 "fragment_splitting", "layer1_extraction", "layer1_conflict", "layer2_extraction", "layer3_pattern"）
            usage: token使用情况 {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
            provider: LLM提供商 ("openai" 或 "deepseek")
            context: 上下文信息（可选，如fragment_id）
        """
        key = f"{call_type}_{provider}"
        
        self.stats[key]["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.stats[key]["completion_tokens"] += usage.get("completion_tokens", 0)
        self.stats[key]["total_tokens"] += usage.get("total_tokens", 0)
        self.stats[key]["calls"] += 1
        
        # 记录详细信息
        self.stats[key]["call_details"].append({
            "usage": usage,
            "context": context
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计结果
        
        Returns:
            统计字典，按调用类型分组
        """
        result = {}
        for key, stats in self.stats.items():
            # 解析key获取call_type和provider
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                call_type, provider = parts
            else:
                call_type, provider = key, "unknown"
            
            if call_type not in result:
                result[call_type] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                    "by_provider": {}
                }
            
            result[call_type]["prompt_tokens"] += stats["prompt_tokens"]
            result[call_type]["completion_tokens"] += stats["completion_tokens"]
            result[call_type]["total_tokens"] += stats["total_tokens"]
            result[call_type]["calls"] += stats["calls"]
            result[call_type]["by_provider"][provider] = {
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "calls": stats["calls"]
            }
        
        return result
    
    def get_total_tokens(self) -> int:
        """获取总token数"""
        total = 0
        for stats in self.stats.values():
            total += stats["total_tokens"]
        return total
    
    def reset(self):
        """重置统计"""
        self.stats.clear()

