"""
Layer3提取器

从事件聚类中提取模式、偏好和行为规则
"""

import json
from typing import Dict, List, Any
from core.infrastructure import LLMClient, parse_llm_json
from .prompt import build_pattern_analysis_prompt


class Layer3Extractor:
    """Layer3模式提取器"""
    
    def __init__(self, llm_client: LLMClient, token_tracker=None, default_provider: str = "deepseek"):
        self.llm_client = llm_client
        self.token_tracker = token_tracker  # Token统计收集器（可选）
        self.default_provider = default_provider  # 默认LLM提供商
    
    def extract_patterns_from_cluster(
        self,
        cluster_events: List[Dict[str, Any]],
        related_states: List[Dict[str, Any]] = None,
        related_contexts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从事件聚类中提取模式
        
        Args:
            cluster_events: 聚类的事件列表
            related_states: 相关的状态列表
            related_contexts: 相关的上下文列表
        
        Returns:
            提取结果 {"event_cluster": {...}, "patterns": [...], "preferences": [...], "behavior_rules": [...]}
        """
        try:
            if not cluster_events:
                return {
                    "event_cluster": None,
                    "patterns": [],
                    "preferences": [],
                    "behavior_rules": []
                }
            
            # 构建prompt
            prompt = build_pattern_analysis_prompt(
                cluster_events,
                related_states or [],
                related_contexts or []
            )
            
            # 调用LLM
            # 如果启用了token追踪，获取usage信息
            if self.token_tracker:
                response, usage = self.llm_client.call_llm(
                    prompt,
                    provider=self.default_provider,
                    return_usage=True
                )
                # 记录token使用情况
                self.token_tracker.record_llm_call("layer3_pattern", usage, provider=self.default_provider)
            else:
                response = self.llm_client.call_llm(
                    prompt,
                    provider=self.default_provider
                )
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            # 添加元数据（记录cluster包含的事件ID）
            result['source_event_ids'] = [e.get('id') for e in cluster_events]
            
            return result
            
        except Exception as e:
            print(f"❌ Layer3提取失败: {e}")
            return {
                "event_cluster": None,
                "patterns": [],
                "preferences": [],
                "behavior_rules": [],
                "error": str(e)
            }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应（使用JSON修复工具）
        
        Args:
            response: LLM响应文本
        
        Returns:
            解析后的结果
        """
        # 使用JSON修复工具解析
        default_result = {
            "event_cluster": None,
            "patterns": [],
            "preferences": [],
            "behavior_rules": []
        }
        
        data = parse_llm_json(
            response,
            expected_keys=['event_cluster', 'patterns', 'preferences', 'behavior_rules'],
            default=default_result
        )
        
        if data is None:
            return default_result
        
        return {
            "event_cluster": data.get('event_cluster'),
            "patterns": data.get('patterns', []),
            "preferences": data.get('preferences', []),
            "behavior_rules": data.get('behavior_rules', [])
        }

