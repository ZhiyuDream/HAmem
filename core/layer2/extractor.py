"""
Layer2提取器

从fragment中提取时间线信息（事件、状态、上下文）
"""

import json
from typing import Dict, List, Any
from core.infrastructure import LLMClient, parse_llm_json
from .prompt import build_layer2_extraction_prompt


class Layer2Extractor:
    """Layer2时间线提取器"""
    
    def __init__(self, llm_client: LLMClient, default_provider: str = "deepseek", token_tracker=None):
        self.llm_client = llm_client
        self.default_provider = default_provider
        self.token_tracker = token_tracker  # Token统计收集器（可选）
    
    def extract_from_fragment(
        self, 
        fragment: Dict[str, Any],
        layer1_entities: List[Dict[str, Any]],
        existing_layer2_nodes: List[Dict[str, Any]] = None,
        provider: str = None
    ) -> Dict[str, Any]:
        """
        从fragment中提取时间线信息（支持已有节点召回和关联）
        
        Args:
            fragment: fragment数据 {"id": "...", "content": "...", "time": "..."}
            layer1_entities: Layer1提取的实体列表
            existing_layer2_nodes: 召回的已有Layer2节点列表（用于关联）
            provider: LLM提供商
        
        Returns:
            提取结果 {"events": [...], "states": [...], "contexts": [...]}
        """
        try:
            fragment_text = fragment.get('content', '')
            session_time = fragment.get('time', 'unknown')
            
            if not fragment_text:
                return {"events": [], "states": [], "contexts": []}
            
            # 构建prompt（包含已有节点信息）
            prompt = build_layer2_extraction_prompt(
                fragment_text, 
                session_time,
                layer1_entities,
                existing_layer2_nodes=existing_layer2_nodes
            )
            
            # 调用LLM
            provider = provider or self.default_provider
            # 如果启用了token追踪，获取usage信息
            if self.token_tracker:
                response, usage = self.llm_client.call_llm(
                    prompt,
                    provider=provider,
                    return_usage=True
                )
                # 记录token使用情况
                self.token_tracker.record_llm_call("layer2_extraction", usage, provider=provider, context=fragment.get('id'))
            else:
                response = self.llm_client.call_llm(
                    prompt,
                    provider=provider
            )
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            # 添加元数据
            result['fragment_id'] = fragment.get('id', 'unknown')
            result['extraction_time'] = session_time
            
            return result
            
        except Exception as e:
            print(f"❌ Layer2提取失败: {e}")
            return {"events": [], "states": [], "contexts": [], "error": str(e)}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应（使用JSON修复工具）
        
        Args:
            response: LLM响应文本
        
        Returns:
            解析后的结果
        """
        # 使用JSON修复工具解析
        default_result = {"events": [], "states": [], "contexts": []}
        data = parse_llm_json(
            response,
            expected_keys=['events', 'states', 'contexts'],
            default=default_result
        )
        
        if data is None:
            return default_result
        
        return {
            "events": data.get('events', []),
            "states": data.get('states', []),
            "contexts": data.get('contexts', [])
        }

