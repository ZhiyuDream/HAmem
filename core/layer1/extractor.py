"""
Layer1实体提取器

从fragment中提取实体和关系
"""

import json
from typing import Dict, List, Any, Optional
from core.infrastructure import LLMClient, parse_llm_json


class Layer1Extractor:
    """Layer1实体提取器"""
    
    def __init__(self, llm_client: LLMClient, default_provider: str = "deepseek", token_tracker=None):
        self.llm_client = llm_client
        self.default_provider = default_provider
        self.token_tracker = token_tracker  # Token统计收集器（可选）
    
    def extract_from_fragment(
        self, 
        fragment: Dict[str, Any], 
        existing_entities: List[Dict[str, Any]] = None,
        provider: str = None
    ) -> Dict[str, Any]:
        """
        从fragment中提取实体和关系（支持已有实体召回和关联）
        
        Args:
            fragment: fragment数据 {"id": "...", "content": "...", ...}
            existing_entities: 召回的已有实体列表（用于关联和补充）
            provider: LLM提供商
        
        Returns:
            提取结果 {"entities": [...], "relationships": [...]}
        """
        try:
            fragment_text = fragment.get('content', '')
            if not fragment_text:
                return {"entities": [], "relationships": []}
            
            # 构建prompt（包含已有实体信息）
            from .prompt import build_layer1_extraction_prompt
            prompt = build_layer1_extraction_prompt(fragment_text, existing_entities=existing_entities)
            
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
                self.token_tracker.record_llm_call("layer1_extraction", usage, provider=provider, context=fragment.get('id'))
            else:
                response = self.llm_client.call_llm(prompt, provider=provider)
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            # 添加元数据
            result['fragment_id'] = fragment.get('id', 'unknown')
            result['extraction_time'] = fragment.get('time', 'unknown')
            
            return result
            
        except Exception as e:
            print(f"❌ Layer1提取失败: {e}")
            return {"entities": [], "relationships": [], "error": str(e)}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应（使用JSON修复工具）
        
        Args:
            response: LLM响应文本
        
        Returns:
            解析后的结果
        """
        # 使用JSON修复工具解析
        default_result = {"entities": [], "relationships": []}
        data = parse_llm_json(
            response, 
            expected_keys=['entities', 'relationships'],
            default=default_result
        )
        
        if data is None:
            return default_result
        
        # 过滤无效的实体和关系
        entities = data.get('entities', [])
        relationships = data.get('relationships', [])
        
        # 验证实体：必须有name
        valid_entities = []
        for entity in entities:
            if entity.get('name'):
                valid_entities.append(entity)
            else:
                print(f"  ⚠️  LLM提取了无效实体（无name）: {entity}")
        
        # 验证关系：必须有source和target
        valid_relationships = []
        for relation in relationships:
            if relation.get('source') and relation.get('target'):
                valid_relationships.append(relation)
            else:
                print(f"  ⚠️  LLM提取了无效关系（缺少source/target）: {relation}")
        
        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }
    
    def batch_extract(self, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量提取多个fragment
        
        Args:
            fragments: fragment列表
        
        Returns:
            提取结果列表
        """
        results = []
        for i, fragment in enumerate(fragments, 1):
            print(f"📝 提取Fragment {i}/{len(fragments)}: {fragment.get('id')}")
            result = self.extract_from_fragment(fragment)
            results.append(result)
        
        return results
