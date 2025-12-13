"""
Layer1冲突解决模块

批量判断实体和关系冲突并生成决策
"""

import json
from typing import Dict, List, Any
from core.infrastructure import LLMClient, parse_llm_json


class Layer1ConflictResolver:
    """Layer1冲突解决器"""
    
    def __init__(self, llm_client: LLMClient, token_tracker=None, default_provider: str = "deepseek"):
        self.llm_client = llm_client
        self.token_tracker = token_tracker  # Token统计收集器（可选）
        self.default_provider = default_provider  # 默认LLM提供商
    
    def resolve_conflicts_batch(
        self,
        fragment: Dict[str, Any],
        new_entities: List[Dict[str, str]],
        new_relationships: List[Dict[str, str]],
        entity_candidates: Dict[str, List[Dict[str, Any]]],
        relation_candidates: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量解决冲突
        
        Args:
            fragment: 原始fragment
            new_entities: 新提取的实体
            new_relationships: 新提取的关系
            entity_candidates: 实体候选集
            relation_candidates: 关系候选集
        
        Returns:
            {
                "entity_decisions": [...],
                "relation_decisions": [...]
            }
        """
        # 分离有候选和无候选的实体/关系
        entities_with_candidates = []
        entities_without_candidates = []
        
        for entity in new_entities:
            entity_name = entity.get('name', '')
            if entity_name in entity_candidates and entity_candidates[entity_name]:
                entities_with_candidates.append(entity)
            else:
                entities_without_candidates.append(entity)
        
        relations_with_candidates = []
        relations_without_candidates = []
        
        for relation in new_relationships:
            if not relation.get('source') or not relation.get('target'):
                continue  # 跳过无效关系
            relation_key = f"{relation['source']}_{relation['target']}"
            if relation_key in relation_candidates and relation_candidates[relation_key]:
                relations_with_candidates.append(relation)
            else:
                relations_without_candidates.append(relation)
        
        print(f"  📊 实体: {len(entities_with_candidates)}个有候选, {len(entities_without_candidates)}个无候选")
        print(f"  📊 关系: {len(relations_with_candidates)}个有候选, {len(relations_without_candidates)}个无候选")
        
        # 无候选的直接创建
        decisions_no_conflict = self._create_all_new_decisions(
            entities_without_candidates, 
            relations_without_candidates
        )
        
        # 如果没有需要LLM判断的，直接返回
        if not entities_with_candidates and not relations_with_candidates:
            print("ℹ️  所有实体/关系无候选，全部创建新节点")
            return decisions_no_conflict
        
        # 有候选的需要LLM判断
        print(f"🤖 调用LLM判断 {len(entities_with_candidates)}个实体 + {len(relations_with_candidates)}个关系...")
        
        # 只传入有候选的实体/关系
        from .prompt import build_conflict_resolution_prompt
        prompt = build_conflict_resolution_prompt(
            fragment_text=fragment.get('content', ''),
            new_entities=entities_with_candidates,
            new_relationships=relations_with_candidates,
            entity_candidates=entity_candidates,
            relation_candidates=relation_candidates
        )
        
        # 调用LLM
        # 如果启用了token追踪，获取usage信息
        if self.token_tracker:
            response, usage = self.llm_client.call_llm(prompt, provider=self.default_provider, return_usage=True)
            # 记录token使用情况
            self.token_tracker.record_llm_call("layer1_conflict", usage, provider=self.default_provider, context=fragment.get('id'))
        else:
            response = self.llm_client.call_llm(prompt, provider=self.default_provider)
        
        # 解析决策
        decisions_with_conflict = self._parse_decisions(response)
        
        # 补充完整的entity_data和relation_data
        decisions_with_conflict = self._enrich_decisions(
            decisions_with_conflict, 
            entities_with_candidates, 
            relations_with_candidates
        )
        
        # 合并两部分决策
        merged_decisions = {
            'entity_decisions': (
                decisions_no_conflict.get('entity_decisions', []) +
                decisions_with_conflict.get('entity_decisions', [])
            ),
            'relation_decisions': (
                decisions_no_conflict.get('relation_decisions', []) +
                decisions_with_conflict.get('relation_decisions', [])
            )
        }
        
        return merged_decisions
    
    def _create_all_new_decisions(
        self,
        new_entities: List[Dict[str, str]],
        new_relationships: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        创建全部新建的决策
        
        Args:
            new_entities: 新实体列表
            new_relationships: 新关系列表
        
        Returns:
            决策结果
        """
        entity_decisions = []
        for entity in new_entities:
            # 验证实体必须有name
            if not entity.get('name'):
                print(f"  ⚠️  跳过空name的实体: {entity}")
                continue
            
            entity_decisions.append({
                'new_entity_name': entity['name'],
                'action': 'create_new',
                'entity_data': entity,
                'reason': '无历史候选，创建新实体'
            })
        
        relation_decisions = []
        for relation in new_relationships:
            # 验证关系必须有source和target
            if 'source' not in relation or 'target' not in relation:
                print(f"  ⚠️  跳过无效关系（缺少source/target）: {relation}")
                continue
            if not relation['source'] or not relation['target']:
                print(f"  ⚠️  跳过空source/target的关系: {relation}")
                continue
            
            relation_decisions.append({
                'new_relation': f"{relation['source']}_{relation['target']}",
                'action': 'create_new',
                'relation_data': relation,
                'reason': '无历史候选，创建新关系'
            })
        
        return {
            'entity_decisions': entity_decisions,
            'relation_decisions': relation_decisions
        }
    
    def _parse_decisions(self, response: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        解析LLM的决策
        
        Args:
            response: LLM响应
        
        Returns:
            决策结果
        """
        # 使用JSON修复工具解析
        default_result = {'entity_decisions': [], 'relation_decisions': []}
        decisions = parse_llm_json(
            response,
            expected_keys=['entity_decisions', 'relation_decisions'],
            default=default_result
        )
        
        if decisions is None:
            return default_result
        
        return decisions
    
    def _enrich_decisions(
        self,
        decisions: Dict[str, List[Dict[str, Any]]],
        new_entities: List[Dict[str, str]],
        new_relationships: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        补充决策中缺失的entity_data和relation_data
        
        LLM返回的决策只包含new_entity_name/new_relation，
        需要根据名字从原始列表中找到完整数据
        
        Args:
            decisions: LLM返回的决策
            new_entities: 原始实体列表
            new_relationships: 原始关系列表
        
        Returns:
            补充完整数据的决策
        """
        # 构建查找字典
        entity_map = {entity['name']: entity for entity in new_entities}
        relation_map = {
            f"{rel['source']}_{rel['target']}": rel 
            for rel in new_relationships
            if rel.get('source') and rel.get('target')
        }
        
        # 补充entity_data
        valid_entity_decisions = []
        for decision in decisions.get('entity_decisions', []):
            if decision.get('action') == 'create_new':
                entity_name = decision.get('new_entity_name', '')
                if entity_name in entity_map:
                    decision['entity_data'] = entity_map[entity_name]
                    valid_entity_decisions.append(decision)
                else:
                    print(f"  ⚠️  LLM幻觉：返回了不存在的实体 '{entity_name}'，已跳过")
            elif decision.get('action') == 'update_existing':
                # update操作不需要entity_data
                valid_entity_decisions.append(decision)
        
        decisions['entity_decisions'] = valid_entity_decisions
        
        # 补充relation_data
        valid_relation_decisions = []
        for decision in decisions.get('relation_decisions', []):
            if decision.get('action') == 'create_new':
                relation_key = decision.get('new_relation', '')
                if relation_key in relation_map:
                    decision['relation_data'] = relation_map[relation_key]
                    valid_relation_decisions.append(decision)
                else:
                    print(f"  ⚠️  LLM幻觉：返回了不存在的关系 '{relation_key}'，已跳过")
            elif decision.get('action') == 'update_existing':
                # update操作不需要relation_data
                valid_relation_decisions.append(decision)
        
        decisions['relation_decisions'] = valid_relation_decisions
        
        return decisions

