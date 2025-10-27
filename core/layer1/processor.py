"""
Layer1处理器

协调提取、召回、冲突解决和存储的完整流程
"""

import os
import json
from typing import List, Dict, Any
from core.infrastructure import LLMClient, UnifiedCache
from .extractor import Layer1Extractor
from .recall import Layer1Recall
from .conflict_resolver import Layer1ConflictResolver
from .storage import Layer1Storage


class Layer1Processor:
    """Layer1处理器"""
    
    def __init__(
        self, 
        llm_client: LLMClient,
        cache: UnifiedCache,
        storage_dir: str = "storage"
    ):
        self.extractor = Layer1Extractor(llm_client)
        self.recall = Layer1Recall(cache)
        self.conflict_resolver = Layer1ConflictResolver(llm_client)
        self.storage = Layer1Storage(storage_dir)
        self.cache = cache
        
        # 计数器
        self.entity_counter = 0
        self.relation_counter = 0
    
    def process_fragment(
        self,
        fragment: Dict[str, Any],
        namespace: str
    ) -> Dict[str, Any]:
        """
        处理单个fragment
        
        Args:
            fragment: fragment数据
            namespace: 命名空间
        
        Returns:
            处理统计信息
        """
        print(f"\n{'='*60}")
        print(f"📝 处理Fragment: {fragment.get('id')}")
        print(f"{'='*60}")
        
        # Phase 1: 提取实体和关系 (LLM #1)
        print("\n🤖 Phase 1: 提取实体和关系...")
        extraction_result = self.extractor.extract_from_fragment(fragment)
        
        entities = extraction_result.get('entities', [])
        relationships = extraction_result.get('relationships', [])
        
        print(f"✅ 提取了 {len(entities)} 个实体, {len(relationships)} 个关系")
        
        if not entities and not relationships:
            print("ℹ️  无提取结果，跳过处理")
            return {'entities_created': 0, 'relations_created': 0}
        
        # Phase 2: 智能召回候选
        print("\n🔍 Phase 2: 智能召回候选...")
        entity_candidates = self.recall.batch_recall_entities(entities)
        relation_candidates = self.recall.batch_recall_relations(relationships)
        
        # Phase 3: 批量冲突解决 (可能LLM #2)
        print("\n🤖 Phase 3: 批量冲突解决...")
        decisions = self.conflict_resolver.resolve_conflicts_batch(
            fragment=fragment,
            new_entities=entities,
            new_relationships=relationships,
            entity_candidates=entity_candidates,
            relation_candidates=relation_candidates
        )
        
        # Phase 4: 执行决策
        print("\n⚡ Phase 4: 执行决策...")
        stats = self._execute_decisions(decisions, namespace)
        
        print(f"\n✅ Fragment处理完成:")
        print(f"  - 创建实体: {stats['entities_created']}")
        print(f"  - 更新实体: {stats['entities_updated']}")
        print(f"  - 创建关系: {stats['relations_created']}")
        print(f"  - 更新关系: {stats['relations_updated']}")
        
        return stats
    
    def _execute_decisions(
        self,
        decisions: Dict[str, List[Dict[str, Any]]],
        namespace: str,
        fragment_id: str = None
    ) -> Dict[str, int]:
        """
        执行决策（批量优化版本）
        
        Args:
            decisions: 决策结果
            namespace: 命名空间
            fragment_id: 源fragment的ID
        
        Returns:
            统计信息
        """
        stats = {
            'entities_created': 0,
            'entities_updated': 0,
            'relations_created': 0,
            'relations_updated': 0
        }
        
        # 收集需要批量创建的实体和关系
        nodes_to_create = []
        edges_to_create = []
        entity_ids_to_create = []  # 用于创建连接边
        
        # 执行实体决策
        for decision in decisions.get('entity_decisions', []):
            action = decision.get('action')
            
            if action == 'create_new':
                # 收集待创建的实体
                entity_data = decision.get('entity_data', {})
                entity_name = entity_data.get('name', '').strip()
                entity_content = entity_data.get('content', '').strip()
                
                # 验证：name不能为空
                if not entity_name:
                    print(f"  ⚠️  跳过空name的实体: {entity_data}")
                    continue
                
                self.entity_counter += 1
                entity_id = f"entity_{self.entity_counter}"
                
                node = {
                    "id": entity_id,
                    "type": "entity",
                    "name": entity_name,
                    "content": entity_content,
                    "layer": 1,
                    "active": True
                }
                nodes_to_create.append(node)
                entity_ids_to_create.append(entity_id)
                stats['entities_created'] += 1
                
            elif action == 'update_existing':
                # 更新实体并创建fragment → entity的连接边
                target_id = decision.get('target_entity_id')
                updated_content = decision.get('updated_content', '')
                
                if target_id and updated_content:
                    self.storage.update_node(target_id, content=updated_content, namespace=namespace, active=True)
                    self.cache.update_node(target_id, new_content=updated_content)
                    stats['entities_updated'] += 1
                    
                    if fragment_id:
                        self._create_fragment_entity_edge(fragment_id, target_id, namespace)
        
        # 执行关系决策
        for decision in decisions.get('relation_decisions', []):
            action = decision.get('action')
            
            if action == 'create_new':
                # 收集待创建的关系
                relation_data = decision.get('relation_data', {})
                source = relation_data.get('source', '').strip()
                target = relation_data.get('target', '').strip()
                content = relation_data.get('content', '').strip()
                
                # 验证：source和target不能为空
                if not source or not target:
                    print(f"  ⚠️  跳过空source/target的关系: {relation_data}")
                    continue
                
                self.relation_counter += 1
                edge_id = f"edge_{self.relation_counter}"
                
                edge = {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "content": content,
                    "layer": 1,
                    "active": True
                }
                edges_to_create.append(edge)
                stats['relations_created'] += 1
                
            elif action == 'update_existing':
                # 更新关系
                target_id = decision.get('target_relation_id')
                updated_content = decision.get('updated_content', '')
                
                if target_id and updated_content:
                    self.storage.update_edge(target_id, content=updated_content, namespace=namespace)
                    
                    # 更新cache中的edge
                    if target_id in self.cache.cache['edges']:
                        edge = self.cache.cache['edges'][target_id]
                        new_embedding, new_idx, _ = self.cache.get_or_generate_embedding(updated_content)
                        edge['content'] = updated_content
                        edge['content_hash'] = self.cache._compute_content_hash(updated_content)
                        edge['embedding_idx'] = new_idx
                    
                    stats['relations_updated'] += 1
        
        # 批量添加到cache（优化：一次性生成所有embedding）
        if nodes_to_create:
            print(f"\n📦 批量创建 {len(nodes_to_create)} 个实体...")
            self.cache.batch_add_nodes(nodes_to_create)
            
            # 批量保存到storage
            for node in nodes_to_create:
                self.storage.save_entity(node, namespace)
                print(f"  ✅ 创建实体: {node['name']}")
            
            # 创建fragment → entity的连接边
            if fragment_id:
                for entity_id in entity_ids_to_create:
                    self._create_fragment_entity_edge(fragment_id, entity_id, namespace)
        
        if edges_to_create:
            print(f"\n📦 批量创建 {len(edges_to_create)} 个关系...")
            self.cache.batch_add_edges(edges_to_create)
            
            # 批量保存到storage
            for edge in edges_to_create:
                self.storage.save_relationship(edge, namespace)
                print(f"  ✅ 创建关系: {edge['source']} -> {edge['target']}")
        
        # 返回统计信息和创建的实体列表（供Layer2使用）
        stats['created_entities'] = nodes_to_create
        return stats
    
    
    def _create_fragment_entity_edge(
        self,
        fragment_id: str,
        entity_id: str,
        namespace: str
    ):
        """创建fragment → entity的连接边（追踪实体在不同fragment中的提及）"""
        edge = {
            "id": f"edge_{fragment_id}_{entity_id}",
            "source": fragment_id,
            "target": entity_id,
            "type": "mentions",  # fragment提及entity
            "active": True
        }
        
        # 保存到storage（不需要content和layer字段）
        storage_path = self.storage.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')
        
        # 不添加到cache，因为这是结构性边，不需要embedding
    
    def process_fragments_batch(
        self,
        fragments: List[Dict[str, Any]],
        namespace: str
    ) -> Dict[str, Any]:
        """
        批量处理fragments
        
        Args:
            fragments: fragment列表
            namespace: 命名空间
        
        Returns:
            总体统计信息
        """
        total_stats = {
            'entities_created': 0,
            'entities_updated': 0,
            'relations_created': 0,
            'relations_updated': 0
        }
        
        for i, fragment in enumerate(fragments, 1):
            print(f"\n{'#'*60}")
            print(f"处理进度: {i}/{len(fragments)}")
            print(f"{'#'*60}")
            
            stats = self.process_fragment(fragment, namespace)
            
            # 累加统计
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
        
        # 保存cache
        print(f"\n💾 保存cache...")
        self.cache.save()
        
        print(f"\n{'='*60}")
        print(f"🎉 全部处理完成！")
        print(f"{'='*60}")
        print(f"总计:")
        print(f"  - 创建实体: {total_stats['entities_created']}")
        print(f"  - 更新实体: {total_stats['entities_updated']}")
        print(f"  - 创建关系: {total_stats['relations_created']}")
        print(f"  - 更新关系: {total_stats['relations_updated']}")
        
        return total_stats
