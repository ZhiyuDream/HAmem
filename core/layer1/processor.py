"""
Layer1处理器

协调提取、召回、冲突解决和存储的完整流程
"""

import os
import json
from typing import List, Dict, Any, Optional
from core.infrastructure import LLMClient, UnifiedCache, EmbeddingManager
from core.infrastructure.neo4j_vector_search import Neo4jVectorSearch
from .extractor import Layer1Extractor
from .recall import Layer1Recall
from .conflict_resolver import Layer1ConflictResolver
from .neo4j_storage import Layer1Neo4jStorage
from ..infrastructure.neo4j_client import Neo4jClient
from config import Config


class Layer1Processor:
    """Layer1处理器"""
    
    def __init__(
        self, 
        llm_client: LLMClient,
        cache: UnifiedCache,
        neo4j_client: Neo4jClient = None,
        namespace: str = "default",
        config: Optional[Config] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        token_tracker=None
    ):
        self.extractor = Layer1Extractor(llm_client, token_tracker=token_tracker)
        
        # 初始化Neo4j向量搜索（如果提供了neo4j_client）
        neo4j_vector_search = None
        if neo4j_client:
            try:
                neo4j_vector_search = Neo4jVectorSearch(
                    neo4j_client=neo4j_client,
                    namespace=namespace,
                    embedding_manager=embedding_manager,
                    config=config
                )
            except Exception as e:
                print(f"  ⚠️  初始化Neo4j向量搜索失败，将使用FAISS: {e}")
        
        # 初始化召回器（支持Neo4j向量搜索或FAISS）
        self.recall = Layer1Recall(
            cache=cache,
            neo4j_vector_search=neo4j_vector_search,
            namespace=namespace
        )
        
        self.conflict_resolver = Layer1ConflictResolver(llm_client, token_tracker=token_tracker, default_provider=config.llm_provider if hasattr(config, 'llm_provider') else "deepseek")
        # 使用Neo4jStorage替代文件存储
        if neo4j_client:
            self.storage = Layer1Neo4jStorage(neo4j_client, namespace)
        else:
            self.storage = None  # 如果没有Neo4j客户端，storage为None（数据将通过统一写入）
        self.cache = cache
        self.neo4j_client = neo4j_client
        self.namespace = namespace
        
        # 计数器
        self.entity_counter = 0
        self.relation_counter = 0
    
    def process_fragment(
        self,
        fragment: Dict[str, Any],
        namespace: str,
        existing_entities: List[Dict[str, Any]] = None
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
        
        # Phase 0: 使用已召回的已有实体（如果未提供，则自己召回）
        if existing_entities is None:
            print("\n🔍 Phase 0: 召回已有实体...")
            fragment_text = fragment.get('content', '')
            existing_entities = []
            if fragment_text:
                try:
                    # 使用fragment content生成embedding并召回已有实体
                    fragment_embedding, _, _ = self.cache.get_or_generate_embedding(fragment_text)
                    existing_candidates = self.cache.filter_and_search(
                        fragment_embedding,
                        filters={'type': 'entity', 'layer': 1},
                        top_k=10  # 召回top-10个已有实体
                    )
                    
                    # 转换为实体格式，并验证节点确实存在于cache中
                    for candidate in existing_candidates:
                        entity_node = candidate.get('node', {})
                        if entity_node:
                            entity_id = entity_node.get('id')
                            # 验证节点确实存在于cache中
                            if entity_id and entity_id in self.cache.cache['nodes']:
                                existing_entities.append(entity_node)
                            else:
                                print(f"  ⚠️  跳过不在cache中的节点: {entity_id}")
                    
                    print(f"✅ 召回了 {len(existing_entities)} 个已有实体")
                    if existing_entities:
                        entity_names = [e.get('name', 'unknown') for e in existing_entities[:5]]
                        print(f"   示例: {', '.join(entity_names)}...")
                except Exception as e:
                    print(f"  ⚠️  召回已有实体失败: {e}")
        else:
            print(f"\n✅ 使用已召回的 {len(existing_entities)} 个已有实体")
        
        # Phase 1: 提取实体和关系（包含关联和补充判断，一次LLM调用）
        print("\n🤖 Phase 1: 提取实体和关系（包含关联和补充判断）...")
        extraction_result = self.extractor.extract_from_fragment(
            fragment, 
            existing_entities=existing_entities
        )
        
        entities = extraction_result.get('entities', [])
        relationships = extraction_result.get('relationships', [])
        
        print(f"✅ 提取了 {len(entities)} 个实体, {len(relationships)} 个关系")
        
        if not entities and not relationships:
            print("ℹ️  无提取结果，跳过处理")
            return {'entities_created': 0, 'relations_created': 0}
        
        # Phase 2: 执行决策（处理create_new/update_existing/link_to_existing）
        print("\n⚡ Phase 2: 执行决策...")
        fragment_id = fragment.get('id')  # 获取fragment ID用于创建连接边
        stats = self._execute_decisions_from_extraction(
            entities, 
            relationships, 
            namespace, 
            fragment_id=fragment_id,
            existing_entities=existing_entities
        )
        
        print(f"\n✅ Fragment处理完成:")
        print(f"  - 创建实体: {stats['entities_created']}")
        print(f"  - 更新实体: {stats['entities_updated']}")
        print(f"  - 创建关系: {stats['relations_created']}")
        print(f"  - 更新关系: {stats['relations_updated']}")
        
        return stats
    
    def _execute_decisions_from_extraction(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        namespace: str,
        fragment_id: str = None,
        existing_entities: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从提取结果执行决策（处理create_new/update_existing/link_to_existing）
        
        Args:
            entities: 提取的实体列表（包含action字段）
            relationships: 提取的关系列表（包含action字段）
            namespace: 命名空间
            fragment_id: 源fragment的ID
            existing_entities: 已有实体列表（用于查找existing_entity_id）
        
        Returns:
            统计信息和创建的节点/边
        """
        stats = {
            'entities_created': 0,
            'entities_updated': 0,
            'relations_created': 0,
            'relations_updated': 0
        }
        
        # 创建已有实体的ID映射（用于查找）
        existing_entity_map = {}
        if existing_entities:
            for entity in existing_entities:
                entity_id = entity.get('id')
                entity_name = entity.get('name', '').lower().strip()
                if entity_id and entity_name:
                    existing_entity_map[entity_name] = entity_id
        
        # 收集需要批量创建的实体和关系
        nodes_to_create = []
        edges_to_create = []
        entity_ids_to_create = []  # 用于创建连接边
        fragment_entity_edges = []  # Fragment到Entity的连接边
        
        # 处理实体
        for entity in entities:
            action = entity.get('action', 'create_new')  # 默认为create_new
            
            if action == 'create_new':
                # 创建新实体
                entity_name = entity.get('name', '').strip()
                entity_content = entity.get('content', '').strip()
                
                if not entity_name:
                    print(f"  ⚠️  跳过空name的实体: {entity}")
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
                
                # 处理link_to_existing（新实体与已有实体建立关系）
                link_to_existing = entity.get('link_to_existing', [])
                if link_to_existing:
                    for link_info in link_to_existing:
                        if isinstance(link_info, dict):
                            existing_entity_id = link_info.get('existing_entity_id')
                            relation_type = link_info.get('relation_type', 'RELATED_TO')
                            relation_content = link_info.get('relation_content', '')
                        else:
                            # 兼容旧格式（只有entity_id字符串）
                            existing_entity_id = link_info
                            relation_type = 'RELATED_TO'
                            relation_content = ''
                        
                        if existing_entity_id:
                            # 创建新实体到已有实体的关系
                            self.relation_counter += 1
                            edge_id = f"edge_{self.relation_counter}"
                            edge = {
                                "id": edge_id,
                                "source": entity_id,
                                "target": existing_entity_id,
                                "content": relation_content,
                                "type": relation_type,
                                "layer": 1,
                                "active": True
                            }
                            edges_to_create.append(edge)
                            stats['relations_created'] += 1
                            print(f"  🔗 新实体 {entity_name} 关联到已有实体 {existing_entity_id}")
                
                # 创建fragment → 新实体的连接边
                if fragment_id:
                    fragment_entity_edges.append((fragment_id, entity_id))
            
            elif action == 'update_existing':
                # 更新已有实体
                existing_entity_id = entity.get('existing_entity_id')
                updated_content = entity.get('content', '')
                
                if existing_entity_id and updated_content:
                    # 验证节点是否存在于cache中
                    if existing_entity_id not in self.cache.cache['nodes']:
                        print(f"  ⚠️  节点 {existing_entity_id} 不在cache中，可能来自Neo4j但尚未写入cache")
                        print(f"  ⚠️  跳过更新，改为创建新实体")
                        # 如果节点不在cache中，改为创建新实体
                        self.entity_counter += 1
                        entity_id = f"entity_{self.entity_counter}"
                        node = {
                            "id": entity_id,
                            "type": "entity",
                            "name": entity.get('name', '').strip(),
                            "content": updated_content,
                            "layer": 1,
                            "active": True
                        }
                        nodes_to_create.append(node)
                        entity_ids_to_create.append(entity_id)
                        stats['entities_created'] += 1
                        if fragment_id:
                            fragment_entity_edges.append((fragment_id, entity_id))
                        print(f"  ✅ 改为创建新实体 {entity_id}")
                    else:
                        # 节点存在于cache中，正常更新
                        if self.storage:
                            self.storage.update_node(existing_entity_id, content=updated_content, namespace=namespace, active=True)
                        self.cache.update_node(existing_entity_id, new_content=updated_content)
                        stats['entities_updated'] += 1
                        
                        # 创建fragment → 已有实体的连接边
                        if fragment_id:
                            fragment_entity_edges.append((fragment_id, existing_entity_id))
                        print(f"  ✏️  更新已有实体 {existing_entity_id}")
                else:
                    print(f"  ⚠️  跳过无效的update_existing实体: {entity}")
        
        # 处理关系
        for relation in relationships:
            action = relation.get('action', 'create_new')  # 默认为create_new
            
            if action == 'create_new':
                # 创建新关系
                source = relation.get('source', '').strip()
                target = relation.get('target', '').strip()
                content = relation.get('content', '').strip()
                
                if not source or not target:
                    print(f"  ⚠️  跳过空source/target的关系: {relation}")
                    continue
                
                # 查找source和target的entity_id（可能是name，需要转换为id）
                source_id = self._find_entity_id_by_name(source, entity_ids_to_create, existing_entity_map)
                target_id = self._find_entity_id_by_name(target, entity_ids_to_create, existing_entity_map)
                
                if not source_id or not target_id:
                    print(f"  ⚠️  跳过无法找到source/target的关系: {source} -> {target}")
                    continue
                
                self.relation_counter += 1
                edge_id = f"edge_{self.relation_counter}"
                
                edge = {
                    "id": edge_id,
                    "source": source_id,
                    "target": target_id,
                    "content": content,
                    "layer": 1,
                    "active": True
                }
                edges_to_create.append(edge)
                stats['relations_created'] += 1
            
            elif action == 'update_existing':
                # 更新已有关系
                existing_relation_id = relation.get('existing_relation_id')
                updated_content = relation.get('content', '')
                
                if existing_relation_id and updated_content:
                    if self.storage:
                        self.storage.update_edge(existing_relation_id, content=updated_content, namespace=namespace)
                    
                    # 更新cache中的edge
                    if existing_relation_id in self.cache.cache['edges']:
                        edge = self.cache.cache['edges'][existing_relation_id]
                        new_embedding, new_idx, _ = self.cache.get_or_generate_embedding(updated_content)
                        edge['content'] = updated_content
                        edge['content_hash'] = self.cache._compute_content_hash(updated_content)
                        edge['embedding_idx'] = new_idx
                    
                    stats['relations_updated'] += 1
                    print(f"  ✏️  更新已有关系 {existing_relation_id}")
                else:
                    print(f"  ⚠️  跳过无效的update_existing关系: {relation}")
        
        # 返回统计信息和创建的节点/边列表
        stats['created_entities'] = nodes_to_create
        stats['created_nodes'] = nodes_to_create
        stats['created_edges'] = edges_to_create
        stats['fragment_entity_edges'] = fragment_entity_edges
        
        return stats
    
    def _find_entity_id_by_name(
        self, 
        entity_name: str, 
        new_entity_ids: List[str],
        existing_entity_map: Dict[str, str]
    ) -> str:
        """
        根据实体名称查找entity_id
        
        Args:
            entity_name: 实体名称
            new_entity_ids: 新创建的实体ID列表（需要从cache中查找）
            existing_entity_map: 已有实体的名称到ID映射
        
        Returns:
            entity_id或None
        """
        entity_name_lower = entity_name.lower().strip()
        
        # 先检查已有实体
        if entity_name_lower in existing_entity_map:
            return existing_entity_map[entity_name_lower]
        
        # 再检查新创建的实体（从cache中查找）
        for entity_id in new_entity_ids:
            entity_node = self.cache.cache['nodes'].get(entity_id)
            if entity_node:
                node_name = entity_node.get('name', '').lower().strip()
                if node_name == entity_name_lower:
                    return entity_id
        
        # 最后检查cache中的所有实体
        for node_id, node in self.cache.cache['nodes'].items():
            if node.get('layer') == 1 and node.get('type') == 'entity':
                node_name = node.get('name', '').lower().strip()
                if node_name == entity_name_lower:
                    return node_id
        
        return None
    
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
        
        # 不立即添加到cache，而是返回节点和边（供后续统一批量生成embedding和写入Neo4j）
        if nodes_to_create:
            print(f"\n📦 准备创建 {len(nodes_to_create)} 个实体...")
            for node in nodes_to_create:
                print(f"  ✅ 实体: {node['name']}")
        
        if edges_to_create:
            print(f"\n📦 准备创建 {len(edges_to_create)} 个关系...")
            for edge in edges_to_create:
                print(f"  ✅ 关系: {edge['source']} -> {edge['target']}")
        
        # 返回统计信息和创建的节点/边列表（供后续统一处理）
        stats['created_entities'] = nodes_to_create
        stats['created_nodes'] = nodes_to_create  # 新增：返回节点列表
        stats['created_edges'] = edges_to_create  # 新增：返回边列表
        stats['fragment_entity_edges'] = [(fragment_id, entity_id) for entity_id in entity_ids_to_create] if fragment_id else []  # Fragment到Entity的连接边
        
        return stats
    
    
    def _create_fragment_entity_edge(
        self,
        fragment_id: str,
        entity_id: str,
        namespace: str
    ):
        """
        创建fragment → entity的连接边（纯Neo4j架构，不再写入文件）
        
        注意：纯Neo4j架构下，这个方法不再执行实际操作
        连接边会在memory.py中统一批量写入Neo4j
        """
        # 纯Neo4j架构：不再写入文件，连接边会在memory.py中统一处理
        # 这个方法保留是为了兼容性，实际不执行操作
        pass
    
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
