"""
Memory Builder and Storage Manager

Provides high-level interfaces for building and managing hierarchical memory
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config import Config
from core.infrastructure import LLMClient, EmbeddingManager, UnifiedCache
from core.fragment.buffer_manager import BufferManager
from core.fragment.fragment_processor import FragmentProcessor
# 纯Neo4j架构，不再使用FragmentStorage
from core.layer1.processor import Layer1Processor
from core.layer2.processor import Layer2Processor
from core.layer3.processor import Layer3Processor


@dataclass
class ConversationData:
    """Conversation data structure"""
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationData':
        """Create from dictionary"""
        return cls(
            messages=data.get('messages', []),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'messages': self.messages,
            'metadata': self.metadata or {}
        }


@dataclass
class MemoryBuildResult:
    """Memory build result"""
    total_fragments: int
    total_entities: int
    total_events: int
    total_clusters: int
    namespace: str
    time_stats: Dict[str, Any] = None  # 时间统计
    token_stats: Dict[str, Any] = None  # Token统计
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'total_fragments': self.total_fragments,
            'total_entities': self.total_entities,
            'total_events': self.total_events,
            'total_clusters': self.total_clusters,
            'namespace': self.namespace
        }
        if self.time_stats:
            result['time_stats'] = self.time_stats
        if self.token_stats:
            result['token_stats'] = self.token_stats
        return result


class MemoryBuilder:
    """Build hierarchical memory from conversation data"""
    
    def __init__(self, config: Config):
        """Initialize memory builder"""
        self.config = config
        self.llm_client = LLMClient(config)
        self.embedding_manager = EmbeddingManager(config)
        
        # Initialize processors（纯Neo4j架构，不再使用文件存储）
        self.buffer_manager = BufferManager(max_length=config.fragment_max_length)
        self.fragment_processor = FragmentProcessor(self.llm_client)
        # 不再使用fragment_storage，数据直接写入Neo4j
        
        # Initialize layer processors（纯Neo4j架构）
        # 注意：neo4j_client和namespace将在build_memory时设置
        self.layer1_processor = Layer1Processor(
            llm_client=self.llm_client,
            cache=None,  # 将在build_memory中设置
            neo4j_client=None,  # 将在build_memory中设置
            namespace="default",  # 将在build_memory中设置
            config=self.config,
            embedding_manager=self.embedding_manager
        )
        # Layer2和Layer3暂时保持原样，后续也需要改为Neo4j
        self.layer2_processor = Layer2Processor(
            self.llm_client,
            None,  # cache will be set per namespace
            storage_dir=config.storage_dir
        )
        self.layer3_processor = Layer3Processor(
            llm_client=self.llm_client,
            cache=None,  # 将在build_memory中设置
            neo4j_client=None,  # 将在build_memory中设置
            namespace="default",  # 将在build_memory中设置
            layer2_threshold=60
        )
    
    def _count_nodes_by_type(self, namespace: str, node_type: str, neo4j_client=None) -> int:
        """Count nodes by type from Neo4j"""
        if neo4j_client and neo4j_client.driver:
            try:
                # 根据node_type构建查询
                if node_type == 'event':
                    query = """
                    MATCH (n:Event)
                    WHERE n.namespace = $namespace
                    RETURN count(n) as count
                    """
                elif node_type == 'event_cluster':
                    query = """
                    MATCH (n:EventCluster)
                    WHERE n.namespace = $namespace
                    RETURN count(n) as count
                    """
                else:
                    query = """
                    MATCH (n)
                    WHERE n.namespace = $namespace AND n.type = $node_type
                    RETURN count(n) as count
                    """
                result = neo4j_client.execute_read(query, {'namespace': namespace, 'node_type': node_type})
                if result:
                    return result[0].get('count', 0)
            except Exception as e:
                print(f"⚠️  从Neo4j统计节点失败: {e}")
        return 0
    
    def build_memory(self, conversation: ConversationData, namespace: str = "default", token_tracker=None, llm_provider: str = "deepseek") -> MemoryBuildResult:
        """
        Build memory from conversation
        
        Args:
            conversation: Conversation data
            namespace: Namespace for storage
            token_tracker: Token统计收集器（可选）
            
        Returns:
            MemoryBuildResult
        """
        # Initialize cache for this namespace
        cache = UnifiedCache(
            cache_dir=self.config.cache_dir,
            namespace=namespace,
            embedding_manager=self.embedding_manager
        )
        
        # 保存cache引用，供后续使用
        self.cache = cache
        
        # 初始化Neo4j客户端（纯Neo4j架构，不再使用文件存储）
        from core.infrastructure.neo4j_client import Neo4jClient
        neo4j_client = Neo4jClient(
            uri=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database
        )
        
        # 纯Neo4j架构：必须成功连接Neo4j
        if not self.config.use_neo4j:
            raise RuntimeError("❌ 纯Neo4j架构要求 use_neo4j=True，请在配置中启用")
        
        if not neo4j_client.connect():
            raise RuntimeError(f"❌ Neo4j连接失败！请检查Neo4j服务是否启动，以及连接配置是否正确。\n"
                             f"   URI: {self.config.neo4j_uri}\n"
                             f"   Username: {self.config.neo4j_username}\n"
                             f"   Database: {self.config.neo4j_database}")
        
        # 更新Processor的storage为Neo4jStorage
        from core.layer1.neo4j_storage import Layer1Neo4jStorage
        # 重新初始化Layer1Processor以使用Neo4j向量搜索
        self.layer1_processor = Layer1Processor(
            llm_client=self.llm_client,
            cache=cache,
            neo4j_client=neo4j_client,
            namespace=namespace,
            config=self.config,
            embedding_manager=self.embedding_manager,
            token_tracker=token_tracker
        )
        
        # Set cache for processors
        self.layer2_processor.cache = cache
        
        # 重新初始化Layer3Processor以使用Neo4j（必须在neo4j_client创建之后）
        self.layer3_processor = Layer3Processor(
            llm_client=self.llm_client,
            cache=cache,
            neo4j_client=neo4j_client,
            namespace=namespace,
            layer2_threshold=60,
            token_tracker=token_tracker
        )
        
        # 更新FragmentProcessor以支持token追踪和provider
        from core.fragment.fragment_processor import FragmentProcessor
        self.fragment_processor = FragmentProcessor(
            self.llm_client,
            default_provider=llm_provider,
            token_tracker=token_tracker
        )
        
        # 更新Layer1Processor的extractor和conflict_resolver以使用正确的provider
        from core.layer1.extractor import Layer1Extractor
        from core.layer1.conflict_resolver import Layer1ConflictResolver
        self.layer1_processor.extractor = Layer1Extractor(
            self.llm_client,
            default_provider=llm_provider,
            token_tracker=token_tracker
        )
        self.layer1_processor.conflict_resolver = Layer1ConflictResolver(
            self.llm_client,
            token_tracker=token_tracker,
            default_provider=llm_provider
        )
        
        # 更新Layer2Processor的extractor以使用正确的provider
        if token_tracker:
            from core.layer2.extractor import Layer2Extractor
            self.layer2_processor.extractor = Layer2Extractor(
                self.llm_client,
                default_provider=llm_provider,
                token_tracker=token_tracker
            )
        
        # 更新Layer3Processor的extractor以使用正确的provider
        from core.layer3.extractor import Layer3Extractor
        self.layer3_processor.extractor = Layer3Extractor(
            self.llm_client,
            token_tracker=token_tracker,
            default_provider=llm_provider
        )
        
        print(f"✅ Neo4j连接成功，使用Neo4j存储 (namespace: {namespace})")
        
        # 保存neo4j_client供后续使用
        self._neo4j_client = neo4j_client
        
        # Convert messages to turns format
        turns = []
        for msg in conversation.messages:
            turns.append({
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', time.time()),
                'metadata': msg.get('metadata', {})
            })
        
        # 时间统计
        time_stats = {
            'fragment_processing': [],  # 每个fragment的总处理时间
            'layer1_processing': [],    # Layer1处理时间
            'layer2_processing': [],    # Layer2处理时间
            'layer3_processing': []      # Layer3处理时间
        }
        
        # 流式处理：每个fragment处理完后立即批量生成embedding并写入Neo4j
        total_fragments = 0
        all_entities = []  # 累积所有实体，供Layer2使用（用于后续fragment的Layer2处理）
        
        def process_fragment_immediately(fragment):
            """立即处理fragment，处理完后立即批量生成embedding并写入Neo4j"""
            nonlocal total_fragments, all_entities
            
            total_fragments += 1
            fragment_start_time = time.time()
            
            print(f"\n{'#'*60}")
            print(f"📄 Fragment {total_fragments}: {fragment.get('id', 'unknown')}")
            print(f"{'#'*60}")
            
            # 收集当前fragment的所有节点和边（用于批量embedding和写入Neo4j）
            fragment_nodes = []
            fragment_edges = []
            
            # 收集Fragment节点
            # 确保conversation_time正确设置（优先使用conversation_time，否则使用time）
            conversation_time = fragment.get('conversation_time') or fragment.get('time', 'unknown')
            fragment_node = {
                "id": fragment.get('id'),
                "type": "fragment",
                "content": fragment.get('content', ''),
                "time": conversation_time,
                "conversation_time": conversation_time,  # 明确设置conversation_time字段
                "layer": 0,
                "active": True
            }
            fragment_nodes.append(fragment_node)
            
            # 并行召回Layer1和Layer2的已有节点（优化：不依赖其他结果，可以并行）
            fragment_text = fragment.get('content', '')
            existing_entities = []
            existing_layer2_nodes = []
            
            if fragment_text:
                import concurrent.futures
                
                def recall_layer1_entities():
                    """召回Layer1已有实体"""
                    try:
                        fragment_embedding, _, _ = self.cache.get_or_generate_embedding(fragment_text)
                        existing_candidates = self.cache.filter_and_search(
                            fragment_embedding,
                            filters={'type': 'entity', 'layer': 1},
                            top_k=10
                        )
                        entities = []
                        for candidate in existing_candidates:
                            entity_node = candidate.get('node', {})
                            if entity_node:
                                entity_id = entity_node.get('id')
                                if entity_id and entity_id in self.cache.cache['nodes']:
                                    entities.append(entity_node)
                        return entities
                    except Exception as e:
                        print(f"  ⚠️  Layer1召回失败: {e}")
                        return []
                
                def recall_layer2_nodes():
                    """召回Layer2已有节点"""
                    try:
                        fragment_embedding, _, _ = self.cache.get_or_generate_embedding(fragment_text)
                        existing_candidates = self.cache.filter_and_search(
                            fragment_embedding,
                            filters={'layer': 2},
                            top_k=10
                        )
                        nodes = []
                        for candidate in existing_candidates:
                            node = candidate.get('node', {})
                            if node:
                                nodes.append(node)
                        return nodes
                    except Exception as e:
                        print(f"  ⚠️  Layer2召回失败: {e}")
                        return []
                
                # 并行执行召回
                print("\n🔍 并行召回已有节点（Layer1和Layer2）...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_layer1 = executor.submit(recall_layer1_entities)
                    future_layer2 = executor.submit(recall_layer2_nodes)
                    
                    existing_entities = future_layer1.result()
                    existing_layer2_nodes = future_layer2.result()
                
                print(f"✅ Layer1: 召回了 {len(existing_entities)} 个已有实体")
                print(f"✅ Layer2: 召回了 {len(existing_layer2_nodes)} 个已有节点")
            
            # Layer1处理（返回节点和边，传入已召回的实体）
            t_layer1 = time.time()
            layer1_stats = self.layer1_processor.process_fragment(
                fragment, 
                namespace,
                existing_entities=existing_entities  # 传入已召回的实体
            )
            layer1_time = time.time() - t_layer1
            time_stats['layer1_processing'].append(layer1_time)
            
            # 收集Layer1节点和边
            layer1_nodes = layer1_stats.get('created_nodes', [])
            layer1_edges = layer1_stats.get('created_edges', [])
            fragment_entity_edges = layer1_stats.get('fragment_entity_edges', [])
            
            fragment_nodes.extend(layer1_nodes)
            fragment_edges.extend(layer1_edges)
            
            # 收集Fragment到Entity的连接边
            if fragment_entity_edges:
                print(f"  📌 收集到 {len(fragment_entity_edges)} 个Fragment->Entity连接边")
                for fragment_id, entity_id in fragment_entity_edges:
                    print(f"    - {fragment_id} -> {entity_id}")
                    fragment_edges.append({
                        "id": f"fragment_entity_{fragment_id}_{entity_id}",
                        "source": fragment_id,
                        "target": entity_id,
                        "type": "contains",
                        "layer": 0,
                        "active": True
                    })
            else:
                print(f"  ⚠️  未收集到Fragment->Entity连接边（fragment_entity_edges为空）")
            
            # 获取创建的实体（供Layer2使用）
            created_entities = layer1_stats.get('created_entities', [])
            all_entities.extend(created_entities)
            
            # 获取所有实体（用于Layer2）- 使用累积的实体列表（因为数据可能还没写入Neo4j）
            current_entities = all_entities
            
            # Layer2处理（返回节点和边，传入已召回的Layer2节点）
            t_layer2 = time.time()
            layer2_stats = self.layer2_processor.process_fragment(
                fragment, 
                namespace, 
                current_entities,
                existing_layer2_nodes=existing_layer2_nodes  # 传入已召回的Layer2节点
            )
            layer2_time = time.time() - t_layer2
            time_stats['layer2_processing'].append(layer2_time)
            
            # 收集Layer2节点和边
            layer2_nodes = layer2_stats.get('created_nodes', [])
            fragment_edges_layer2 = layer2_stats.get('fragment_edges', [])  # Fragment到Layer2节点的连接边
            structural_edges = layer2_stats.get('structural_edges', [])  # Layer2节点到Entity的结构性边
            
            fragment_nodes.extend(layer2_nodes)
            
            # 收集Fragment到Layer2节点的连接边
            for fragment_id, node_id, rel_type in fragment_edges_layer2:
                fragment_edges.append({
                    "id": f"fragment_layer2_{fragment_id}_{node_id}",
                    "source": fragment_id,
                    "target": node_id,
                    "type": rel_type,
                    "layer": 0,
                    "active": True
                })
            
            # 收集Layer2节点到Entity的结构性边
            for source_id, target_id, rel_type in structural_edges:
                fragment_edges.append({
                    "id": f"structural_{source_id}_{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "layer": 2,
                    "active": True
                })
            
            # Layer3处理（检查是否触发分析）
            t_layer3 = time.time()
            layer3_stats = {}
            layer3_nodes = []
            layer3_edges = []
            if self.layer3_processor.should_trigger_analysis(namespace):
                layer3_stats = self.layer3_processor.analyze_patterns(namespace)
                # 收集Layer3节点（如果有）
                if 'created_nodes' in layer3_stats:
                    layer3_nodes = layer3_stats.get('created_nodes', [])
                    layer3_edges = layer3_stats.get('created_edges', [])
                    fragment_nodes.extend(layer3_nodes)
                    fragment_edges.extend(layer3_edges)
            layer3_time = time.time() - t_layer3
            if layer3_time > 0.001:  # 只记录有实际处理的时间
                time_stats['layer3_processing'].append(layer3_time)
            
            fragment_total_time = time.time() - fragment_start_time
            time_stats['fragment_processing'].append(fragment_total_time)
            
            print(f"\n⏱️  Fragment处理耗时: {fragment_total_time:.3f}秒")
            print(f"  - Layer1: {layer1_time:.3f}秒")
            print(f"  - Layer2: {layer2_time:.3f}秒")
            if layer3_time > 0.001:
                print(f"  - Layer3: {layer3_time:.3f}秒")
            
            # ===== 关键改动：每个fragment处理完后立即批量生成embedding并直接写入Neo4j =====
            if fragment_nodes or fragment_edges:
                print(f"\n{'='*60}")
                print(f"🚀 Fragment {total_fragments} 批量生成embedding并直接写入Neo4j")
                print(f"{'='*60}")
                print(f"📊 当前Fragment统计:")
                print(f"  - 节点: {len(fragment_nodes)} (Fragment: {len([n for n in fragment_nodes if n.get('layer') == 0])}, "
                      f"Layer1: {len([n for n in fragment_nodes if n.get('layer') == 1])}, "
                      f"Layer2: {len([n for n in fragment_nodes if n.get('layer') == 2])}, "
                      f"Layer3: {len([n for n in fragment_nodes if n.get('layer') == 3])})")
                print(f"  - 边: {len(fragment_edges)}")
                
                # 批量生成embedding（使用cache做去重优化）
                t_embedding = time.time()
                if fragment_nodes:
                    print(f"\n📦 批量生成 {len(fragment_nodes)} 个节点的embedding...")
                    cache.batch_add_nodes(fragment_nodes)
                if fragment_edges:
                    print(f"\n📦 批量生成 {len(fragment_edges)} 个边的embedding...")
                    cache.batch_add_edges(fragment_edges)
                embedding_time = time.time() - t_embedding
                print(f"✅ Embedding生成完成，耗时: {embedding_time:.3f}秒")
                
                # 直接写入Neo4j（不使用"同步"概念）
                try:
                    from core.fragment.neo4j_storage import FragmentNeo4jStorage
                    from core.layer1.neo4j_storage import Layer1Neo4jStorage
                    from core.layer2.neo4j_storage import Layer2Neo4jStorage
                    from core.layer3.neo4j_storage import Layer3Neo4jStorage
                    from core.infrastructure.neo4j_vector_search import Neo4jVectorSearch
                    
                    # 初始化各个Storage
                    fragment_storage = FragmentNeo4jStorage(self._neo4j_client, namespace)
                    layer1_storage = Layer1Neo4jStorage(self._neo4j_client, namespace)
                    layer2_storage = Layer2Neo4jStorage(self._neo4j_client, namespace)
                    layer3_storage = Layer3Neo4jStorage(self._neo4j_client, namespace)
                    vector_search = Neo4jVectorSearch(self._neo4j_client, namespace, self.embedding_manager, self.config)
                    
                    input_filename = f"{namespace}.json"  # 用于Storage接口兼容性
                    
                    nodes_written = 0
                    edges_written = 0
                    embeddings_set = 0
                    
                    print(f"\n💾 直接写入Neo4j...")
                    
                    # 按layer分组写入节点
                    nodes_by_layer = {0: [], 1: [], 2: [], 3: []}
                    for node in fragment_nodes:
                        layer = node.get('layer', 0)
                        nodes_by_layer[layer].append(node)
                    
                    # 写入Fragment节点（Layer0）
                    for node in nodes_by_layer[0]:
                        fragment_storage.save_fragment(node, input_filename)
                        nodes_written += 1
                        # 设置embedding
                        node_id = node.get('id')
                        if node_id in cache.cache['nodes']:
                            cached_node = cache.cache['nodes'][node_id]
                            embedding_idx = cached_node.get('embedding_idx', -1)
                            if embedding_idx >= 0:
                                embedding = cache.embeddings[embedding_idx]
                                vector_search.set_node_embedding(node_id, embedding)
                                embeddings_set += 1
                    
                    # 写入Layer1节点（实体）
                    for node in nodes_by_layer[1]:
                        layer1_storage.save_entity(node, input_filename)
                        nodes_written += 1
                        # 设置embedding
                        node_id = node.get('id')
                        if node_id in cache.cache['nodes']:
                            cached_node = cache.cache['nodes'][node_id]
                            embedding_idx = cached_node.get('embedding_idx', -1)
                            if embedding_idx >= 0:
                                embedding = cache.embeddings[embedding_idx]
                                vector_search.set_node_embedding(node_id, embedding)
                                embeddings_set += 1
                    
                    # 写入Layer2节点（事件、状态、上下文）
                    for node in nodes_by_layer[2]:
                        node_type = node.get('type', 'event')
                        layer2_storage.save_timeline_node(node, namespace, node_type)
                        nodes_written += 1
                        # 设置embedding
                        node_id = node.get('id')
                        if node_id in cache.cache['nodes']:
                            cached_node = cache.cache['nodes'][node_id]
                            embedding_idx = cached_node.get('embedding_idx', -1)
                            if embedding_idx >= 0:
                                embedding = cache.embeddings[embedding_idx]
                                vector_search.set_node_embedding(node_id, embedding)
                                embeddings_set += 1
                    
                    # 写入Layer3节点（聚类、模式等）
                    for node in nodes_by_layer[3]:
                        node_type = node.get('type', 'event_cluster')
                        if node_type == 'event_cluster':
                            layer3_storage.save_event_cluster(node, namespace)
                        elif node_type == 'pattern':
                            layer3_storage.save_pattern(node, namespace)
                        elif node_type == 'preference':
                            layer3_storage.save_preference(node, namespace)
                        elif node_type == 'behavior_rule':
                            layer3_storage.save_behavior_rule(node, namespace)
                        nodes_written += 1
                        # 设置embedding
                        node_id = node.get('id')
                        if node_id in cache.cache['nodes']:
                            cached_node = cache.cache['nodes'][node_id]
                            embedding_idx = cached_node.get('embedding_idx', -1)
                            if embedding_idx >= 0:
                                embedding = cache.embeddings[embedding_idx]
                                vector_search.set_node_embedding(node_id, embedding)
                                embeddings_set += 1
                    
                    # 构建实体名称到ID的映射（用于将关系中的实体名称转换为ID）
                    entity_name_to_id = {}
                    for node in fragment_nodes:
                        if node.get('layer') == 1 and node.get('type') == 'entity':
                            node_id = node.get('id')
                            node_name = node.get('name')
                            if node_id and node_name:
                                entity_name_to_id[node_name] = node_id
                    
                    # 写入边
                    fragment_to_entity_count = 0
                    fragment_to_layer2_count = 0
                    layer1_relation_count = 0
                    layer2_structural_count = 0
                    
                    for edge in fragment_edges:
                        source_id = edge.get('source')
                        target_id = edge.get('target')
                        edge_type = edge.get('type', 'RELATED_TO')
                        edge_props = edge.get('properties', {})
                        edge_layer = edge.get('layer', 0)
                        
                        # 根据边的类型和layer选择存储方式
                        if edge_layer == 1:
                            # Layer1关系 - source和target可能是实体名称，需要转换为ID
                            # 如果source/target不是以entity_开头，尝试从映射中查找
                            if source_id and not source_id.startswith('entity_'):
                                source_id = entity_name_to_id.get(source_id, source_id)
                            if target_id and not target_id.startswith('entity_'):
                                target_id = entity_name_to_id.get(target_id, target_id)
                            
                            # 验证source和target都是有效的实体ID
                            if source_id and target_id and (source_id.startswith('entity_') or source_id in entity_name_to_id.values()) and (target_id.startswith('entity_') or target_id in entity_name_to_id.values()):
                                # Layer1关系 - 使用save_relationship方法（需要传递字典）
                                relationship_dict = {
                                    'source': source_id,
                                    'target': target_id,
                                    'type': edge_type,
                                    'layer': 1,
                                    'content': edge.get('content', ''),
                                    **edge_props
                                }
                                layer1_storage.save_relationship(relationship_dict, namespace)
                                layer1_relation_count += 1
                                print(f"  🔗 创建Layer1关系: {source_id} -> {target_id} (type: {edge_type})")
                            else:
                                print(f"  ⚠️  跳过无效的Layer1关系（无法找到实体ID）: {edge.get('source')} -> {edge.get('target')}")
                        elif edge_layer == 2:
                            # Layer2结构性边
                            layer2_storage.create_structural_edge(source_id, target_id, edge_type, edge_props)
                            layer2_structural_count += 1
                        elif edge_layer == 3:
                            # Layer3边 - pattern/preference/behavior_rule到event的连接
                            # 使用Layer3Storage的通用方法创建关系
                            layer3_storage.create_relationship(
                                source_id=source_id,
                                target_id=target_id,
                                rel_type=edge_type,
                                properties=edge_props
                            )
                            print(f"  🔗 创建Layer3关系: {source_id} -> {target_id} (type: {edge_type})")
                        else:
                            # Fragment连接边（layer=0）- Fragment到Entity/Layer2节点的连接
                            if source_id and source_id.startswith('fragment_'):
                                # Fragment到Entity或Layer2节点的连接
                                # 使用Layer2Storage的create_fragment_connection_edge方法
                                # 这个方法会创建Fragment -> Entity/Event/State/Context的连接
                                print(f"  🔗 创建Fragment连接边: {source_id} -> {target_id} (type: {edge_type})")
                                layer2_storage.create_fragment_connection_edge(source_id, target_id, edge_type, namespace)
                                # 统计：检查target是Entity还是Layer2节点
                                if target_id.startswith('entity_'):
                                    fragment_to_entity_count += 1
                                else:
                                    fragment_to_layer2_count += 1
                            else:
                                # 其他连接边 - 使用Neo4jStorageBase的通用方法
                                if source_id:
                                    print(f"  🔗 创建其他连接边: {source_id} -> {target_id} (type: {edge_type}, layer: {edge_layer})")
                                    layer1_storage.create_relationship(source_id, target_id, edge_type, edge_props)
                                else:
                                    print(f"  ⚠️  跳过无效边（source_id为空）: {edge}")
                        
                        edges_written += 1
                        # 设置边的embedding（如果有）
                        edge_id = edge.get('id')
                        if edge_id in cache.cache['edges']:
                            cached_edge = cache.cache['edges'][edge_id]
                            embedding_idx = cached_edge.get('embedding_idx', -1)
                            if embedding_idx >= 0:
                                embedding = cache.embeddings[embedding_idx]
                                # 注意：Neo4j中边的embedding可能需要特殊处理，这里先跳过
                    
                    print(f"✅ Neo4j写入完成:")
                    print(f"  - 节点: {nodes_written}")
                    print(f"  - 边: {edges_written}")
                    print(f"    * Fragment -> Entity: {fragment_to_entity_count}")
                    print(f"    * Fragment -> Layer2: {fragment_to_layer2_count}")
                    print(f"    * Layer1关系: {layer1_relation_count}")
                    print(f"    * Layer2结构性边: {layer2_structural_count}")
                    print(f"  - Embedding: {embeddings_set}")
                    
                    # 创建向量索引（如果还没有）
                    if embeddings_set > 0:
                        try:
                            print(f"\n📌 创建/检查向量索引...")
                            vector_search.create_vector_index(
                                index_name=f"layer1_entity_vector_idx_{namespace}",
                                label="Entity",
                                dimension=1536,  # text-embedding-3-small的维度
                                similarity_function='cosine'
                            )
                            print(f"  ✅ 向量索引已就绪")
                        except Exception as e:
                            print(f"  ⚠️  向量索引创建失败（可能已存在）: {e}")
                    
                except Exception as e:
                    print(f"❌ Neo4j写入失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Fragment {total_fragments} Neo4j写入失败: {e}")
            
            return layer1_stats, layer2_stats, layer3_stats
        
        # 处理turns，分片后立即处理
        for turn in turns:
            # 从turn中提取timestamp（优先使用metadata中的session_time，否则使用timestamp字段）
            timestamp = turn.get('timestamp')
            metadata = turn.get('metadata', {})
            if metadata.get('session_time'):
                timestamp = metadata.get('session_time')
            
            # Add turn to buffer（传递timestamp，如果时间戳变化会自动分片）
            fragment, needs_llm = self.buffer_manager.add_turn(turn, timestamp=timestamp)
            
            if fragment:
                # 时间戳变化导致的分片，立即处理fragment
                process_fragment_immediately(fragment)
            
            # If needs LLM, process split（长度超限，需要LLM判断）
            if needs_llm:
                split_point = self.fragment_processor.should_split(self.buffer_manager.turns)
                if split_point is not None and split_point > 0:
                    # Create fragment from turns up to split point
                    fragment_turns = self.buffer_manager.turns[:split_point]
                    fragment = {
                        "id": f"fragment_{total_fragments + 1}",
                        "type": "fragment",
                        "content": "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in fragment_turns]),
                        "time": fragment_turns[0].get('timestamp', time.time()) if fragment_turns else time.time(),
                        "layer": 0,
                        "active": True
                    }
                    # 立即处理fragment
                    process_fragment_immediately(fragment)
                    # Remove processed turns
                    self.buffer_manager.turns = self.buffer_manager.turns[split_point:]
        
        # 处理剩余的turns
        if self.buffer_manager.turns:
            fragment = {
                "id": f"fragment_{total_fragments + 1}",
                "type": "fragment",
                "content": "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in self.buffer_manager.turns]),
                "time": self.buffer_manager.turns[0].get('timestamp', time.time()) if self.buffer_manager.turns else time.time(),
                "layer": 0,
                "active": True
            }
            process_fragment_immediately(fragment)
        
        # 注意：每个fragment处理完后已经立即写入Neo4j了
        # 这里只需要统计最终结果
        print(f"\n{'='*60}")
        print(f"✅ 所有Fragment处理完成，共 {total_fragments} 个")
        print(f"{'='*60}")
        
        # 统计最终结果（从Neo4j读取，纯Neo4j架构）
        neo4j_client = self._neo4j_client
        if not neo4j_client or not neo4j_client.driver:
            raise RuntimeError("❌ Neo4j客户端未初始化")
        
        try:
            # 从Neo4j统计
            stats_query = """
            MATCH (n)
            WHERE n.namespace = $namespace
            RETURN 
                sum(CASE WHEN 'Entity' IN labels(n) THEN 1 ELSE 0 END) as entities,
                sum(CASE WHEN 'Event' IN labels(n) THEN 1 ELSE 0 END) as events,
                sum(CASE WHEN 'EventCluster' IN labels(n) THEN 1 ELSE 0 END) as clusters
            """
            result = neo4j_client.execute_read(stats_query, {'namespace': namespace})
            if result:
                total_entities = result[0].get('entities', 0) or 0
                total_events = result[0].get('events', 0) or 0
                total_clusters = result[0].get('clusters', 0) or 0
                print(f"\n📊 Neo4j统计结果:")
                print(f"  - 实体 (Layer1): {total_entities}")
                print(f"  - 事件 (Layer2): {total_events}")
                print(f"  - 聚类 (Layer3): {total_clusters}")
            else:
                raise RuntimeError("从Neo4j统计失败：查询无结果")
        except Exception as e:
            print(f"❌ 从Neo4j统计失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"从Neo4j统计失败: {e}")
        
        # 获取token统计（如果有）
        token_stats = None
        if token_tracker:
            token_stats = token_tracker.get_stats()
        
        # 保存cache到磁盘（供QA系统使用）
        try:
            cache.save()
            print(f"\n💾 Cache已保存到磁盘（{len(cache.cache['nodes'])} 个节点, {len(cache.cache['edges'])} 个边），供QA系统使用")
        except Exception as e:
            print(f"\n⚠️  保存cache失败: {e}")
            import traceback
            traceback.print_exc()
        
        return MemoryBuildResult(
            total_fragments=total_fragments,
            total_entities=total_entities,
            total_events=total_events,
            total_clusters=total_clusters,
            namespace=namespace,
            time_stats=time_stats,
            token_stats=token_stats
        )


class StorageManager:
    """Manage storage operations（纯Neo4j架构）"""
    
    def __init__(self, config: Config):
        """Initialize storage manager"""
        self.config = config
        # 纯Neo4j架构，不再使用文件存储
    
    def get_stats(self, namespace: str = "default", neo4j_client=None) -> Dict[str, Any]:
        """Get storage statistics from Neo4j"""
        if not neo4j_client or not neo4j_client.driver:
            raise RuntimeError("❌ Neo4j客户端未初始化，无法读取统计信息")
        
        try:
            # 从Neo4j统计
            stats_query = """
            MATCH (n)
            WHERE n.namespace = $namespace
            RETURN 
                sum(CASE WHEN 'Entity' IN labels(n) THEN 1 ELSE 0 END) as entities,
                sum(CASE WHEN 'Event' IN labels(n) THEN 1 ELSE 0 END) as events,
                sum(CASE WHEN 'EventCluster' IN labels(n) THEN 1 ELSE 0 END) as clusters
            """
            result = neo4j_client.execute_read(stats_query, {'namespace': namespace})
            if result:
                entities = result[0].get('entities', 0) or 0
                events = result[0].get('events', 0) or 0
                clusters = result[0].get('clusters', 0) or 0
            else:
                entities = events = clusters = 0
            
            return {
                'namespace': namespace,
                'layer1_entities': entities,
                'layer2_events': events,
                'layer3_clusters': clusters
            }
        except Exception as e:
            print(f"❌ 从Neo4j读取统计失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"从Neo4j读取统计失败: {e}")

