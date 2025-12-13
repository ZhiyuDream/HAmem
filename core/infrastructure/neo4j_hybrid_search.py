"""
Neo4j 混合检索模块

结合 FAISS 向量搜索和 Neo4j 图扩展的优势
- FAISS: 毫秒级向量搜索（利用 UnifiedCache）
- Neo4j: 高效的图扩展和关系查询
"""

from typing import List, Dict, Any, Optional, Tuple
from .neo4j_client import Neo4jClient
from .neo4j_vector_search import Neo4jVectorSearch
from .cache import UnifiedCache
from .embedding import EmbeddingManager
from config import Config
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Neo4jHybridSearch:
    """
    混合检索：FAISS 向量搜索 + Neo4j 图扩展
    
    优势：
    1. 使用 UnifiedCache 的 FAISS 索引进行快速向量搜索
    2. 使用 Neo4j 进行高效的图扩展
    3. 保留所有现有的性能优化（批量处理、去重等）
    """
    
    def __init__(
        self,
        cache: UnifiedCache,
        neo4j_client: Neo4jClient,
        namespace: str = "default"
    ):
        """
        初始化混合检索
        
        Args:
            cache: UnifiedCache（包含 FAISS 索引和 embedding 管理）
            neo4j_client: Neo4j 客户端
            namespace: 命名空间
        """
        self.cache = cache
        self.neo4j_client = neo4j_client
        self.namespace = namespace
        self.vector_search = Neo4jVectorSearch(
            neo4j_client,
            namespace,
            embedding_manager=cache.embedding_manager
        )
    
    def hybrid_search(
        self,
        query: str,
        vector_top_k: int = 10,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        expand_limit: int = 50,
        similarity_threshold: float = 0.0,
        layer: Optional[int] = None,
        node_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        混合搜索：FAISS 向量搜索 + Neo4j 图扩展
        
        Args:
            query: 查询文本
            vector_top_k: 向量搜索返回的初始节点数
            max_hops: 图扩展的最大跳数
            relationship_types: 关系类型过滤
            expand_limit: 扩展节点数量限制
            similarity_threshold: 相似度阈值
            layer: 节点层级过滤（可选）
            node_type: 节点类型过滤（可选）
            
        Returns:
            Dict: 包含初始节点和扩展节点的结果
        """
        # 第一步：使用 Neo4j 向量搜索（优先）或 FAISS 向量搜索
        query_embedding = self._get_query_embedding(query)
        
        # 构建过滤条件
        filters = {}
        if layer is not None:
            filters['layer'] = layer
        if node_type:
            filters['type'] = node_type
        
        # 优先使用Neo4j向量搜索（如果可用）
        initial_nodes = []
        initial_node_ids = []
        use_neo4j_vector = False
        
        # 尝试使用Neo4j向量搜索
        try:
            # 根据layer和node_type确定label和index_name
            label = None
            index_name = None
            if layer == 1:
                label = "Entity"
                index_name = f"layer1_entity_vector_idx_{self.namespace}"
            elif layer == 2:
                label = "Event"  # 或其他Layer2标签
            elif layer == 3:
                label = "EventCluster"
            
            neo4j_results = self.vector_search.vector_search(
                query_embedding=query_embedding,
                index_name=index_name if index_name else None,
                label=label,
                top_k=vector_top_k,
                similarity_threshold=similarity_threshold
            )
            
            if neo4j_results:
                # 进一步过滤（如果需要）
                for node in neo4j_results:
                    # 检查layer和type过滤（不再过滤layer=0，因为Layer0也需要召回）
                    if layer is not None and node.get('layer') != layer:
                        continue
                    if node_type and node.get('type') != node_type:
                        continue
                    
                    node_dict = dict(node)
                    initial_nodes.append(node_dict)
                    node_id = node_dict.get('id')
                    if node_id:
                        initial_node_ids.append(node_id)
                
                use_neo4j_vector = True
        except Exception as e:
            logger.debug(f"Neo4j向量搜索失败，降级到FAISS: {e}")
        
        # 如果Neo4j向量搜索没有结果，降级到FAISS
        if not initial_nodes:
            # 使用 UnifiedCache 的 FAISS 索引进行快速搜索
            initial_candidates = self.cache.filter_and_search(
                query_embedding,
                filters=filters if filters else None,
                top_k=vector_top_k
            )
            
            # 提取初始节点
            for candidate in initial_candidates:
                node = candidate.get('node', {})
                similarity = candidate.get('similarity', 0.0)
                
                if similarity >= similarity_threshold and node:
                    # 不再过滤layer=0，因为Layer0也需要召回
                    node_dict = dict(node)
                    node_dict['similarity_score'] = similarity
                    initial_nodes.append(node_dict)
                    node_id = node_dict.get('id')
                    if node_id:
                        initial_node_ids.append(node_id)
        
        if not initial_node_ids:
            return {
                'initial_nodes': [],
                'expanded_nodes': [],
                'all_nodes': [],
                'total_nodes': 0,
                'search_method': 'FAISS' if not use_neo4j_vector else 'Neo4j'
            }
        
        # 第二步：从初始节点通过 Neo4j 进行图扩展
        expanded_nodes = self.vector_search.expand_from_nodes(
            node_ids=initial_node_ids,
            max_hops=max_hops,
            relationship_types=relationship_types,
            direction='both',
            limit=expand_limit
        )
        
        # 根据layer参数过滤节点（不再过滤layer=0，因为Layer0也需要召回）
        if layer is not None:
            # 如果指定了layer，只保留该layer的节点
            expanded_nodes = [n for n in expanded_nodes if n.get('layer') == layer]
        # 如果没有指定layer，保留所有节点（包括layer=0）
        
        # 分离初始节点和扩展节点
        expanded_only = [n for n in expanded_nodes if not n.get('is_initial', False)]
        
        search_method = 'Neo4j + Neo4j' if use_neo4j_vector else 'FAISS + Neo4j'
        return {
            'initial_nodes': initial_nodes,
            'expanded_nodes': expanded_only,
            'all_nodes': expanded_nodes,
            'total_nodes': len(expanded_nodes),
            'search_method': search_method
        }
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        获取查询的 embedding（使用 UnifiedCache 的缓存）
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: embedding 向量
        """
        embedding, _, _ = self.cache.get_or_generate_embedding(query)
        return embedding
    
    def sync_cache_to_neo4j(
        self,
        batch_size: int = 100,
        update_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        将 UnifiedCache 中的数据同步到 Neo4j
        
        这个操作可以：
        1. 批量将节点写入 Neo4j
        2. 批量将关系写入 Neo4j
        3. 批量设置 embedding（如果启用）
        
        Args:
            batch_size: 批量处理大小
            update_embeddings: 是否同步 embedding
            
        Returns:
            Dict: 同步统计信息
        """
        from ..layer1.neo4j_storage import Layer1Neo4jStorage
        from ..layer2.neo4j_storage import Layer2Neo4jStorage
        from ..layer3.neo4j_storage import Layer3Neo4jStorage
        
        stats = {
            'nodes_synced': 0,
            'edges_synced': 0,
            'embeddings_synced': 0,
            'errors': []
        }
        
        print(f"\n{'='*60}")
        print(f"🔄 开始同步数据到Neo4j...")
        print(f"{'='*60}")
        
        # 按 layer 分组节点
        nodes_by_layer = {1: [], 2: [], 3: [], 0: []}
        for node_id, node in self.cache.cache['nodes'].items():
            layer = node.get('layer', 0)
            nodes_by_layer[layer].append(node)
        
        print(f"\n📊 节点统计:")
        print(f"  - Fragment (Layer0): {len(nodes_by_layer[0])}")
        print(f"  - Entity (Layer1): {len(nodes_by_layer[1])}")
        print(f"  - Timeline (Layer2): {len(nodes_by_layer[2])}")
        print(f"  - Cluster (Layer3): {len(nodes_by_layer[3])}")
        
        # 同步 Layer1 节点（实体）
        if nodes_by_layer[1]:
            print(f"\n📦 同步Layer1节点（实体）: {len(nodes_by_layer[1])} 个...")
            layer1_storage = Layer1Neo4jStorage(self.neo4j_client, self.namespace)
            for i in range(0, len(nodes_by_layer[1]), batch_size):
                batch = nodes_by_layer[1][i:i + batch_size]
                print(f"  处理批次 {i//batch_size + 1}/{(len(nodes_by_layer[1])-1)//batch_size + 1} ({len(batch)} 个节点)...")
                for node in batch:
                    try:
                        # 构建实体数据
                        entity = {
                            'id': node.get('id'),
                            'name': node.get('name', ''),
                            'content': node.get('content', ''),
                            'layer': 1,
                            'active': node.get('active', True)
                        }
                        layer1_storage.save_entity(entity, f"{self.namespace}.json")
                        
                        # 同步 embedding
                        if update_embeddings:
                            embedding_idx = node.get('embedding_idx')
                            if embedding_idx is not None and embedding_idx != -1:
                                embedding = self.cache.embeddings[embedding_idx]
                                self.vector_search.set_node_embedding(
                                    node.get('id'),
                                    embedding.tolist(),
                                    labels=['Entity', 'Layer1']
                                )
                                stats['embeddings_synced'] += 1
                        
                        stats['nodes_synced'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Node {node.get('id')}: {e}")
        
        # 同步 Layer2 节点（事件、状态、上下文）
        if nodes_by_layer[2]:
            print(f"\n📦 同步Layer2节点（事件/状态/上下文）: {len(nodes_by_layer[2])} 个...")
            layer2_storage = Layer2Neo4jStorage(self.neo4j_client, self.namespace)
            for node in nodes_by_layer[2]:
                try:
                    node_type = node.get('type', 'event')
                    layer2_storage.save_timeline_node(node, self.namespace, node_type)
                    
                    if update_embeddings:
                        embedding_idx = node.get('embedding_idx')
                        if embedding_idx is not None and embedding_idx != -1:
                            embedding = self.cache.embeddings[embedding_idx]
                            labels = [node_type.capitalize(), 'Layer2']
                            self.vector_search.set_node_embedding(
                                node.get('id'),
                                embedding.tolist(),
                                labels=labels
                            )
                            stats['embeddings_synced'] += 1
                    
                    stats['nodes_synced'] += 1
                except Exception as e:
                    stats['errors'].append(f"Node {node.get('id')}: {e}")
        
        # 同步 Layer3 节点
        if nodes_by_layer[3]:
            print(f"\n📦 同步Layer3节点（聚类/模式）: {len(nodes_by_layer[3])} 个...")
            layer3_storage = Layer3Neo4jStorage(self.neo4j_client, self.namespace)
            for node in nodes_by_layer[3]:
                try:
                    node_type = node.get('type', 'pattern')
                    if node_type == 'event_cluster':
                        layer3_storage.save_event_cluster(node, self.namespace)
                    elif node_type == 'pattern':
                        layer3_storage.save_pattern(node, self.namespace)
                    elif node_type == 'preference':
                        layer3_storage.save_preference(node, self.namespace)
                    elif node_type == 'behavior_rule':
                        layer3_storage.save_behavior_rule(node, self.namespace)
                    
                    if update_embeddings:
                        embedding_idx = node.get('embedding_idx')
                        if embedding_idx is not None and embedding_idx != -1:
                            embedding = self.cache.embeddings[embedding_idx]
                            labels = [node_type.capitalize(), 'Layer3']
                            self.vector_search.set_node_embedding(
                                node.get('id'),
                                embedding.tolist(),
                                labels=labels
                            )
                            stats['embeddings_synced'] += 1
                    
                    stats['nodes_synced'] += 1
                except Exception as e:
                    stats['errors'].append(f"Node {node.get('id')}: {e}")
        
        # 同步 Fragment 节点（Layer0）
        if nodes_by_layer[0]:
            print(f"\n📦 同步Fragment节点（Layer0）: {len(nodes_by_layer[0])} 个...")
            from ..fragment.neo4j_storage import FragmentNeo4jStorage
            fragment_storage = FragmentNeo4jStorage(self.neo4j_client, self.namespace)
            for i in range(0, len(nodes_by_layer[0]), batch_size):
                batch = nodes_by_layer[0][i:i + batch_size]
                if len(nodes_by_layer[0]) > batch_size:
                    print(f"  处理批次 {i//batch_size + 1}/{(len(nodes_by_layer[0])-1)//batch_size + 1} ({len(batch)} 个节点)...")
                for node in batch:
                    try:
                        fragment_storage.save_fragment(node, self.namespace)
                        
                        # 同步 embedding
                        if update_embeddings:
                            embedding_idx = node.get('embedding_idx')
                            if embedding_idx is not None and embedding_idx != -1:
                                embedding = self.cache.embeddings[embedding_idx]
                                self.vector_search.set_node_embedding(
                                    node.get('id'),
                                    embedding.tolist(),
                                    labels=['Fragment', 'Layer0']
                                )
                                stats['embeddings_synced'] += 1
                        
                        stats['nodes_synced'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Fragment {node.get('id')}: {e}")
        
        # 同步边
        all_edges = self.cache.get_all_edges()
        if all_edges:
            print(f"\n📦 同步关系（边）: {len(all_edges)} 条...")
        
        layer1_storage = Layer1Neo4jStorage(self.neo4j_client, self.namespace)
        layer2_storage = Layer2Neo4jStorage(self.neo4j_client, self.namespace)
        
        for i in range(0, len(all_edges), batch_size):
            batch = all_edges[i:i + batch_size]
            if len(all_edges) > batch_size:
                print(f"  处理批次 {i//batch_size + 1}/{(len(all_edges)-1)//batch_size + 1} ({len(batch)} 条边)...")
            for edge in batch:
                try:
                    source_id = edge.get('source')
                    target_id = edge.get('target')
                    rel_type = edge.get('type', 'RELATED_TO')
                    edge_layer = edge.get('layer', 1)
                    
                    # 判断边的类型
                    if edge.get('embedding_idx') == -1:
                        # 结构性边（没有 content）
                        # 检查是否是Fragment到Layer2节点的连接边
                        if edge_layer == 0 and rel_type in ['contains', 'occurs_in']:
                            # Fragment到Layer2节点的连接边，使用Layer2Storage
                            layer2_storage.create_fragment_connection_edge(
                                source_id, target_id, rel_type, self.namespace
                            )
                        elif edge_layer == 2:
                            # Layer2节点到Entity的结构性边，使用Layer2Storage
                            layer2_storage.create_structural_edge(
                                source_id, target_id, rel_type, self.namespace
                            )
                        else:
                            # 其他结构性边，使用Layer1Storage
                            layer1_storage.create_relationship(
                                source_id,
                                target_id,
                                rel_type,
                                properties={'active': edge.get('active', True)}
                            )
                    else:
                        # 有 content 的边（Layer1关系）
                        relationship = {
                            'id': edge.get('id'),
                            'source': source_id,
                            'target': target_id,
                            'type': rel_type,
                            'content': edge.get('content', ''),
                            'layer': edge_layer,
                            'active': edge.get('active', True)
                        }
                        layer1_storage.save_relationship(relationship, self.namespace)
                        
                        # 同步 embedding（如果有）
                        if update_embeddings:
                            embedding_idx = edge.get('embedding_idx')
                            if embedding_idx is not None and embedding_idx != -1:
                                embedding = self.cache.embeddings[embedding_idx]
                                # 注意：Neo4j关系不支持直接存储embedding，这里跳过
                                # 如果需要，可以考虑将embedding存储在关系的properties中
                    
                    stats['edges_synced'] += 1
                except Exception as e:
                    stats['errors'].append(f"Edge {edge.get('id')}: {e}")
        
        print(f"\n{'='*60}")
        print(f"✅ Neo4j同步完成!")
        print(f"  - 节点: {stats['nodes_synced']}")
        print(f"  - 边: {stats['edges_synced']}")
        print(f"  - Embedding: {stats['embeddings_synced']}")
        if stats['errors']:
            print(f"  - 错误: {len(stats['errors'])}")
        print(f"{'='*60}")
        
        return stats
    
    def batch_sync_embeddings_to_neo4j(
        self,
        node_ids: Optional[List[str]] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        批量将 UnifiedCache 中的 embedding 同步到 Neo4j
        
        Args:
            node_ids: 要同步的节点ID列表（如果为 None，同步所有节点）
            batch_size: 批量大小
            
        Returns:
            Dict: 同步统计信息
        """
        stats = {
            'embeddings_synced': 0,
            'nodes_processed': 0,
            'errors': []
        }
        
        # 确定要处理的节点
        if node_ids is None:
            nodes_to_sync = list(self.cache.cache['nodes'].values())
        else:
            nodes_to_sync = [
                self.cache.cache['nodes'][nid]
                for nid in node_ids
                if nid in self.cache.cache['nodes']
            ]
        
        # 批量处理
        for i in range(0, len(nodes_to_sync), batch_size):
            batch = nodes_to_sync[i:i + batch_size]
            
            for node in batch:
                try:
                    node_id = node.get('id')
                    embedding_idx = node.get('embedding_idx')
                    
                    if embedding_idx is None or embedding_idx == -1:
                        continue
                    
                    if embedding_idx >= len(self.cache.embeddings):
                        continue
                    
                    # 获取 embedding
                    embedding = self.cache.embeddings[embedding_idx]
                    
                    # 确定节点标签
                    layer = node.get('layer', 0)
                    node_type = node.get('type', '')
                    
                    labels = []
                    if layer == 1:
                        labels = ['Entity', 'Layer1']
                    elif layer == 2:
                        labels = [node_type.capitalize() if node_type else 'Event', 'Layer2']
                    elif layer == 3:
                        labels = [node_type.capitalize() if node_type else 'Pattern', 'Layer3']
                    else:
                        labels = ['Fragment', 'Layer0']
                    
                    # 设置 embedding
                    success = self.vector_search.set_node_embedding(
                        node_id,
                        embedding.tolist(),
                        labels=labels
                    )
                    
                    if success:
                        stats['embeddings_synced'] += 1
                    stats['nodes_processed'] += 1
                    
                except Exception as e:
                    stats['errors'].append(f"Node {node.get('id')}: {e}")
        
        return stats

