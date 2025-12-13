"""
Neo4j 混合召回模块

结合 FAISS 向量搜索和 Neo4j 图扩展
保留所有现有性能优化
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..infrastructure.neo4j_hybrid_search import Neo4jHybridSearch
from ..infrastructure.cache import UnifiedCache
from ..infrastructure.neo4j_client import Neo4jClient


class Neo4jHybridRecall:
    """
    混合召回：FAISS + Neo4j
    
    使用 FAISS 进行快速向量搜索，使用 Neo4j 进行图扩展
    """
    
    def __init__(
        self,
        cache: UnifiedCache,
        neo4j_client: Neo4jClient,
        namespace: str = "default"
    ):
        """
        初始化混合召回
        
        Args:
            cache: UnifiedCache（包含 FAISS 索引）
            neo4j_client: Neo4j 客户端
            namespace: 命名空间
        """
        self.cache = cache
        self.hybrid_search = Neo4jHybridSearch(cache, neo4j_client, namespace)
        self._embedding_cache = {}  # 查询 embedding 缓存
    
    def recall_with_expansion(
        self,
        query: str,
        vector_top_k: int = 10,
        max_hops: int = 2,
        expand_limit: int = 50,
        layer: Optional[int] = None,
        node_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        召回并扩展：使用 FAISS 找到初始节点，然后通过 Neo4j 扩展
        
        Args:
            query: 查询文本
            vector_top_k: 向量搜索返回的初始节点数
            max_hops: 图扩展的最大跳数
            expand_limit: 扩展节点数量限制
            layer: 节点层级过滤
            node_type: 节点类型过滤
            
        Returns:
            Dict: 包含初始节点和扩展节点的结果
        """
        return self.hybrid_search.hybrid_search(
            query=query,
            vector_top_k=vector_top_k,
            max_hops=max_hops,
            expand_limit=expand_limit,
            layer=layer,
            node_type=node_type
        )
    
    def multi_layer_recall_with_expansion(
        self,
        query: str,
        layer0_top_k: int = 2,
        layer1_top_k: int = 10,
        layer2_top_k: int = 20,
        layer3_top_k: int = 5,
        max_hops: int = 2,
        expand_limit: int = 50
    ) -> Dict[str, Any]:
        """
        多层召回并扩展（包括Layer0 Fragment）
        
        Args:
            query: 查询文本
            layer0_top_k: Layer0 (Fragment) 初始节点数（使用向量相似度）
            layer1_top_k: Layer1 初始节点数
            layer2_top_k: Layer2 初始节点数
            layer3_top_k: Layer3 初始节点数
            max_hops: 图扩展的最大跳数
            expand_limit: 扩展节点数量限制
            
        Returns:
            Dict: 各层的召回和扩展结果
        """
        results = {}
        
        # Layer0 (Fragment): 使用向量相似度搜索，不进行图扩展
        if layer0_top_k > 0:
            try:
                query_embedding = self.hybrid_search._get_query_embedding(query)
                fragment_candidates = self.cache.filter_and_search(
                    query_embedding,
                    filters={'layer': 0},  # 只搜索Fragment
                    top_k=layer0_top_k
                )
                
                # 转换为节点列表格式
                fragment_nodes = []
                for candidate in fragment_candidates:
                    frag_node = candidate.get('node', {})
                    if frag_node:
                        frag_node['similarity_score'] = candidate.get('similarity', 0.0)
                        fragment_nodes.append(frag_node)
                
                results['layer0'] = {
                    'initial_nodes': fragment_nodes,
                    'expanded_nodes': [],
                    'all_nodes': fragment_nodes,
                    'total_nodes': len(fragment_nodes),
                    'search_method': 'FAISS'
                }
            except Exception as e:
                print(f"⚠️  Layer0 (Fragment) recall failed: {e}")
                results['layer0'] = {
                    'initial_nodes': [],
                    'expanded_nodes': [],
                    'all_nodes': [],
                    'total_nodes': 0,
                    'search_method': 'FAISS'
                }
        
        # Layer1/Layer2/Layer3: 使用向量搜索 + 图扩展
        for layer, top_k in [(1, layer1_top_k), (2, layer2_top_k), (3, layer3_top_k)]:
            layer_result = self.recall_with_expansion(
                query=query,
                vector_top_k=top_k,
                max_hops=max_hops,
                expand_limit=expand_limit,
                layer=layer
            )
            results[f'layer{layer}'] = layer_result
        
        return results
    
    def recall_by_type_with_expansion(
        self,
        query: str,
        node_type: str,
        top_k: int = 20,
        max_hops: int = 2,
        expand_limit: int = 50
    ) -> Dict[str, Any]:
        """
        按类型召回并扩展
        
        Args:
            query: 查询文本
            node_type: 节点类型
            top_k: 初始节点数
            max_hops: 图扩展的最大跳数
            expand_limit: 扩展节点数量限制
            
        Returns:
            Dict: 召回和扩展结果
        """
        return self.recall_with_expansion(
            query=query,
            vector_top_k=top_k,
            max_hops=max_hops,
            expand_limit=expand_limit,
            node_type=node_type
        )
    
    def get_fragments_by_nodes(
        self,
        node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        通过节点ID获取相关的Fragment（从Neo4j查询）
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            List[Dict]: Fragment节点列表
        """
        if not node_ids:
            return []
        
        # 从Neo4j查询：找到所有连接到这些节点的Fragment
        # Fragment -> (CONTAINS/OCCURS_IN) -> Entity/Event/State/Context
        query = """
        MATCH (f:Fragment:Layer0)-[r:CONTAINS|OCCURS_IN]->(n)
        WHERE n.id IN $node_ids AND f.namespace = $namespace
        RETURN DISTINCT properties(f) as props
        """
        
        try:
            result = self.hybrid_search.neo4j_client.execute_read(
                query, 
                {'node_ids': node_ids, 'namespace': self.hybrid_search.namespace}
            )
            
            fragments = []
            for record in result:
                props = record.get('props', {})
                if props:
                    fragment_node = dict(props)  # props已经是字典
                    fragments.append(fragment_node)
            
            return fragments
        except Exception as e:
            print(f"⚠️  从Neo4j查询Fragment失败: {e}")
            return []
    
    def multi_layer_recall(
        self,
        query: str,
        layer1_top_k: int = 10,
        layer2_top_k: int = 20,
        layer3_top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        多层召回（兼容原有接口）
        
        Args:
            query: 查询文本
            layer1_top_k: Layer1 节点数
            layer2_top_k: Layer2 节点数
            layer3_top_k: Layer3 节点数
            
        Returns:
            Dict: 各层的节点列表
        """
        results = {}
        
        for layer, top_k in [(1, layer1_top_k), (2, layer2_top_k), (3, layer3_top_k)]:
            layer_result = self.recall_with_expansion(
                query=query,
                vector_top_k=top_k,
                max_hops=0,  # 不扩展，只返回初始节点
                expand_limit=0,
                layer=layer
            )
            # 提取节点列表
            nodes = layer_result.get('all_nodes', [])
            results[f'layer{layer}'] = nodes
        
        return results

