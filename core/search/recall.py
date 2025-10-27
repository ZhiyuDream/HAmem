"""
Search Recall模块

负责从UnifiedCache中召回相关节点
✅ 复用已有的embedding和FAISS索引
✅ 不重复生成embedding
"""

from typing import List, Dict, Any
import numpy as np


class SearchRecall:
    """
    检索召回模块
    
    直接使用UnifiedCache的FAISS索引进行高效检索
    """
    
    def __init__(self, cache, storage):
        """
        Args:
            cache: UnifiedCache实例（包含所有节点的embedding和FAISS索引）
            storage: Storage实例（用于读取节点详情）
        """
        self.cache = cache
        self.storage = storage
        # 添加embedding缓存，避免重复生成
        self._embedding_cache = {}
    
    def multi_layer_recall(
        self,
        query: str,
        layer1_top_k: int = 10,
        layer2_top_k: int = 20,
        layer3_top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        分层召回：同时检索多个层级的节点
        
        Args:
            query: 用户问题
            layer1_top_k: Layer1召回数量（实体/关系）
            layer2_top_k: Layer2召回数量（事件/状态/上下文）
            layer3_top_k: Layer3召回数量（模式/规则）
        
        Returns:
            {
                'layer1': [nodes],
                'layer2': [nodes],
                'layer3': [nodes]
            }
        """
        print(f"\n🔍 分层召回: query='{query[:50]}...'")
        
        results = {
            'layer1': self.recall_by_layer(query, layer=1, top_k=layer1_top_k),
            'layer2': self.recall_by_layer(query, layer=2, top_k=layer2_top_k),
            'layer3': self.recall_by_layer(query, layer=3, top_k=layer3_top_k)
        }
        
        print(f"  ✅ Layer1: {len(results['layer1'])} 个节点")
        print(f"  ✅ Layer2: {len(results['layer2'])} 个节点")
        print(f"  ✅ Layer3: {len(results['layer3'])} 个节点")
        
        return results
    
    def recall_by_layer(
        self,
        query: str,
        layer: int,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按层级召回节点
        
        Args:
            query: 用户问题
            layer: 层级 (0=fragment, 1=entity/relation, 2=event/state/context, 3=pattern/rule)
            top_k: 召回数量
        
        Returns:
            节点列表（按相似度排序）
        """
        # 使用缓存的embedding，避免重复生成
        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.cache.embedding_manager.get_embedding(query)
        query_embedding = self._embedding_cache[query]
        
        # 使用cache的FAISS索引检索（复用已有embedding）
        candidates = self.cache.filter_and_search(
            query_embedding,
            filters={'layer': layer},
            top_k=top_k
        )
        
        # 返回节点（已按相似度排序）
        return [c['node'] for c in candidates if c.get('node')]
    
    def recall_by_type(
        self,
        query: str,
        node_type: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        按类型召回节点
        
        Args:
            query: 用户问题
            node_type: 节点类型（entity, event, state, context, pattern, etc.）
            top_k: 召回数量
        
        Returns:
            节点列表
        """
        # 使用缓存的embedding，避免重复生成
        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.cache.embedding_manager.get_embedding(query)
        query_embedding = self._embedding_cache[query]
        
        candidates = self.cache.filter_and_search(
            query_embedding,
            filters={'type': node_type},
            top_k=top_k
        )
        
        return [c['node'] for c in candidates if c.get('node')]
    
    def recall_by_entity_name(
        self,
        entity_names: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        按实体名称召回（用于提取问题中的实体）
        
        Args:
            entity_names: 实体名称列表
            top_k: 每个实体召回数量
        
        Returns:
            节点列表
        """
        results = []
        
        for entity_name in entity_names:
            # 使用缓存的embedding，避免重复生成
            if entity_name not in self._embedding_cache:
                self._embedding_cache[entity_name] = self.cache.embedding_manager.get_embedding(entity_name)
            entity_embedding = self._embedding_cache[entity_name]
            
            # 只检索entity类型
            candidates = self.cache.filter_and_search(
                entity_embedding,
                filters={'type': 'entity', 'layer': 1},
                top_k=top_k
            )
            
            results.extend([c['node'] for c in candidates if c.get('node')])
        
        # 去重
        seen_ids = set()
        unique_results = []
        for node in results:
            if node['id'] not in seen_ids:
                seen_ids.add(node['id'])
                unique_results.append(node)
        
        return unique_results
    
    def get_fragments_by_nodes(
        self,
        node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据节点ID获取相关的Fragments
        
        通过mentions edges追溯到原始Fragment
        
        Args:
            node_ids: 节点ID列表
        
        Returns:
            Fragment列表
        """
        fragment_ids = set()
        
        # 查找所有指向这些节点的mentions edges
        all_edges = self.cache.get_all_edges()
        
        for edge in all_edges:
            if edge.get('type') == 'mentions' and edge.get('target') in node_ids:
                # fragment --mentions--> entity/event/...
                fragment_ids.add(edge.get('source'))
        
        # 获取fragment节点
        fragments = []
        for frag_id in fragment_ids:
            if frag_id in self.cache.cache['nodes']:
                fragments.append(self.cache.cache['nodes'][frag_id])
        
        return fragments

