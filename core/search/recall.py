"""
Search Recall module

Recall relevant nodes from UnifiedCache
✅ Reuse existing embeddings and FAISS indexes
✅ Avoid duplicate embedding generation
"""

from typing import List, Dict, Any
import numpy as np


class SearchRecall:
    """
    Recall via UnifiedCache FAISS index
    """
    
    def __init__(self, cache, storage):
        """
        Args:
            cache: UnifiedCache (embeddings + FAISS index)
            storage: Storage (read node details)
        """
        self.cache = cache
        self.storage = storage
        # 添加embedding缓存，避免重复生成
        self._embedding_cache = {}
    
    def multi_layer_recall(
        self,
        query: str,
        layer0_top_k: int = 2,
        layer1_top_k: int = 10,
        layer2_top_k: int = 20,
        layer3_top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Multi-layer recall: search multiple layers at once (including Layer0 Fragment)
        """
        print(f"\n🔍 Multi-layer recall: query='{query[:50]}...'")
        
        results = {}
        
        # Layer0 (Fragment): 使用向量相似度搜索
        if layer0_top_k > 0:
            try:
                from ..infrastructure.embedding import EmbeddingManager
                from config import Config
                config = Config()
                embedding_manager = EmbeddingManager(config)
                query_embedding, _, _ = self.cache.get_or_generate_embedding(query)
                
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
                        fragment_nodes.append(frag_node)
                
                results['layer0'] = fragment_nodes
                print(f"  ✅ Layer0: {len(results['layer0'])} nodes")
            except Exception as e:
                print(f"  ⚠️  Layer0 recall failed: {e}")
                results['layer0'] = []
        
        # Layer1/Layer2/Layer3
        results['layer1'] = self.recall_by_layer(query, layer=1, top_k=layer1_top_k)
        results['layer2'] = self.recall_by_layer(query, layer=2, top_k=layer2_top_k)
        results['layer3'] = self.recall_by_layer(query, layer=3, top_k=layer3_top_k)
        
        print(f"  ✅ Layer1: {len(results['layer1'])} nodes")
        print(f"  ✅ Layer2: {len(results['layer2'])} nodes")
        print(f"  ✅ Layer3: {len(results['layer3'])} nodes")
        
        return results
    
    def recall_by_layer(
        self,
        query: str,
        layer: int,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Recall nodes by layer
        """
        # Use cached embedding to avoid duplicates
        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.cache.embedding_manager.get_embedding(query)
        query_embedding = self._embedding_cache[query]
        
        # Use FAISS index with cached embedding
        candidates = self.cache.filter_and_search(
            query_embedding,
            filters={'layer': layer},
            top_k=top_k
        )
        
        # Return nodes (sorted by similarity)
        return [c['node'] for c in candidates if c.get('node')]
    
    def recall_by_type(
        self,
        query: str,
        node_type: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Recall nodes by type
        """
        # Use cached embedding
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
        Recall by entity names (extract question entities)
        """
        results = []
        
        for entity_name in entity_names:
            # Use cached embedding
            if entity_name not in self._embedding_cache:
                self._embedding_cache[entity_name] = self.cache.embedding_manager.get_embedding(entity_name)
            entity_embedding = self._embedding_cache[entity_name]
            
            # Search entity type only
            candidates = self.cache.filter_and_search(
                entity_embedding,
                filters={'type': 'entity', 'layer': 1},
                top_k=top_k
            )
            
            results.extend([c['node'] for c in candidates if c.get('node')])
        
        # Deduplicate
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
        Get related fragments by node IDs via mentions edges
        """
        fragment_ids = set()
        
        # Find all mentions edges pointing to these nodes
        all_edges = self.cache.get_all_edges()
        
        for edge in all_edges:
            if edge.get('type') == 'mentions' and edge.get('target') in node_ids:
                # fragment --mentions--> entity/event/...
                fragment_ids.add(edge.get('source'))
        
        # Fetch fragment nodes
        fragments = []
        for frag_id in fragment_ids:
            if frag_id in self.cache.cache['nodes']:
                fragments.append(self.cache.cache['nodes'][frag_id])
        
        return fragments

