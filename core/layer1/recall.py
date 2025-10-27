"""
Layer1智能召回模块

为冲突检测召回候选实体和关系
"""

from typing import Dict, List, Any
from core.infrastructure import UnifiedCache


class Layer1Recall:
    """Layer1智能召回器"""
    
    def __init__(self, cache: UnifiedCache):
        self.cache = cache
        self.entity_similarity_threshold = 0.85
        self.relation_similarity_threshold = 0.85
    
    def recall_entity_candidates(
        self, 
        entity: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        召回实体候选
        
        Args:
            entity: 新提取的实体 {"name": "...", "description": "..."}
        
        Returns:
            候选实体列表
        """
        # 构建搜索文本
        search_text = entity['name'] + ' ' + entity['description']
        
        # 生成embedding
        entity_embedding, _, _ = self.cache.get_or_generate_embedding(search_text)
        
        # FAISS检索相似实体（只在Layer1的实体中搜索）
        similar_entities = self.cache.filter_and_search(
            entity_embedding,
            filters={'type': 'entity', 'layer': 1},
            top_k=10  # 多召回一些，后面过滤
        )
        
        # 过滤：相似度阈值
        candidates = [
            c for c in similar_entities
            if c['similarity'] > self.entity_similarity_threshold
        ]
        
        # 限制最多5个候选
        return candidates[:5]
    
    def recall_relation_candidates(
        self,
        relation: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        召回关系候选
        
        Args:
            relation: 新提取的关系 {"source": "...", "target": "...", "description": "..."}
        
        Returns:
            候选关系列表
        """
        # 容错处理：检查必需字段
        source = relation.get('source', '')
        target = relation.get('target', '')
        description = relation.get('description', '')
        
        if not source or not target:
            print(f"  ⚠️  跳过无效关系: {relation}")
            return []
        
        # 构建搜索文本
        search_text = f"{source} {target} {description}"
        
        # 生成embedding
        relation_embedding, _, _ = self.cache.get_or_generate_embedding(search_text)
        
        # 获取所有Layer1的边
        # 注意：edges没有type字段，所以filter_and_search需要特殊处理
        # 这里我们先用一个简化的实现
        all_edges = self.cache.get_all_edges(layer=1)
        
        if not all_edges:
            return []
        
        # 计算相似度
        candidates = []
        for edge in all_edges:
            edge_content = edge.get('content', '')
            edge_embedding_idx = edge.get('embedding_idx')
            
            if edge_embedding_idx is None:
                continue
            
            # 获取edge的embedding
            if edge_embedding_idx < len(self.cache.embeddings):
                edge_embedding = self.cache.embeddings[edge_embedding_idx]
                
                # 计算相似度
                import numpy as np
                similarity = self._cosine_similarity(relation_embedding, edge_embedding)
                
                if similarity > self.relation_similarity_threshold:
                    candidates.append({
                        'edge': edge,
                        'similarity': float(similarity)
                    })
        
        # 按相似度排序
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 限制最多3个候选
        return candidates[:3]
    
    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_recall_entities(
        self,
        entities: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量并行召回实体候选（一次embedding生成，一次FAISS搜索）
        
        Args:
            entities: 实体列表
        
        Returns:
            {entity_name: [candidates]}
        """
        if not entities:
            return {}
        
        print(f"🔍 批量召回 {len(entities)} 个实体的候选...")
        
        entity_candidates = {}
        
        # Step 1: 批量生成所有实体的embedding
        entity_embeddings = []
        entity_names = []
        for entity in entities:
            # 容错处理：检查必需字段
            name = entity.get('name', '')
            description = entity.get('description', '')
            
            if not name:
                print(f"  ⚠️  跳过无效实体: {entity}")
                continue
                
            search_text = f"{name} {description}"
            embedding, _, _ = self.cache.get_or_generate_embedding(search_text)
            entity_embeddings.append(embedding)
            entity_names.append(name)
        
        # Step 2: 对每个实体并行搜索（FAISS支持批量搜索）
        for i, (entity_name, entity_embedding) in enumerate(zip(entity_names, entity_embeddings)):
            similar_entities = self.cache.filter_and_search(
                entity_embedding,
                filters={'type': 'entity', 'layer': 1},
                top_k=10
            )
            
            # 过滤：相似度阈值
            candidates = [
                c for c in similar_entities
                if c['similarity'] > self.entity_similarity_threshold
            ][:5]  # 最多5个候选
            
            entity_candidates[entity_name] = candidates
            
            if candidates:
                print(f"  ✅ {entity_name}: 找到 {len(candidates)} 个候选")
            else:
                print(f"  ℹ️  {entity_name}: 无候选")
        
        return entity_candidates
    
    def batch_recall_relations(
        self,
        relationships: List[Dict[str, str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量并行召回关系候选
        
        Args:
            relationships: 关系列表
        
        Returns:
            {relation_key: [candidates]}
        """
        if not relationships:
            return {}
        
        print(f"🔍 批量召回 {len(relationships)} 个关系的候选...")
        
        relation_candidates = {}
        
        # Step 1: 获取所有Layer1的边
        all_edges = self.cache.get_all_edges(layer=1)
        
        if not all_edges:
            # 没有现有的边，所有关系都是新建
            for relation in relationships:
                source = relation.get('source', '')
                target = relation.get('target', '')
                if source and target:
                    relation_key = f"{source}_{target}"
                    relation_candidates[relation_key] = []
                    print(f"  ℹ️  {relation_key}: 无候选")
            return relation_candidates
        
        # Step 2: 批量生成所有关系的embedding
        relation_embeddings = []
        relation_keys = []
        for relation in relationships:
            # 容错处理：检查必需字段
            source = relation.get('source', '')
            target = relation.get('target', '')
            description = relation.get('description', '')
            
            if not source or not target:
                print(f"  ⚠️  跳过无效关系: {relation}")
                continue
                
            search_text = f"{source} {target} {description}"
            embedding, _, _ = self.cache.get_or_generate_embedding(search_text)
            relation_embeddings.append(embedding)
            relation_keys.append(f"{source}_{target}")
        
        # Step 3: 预先生成所有边的embedding
        import numpy as np
        edge_embeddings = []
        valid_edges = []
        for edge in all_edges:
            edge_embedding_idx = edge.get('embedding_idx')
            if edge_embedding_idx is not None and edge_embedding_idx < len(self.cache.embeddings):
                edge_embeddings.append(self.cache.embeddings[edge_embedding_idx])
                valid_edges.append(edge)
        
        if not edge_embeddings:
            # 没有有效的边embedding
            for relation_key in relation_keys:
                relation_candidates[relation_key] = []
                print(f"  ℹ️  {relation_key}: 无候选")
            return relation_candidates
        
        # Step 4: 批量计算相似度
        edge_embeddings_array = np.array(edge_embeddings)
        
        for relation_key, relation_embedding in zip(relation_keys, relation_embeddings):
            # 计算当前关系与所有边的相似度
            relation_embedding_array = np.array(relation_embedding)
            
            # 批量计算余弦相似度
            dot_products = np.dot(edge_embeddings_array, relation_embedding_array)
            norm1 = np.linalg.norm(edge_embeddings_array, axis=1)
            norm2 = np.linalg.norm(relation_embedding_array)
            
            similarities = dot_products / (norm1 * norm2 + 1e-8)
            
            # 过滤并排序
            candidates = []
            for i, similarity in enumerate(similarities):
                if similarity > self.relation_similarity_threshold:
                    candidates.append({
                        'edge': valid_edges[i],
                        'similarity': float(similarity)
                    })
            
            # 按相似度排序，取前3个
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            relation_candidates[relation_key] = candidates[:3]
            
            if candidates:
                print(f"  ✅ {relation_key}: 找到 {len(candidates[:3])} 个候选")
            else:
                print(f"  ℹ️  {relation_key}: 无候选")
        
        return relation_candidates

