"""
Layer1智能召回模块

为冲突检测召回候选实体和关系
优先使用cache（FAISS）召回，因为cache中有当前处理过程中累积的实体
如果cache结果不足，再从Neo4j补充
"""

from typing import Dict, List, Any, Optional
import logging
from core.infrastructure import UnifiedCache
from core.infrastructure.neo4j_vector_search import Neo4jVectorSearch
from core.infrastructure.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class Layer1Recall:
    """Layer1智能召回器"""
    
    def __init__(
        self, 
        cache: UnifiedCache,
        neo4j_vector_search: Optional[Neo4jVectorSearch] = None,
        namespace: str = "default"
    ):
        """
        初始化召回器
        
        Args:
            cache: UnifiedCache（用于生成embedding）
            neo4j_vector_search: Neo4j向量搜索实例（如果提供，使用Neo4j向量搜索；否则使用FAISS）
            namespace: 命名空间
        """
        self.cache = cache
        self.neo4j_vector_search = neo4j_vector_search
        self.namespace = namespace
        self.entity_similarity_threshold = 0.85
        self.relation_similarity_threshold = 0.85
        
        # 检查Neo4j是否支持向量索引（社区版不支持）
        if neo4j_vector_search:
            # 检查是否支持向量索引
            self.use_neo4j_search = getattr(neo4j_vector_search, 'supports_vector_index', False)
        else:
            self.use_neo4j_search = False
        
        # 如果使用Neo4j向量搜索，尝试创建向量索引
        if self.use_neo4j_search:
            self._ensure_vector_index()
        # 社区版会自动使用cache（FAISS），无需输出提示
    
    def _ensure_vector_index(self):
        """确保向量索引存在"""
        if not self.use_neo4j_search:
            return
        
        try:
            # 尝试创建Layer1实体的向量索引
            # embedding维度：text-embedding-3-small 是 1536
            self.neo4j_vector_search.create_vector_index(
                index_name=f"layer1_entity_vector_idx_{self.namespace}",
                label="Entity",
                dimension=1536,
                similarity_function='cosine'
            )
        except Exception as e:
            print(f"  ⚠️  创建向量索引失败（可能已存在）: {e}")
    
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
        # 构建搜索文本（使用实体的name和description）
        name = entity.get('name', '')
        description = entity.get('description', '')
        search_text = f"{name} {description}".strip()
        
        # 生成embedding
        entity_embedding, _, _ = self.cache.get_or_generate_embedding(search_text)
        
        candidates = []
        
        # 首先尝试从cache（FAISS）召回（优先，因为cache中有当前处理过程中累积的实体）
        try:
            similar_entities = self.cache.filter_and_search(
                entity_embedding,
                filters={'type': 'entity', 'layer': 1},
                top_k=10
            )
        
        # 过滤：相似度阈值
            cache_candidates = [
            c for c in similar_entities
            if c['similarity'] > self.entity_similarity_threshold
        ]
            candidates.extend(cache_candidates)
        except Exception as e:
            logger.debug(f"从cache召回失败: {e}")
        
        # 如果使用Neo4j搜索且cache召回结果不足，尝试从Neo4j补充
        if self.use_neo4j_search and len(candidates) < 5:
            try:
                similar_nodes = self.neo4j_vector_search.vector_search(
                    query_embedding=entity_embedding,
                    index_name=f"layer1_entity_vector_idx_{self.namespace}",
                    label="Entity",
                    top_k=10,
                    similarity_threshold=self.entity_similarity_threshold
                )
                
                # 转换为统一格式并去重
                existing_ids = {c.get('id') for c in candidates}
                for node in similar_nodes:
                    if node.get('layer') == 1 and node.get('type') == 'entity':
                        node_id = node.get('id')
                        if node_id not in existing_ids:
                            candidates.append({
                                'id': node_id,
                                'name': node.get('name', ''),
                                'content': node.get('content', ''),
                                'similarity': node.get('similarity_score', 0.0)
                            })
                            existing_ids.add(node_id)
            except Exception as e:
                logger.debug(f"从Neo4j召回失败: {e}")
        
        # 按相似度排序并限制数量
        candidates.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
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
        
        # Step 2: 对每个实体进行搜索
        # 优先使用cache（FAISS）召回，因为cache中已经有当前处理过程中累积的实体
        # Neo4j中的实体可能还没有写入（流式处理）
        for i, (entity_name, entity_embedding) in enumerate(zip(entity_names, entity_embeddings)):
            candidates = []
            
            # 首先尝试从cache（FAISS）召回
            try:
                similar_entities = self.cache.filter_and_search(
                    entity_embedding,
                    filters={'type': 'entity', 'layer': 1},
                    top_k=10
                )
            
            # 过滤：相似度阈值
                cache_candidates = [
                c for c in similar_entities
                if c['similarity'] > self.entity_similarity_threshold
                ]
                candidates.extend(cache_candidates)
            except Exception as e:
                logger.debug(f"从cache召回失败: {e}")
            
            # 如果使用Neo4j搜索且cache召回结果不足，尝试从Neo4j补充
            if self.use_neo4j_search and len(candidates) < 5:
                try:
                    similar_nodes = self.neo4j_vector_search.vector_search(
                        query_embedding=entity_embedding,
                        index_name=f"layer1_entity_vector_idx_{self.namespace}",
                        label="Entity",
                        top_k=10,
                        similarity_threshold=self.entity_similarity_threshold
                    )
                    
                    # 转换为统一格式并去重（避免与cache结果重复）
                    existing_ids = {c.get('id') for c in candidates}
                    for node in similar_nodes:
                        # 只返回Layer1的实体
                        if node.get('layer') == 1 and node.get('type') == 'entity':
                            node_id = node.get('id')
                            if node_id not in existing_ids:
                                candidates.append({
                                    'id': node_id,
                                    'name': node.get('name', ''),
                                    'content': node.get('content', ''),
                                    'similarity': node.get('similarity_score', 0.0)
                                })
                                existing_ids.add(node_id)
                except Exception as e:
                    logger.debug(f"从Neo4j召回失败: {e}")
            
            # 按相似度排序并限制数量
            candidates.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            candidates = candidates[:5]  # 最多5个候选
            
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

