"""
Neo4j 向量搜索和图扩展模块

支持通过 embedding 进行向量相似性搜索，然后通过图扩展获取相关节点
集成 OpenAI embedding API 自动生成向量
"""

from typing import List, Dict, Any, Optional, Tuple
from .neo4j_client import Neo4jClient
from .embedding import EmbeddingManager
from config import Config
import logging
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        float: 余弦相似度 (0-1)
    """
    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)
    
    dot_product = np.dot(vec1_array, vec2_array)
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


class Neo4jVectorSearch:
    """Neo4j 向量搜索和图扩展"""
    
    def __init__(
        self, 
        neo4j_client: Neo4jClient, 
        namespace: str = "default",
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[Config] = None
    ):
        """
        初始化向量搜索
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间
            embedding_manager: Embedding管理器（如果提供，可以自动生成embedding）
            config: 配置对象（如果提供embedding_manager，会自动创建）
        """
        self.client = neo4j_client
        self.namespace = namespace
        
        # 检查是否为社区版（社区版不支持原生向量索引）
        self.supports_vector_index = not getattr(neo4j_client, 'is_community_edition', False)
        if not self.supports_vector_index:
            logger.debug("Neo4j社区版不支持原生向量索引，向量搜索功能将被禁用，使用cache（FAISS）进行向量搜索")
        
        # 初始化 EmbeddingManager（如果未提供）
        if embedding_manager is None and config is not None:
            try:
                self.embedding_manager = EmbeddingManager(config)
            except Exception as e:
                logger.warning(f"无法初始化 EmbeddingManager: {e}")
                self.embedding_manager = None
        else:
            self.embedding_manager = embedding_manager
    
    def generate_and_set_embedding(
        self,
        node_id: str,
        text: str,
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        使用 OpenAI API 生成 embedding 并设置到节点
        
        Args:
            node_id: 节点ID
            text: 要生成 embedding 的文本（通常是节点的 content）
            labels: 节点标签（可选，用于优化查询）
            
        Returns:
            bool: 是否设置成功
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager 未初始化，无法生成 embedding")
        
        try:
            # 使用 OpenAI API 生成 embedding
            embedding = self.embedding_manager.get_embedding(text)
            logger.debug(f"✅ 成功生成 embedding (维度: {len(embedding)})")
            success = self.set_node_embedding(node_id, embedding, labels)
            if success:
                logger.info(f"✅ 节点 {node_id} 的 embedding 设置成功")
            else:
                logger.warning(f"⚠️  节点 {node_id} 的 embedding 设置失败")
            return success
        except Exception as e:
            logger.error(f"生成 embedding 失败: {e}", exc_info=True)
            return False
    
    def set_node_embedding(
        self,
        node_id: str,
        embedding: List[float],
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        设置节点的 embedding 向量
        
        Args:
            node_id: 节点ID
            embedding: embedding 向量
            labels: 节点标签（可选，用于优化查询）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            # 使用 db.create.setNodeVectorProperty 设置向量属性
            if labels:
                labels_str = ":".join(labels)
                query = f"""
                MATCH (n:{labels_str} {{id: $node_id, namespace: $namespace}})
                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                RETURN n
                """
            else:
                query = """
                MATCH (n {id: $node_id, namespace: $namespace})
                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                RETURN n
                """
            
            result = self.client.execute_write(query, {
                'node_id': node_id,
                'namespace': self.namespace,
                'embedding': embedding
            })
            
            if len(result) > 0:
                logger.debug(f"✅ 使用向量属性设置 embedding 成功")
                return True
            else:
                logger.warning(f"⚠️  向量属性设置返回空结果，节点可能不存在: {node_id}")
                # 降级到普通属性存储
                return self._set_embedding_fallback(node_id, embedding, labels)
        except Exception as e:
            logger.warning(f"向量属性设置失败，降级到普通属性存储: {e}")
            # 如果 db.create.setNodeVectorProperty 不可用，使用普通属性存储
            return self._set_embedding_fallback(node_id, embedding, labels)
    
    def _set_embedding_fallback(
        self,
        node_id: str,
        embedding: List[float],
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        降级方法：使用普通属性存储 embedding
        
        Args:
            node_id: 节点ID
            embedding: embedding 向量
            labels: 节点标签
            
        Returns:
            bool: 是否设置成功
        """
        try:
            if labels:
                labels_str = ":".join(labels)
                query = f"""
                MATCH (n:{labels_str} {{id: $node_id, namespace: $namespace}})
                SET n.embedding = $embedding
                RETURN n
                """
            else:
                query = """
                MATCH (n {id: $node_id, namespace: $namespace})
                SET n.embedding = $embedding
                RETURN n
                """
            
            result = self.client.execute_write(query, {
                'node_id': node_id,
                'namespace': self.namespace,
                'embedding': embedding
            })
            
            if len(result) > 0:
                logger.info(f"✅ 使用普通属性设置 embedding 成功: {node_id}")
                return True
            else:
                logger.error(f"❌ 普通属性设置也失败，节点可能不存在: {node_id}")
                return False
        except Exception as e2:
            logger.error(f"❌ 普通属性设置也失败: {e2}", exc_info=True)
            return False
    
    def _check_index_exists(self, index_name: str) -> bool:
        """
        检查向量索引是否存在（使用SHOW INDEXES，兼容所有Neo4j版本）
        
        Args:
            index_name: 索引名称
            
        Returns:
            bool: 索引是否存在
        """
        try:
            # 使用SHOW INDEXES检查索引（Neo4j标准命令）
            query = """
            SHOW INDEXES
            YIELD name, type, state
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN name, state
            """
            result = self.client.execute_read(query, {'index_name': index_name})
            if result and len(result) > 0:
                state = result[0].get('state', '')
                logger.debug(f"索引 {index_name} 存在，状态: {state}")
                return True
            else:
                logger.debug(f"索引 {index_name} 不存在")
                return False
        except Exception as e:
            logger.warning(f"检查索引是否存在失败: {e}")
            return False
    
    def create_vector_index(
        self,
        index_name: str,
        label: str,
        dimension: int,
        similarity_function: str = 'cosine'
    ) -> bool:
        """
        创建向量索引（使用Neo4j 5.18+语法）
        
        注意：Neo4j社区版不支持原生向量索引，此方法在社区版中会返回False
        
        Args:
            index_name: 索引名称
            label: 节点标签
            dimension: 向量维度
            similarity_function: 相似性函数（cosine, euclidean）
            
        Returns:
            bool: 是否创建成功
        """
        # 检查是否支持向量索引（社区版直接返回，不输出警告）
        if not self.supports_vector_index:
            logger.debug(f"Neo4j社区版不支持原生向量索引，跳过创建 {index_name}")
            return False
        
        try:
            # 先检查索引是否已存在
            if self._check_index_exists(index_name):
                logger.debug(f"向量索引 {index_name} 已存在，检查状态...")
                # 等待索引就绪
                if self._wait_for_index_ready(index_name):
                    logger.info(f"✅ 向量索引 {index_name} 已就绪")
                    return True
                else:
                    logger.warning(f"⚠️  向量索引 {index_name} 存在但未就绪")
                    return False
            
            # Neo4j 5.18+ 向量索引创建语法
            # 注意：索引名称用反引号包裹，ON后面用括号
            escaped_index_name = index_name.replace('`', '``')  # 转义反引号
            query = f"""
            CREATE VECTOR INDEX `{escaped_index_name}`
            IF NOT EXISTS
            FOR (n:{label})
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """
            
            logger.info(f"📌 创建向量索引: {index_name} (label: {label}, dimension: {dimension})")
            logger.debug(f"执行查询: {query[:200]}...")
            
            try:
                result = self.client.execute_write(query, {})
                logger.debug(f"索引创建命令执行完成，结果: {result}")
            except Exception as create_error:
                logger.error(f"执行索引创建命令失败: {create_error}")
                # 检查是否是因为索引已存在
                if "already exists" in str(create_error).lower() or "exists" in str(create_error).lower():
                    logger.info(f"索引 {index_name} 可能已存在，继续验证...")
                else:
                    raise
            
            # 等待一小段时间让索引创建完成
            import time
            time.sleep(0.5)
            
            # 验证索引是否真的创建成功
            if not self._check_index_exists(index_name):
                logger.error(f"❌ 向量索引 {index_name} 创建失败：索引不存在")
                # 尝试列出所有向量索引以调试
                try:
                    debug_query = """
                    SHOW INDEXES
                    YIELD name, type
                    WHERE type = 'VECTOR'
                    RETURN name
                    """
                    all_indexes = self.client.execute_read(debug_query, {})
                    logger.debug(f"当前所有向量索引: {[r.get('name') for r in all_indexes]}")
                except:
                    pass
                return False
            
            logger.info(f"✅ 向量索引 {index_name} 创建成功，等待索引就绪...")
            
            # 等待索引就绪（最多等待30秒）
            if self._wait_for_index_ready(index_name, max_wait=30):
                logger.info(f"✅ 向量索引 {index_name} 已就绪，可以使用")
                return True
            else:
                logger.error(f"❌ 向量索引 {index_name} 创建但未就绪（超时）")
                return False
        except Exception as e:
            logger.error(f"创建向量索引失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _wait_for_index_ready(self, index_name: str, max_wait: int = 10) -> bool:
        """
        等待向量索引就绪（使用SHOW INDEXES检查状态）
        
        Args:
            index_name: 索引名称
            max_wait: 最大等待秒数
            
        Returns:
            bool: 索引是否就绪
        """
        import time
        for i in range(max_wait):
            try:
                # 使用SHOW INDEXES检查索引状态
                query = """
                SHOW INDEXES
                YIELD name, type, state
                WHERE name = $index_name AND type = 'VECTOR'
                RETURN state
                """
                result = self.client.execute_read(query, {'index_name': index_name})
                if result and len(result) > 0:
                    state = result[0].get('state', '')
                    if state == 'ONLINE':
                        return True
                    elif state in ['POPULATING', 'BUILDING']:
                        logger.debug(f"索引 {index_name} 状态: {state}，等待中... ({i+1}/{max_wait})")
                        time.sleep(1)
                    else:
                        logger.warning(f"索引 {index_name} 状态: {state}")
                        # 如果状态不是ONLINE/POPULATING/BUILDING，可能索引有问题
                        if i == 0:  # 第一次检查就发现异常状态
                            return False
                else:
                    # 索引不存在
                    logger.warning(f"索引 {index_name} 不存在")
                    return False
            except Exception as e:
                logger.debug(f"检查索引状态失败: {e}")
                time.sleep(1)
        
        # 超时后再次检查一次
        try:
            query = """
            SHOW INDEXES
            YIELD name, type, state
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN state
            """
            result = self.client.execute_read(query, {'index_name': index_name})
            if result and len(result) > 0:
                state = result[0].get('state', '')
                if state == 'ONLINE':
                    return True
        except:
            pass
        
        return False
    
    def vector_search(
        self,
        query_embedding: List[float],
        index_name: Optional[str] = None,
        label: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        向量相似性搜索（仅使用向量索引，不降级到普通查询）
        
        注意：Neo4j社区版不支持原生向量索引，此方法在社区版中会返回空列表
        
        Args:
            query_embedding: 查询向量
            index_name: 向量索引名称（必需）
            label: 节点标签（用于创建索引）
            top_k: 返回最相似的 k 个节点
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict]: 相似节点列表，包含节点数据和相似度分数
        """
        results = []
        
        # 检查是否支持向量索引
        if not self.supports_vector_index:
            logger.debug("Neo4j社区版不支持原生向量索引，返回空结果（应使用cache/FAISS进行召回）")
            return results
        
        # 必须提供索引名称
        if not index_name:
            logger.error("向量搜索需要提供 index_name 参数")
            return results
        
        try:
            # 先检查索引是否存在，如果不存在则尝试创建
            if not self._check_index_exists(index_name):
                logger.info(f"向量索引 {index_name} 不存在，尝试创建...")
                if label:
                    # 从index_name推断维度（默认1536，text-embedding-3-small）
                    dimension = 1536
                    if not self.create_vector_index(index_name, label, dimension, 'cosine'):
                        logger.error(f"无法创建向量索引 {index_name}，查询失败")
                        return results
                else:
                    logger.error(f"无法创建向量索引 {index_name}：缺少 label 参数")
                    return results
            
            # 再次确认索引存在且就绪
            if not self._wait_for_index_ready(index_name, max_wait=5):
                logger.error(f"向量索引 {index_name} 未就绪，无法查询")
                return results
            
            # 使用向量索引查询
            # 注意：根据Neo4j官方文档，索引名称应该作为字符串字面量传递
            # 转义单引号以避免SQL注入
            escaped_index_name = index_name.replace("'", "''")
            query = f"""
            CALL db.index.vector.queryNodes('{escaped_index_name}', $top_k, $query_embedding)
            YIELD node, score
            WHERE node.namespace = $namespace AND score >= $threshold
            RETURN properties(node) as props, score
            ORDER BY score DESC
            """
            
            result = self.client.execute_read(query, {
                'top_k': top_k,
                'query_embedding': query_embedding,
                'namespace': self.namespace,
                'threshold': similarity_threshold
            })
            
            for record in result:
                props = record.get('props', {})
                score = record.get('score', 0.0)
                if props:
                    node_dict = dict(props)  # props已经是字典
                    node_dict['similarity_score'] = score
                    results.append(node_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"向量索引查询失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 不降级到普通查询，直接返回空结果
            return results
    
    def expand_from_nodes(
        self,
        node_ids: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        direction: str = 'both',  # 'outgoing', 'incoming', 'both'
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        从初始节点通过边扩展获取更多节点
        
        Args:
            node_ids: 初始节点ID列表
            max_hops: 最大跳数
            relationship_types: 关系类型过滤（None 表示所有类型）
            direction: 扩展方向
            limit: 返回节点数量限制
            
        Returns:
            List[Dict]: 扩展后的节点列表（包含初始节点）
        """
        # 构建关系类型过滤
        if relationship_types:
            rel_types_str = "|".join([f":{rt}" for rt in relationship_types])
            rel_filter = f"*1..{max_hops}"
            rel_type_filter = f":{rel_types_str}"
        else:
            rel_filter = f"*1..{max_hops}"
            rel_type_filter = ""
        
        # 构建方向
        if direction == 'outgoing':
            pattern = f"-[{rel_type_filter}*1..{max_hops}]->"
        elif direction == 'incoming':
            pattern = f"<-[{rel_type_filter}*1..{max_hops}]-"
        else:  # both
            pattern = f"-[{rel_type_filter}*1..{max_hops}]-"
        
        # 构建查询
        node_ids_str = "', '".join(node_ids)
        query = f"""
        MATCH (start {{id: $node_ids[0], namespace: $namespace}})
        MATCH path = (start){pattern}(connected {{namespace: $namespace}})
        WHERE ALL(id IN $node_ids WHERE connected.id <> id)
        WITH DISTINCT connected, length(path) as hops
        ORDER BY hops ASC
        LIMIT $limit
        RETURN connected, hops
        """
        
        # 如果只有一个节点，简化查询
        if len(node_ids) == 1:
            query = f"""
            MATCH (start {{id: $node_id, namespace: $namespace}})
            MATCH path = (start){pattern}(connected {{namespace: $namespace}})
            WHERE connected.id <> $node_id
            WITH DISTINCT connected, length(path) as hops
            ORDER BY hops ASC
            LIMIT $limit
            RETURN properties(connected) as props, hops
            """
            result = self.client.execute_read(query, {
                'node_id': node_ids[0],
                'namespace': self.namespace,
                'limit': limit
            })
        else:
            # 多个初始节点的情况
            query = f"""
            MATCH (start {{namespace: $namespace}})
            WHERE start.id IN $node_ids
            MATCH path = (start){pattern}(connected {{namespace: $namespace}})
            WHERE NOT connected.id IN $node_ids
            WITH DISTINCT connected, length(path) as hops
            ORDER BY hops ASC
            LIMIT $limit
            RETURN properties(connected) as props, hops
            """
            result = self.client.execute_read(query, {
                'node_ids': node_ids,
                'namespace': self.namespace,
                'limit': limit
            })
        
        nodes = []
        initial_nodes_dict = {nid: True for nid in node_ids}
        
        # 添加初始节点
        for node_id in node_ids:
            node_query = """
            MATCH (n {id: $node_id, namespace: $namespace})
            RETURN properties(n) as props
            """
            node_result = self.client.execute_read(node_query, {
                'node_id': node_id,
                'namespace': self.namespace
            })
            if node_result:
                props = node_result[0].get('props', {})
                if props:
                    node = dict(props)  # props已经是字典
                    node['hops'] = 0
                    node['is_initial'] = True
                    nodes.append(node)
        
        # 添加扩展节点
        for record in result:
            props = record.get('props', {})
            hops = record.get('hops', 1)
            if props:
                node_dict = dict(props)  # props已经是字典
                node_dict['hops'] = hops
                node_dict['is_initial'] = False
                # 避免重复添加初始节点
                if node_dict.get('id') not in initial_nodes_dict:
                    nodes.append(node_dict)
        
        return nodes
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        index_name: Optional[str] = None,
        label: Optional[str] = None,
        vector_top_k: int = 10,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        expand_limit: int = 50,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        混合搜索：向量搜索 + 图扩展
        
        Args:
            query_embedding: 查询向量
            index_name: 向量索引名称
            label: 节点标签
            vector_top_k: 向量搜索返回的初始节点数
            max_hops: 图扩展的最大跳数
            relationship_types: 关系类型过滤
            expand_limit: 扩展节点数量限制
            similarity_threshold: 相似度阈值
            
        Returns:
            Dict: 包含初始节点和扩展节点的结果
        """
        # 第一步：向量搜索找到初始节点
        initial_nodes = self.vector_search(
            query_embedding=query_embedding,
            index_name=index_name,
            label=label,
            top_k=vector_top_k,
            similarity_threshold=similarity_threshold
        )
        
        if not initial_nodes:
            return {
                'initial_nodes': [],
                'expanded_nodes': [],
                'total_nodes': 0
            }
        
        # 提取初始节点ID
        initial_node_ids = [node.get('id') for node in initial_nodes if node.get('id')]
        
        # 第二步：从初始节点扩展
        expanded_nodes = self.expand_from_nodes(
            node_ids=initial_node_ids,
            max_hops=max_hops,
            relationship_types=relationship_types,
            limit=expand_limit
        )
        
        # 分离初始节点和扩展节点
        expanded_only = [n for n in expanded_nodes if not n.get('is_initial', False)]
        
        return {
            'initial_nodes': initial_nodes,
            'expanded_nodes': expanded_only,
            'all_nodes': expanded_nodes,
            'total_nodes': len(expanded_nodes)
        }

