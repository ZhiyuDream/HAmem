"""
Neo4j 存储增强模块

在保存节点时自动生成并设置 embedding
"""

from typing import Dict, Any, Optional, List
from .neo4j_storage_base import Neo4jStorageBase
from .neo4j_client import Neo4jClient
from .embedding import EmbeddingManager
from config import Config
import logging

logger = logging.getLogger(__name__)


class Neo4jStorageWithEmbedding(Neo4jStorageBase):
    """带自动 embedding 生成的 Neo4j 存储基类"""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        namespace: str = "default",
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[Config] = None,
        auto_generate_embedding: bool = True
    ):
        """
        初始化存储（带 embedding 支持）
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间
            embedding_manager: Embedding管理器
            config: 配置对象（如果未提供 embedding_manager）
            auto_generate_embedding: 是否自动生成 embedding
        """
        super().__init__(neo4j_client, namespace)
        
        # 初始化 EmbeddingManager
        if embedding_manager is None and config is not None:
            try:
                self.embedding_manager = EmbeddingManager(config)
            except Exception as e:
                logger.warning(f"无法初始化 EmbeddingManager: {e}")
                self.embedding_manager = None
        else:
            self.embedding_manager = embedding_manager
        
        self.auto_generate_embedding = auto_generate_embedding and self.embedding_manager is not None
    
    def create_node_with_embedding(
        self,
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any],
        embedding_text: Optional[str] = None
    ) -> None:
        """
        创建节点并自动生成 embedding
        
        Args:
            node_id: 节点ID
            labels: 节点标签列表
            properties: 节点属性
            embedding_text: 用于生成 embedding 的文本（如果为 None，使用 content 字段）
        """
        # 创建节点
        self.create_node(node_id, labels, properties)
        
        # 自动生成 embedding
        if self.auto_generate_embedding:
            try:
                # 确定用于生成 embedding 的文本
                text = embedding_text or properties.get('content', '')
                
                if text:
                    # 使用 OpenAI API 生成 embedding
                    embedding = self.embedding_manager.get_embedding(text)
                    
                    # 设置 embedding 到节点
                    from .neo4j_vector_search import Neo4jVectorSearch
                    vector_search = Neo4jVectorSearch(self.client, self.namespace)
                    vector_search.set_node_embedding(node_id, embedding, labels)
                    
                    logger.debug(f"✅ 为节点 {node_id} 生成并设置 embedding")
                else:
                    logger.warning(f"⚠️  节点 {node_id} 没有可用于生成 embedding 的文本")
            except Exception as e:
                logger.error(f"❌ 为节点 {node_id} 生成 embedding 失败: {e}")

