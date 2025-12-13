"""
Layer2 Neo4j 存储管理

保存事件、状态、上下文节点到 Neo4j
创建 fragment 连接边
"""

from typing import Dict, Any, List
from ..infrastructure.neo4j_storage_base import Neo4jStorageBase
from ..infrastructure.neo4j_client import Neo4jClient
import logging
import os

logger = logging.getLogger(__name__)


class Layer2Neo4jStorage(Neo4jStorageBase):
    """Layer2 Neo4j 存储管理器"""
    
    def __init__(self, neo4j_client: Neo4jClient, namespace: str = "default"):
        """
        初始化 Layer2 存储
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间（通常使用输入文件名）
        """
        super().__init__(neo4j_client, namespace)
    
    def get_storage_path(self, namespace: str) -> str:
        """根据命名空间获取存储路径（Neo4j 中不需要，保留接口兼容性）"""
        return namespace
    
    def save_timeline_node(
        self, 
        node: Dict[str, Any], 
        namespace: str,
        node_type: str  # "event", "state", "context"
    ) -> None:
        """
        保存时间线节点到 Neo4j
        
        Args:
            node: 节点数据
            namespace: 命名空间
            node_type: 节点类型
        """
        self.namespace = namespace
        
        node_id = node.get('id')
        if not node_id:
            logger.warning("Timeline node missing id, skipping")
            return
        
        # 构建节点属性
        properties = {
            "content": node.get('content', ''),
            "conversation_time": node.get('conversation_time'),
            "relative_time": node.get('relative_time'),
            "layer": 2,
            "active": True,
            "type": node_type
        }
        
        # 根据类型添加特定字段
        if node_type == "event":
            participants = node.get('participants', [])
            if participants:
                properties['participants'] = participants
            if node.get('location'):
                properties['location'] = node.get('location')
        elif node_type == "state":
            participants = node.get('participants', [])
            if participants:
                properties['participants'] = participants
            if node.get('duration'):
                properties['duration'] = node.get('duration')
        elif node_type == "context":
            affected_entities = node.get('affected_entities', [])
            if affected_entities:
                properties['affected_entities'] = affected_entities
            if node.get('impact'):
                properties['impact'] = node.get('impact')
        
        # 移除 None 值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # 创建节点，标签为对应的类型和 Layer2
        labels = [node_type.capitalize(), "Layer2"]
        self.create_node(
            node_id=node_id,
            labels=labels,
            properties=properties
        )
    
    def create_fragment_connection_edge(
        self,
        fragment_id: str,
        node_id: str,
        edge_type: str,  # "contains" or "occurs_in"
        namespace: str
    ) -> None:
        """
        创建 fragment → timeline 节点的连接边
        
        Args:
            fragment_id: fragment 的 ID
            node_id: timeline 节点的 ID
            edge_type: 边类型（contains/occurs_in）
            namespace: 命名空间
        """
        self.namespace = namespace
        
        # 将边类型转换为大写（Neo4j 关系类型通常使用大写）
        rel_type = edge_type.upper()
        
        self.create_relationship(
            source_id=fragment_id,
            target_id=node_id,
            rel_type=rel_type,
            properties={"active": True}
        )
    
    def create_structural_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        namespace: str
    ) -> None:
        """
        创建结构性边（Layer2 节点到 entity 的连接）
        
        这些边只表示关系，不需要 content 和 embedding
        
        Args:
            source_id: 源节点 ID（event/state/context）
            target_id: 目标节点 ID（entity）
            edge_type: 边类型（involves/describes/affects）
            namespace: 命名空间
        """
        self.namespace = namespace
        
        # 将边类型转换为大写
        rel_type = edge_type.upper()
        
        self.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            properties={"active": True}
        )

