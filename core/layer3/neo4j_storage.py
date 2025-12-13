"""
Layer3 Neo4j 存储管理

保存事件聚类、模式、偏好、行为规则节点
"""

from typing import Dict, Any
from ..infrastructure.neo4j_storage_base import Neo4jStorageBase
from ..infrastructure.neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


class Layer3Neo4jStorage(Neo4jStorageBase):
    """Layer3 Neo4j 存储管理器"""
    
    def __init__(self, neo4j_client: Neo4jClient, namespace: str = "default"):
        """
        初始化 Layer3 存储
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间（通常使用输入文件名）
        """
        super().__init__(neo4j_client, namespace)
    
    def get_storage_path(self, namespace: str) -> str:
        """根据命名空间获取存储路径（Neo4j 中不需要，保留接口兼容性）"""
        return namespace
    
    def save_event_cluster(
        self,
        cluster: Dict[str, Any],
        namespace: str
    ) -> None:
        """
        保存事件聚类节点到 Neo4j
        
        Args:
            cluster: 聚类数据
            namespace: 命名空间
        """
        self.namespace = namespace
        
        cluster_id = cluster.get('id')
        if not cluster_id:
            logger.warning("Event cluster missing id, skipping")
            return
        
        # 构建节点属性
        properties = {
            "content": cluster.get('content', ''),
            "cluster_type": cluster.get('cluster_type'),
            "participants": cluster.get('participants', []),
            "time_span": cluster.get('time_span'),
            "significance": cluster.get('significance'),
            "layer": 3,
            "active": True,
            "type": "event_cluster"
        }
        
        # 移除 None 值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # 创建节点，标签为 EventCluster 和 Layer3
        self.create_node(
            node_id=cluster_id,
            labels=["EventCluster", "Layer3"],
            properties=properties
        )
    
    def save_pattern(
        self,
        pattern: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存模式节点"""
        self.namespace = namespace
        
        pattern_id = pattern.get('id')
        if not pattern_id:
            logger.warning("Pattern missing id, skipping")
            return
        
        properties = {
            "person": pattern.get('person'),
            "pattern_type": pattern.get('pattern_type'),
            "content": pattern.get('content', ''),
            "layer": 3,
            "active": True,
            "type": "pattern"
        }
        
        properties = {k: v for k, v in properties.items() if v is not None}
        
        self.create_node(
            node_id=pattern_id,
            labels=["Pattern", "Layer3"],
            properties=properties
        )
    
    def save_preference(
        self,
        preference: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存偏好节点"""
        self.namespace = namespace
        
        preference_id = preference.get('id')
        if not preference_id:
            logger.warning("Preference missing id, skipping")
            return
        
        properties = {
            "person": preference.get('person'),
            "category": preference.get('category'),
            "content": preference.get('content', ''),
            "layer": 3,
            "active": True,
            "type": "preference"
        }
        
        properties = {k: v for k, v in properties.items() if v is not None}
        
        self.create_node(
            node_id=preference_id,
            labels=["Preference", "Layer3"],
            properties=properties
        )
    
    def save_behavior_rule(
        self,
        rule: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存行为规则节点"""
        self.namespace = namespace
        
        rule_id = rule.get('id')
        if not rule_id:
            logger.warning("Behavior rule missing id, skipping")
            return
        
        properties = {
            "person": rule.get('person'),
            "rule_type": rule.get('rule_type'),
            "content": rule.get('content', ''),
            "layer": 3,
            "active": True,
            "type": "behavior_rule"
        }
        
        properties = {k: v for k, v in properties.items() if v is not None}
        
        self.create_node(
            node_id=rule_id,
            labels=["BehaviorRule", "Layer3"],
            properties=properties
        )
    
    def create_cluster_event_edge(
        self,
        cluster_id: str,
        event_id: str,
        namespace: str
    ) -> None:
        """
        创建 cluster → event 的连接边
        
        Args:
            cluster_id: cluster 的 ID
            event_id: event 的 ID
            namespace: 命名空间
        """
        self.namespace = namespace
        
        self.create_relationship(
            source_id=cluster_id,
            target_id=event_id,
            rel_type="CONTAINS",
            properties={"active": True}
        )
    
    def create_pattern_person_edge(
        self,
        pattern_id: str,
        person_name: str,
        namespace: str
    ) -> None:
        """
        创建 pattern → person 的连接边
        
        Args:
            pattern_id: pattern/preference/rule 的 ID
            person_name: 人名（作为 entity 节点的 id）
            namespace: 命名空间
        """
        self.namespace = namespace
        
        self.create_relationship(
            source_id=pattern_id,
            target_id=person_name,
            rel_type="DESCRIBES",
            properties={"active": True}
        )

