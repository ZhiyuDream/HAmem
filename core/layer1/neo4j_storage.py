"""
Layer1 Neo4j 存储管理

管理实体和关系的 Neo4j 存储
"""

from typing import Dict, List, Any, Optional
from ..infrastructure.neo4j_storage_base import Neo4jStorageBase
from ..infrastructure.neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


class Layer1Neo4jStorage(Neo4jStorageBase):
    """Layer1 Neo4j 存储管理器"""
    
    def __init__(self, neo4j_client: Neo4jClient, namespace: str = "default"):
        """
        初始化 Layer1 存储
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间（通常使用输入文件名）
        """
        super().__init__(neo4j_client, namespace)
    
    def initialize_storage(self, input_filename: str) -> None:
        """
        初始化存储（Neo4j 中不需要创建文件，只需设置命名空间）
        
        Args:
            input_filename: 输入文件名
        """
        # 从文件名中提取基础名称作为命名空间
        import os
        base_name = os.path.splitext(input_filename)[0]
        self.namespace = base_name
        logger.info(f"Layer1 storage initialized with namespace: {self.namespace}")
    
    def save_entity(self, entity: Dict[str, Any], input_filename: str) -> None:
        """
        保存实体到 Neo4j
        
        Args:
            entity: 实体数据
            input_filename: 输入文件名（namespace）
        """
        # 更新命名空间
        import os
        base_name = os.path.splitext(input_filename)[0]
        self.namespace = base_name
        
        entity_id = entity.get('id')
        if not entity_id:
            logger.warning("Entity missing id, skipping")
            return
        
        # 构建节点属性
        properties = {
            "name": entity.get('name', ''),
            "content": entity.get('content', ''),
            "layer": entity.get('layer', 1),
            "active": entity.get('active', True),
            "type": "entity"
        }
        
        # 移除 None 值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # 创建节点，标签为 Entity 和 Layer1
        self.create_node(
            node_id=entity_id,
            labels=["Entity", "Layer1"],
            properties=properties
        )
    
    def update_node(self, node_id: str, content: str = None, namespace: str = None, **kwargs) -> None:
        """
        更新节点
        
        Args:
            node_id: 节点ID
            content: 新的content
            namespace: 命名空间
            **kwargs: 其他要更新的字段
        """
        # 如果提供了 namespace，更新存储的 namespace
        if namespace:
            self.namespace = namespace
        else:
            # 如果没有提供，使用当前的 namespace
            pass
        
        # 构建更新属性
        properties = {}
        if content is not None:
            properties["content"] = content
        properties.update(kwargs)
        
        if not properties:
            return
        
        # 更新节点
        set_clauses = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"""
        MATCH (n {{id: $node_id, namespace: $namespace}})
        SET {set_clauses}
        RETURN n
        """
        
        params = {
            "node_id": node_id,
            "namespace": self.namespace,
            **properties
        }
        result = self.client.execute_write(query, params)
        
        if result:
            logger.info(f"✅ 节点 {node_id} 更新成功")
        else:
            logger.warning(f"⚠️  节点 {node_id} 更新失败或节点不存在")
    
    def update_edge(self, edge_id: str, content: str = None, namespace: str = None, **kwargs) -> None:
        """
        更新边
        
        Args:
            edge_id: 边ID（格式：edge_source_target）
            content: 新的content
            namespace: 命名空间
            **kwargs: 其他要更新的字段
        """
        if namespace:
            self.namespace = namespace
        
        # 构建更新属性
        properties = {}
        if content is not None:
            properties["content"] = content
        properties.update(kwargs)
        
        if not properties:
            return
        
        # 更新关系（通过 source 和 target 查找）
        # edge_id 格式通常是 "edge_source_target"
        parts = edge_id.split("_", 2)
        if len(parts) >= 3:
            source_id = parts[1]
            target_id = parts[2]
            
            set_clauses = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (a {{id: $source_id, namespace: $namespace}})-[r {{namespace: $namespace}}]->(b {{id: $target_id, namespace: $namespace}})
            SET {set_clauses}
            """
            
            params = {
                "source_id": source_id,
                "target_id": target_id,
                "namespace": self.namespace,
                **properties
            }
            self.client.execute_write(query, params)
    
    def save_relationship(self, relationship: Dict[str, Any], namespace: str) -> None:
        """
        保存关系到 Neo4j
        
        Args:
            relationship: 关系数据
            namespace: 命名空间
        """
        self.namespace = namespace
        
        source_id = relationship.get('source')
        target_id = relationship.get('target')
        rel_type = relationship.get('type', 'RELATED_TO')
        
        if not source_id or not target_id:
            logger.warning("Relationship missing source or target, skipping")
            return
        
        # 构建关系属性
        properties = {
            "content": relationship.get('content', ''),
            "layer": relationship.get('layer', 1),
            "active": relationship.get('active', True)
        }
        
        # 移除 None 值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # 创建关系
        self.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            properties=properties
        )
    
    def get_entities(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取所有实体
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            实体列表
        """
        import os
        base_name = os.path.splitext(input_filename)[0]
        self.namespace = base_name
        
        return self.query_nodes(
            labels=["Entity", "Layer1"],
            filters={"type": "entity", "active": True}
        )
    
    def get_relationships(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取所有关系
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            关系列表
        """
        import os
        base_name = os.path.splitext(input_filename)[0]
        self.namespace = base_name
        
        # 查询关系：通过节点的 namespace 来过滤，关系可能没有 namespace 属性
        query = """
        MATCH (a {namespace: $namespace})-[r]->(b {namespace: $namespace})
        RETURN a.id as source, b.id as target, type(r) as type, r as properties
        """
        result = self.client.execute_read(query, {"namespace": self.namespace})
        
        relationships = []
        for record in result:
            rel = {
                "source": record.get("source"),
                "target": record.get("target"),
                "type": record.get("type"),
            }
            # 添加关系属性
            props = record.get("properties")
            if props:
                rel.update(dict(props))
            relationships.append(rel)
        
        return relationships
    
    def get_storage_stats(self, input_filename: str) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            统计信息
        """
        import os
        base_name = os.path.splitext(input_filename)[0]
        self.namespace = base_name
        
        stats = self.get_stats()
        
        # 添加实体和关系的详细统计
        entities = self.get_entities(input_filename)
        relationships = self.get_relationships(input_filename)
        
        stats.update({
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "namespace": self.namespace
        })
        
        return stats

