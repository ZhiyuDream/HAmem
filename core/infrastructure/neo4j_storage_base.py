"""
Neo4j 存储基类

提供通用的 Neo4j 存储操作方法
"""

from typing import Dict, Any, List, Optional
from .neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


class Neo4jStorageBase:
    """Neo4j 存储基类"""
    
    def __init__(self, neo4j_client: Neo4jClient, namespace: str = "default"):
        """
        初始化存储基类
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间（用于数据隔离）
        """
        self.client = neo4j_client
        self.namespace = namespace
    
    def create_node(
        self,
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> None:
        """
        创建节点
        
        Args:
            node_id: 节点ID（作为唯一标识）
            labels: 节点标签列表
            properties: 节点属性
        """
        # 使用 MERGE 确保节点唯一性（基于 id 属性）
        labels_str = ":".join(labels)
        props = properties.copy()
        
        # 构建 SET 子句（只包含 properties 中的字段）
        set_clauses = ", ".join([f"n.{k} = ${k}" for k in properties.keys()]) if properties else ""
        
        if set_clauses:
            query = f"""
            MERGE (n:{labels_str} {{id: $node_id, namespace: $namespace}})
            SET {set_clauses}
            """
        else:
            query = f"""
            MERGE (n:{labels_str} {{id: $node_id, namespace: $namespace}})
            """
        
        # 构建参数字典：包含 node_id, namespace 和所有 properties
        params = {'node_id': node_id, 'namespace': self.namespace, **props}
        
        self.client.execute_write(query, params)
    
    def update_node(
        self,
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> None:
        """
        更新节点（与 create_node 相同，使用 MERGE）
        
        Args:
        node_id: 节点ID
        labels: 节点标签列表
        properties: 要更新的属性
        """
        self.create_node(node_id, labels, properties)
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        创建关系
        
        Args:
        source_id: 源节点ID
        target_id: 目标节点ID
        rel_type: 关系类型
        properties: 关系属性（可选）
        """
        props = properties or {}
        props['namespace'] = self.namespace
        
        # 构建关系属性字符串
        if props:
            rel_props_str = " {" + ", ".join([f"{k}: ${k}" for k in props.keys()]) + "}"
        else:
            rel_props_str = ""
        
        query = f"""
        MATCH (a {{id: $source_id, namespace: $namespace}})
        MATCH (b {{id: $target_id, namespace: $namespace}})
        MERGE (a)-[r:{rel_type.upper()}{rel_props_str}]->(b)
        """
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "namespace": self.namespace,
            **props
        }
        
        self.client.execute_write(query, params)
    
    def get_node(self, node_id: str, labels: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        获取节点
        
        Args:
        node_id: 节点ID
        labels: 可选的标签过滤
        
        Returns:
        节点数据，如果不存在返回 None
        """
        if labels:
            labels_str = ":".join(labels)
            query = f"""
            MATCH (n:{labels_str} {{id: $node_id, namespace: $namespace}})
            RETURN n
            """
        else:
            query = """
            MATCH (n {id: $node_id, namespace: $namespace})
            RETURN n
            """
        
        result = self.client.execute_read(query, {"node_id": node_id, "namespace": self.namespace})
        if result:
            node = result[0].get('n')
        if node:
            return dict(node)
        return None
    
    def query_nodes(
        self,
        labels: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        查询节点
        
        Args:
        labels: 节点标签过滤
        filters: 属性过滤条件
        limit: 返回数量限制
        
        Returns:
        节点列表
        """
        if labels:
            labels_str = ":".join(labels)
            query = f"MATCH (n:{labels_str} {{namespace: $namespace}})"
        else:
            query = "MATCH (n {namespace: $namespace})"
        
            # 添加过滤条件
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(f"n.{key} = ${key}")
            if filter_conditions:
                query += " WHERE " + " AND ".join(filter_conditions)
        
        query += " RETURN n"
        
        if limit:
            query += f" LIMIT $limit"
        
        params = {"namespace": self.namespace}
        if filters:
            params.update(filters)
        if limit:
            params["limit"] = limit
        
        result = self.client.execute_read(query, params)
        nodes = []
        for record in result:
            node = record.get('n')
        if node:
            nodes.append(dict(node))
        return nodes
    
    def delete_node(self, node_id: str) -> None:
        """
        删除节点（同时删除相关关系）
        
        Args:
        node_id: 节点ID
        """
        query = """
        MATCH (n {id: $node_id, namespace: $namespace})
        DETACH DELETE n
            """
        self.client.execute_write(query, {"node_id": node_id, "namespace": self.namespace})
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
        统计信息字典
        """
            # 统计节点数量
        node_query = """
        MATCH (n {namespace: $namespace})
        RETURN count(n) as node_count
            """
        node_result = self.client.execute_read(node_query, {"namespace": self.namespace})
        node_count = node_result[0].get("node_count", 0) if node_result else 0
        
            # 统计关系数量
        rel_query = """
        MATCH ()-[r {namespace: $namespace}]->()
        RETURN count(r) as rel_count
            """
        rel_result = self.client.execute_read(rel_query, {"namespace": self.namespace})
        rel_count = rel_result[0].get("rel_count", 0) if rel_result else 0
        
        return {
            "namespace": self.namespace,
            "node_count": node_count,
            "relationship_count": rel_count
        }

