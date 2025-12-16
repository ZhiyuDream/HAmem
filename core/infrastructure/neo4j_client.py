"""
Neo4j 连接管理模块

提供 Neo4j 数据库连接和基本操作
"""

import os
from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List
import logging

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有 python-dotenv，手动加载 .env 文件
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


class Neo4jClient:
    """Neo4j 客户端"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        初始化 Neo4j 客户端
        
        Args:
            uri: Neo4j 连接 URI（如果为 None，从环境变量 NEO4J_URI 读取）
            username: 用户名（如果为 None，从环境变量 NEO4J_USERNAME 读取）
            password: 密码（如果为 None，从环境变量 NEO4J_PASSWORD 读取）
            database: 数据库名称（如果为 None，从环境变量 NEO4J_DATABASE 读取，默认 neo4j）
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', '')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver: Optional[GraphDatabase.driver] = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        连接到 Neo4j 数据库
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # 验证连接
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            # 检测Neo4j版本（社区版不支持向量索引）
            self.is_community_edition = self._check_community_edition()
            if self.is_community_edition:
                self.logger.info(f"✅ Connected to Neo4j Community Edition at {self.uri}")
                self.logger.warning("⚠️  Neo4j社区版不支持原生向量索引，将使用cache（FAISS）进行向量搜索")
            else:
                self.logger.info(f"✅ Connected to Neo4j Enterprise Edition at {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def _check_community_edition(self) -> bool:
        """
        检查Neo4j是否为社区版
        
        Returns:
            bool: 如果是社区版返回True，否则返回False
        """
        try:
            # 方法1: 检查版本信息
            query = "CALL dbms.components() YIELD name, versions, edition WHERE name = 'Neo4j Kernel' RETURN edition"
            result = self.execute_read(query, {})
            if result and len(result) > 0:
                edition = result[0].get('edition', '').lower()
                if 'community' in edition:
                    return True
                # 如果是企业版，返回False
                if 'enterprise' in edition:
                    return False
            
            # 方法2: 尝试检查是否支持向量索引（更直接的方法）
            # 如果无法创建向量索引，很可能是社区版
            test_query = """
            CREATE VECTOR INDEX test_vector_index_check IF NOT EXISTS
            FOR (n:TestNode)
            ON (n.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 128,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
            try:
                self.execute_write(test_query, {})
                # 如果成功创建，说明是企业版，清理测试索引
                cleanup_query = "DROP INDEX test_vector_index_check IF EXISTS"
                try:
                    self.execute_write(cleanup_query, {})
                except:
                    pass
                return False  # 企业版
            except Exception as e:
                error_msg = str(e).lower()
                if 'vector' in error_msg or 'index' in error_msg:
                    # 如果错误与向量索引相关，很可能是社区版
                    return True
                # 其他错误，保守假设是社区版
                self.logger.warning(f"无法确定Neo4j版本，假设为社区版: {e}")
                return True
        except Exception as e:
            # 如果所有检测方法都失败，保守假设是社区版（更安全）
            self.logger.warning(f"无法检测Neo4j版本，假设为社区版: {e}")
            return True
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def execute_write(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        执行写操作（CREATE, UPDATE, DELETE）
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            List[Dict]: 查询结果
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, parameters or {}))
            )
            return [record.data() for record in result]
    
    def execute_read(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        执行读操作（MATCH, RETURN）
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            List[Dict]: 查询结果
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        
        with self.driver.session(database=self.database) as session:
            result = session.execute_read(
                lambda tx: list(tx.run(query, parameters or {}))
            )
            return [record.data() for record in result]
    
    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> str:
        """
        创建节点
        
        Args:
            labels: 节点标签列表（如 ['Person', 'Entity']）
            properties: 节点属性
            node_id: 可选的节点ID（如果提供，将作为 id 属性）
            
        Returns:
            str: 创建的节点ID
        """
        labels_str = ":".join(labels)
        props = properties.copy()
        
        if node_id:
            props['id'] = node_id
        
        # 构建属性字符串
        props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        
        # 使用 elementId 替代已弃用的 id() 函数
        # 如果提供了 node_id，直接使用它；否则使用 elementId 作为后备
        if node_id:
            query = f"CREATE (n:{labels_str} {{{props_str}}}) RETURN n.id as id, elementId(n) as element_id"
        else:
            query = f"CREATE (n:{labels_str} {{{props_str}}}) RETURN elementId(n) as element_id"
        
        result = self.execute_write(query, props)
        
        if result:
            # 优先使用应用生成的 id，如果没有则使用 elementId
            return result[0].get('id') or result[0].get('element_id')
        return None
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        创建关系
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型
            properties: 关系属性
            
        Returns:
            bool: 是否创建成功
        """
        props = properties or {}
        props_str = ""
        if props:
            props_str = " {" + ", ".join([f"{k}: ${k}" for k in props.keys()]) + "}"
        
        query = (
            f"MATCH (a), (b) "
            f"WHERE a.id = $source_id AND b.id = $target_id "
            f"CREATE (a)-[r:{rel_type}{props_str}]->(b) "
            f"RETURN r"
        )
        
        params = {
            'source_id': source_id,
            'target_id': target_id,
            **props
        }
        
        result = self.execute_write(query, params)
        return len(result) > 0
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            Dict: 节点数据，如果不存在返回 None
        """
        query = "MATCH (n) WHERE n.id = $node_id RETURN n"
        result = self.execute_read(query, {'node_id': node_id})
        
        if result:
            node = result[0]['n']
            return dict(node)
        return None
    
    def get_nodes_by_type(self, node_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据类型获取节点列表
        
        Args:
            node_type: 节点类型（标签）
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 节点列表
        """
        query = f"MATCH (n:{node_type}) RETURN n LIMIT $limit"
        result = self.execute_read(query, {'limit': limit})
        
        return [dict(record['n']) for record in result]
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        更新节点属性
        
        Args:
            node_id: 节点ID
            properties: 要更新的属性
            
        Returns:
            bool: 是否更新成功
        """
        if not properties:
            return False
        
        set_clauses = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MATCH (n) WHERE n.id = $node_id SET {set_clauses} RETURN n"
        
        params = {'node_id': node_id, **properties}
        result = self.execute_write(query, params)
        
        return len(result) > 0
    
    def delete_node(self, node_id: str, delete_relationships: bool = True) -> bool:
        """
        删除节点
        
        Args:
            node_id: 节点ID
            delete_relationships: 是否同时删除关系
            
        Returns:
            bool: 是否删除成功
        """
        if delete_relationships:
            query = "MATCH (n) WHERE n.id = $node_id DETACH DELETE n RETURN count(n) as deleted"
        else:
            query = "MATCH (n) WHERE n.id = $node_id DELETE n RETURN count(n) as deleted"
        
        result = self.execute_write(query, {'node_id': node_id})
        
        if result:
            return result[0].get('deleted', 0) > 0
        return False
    
    def clear_database(self):
        """清空数据库（谨慎使用！）"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write(query)
        self.logger.warning("Database cleared!")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        node_count_query = "MATCH (n) RETURN count(n) as node_count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        
        node_count = self.execute_read(node_count_query)[0].get('node_count', 0)
        rel_count = self.execute_read(rel_count_query)[0].get('rel_count', 0)
        
        return {
            'node_count': node_count,
            'relationship_count': rel_count,
            'uri': self.uri,
            'database': self.database
        }
    
