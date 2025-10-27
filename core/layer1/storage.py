"""
Layer1存储管理

管理实体和关系的存储
"""

import os
import json
from typing import Dict, List, Any, Optional


class Layer1Storage:
    """Layer1存储管理器"""
    
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = storage_dir
    
    def get_storage_path(self, input_filename: str) -> str:
        """
        获取存储路径
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            存储路径
        """
        # 从文件名中提取基础名称
        base_name = os.path.splitext(input_filename)[0]
        return os.path.join(self.storage_dir, base_name)
    
    def initialize_storage(self, input_filename: str) -> None:
        """
        初始化存储目录
        
        Args:
            input_filename: 输入文件名
        """
        storage_path = self.get_storage_path(input_filename)
        os.makedirs(storage_path, exist_ok=True)
        
        # 创建nodes和edges文件（与fragment模块保持一致）
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        # 如果文件不存在，创建空文件
        if not os.path.exists(nodes_file):
            with open(nodes_file, 'w', encoding='utf-8') as f:
                pass
        
        if not os.path.exists(edges_file):
            with open(edges_file, 'w', encoding='utf-8') as f:
                pass
    
    def save_entity(self, entity: Dict[str, Any], input_filename: str) -> None:
        """
        保存实体到nodes.jsonl
        
        Args:
            entity: 实体数据
            input_filename: 输入文件名（namespace）
        """
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 构建实体记录（使用传入的完整entity数据）
        entity_record = {
            "id": entity.get('id'),  # 使用传入的ID
            "type": "entity",
            "name": entity.get('name', ''),
            "content": entity.get('content', ''),  # 使用content字段
            "layer": entity.get('layer', 1),
            "active": entity.get('active', True)
        }
        
        # 追加到nodes.jsonl文件
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entity_record, ensure_ascii=False) + '\n')
    
    def update_node(self, node_id: str, content: str = None, namespace: str = None, **kwargs) -> None:
        """
        更新节点（追加写入新版本）
        
        Args:
            node_id: 节点ID
            content: 新的content
            namespace: 命名空间
            **kwargs: 其他要更新的字段
        """
        storage_path = self.get_storage_path(namespace or "default")
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 读取现有节点
        existing_node = None
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        node = json.loads(line.strip())
                        if node.get('id') == node_id:
                            existing_node = node
                            break
        
        if not existing_node:
            print(f"⚠️  节点不存在: {node_id}")
            return
        
        # 更新节点
        updated_node = {**existing_node}
        if content:
            updated_node['content'] = content
        for key, value in kwargs.items():
            updated_node[key] = value
        
        # 追加写入（新版本）
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(updated_node, ensure_ascii=False) + '\n')
    
    def update_edge(self, edge_id: str, content: str = None, namespace: str = None, **kwargs) -> None:
        """
        更新边（追加写入新版本）
        
        Args:
            edge_id: 边ID
            content: 新的content
            namespace: 命名空间
            **kwargs: 其他要更新的字段
        """
        storage_path = self.get_storage_path(namespace or "default")
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        # 读取现有边
        existing_edge = None
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        edge = json.loads(line.strip())
                        if edge.get('id') == edge_id:
                            existing_edge = edge
                            break
        
        if not existing_edge:
            print(f"⚠️  边不存在: {edge_id}")
            return
        
        # 更新边
        updated_edge = {**existing_edge}
        if content:
            updated_edge['content'] = content
        for key, value in kwargs.items():
            updated_edge[key] = value
        
        # 追加写入（新版本）
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(updated_edge, ensure_ascii=False) + '\n')
    
    def save_relationship(self, relationship: Dict[str, Any], namespace: str) -> None:
        """
        保存关系到edges.jsonl
        
        Args:
            relationship: 关系数据
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        # 构建关系记录（使用传入的ID）
        relationship_record = {
            "id": relationship.get('id'),  # 使用传入的ID（edge_1, edge_2等）
            "source": relationship.get('source', ''),
            "target": relationship.get('target', ''),
            "type": "relationship",  # 添加type字段
            "content": relationship.get('content', ''),
            "layer": relationship.get('layer', 1),
            "active": relationship.get('active', True)
        }
        
        # 追加到edges.jsonl文件
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(relationship_record, ensure_ascii=False) + '\n')
    
    def get_entities(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取所有实体（从nodes.jsonl中筛选type=entity的节点）
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            实体列表
        """
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        entities = []
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            node = json.loads(line.strip())
                            if node.get('type') == 'entity':
                                entities.append(node)
                        except json.JSONDecodeError:
                            continue
        
        return entities
    
    def get_relationships(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取所有关系（从edges.jsonl中读取）
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            关系列表
        """
        storage_path = self.get_storage_path(input_filename)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        relationships = []
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            relationship = json.loads(line.strip())
                            relationships.append(relationship)
                        except json.JSONDecodeError:
                            continue
        
        return relationships
    
    def get_storage_stats(self, input_filename: str) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            统计信息
        """
        entities = self.get_entities(input_filename)
        relationships = self.get_relationships(input_filename)
        
        return {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "storage_path": self.get_storage_path(input_filename),
            "nodes_file_exists": os.path.exists(os.path.join(self.get_storage_path(input_filename), "nodes.jsonl")),
            "edges_file_exists": os.path.exists(os.path.join(self.get_storage_path(input_filename), "edges.jsonl"))
        }
