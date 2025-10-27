"""
Layer2存储管理

保存事件、状态、上下文节点到nodes.jsonl
创建fragment连接边到edges.jsonl
"""

import os
import json
from typing import Dict, Any, List


class Layer2Storage:
    """Layer2存储管理器"""
    
    def __init__(self, base_storage_dir: str = "storage"):
        self.base_storage_dir = base_storage_dir
    
    def get_storage_path(self, namespace: str) -> str:
        """根据命名空间获取存储路径"""
        return os.path.join(self.base_storage_dir, namespace)
    
    def save_timeline_node(
        self, 
        node: Dict[str, Any], 
        namespace: str,
        node_type: str  # "event", "state", "context"
    ) -> None:
        """
        保存时间线节点到nodes.jsonl
        
        Args:
            node: 节点数据
            namespace: 命名空间
            node_type: 节点类型
        """
        storage_path = self.get_storage_path(namespace)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 构建节点记录
        node_record = {
            "id": node.get('id'),
            "type": node_type,
            "content": node.get('content', ''),
            "conversation_time": node.get('conversation_time'),
            "relative_time": node.get('relative_time'),
            "layer": 2,
            "active": True
        }
        
        # 根据类型添加特定字段
        if node_type == "event":
            node_record['participants'] = node.get('participants', [])
            node_record['location'] = node.get('location')
        elif node_type == "state":
            node_record['participants'] = node.get('participants', [])
            node_record['duration'] = node.get('duration')
        elif node_type == "context":
            node_record['affected_entities'] = node.get('affected_entities', [])
            node_record['impact'] = node.get('impact')
        
        # 移除None值的字段
        node_record = {k: v for k, v in node_record.items() if v is not None}
        
        # 追加到nodes.jsonl
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
    
    def create_fragment_connection_edge(
        self,
        fragment_id: str,
        node_id: str,
        edge_type: str,  # "contains" or "occurs_in"
        namespace: str
    ) -> None:
        """
        创建fragment → timeline节点的连接边
        
        Args:
            fragment_id: fragment的ID
            node_id: timeline节点的ID
            edge_type: 边类型（contains/occurs_in）
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        edge = {
            "id": f"edge_{fragment_id}_{node_id}",
            "source": fragment_id,
            "target": node_id,
            "type": edge_type,
            "active": True
        }
        
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')
    
    def create_structural_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        namespace: str
    ) -> None:
        """
        创建结构性边（Layer2节点到entity的连接）
        
        这些边只表示关系，不需要content和embedding
        
        Args:
            source_id: 源节点ID（event/state/context）
            target_id: 目标节点ID（entity）
            edge_type: 边类型（involves/describes/affects）
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        edge = {
            "id": f"edge_{source_id}_{target_id}",
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "active": True
            # 注意：结构性边不需要content、layer等字段
        }
        
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')

