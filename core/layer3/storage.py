"""
Layer3存储管理

保存事件聚类、模式、偏好、行为规则节点
"""

import os
import json
from typing import Dict, Any, List


class Layer3Storage:
    """Layer3存储管理器"""
    
    def __init__(self, base_storage_dir: str = "storage"):
        self.base_storage_dir = base_storage_dir
    
    def get_storage_path(self, namespace: str) -> str:
        """根据命名空间获取存储路径"""
        return os.path.join(self.base_storage_dir, namespace)
    
    def save_event_cluster(
        self,
        cluster: Dict[str, Any],
        namespace: str
    ) -> None:
        """
        保存事件聚类节点到nodes.jsonl
        
        Args:
            cluster: 聚类数据
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 构建节点记录
        node_record = {
            "id": cluster.get('id'),
            "type": "event_cluster",
            "content": cluster.get('content', ''),
            "cluster_type": cluster.get('cluster_type'),
            "participants": cluster.get('participants', []),
            "time_span": cluster.get('time_span'),
            "significance": cluster.get('significance'),
            "layer": 3,
            "active": True
        }
        
        # 移除None值的字段
        node_record = {k: v for k, v in node_record.items() if v is not None}
        
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
    
    def save_pattern(
        self,
        pattern: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存模式节点"""
        storage_path = self.get_storage_path(namespace)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        node_record = {
            "id": pattern.get('id'),
            "type": "pattern",
            "person": pattern.get('person'),
            "pattern_type": pattern.get('pattern_type'),
            "content": pattern.get('content', ''),
            "layer": 3,
            "active": True
        }
        
        node_record = {k: v for k, v in node_record.items() if v is not None}
        
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
    
    def save_preference(
        self,
        preference: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存偏好节点"""
        storage_path = self.get_storage_path(namespace)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        node_record = {
            "id": preference.get('id'),
            "type": "preference",
            "person": preference.get('person'),
            "category": preference.get('category'),
            "content": preference.get('content', ''),
            "layer": 3,
            "active": True
        }
        
        node_record = {k: v for k, v in node_record.items() if v is not None}
        
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
    
    def save_behavior_rule(
        self,
        rule: Dict[str, Any],
        namespace: str
    ) -> None:
        """保存行为规则节点"""
        storage_path = self.get_storage_path(namespace)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        node_record = {
            "id": rule.get('id'),
            "type": "behavior_rule",
            "person": rule.get('person'),
            "rule_type": rule.get('rule_type'),
            "content": rule.get('content', ''),
            "layer": 3,
            "active": True
        }
        
        node_record = {k: v for k, v in node_record.items() if v is not None}
        
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_record, ensure_ascii=False) + '\n')
    
    def create_cluster_event_edge(
        self,
        cluster_id: str,
        event_id: str,
        namespace: str
    ) -> None:
        """
        创建 cluster → event 的连接边
        
        Args:
            cluster_id: cluster的ID
            event_id: event的ID
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        edge = {
            "id": f"edge_{cluster_id}_{event_id}",
            "source": cluster_id,
            "target": event_id,
            "type": "contains",
            "active": True
        }
        
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')
    
    def create_pattern_person_edge(
        self,
        pattern_id: str,
        person_name: str,
        namespace: str
    ) -> None:
        """
        创建 pattern → person 的连接边
        
        Args:
            pattern_id: pattern/preference/rule的ID
            person_name: 人名
            namespace: 命名空间
        """
        storage_path = self.get_storage_path(namespace)
        edges_file = os.path.join(storage_path, "edges.jsonl")
        
        edge = {
            "id": f"edge_{pattern_id}_{person_name.replace(' ', '_').lower()}",
            "source": pattern_id,
            "target": person_name,
            "type": "describes",
            "active": True
        }
        
        with open(edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')

