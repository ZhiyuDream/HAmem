"""
Fragment Neo4j 存储管理

保存 fragment 到 Neo4j
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from ..infrastructure.neo4j_storage_base import Neo4jStorageBase
from ..infrastructure.neo4j_client import Neo4jClient
import logging
import os

logger = logging.getLogger(__name__)


class FragmentNeo4jStorage(Neo4jStorageBase):
    """Fragment Neo4j 存储管理器"""
    
    def __init__(self, neo4j_client: Neo4jClient, namespace: str = "default"):
        """
        初始化 Fragment 存储
        
        Args:
            neo4j_client: Neo4j 客户端
            namespace: 命名空间（通常使用输入文件名）
        """
        super().__init__(neo4j_client, namespace)
    
    def get_storage_path(self, input_filename: str) -> str:
        """
        根据输入文件名获取存储路径（Neo4j 中不需要，保留接口兼容性）
        
        Args:
            input_filename: 输入文件名（如 s_item_410.json）
        
        Returns:
            str: 命名空间
        """
        # 提取文件名（不含扩展名）作为命名空间
        name_without_ext = Path(input_filename).stem
        return name_without_ext
    
    def save_fragment(self, fragment: Dict[str, Any], input_filename: str) -> None:
        """
        保存单个片段
        
        Args:
            fragment: 片段数据
            input_filename: 输入文件名
        """
        # 更新命名空间
        name_without_ext = Path(input_filename).stem
        self.namespace = name_without_ext
        
        fragment_id = fragment.get('id')
        if not fragment_id:
            logger.warning("Fragment missing id, skipping")
            return
        
        # 构建节点属性（保留所有 fragment 的原始属性）
        properties = fragment.copy()
        properties['type'] = 'fragment'
        properties['layer'] = properties.get('layer', 0)
        properties['active'] = properties.get('active', True)
        
        # 移除 id（因为会单独传入）
        if 'id' in properties:
            del properties['id']
        
        # 移除 None 值
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # 创建节点，标签为 Fragment 和 Layer0
        self.create_node(
            node_id=fragment_id,
            labels=["Fragment", "Layer0"],
            properties=properties
        )
    
    def save_fragments(self, fragments: List[Dict[str, Any]], input_filename: str) -> None:
        """
        批量保存片段
        
        Args:
            fragments: 片段列表
            input_filename: 输入文件名
        """
        if not fragments:
            return
        
        # 更新命名空间
        name_without_ext = Path(input_filename).stem
        self.namespace = name_without_ext
        
        for fragment in fragments:
            self.save_fragment(fragment, input_filename)
    
    def initialize_storage(self, input_filename: str) -> None:
        """
        初始化存储（Neo4j 中不需要创建文件，只需设置命名空间）
        
        Args:
            input_filename: 输入文件名
        """
        name_without_ext = Path(input_filename).stem
        self.namespace = name_without_ext
        logger.info(f"Fragment storage initialized with namespace: {self.namespace}")
    
    def get_fragments(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取已保存的片段
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            List[Dict]: 片段列表
        """
        name_without_ext = Path(input_filename).stem
        self.namespace = name_without_ext
        
        fragments = self.query_nodes(
            labels=["Fragment", "Layer0"],
            filters={"type": "fragment", "active": True}
        )
        
        return fragments
    
    def get_storage_stats(self, input_filename: str) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            Dict: 统计信息
        """
        name_without_ext = Path(input_filename).stem
        self.namespace = name_without_ext
        
        stats = self.get_stats()
        fragments = self.get_fragments(input_filename)
        
        stats.update({
            "total_fragments": len(fragments),
            "namespace": self.namespace,
            "fragment_ids": [f.get('id') for f in fragments]
        })
        
        return stats

