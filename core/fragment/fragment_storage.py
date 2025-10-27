"""
片段存储管理器

保存fragment到nodes.jsonl格式，兼容现有系统
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class FragmentStorage:
    """片段存储管理器"""
    
    def __init__(self, base_storage_dir: str = "storage"):
        self.base_storage_dir = base_storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.base_storage_dir, exist_ok=True)
    
    def get_storage_path(self, input_filename: str) -> str:
        """
        根据输入文件名获取存储路径
        
        Args:
            input_filename: 输入文件名（如s_item_410.json）
        
        Returns:
            str: 存储目录路径
        """
        # 提取文件名（不含扩展名）
        name_without_ext = Path(input_filename).stem
        storage_path = os.path.join(self.base_storage_dir, name_without_ext)
        os.makedirs(storage_path, exist_ok=True)
        return storage_path
    
    def save_fragment(self, fragment: Dict[str, Any], input_filename: str):
        """
        保存单个片段
        
        Args:
            fragment: 片段数据
            input_filename: 输入文件名
        """
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 追加到nodes.jsonl文件
        with open(nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(fragment, ensure_ascii=False) + '\n')
    
    def save_fragments(self, fragments: List[Dict[str, Any]], input_filename: str):
        """
        批量保存片段
        
        Args:
            fragments: 片段列表
            input_filename: 输入文件名
        """
        if not fragments:
            return
        
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        # 追加到nodes.jsonl文件
        with open(nodes_file, 'a', encoding='utf-8') as f:
            for fragment in fragments:
                f.write(json.dumps(fragment, ensure_ascii=False) + '\n')
    
    def initialize_storage(self, input_filename: str):
        """
        初始化存储文件
        
        Args:
            input_filename: 输入文件名
        """
        storage_path = self.get_storage_path(input_filename)
        
        # 初始化nodes.jsonl（如果不存在）
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        if not os.path.exists(nodes_file):
            with open(nodes_file, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
        
        # 初始化edges.jsonl（如果不存在）
        edges_file = os.path.join(storage_path, "edges.jsonl")
        if not os.path.exists(edges_file):
            with open(edges_file, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
    
    def get_fragments(self, input_filename: str) -> List[Dict[str, Any]]:
        """
        获取已保存的片段
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            List[Dict]: 片段列表
        """
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        if not os.path.exists(nodes_file):
            return []
        
        fragments = []
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'fragment':
                            fragments.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return fragments
    
    def get_storage_stats(self, input_filename: str) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Args:
            input_filename: 输入文件名
        
        Returns:
            Dict: 统计信息
        """
        storage_path = self.get_storage_path(input_filename)
        nodes_file = os.path.join(storage_path, "nodes.jsonl")
        
        if not os.path.exists(nodes_file):
            return {
                "total_fragments": 0,
                "storage_path": storage_path,
                "nodes_file_exists": False
            }
        
        fragments = self.get_fragments(input_filename)
        
        return {
            "total_fragments": len(fragments),
            "storage_path": storage_path,
            "nodes_file_exists": True,
            "fragment_ids": [f.get('id') for f in fragments]
        }
