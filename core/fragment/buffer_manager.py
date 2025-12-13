"""
缓冲区管理器

管理对话轮次的缓冲区，检测长度和时间变化
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import re


class BufferManager:
    """对话缓冲区管理器"""
    
    def __init__(self, max_length: int = 5000):
        self.max_length = max_length
        self.turns: List[Dict[str, Any]] = []
        self.current_timestamp: Optional[str] = None
        self.fragment_counter = 0
    
    def add_turn(self, turn: Dict[str, Any], timestamp: str = None) -> tuple:
        """
        添加对话轮次到缓冲区
        
        Args:
            turn: 对话轮次 {"role": "user/assistant", "content": "..."}
            timestamp: 时间戳
        
        Returns:
            tuple: (fragment_or_none, needs_llm)
                - fragment_or_none: 如果时间戳变化产生的fragment，否则None
                - needs_llm: 是否需要LLM判断分片（长度超限）
        """
        # 检查时间戳变化
        if timestamp and timestamp != self.current_timestamp:
            # 时间戳变化，直接分片（不需要LLM判断）
            fragment = None
            if self.turns:  # 如果缓冲区有内容，先保存为片段
                fragment = self._save_timestamp_fragment()
            # 清空缓冲区，添加新轮次
            self.turns = [turn]
            self.current_timestamp = timestamp
            return (fragment, False)  # 返回fragment和False（不需要LLM）
        
        # 添加轮次
        self.turns.append(turn)
        
        # 检查长度是否超限
        current_length = self._calculate_length()
        needs_llm = current_length > self.max_length
        return (None, needs_llm)  # 返回None和是否需要LLM
    
    def _save_timestamp_fragment(self):
        """时间戳变化时保存当前缓冲区为片段"""
        if not self.turns:
            return
        
        # 生成片段ID
        self.fragment_counter += 1
        fragment_id = f"fragment_{self.fragment_counter}"
        
        # 从turns中提取conversation_time（优先使用metadata中的session_time）
        conversation_time = self.current_timestamp or "unknown"
        if self.turns:
            # 尝试从第一个turn的metadata中获取session_time
            first_turn = self.turns[0]
            metadata = first_turn.get('metadata', {})
            session_time = metadata.get('session_time', '')
            if session_time:
                conversation_time = session_time
        
        # 构建片段数据
        fragment = {
            "id": fragment_id,
            "type": "fragment",
            "content": self._format_fragment_content(self.turns),
            "time": conversation_time,  # 使用session_time作为conversation_time
            "conversation_time": conversation_time,  # 同时设置conversation_time字段
            "layer": 0,
            "active": True
        }
        
        # 保存片段（这里需要存储系统支持）
        # 暂时先清空缓冲区
        self.turns = []
        
        return fragment
    
    def _calculate_length(self) -> int:
        """计算当前缓冲区内容长度"""
        total_length = 0
        for turn in self.turns:
            total_length += len(turn.get('content', ''))
        return total_length
    
    def get_turns_for_llm(self) -> List[Dict[str, Any]]:
        """获取用于LLM判断的对话轮次"""
        return self.turns.copy()
    
    def extract_fragment(self, split_point: int) -> Dict[str, Any]:
        """
        提取已完成的片段
        
        Args:
            split_point: 分片点（0-based索引）
        
        Returns:
            Dict: 片段数据
        """
        if split_point <= 0 or split_point >= len(self.turns):
            return None
        
        # 提取片段内容
        fragment_turns = self.turns[:split_point]
        content = self._format_fragment_content(fragment_turns)
        
        # 生成片段ID
        self.fragment_counter += 1
        fragment_id = f"fragment_{self.fragment_counter}"
        
        # 从turns中提取conversation_time（优先使用metadata中的session_time）
        conversation_time = self.current_timestamp or "unknown"
        if fragment_turns:
            # 尝试从第一个turn的metadata中获取session_time
            first_turn = fragment_turns[0]
            metadata = first_turn.get('metadata', {})
            session_time = metadata.get('session_time', '')
            if session_time:
                conversation_time = session_time
        
        # 构建片段数据
        fragment = {
            "id": fragment_id,
            "type": "fragment",
            "content": content,
            "time": conversation_time,  # 使用session_time作为conversation_time
            "conversation_time": conversation_time,  # 同时设置conversation_time字段
            "layer": 0,
            "active": True
        }
        
        return fragment
    
    def keep_remaining(self, split_point: int):
        """
        保留剩余部分在缓冲区
        
        Args:
            split_point: 分片点
        """
        if split_point > 0 and split_point < len(self.turns):
            self.turns = self.turns[split_point:]
        else:
            self.turns = []
    
    def _format_fragment_content(self, turns: List[Dict[str, Any]]) -> str:
        """
        格式化片段内容为兼容格式
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            str: 格式化后的内容
        """
        formatted_lines = []
        for turn in turns:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted_lines.append(f"[{role}][] {content}")
        
        return "\n".join(formatted_lines)
    
    def get_content(self) -> str:
        """
        获取当前buffer的完整内容（格式化后）
        
        Returns:
            str: 格式化后的对话内容
        """
        return self._format_fragment_content(self.turns)
    
    def get_buffer_length(self) -> int:
        """获取当前缓冲区长度"""
        return self._calculate_length()
    
    def get_turn_count(self) -> int:
        """获取当前轮次数量"""
        return len(self.turns)
    
    def clear(self):
        """清空缓冲区"""
        self.turns = []
        self.current_timestamp = None
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return len(self.turns) == 0
