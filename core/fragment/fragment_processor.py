"""
分片处理器

调用LLM判断分片点
"""

import json
from typing import List, Dict, Any, Optional
from core.infrastructure import LLMClient, parse_llm_json
from .prompt import build_batch_split_fragment_prompt


class FragmentProcessor:
    """分片处理器"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def should_split(self, turns: List[Dict[str, Any]]) -> Optional[int]:
        """
        判断是否需要分片以及分片点
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            Optional[int]: 分片点（0-based索引），None表示不需要分片
        """
        if len(turns) <= 1:
            return None
        
        try:
            # 转换格式以匹配prompt期望的格式
            formatted_turns = []
            for turn in turns:
                formatted_turns.append({
                    'speaker': turn.get('role', 'unknown'),
                    'text': turn.get('content', '')
                })
            
            # 构建prompt
            prompt = build_batch_split_fragment_prompt(formatted_turns)
            
            print(f"🔍 调用LLM进行分片判断...")
            print(f"📝 Prompt长度: {len(prompt)} 字符")
            
            # 调用LLM
            response = self.llm_client.call_llm(
                prompt, 
                provider="deepseek"
            )
            
            print(f"🤖 LLM响应: {response[:200]}...")
            
            # 解析响应
            split_point = self._parse_llm_response(response)
            return split_point
            
        except Exception as e:
            print(f"❌ 分片判断失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[int]:
        """
        解析LLM响应（使用JSON修复工具）
        
        Args:
            response: LLM响应文本
        
        Returns:
            Optional[int]: 分片点，None表示不需要分片
        """
        # 使用JSON修复工具解析
        default_result = {"split_point": -1}
        data = parse_llm_json(
            response,
            expected_keys=['split_point'],
            default=default_result
        )
        
        if data is None:
            return None
        
        split_point = data.get('split_point', -1)
        
        if split_point == -1 or split_point <= 0:
            return None  # 不需要分片
        else:
            return split_point
    
    def process_fragment(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理分片逻辑
        
        Args:
            turns: 对话轮次列表
        
        Returns:
            Dict: 处理结果
        """
        split_point = self.should_split(turns)
        
        if split_point is None:
            return {
                "should_split": False,
                "split_point": None,
                "reason": "LLM判断不需要分片"
            }
        else:
            return {
                "should_split": True,
                "split_point": split_point,
                "reason": f"LLM判断在位置{split_point}分片"
            }
