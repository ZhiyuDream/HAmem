"""
Question Router模块

使用LLM路由问题到合适的答案生成模块
"""

import json
from typing import List, Dict, Any
from core.infrastructure import LLMClient, parse_llm_json


class QuestionRouter:
    """
    问题路由模块
    
    使用LLM分析问题类型，选择合适的specialized modules
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def route(self, question: str) -> List[str]:
        """
        路由问题到合适的模块
        
        Args:
            question: 用户问题
        
        Returns:
            选择的模块列表 ['time_handling', 'opinion_sentiment', ...]
        """
        print(f"\n🧭 路由问题: '{question[:50]}...'")
        
        # 构建路由prompt
        prompt = self._build_router_prompt(question)
        
        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider="deepseek")
        
        # 调试：显示原始响应
        print(f"  🔍 LLM原始响应: {response[:200]}...")
        
        # 解析结果
        result = self._parse_router_response(response)
        
        # 调试：显示解析结果
        print(f"  🔍 解析结果: {result}")
        
        modules = result.get('selected_modules', ['inference_prediction'])
        reasoning = result.get('reasoning', '')
        
        print(f"  ✅ 选择模块: {modules}")
        print(f"  💡 推理: {reasoning[:100]}...")
        
        return modules
    
    def _build_router_prompt(self, question: str) -> str:
        """
        构建路由prompt
        
        Args:
            question: 用户问题
        
        Returns:
            prompt字符串
        """
        return f"""
Analyze the question and route it to appropriate answer modules.

Question: "{question}"

Available Modules:
1. **time_handling**: For temporal questions (when, schedule, time, date, duration)
2. **opinion_sentiment**: For opinion/emotion questions (think, feel, like, dislike, opinion)
3. **inference_prediction**: For inference/prediction questions (will, would, might, predict, future)
4. **state_analysis**: For state/condition questions (is doing, was doing, current state)
5. **detail_extraction**: For detail questions (how, why, explain, describe)

Select the SINGLE BEST module for the question.

Return JSON format:
{{
  "selected_modules": ["module_name"],
  "reasoning": "Brief explanation of why this module is selected"
}}

Examples:
- "What was Admon's shift on Sunday?" → ["time_handling"]
- "How does Caroline feel about the policy?" → ["opinion_sentiment"]
- "Will she accept the new project?" → ["inference_prediction"]
- "What did they discuss about the timeline?" → ["detail_extraction"]

Now analyze and route the question:
"""
    
    def _parse_router_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的路由结果
        
        Args:
            response: LLM响应
        
        Returns:
            解析后的结果
        """
        default_result = {
            'selected_modules': ['detail_extraction'],
            'reasoning': 'Default to detail extraction'
        }
        
        result = parse_llm_json(
            response,
            expected_keys=['selected_modules', 'reasoning'],
            default=default_result
        )
        
        if result is None:
            return default_result
        
        # 验证modules
        valid_modules = [
            'time_handling',
            'opinion_sentiment',
            'inference_prediction',
            'state_analysis',
            'detail_extraction'
        ]
        
        selected = result.get('selected_modules', [])
        if not isinstance(selected, list) or not selected:
            selected = ['detail_extraction']
        
        # 过滤无效模块
        selected = [m for m in selected if m in valid_modules]
        if not selected:
            selected = ['inference_prediction']
        
        result['selected_modules'] = selected
        
        return result

