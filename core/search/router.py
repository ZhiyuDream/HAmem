"""
Question Router module

Use LLM to route questions to appropriate answer modules
"""

import json
from typing import List, Dict, Any
from core.infrastructure import LLMClient, parse_llm_json


class QuestionRouter:
    """
    Analyze question type and select specialized modules
    """
    
    def __init__(self, llm_client: LLMClient, default_provider: str = "deepseek"):
        """
        Args:
            llm_client: LLM client
            default_provider: 默认LLM提供商 ("openai" 或 "deepseek")
        """
        self.llm_client = llm_client
        self.default_provider = default_provider
    
    def route(self, question: str) -> List[str]:
        """
        Route the question to best-fit modules
        """
        print(f"\n🧭 Routing question: '{question[:50]}...'")
        
        # 构建路由prompt
        prompt = self._build_router_prompt(question)
        
        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider=self.default_provider)
        
        # Debug: raw response
        print(f"  🔍 LLM raw response: {response[:200]}...")
        
        # Parse result
        result = self._parse_router_response(response)
        
        # Debug: parsed result
        print(f"  🔍 Parsed: {result}")
        
        modules = result.get('selected_modules', ['inference_prediction'])
        reasoning = result.get('reasoning', '')
        
        print(f"  ✅ Selected modules: {modules}")
        print(f"  💡 Reasoning: {reasoning[:100]}...")
        
        return modules
    
    def _build_router_prompt(self, question: str) -> str:
        """
        Build routing prompt
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
        Parse LLM routing result
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
        
        # Validate modules
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
        
        # Filter invalid modules
        selected = [m for m in selected if m in valid_modules]
        if not selected:
            selected = ['inference_prediction']
        
        result['selected_modules'] = selected
        
        return result

