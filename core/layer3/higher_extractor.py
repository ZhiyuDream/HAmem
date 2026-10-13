"""
高层抽象提取器（Layer N -> Layer N+1，N>=3）

对每个聚类调用一次LLM，提炼出一个更高层的抽象节点
"""

from typing import Dict, List, Any
from core.infrastructure import LLMClient, parse_llm_json
from .prompt import build_abstraction_prompt


class HigherLayerExtractor:
    """递归高层抽象提取器"""

    def __init__(self, llm_client: LLMClient, token_tracker=None, default_provider: str = "openai"):
        self.llm_client = llm_client
        self.token_tracker = token_tracker
        self.default_provider = default_provider

    def abstract_cluster(
        self,
        nodes: List[Dict[str, Any]],
        source_layer: int,
        target_layer: int
    ) -> Dict[str, Any]:
        """
        把一个cluster的Layer N节点抽象为Layer N+1描述

        Args:
            nodes: cluster内的Layer N节点列表
            source_layer: 来源层（N）
            target_layer: 目标层（N+1）

        Returns:
            {"abstraction": {"summary": str, "themes": [...], "significance": str}}
        """
        default_result = {"abstraction": None}

        if not nodes:
            return default_result

        try:
            prompt = build_abstraction_prompt(source_layer, target_layer, nodes)

            if self.token_tracker:
                response, usage = self.llm_client.call_llm(
                    prompt,
                    provider=self.default_provider,
                    return_usage=True
                )
                self.token_tracker.record_llm_call(
                    f"layer{target_layer}_abstraction", usage, provider=self.default_provider
                )
            else:
                response = self.llm_client.call_llm(
                    prompt,
                    provider=self.default_provider
                )

            return self._parse_llm_response(response)

        except Exception as e:
            print(f"❌ Layer{target_layer}抽象提取失败: {e}")
            return {"abstraction": None, "error": str(e)}

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        default_result = {"abstraction": None}

        data = parse_llm_json(
            response,
            expected_keys=['abstraction'],
            default=default_result
        )

        if data is None:
            return default_result

        return {"abstraction": data.get('abstraction')}
