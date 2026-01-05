"""
LLM调用模块

支持多种LLM提供商（OpenAI、DeepSeek、Anthropic、Ollama等）
"""

import os
import time
from typing import Optional, List
from openai import OpenAI
from config import Config


class LLMClient:
    """统一的LLM调用客户端"""
    
    def __init__(self, config: Config, token_tracker=None):
        """
        初始化LLM客户端
        
        Args:
            config: 配置对象
            token_tracker: Token统计收集器（可选），用于自动记录token使用情况
        """
        self.config = config
        self.llm_config = config.llm_config
        self.clients = {}  # 存储不同提供商的客户端
        self.token_tracker = token_tracker
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        if not self.llm_config:
            raise ValueError("LLM配置未提供")
        
        provider = self.llm_config.provider
        provider_config = self.llm_config.config
        
        # 根据提供商初始化客户端
        if provider in ['openai', 'deepseek', 'anthropic', 'groq', 'together', 'azure_openai', 'xai', 'lmstudio', 'litellm']:
            # 这些提供商使用OpenAI兼容的API
            base_url = self.llm_config.get_base_url()
            if not base_url:
                # 根据提供商设置默认base_url
                if provider == 'openai':
                    base_url = 'https://api.openai.com/v1'
                elif provider == 'deepseek':
                    base_url = 'https://api.deepseek.com'
                elif provider == 'anthropic':
                    base_url = 'https://api.anthropic.com'
                elif provider == 'groq':
                    base_url = 'https://api.groq.com/openai/v1'
                elif provider == 'together':
                    base_url = 'https://api.together.xyz/v1'
                elif provider == 'xai':
                    base_url = 'https://api.x.ai/v1'
                elif provider == 'lmstudio':
                    base_url = 'http://localhost:1234/v1'
              
            self.clients[provider] = OpenAI(
                api_key=self.llm_config.get_api_key(),
                base_url=base_url,
                timeout=provider_config.timeout or 60.0
            )
        elif provider == 'ollama':
            # Ollama使用本地API
            base_url = self.llm_config.get_base_url() or 'http://localhost:11434/v1'
            self.clients[provider] = OpenAI(
                api_key='ollama',  # Ollama不需要真实的API key
                base_url=base_url,
                timeout=provider_config.timeout or 60.0
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    def call_llm(self, prompt: str, model: str = None, provider: str = None, return_usage: bool = False):
        """
        调用LLM API
        
        Args:
            prompt: 输入提示
            model: 模型名称，如果为None则使用配置中的默认模型
            provider: 提供商，如果为None则使用配置中的提供商
            return_usage: 是否返回token使用信息
        
        Returns:
            str 或 tuple: 如果return_usage=False，返回模型回复内容；如果return_usage=True，返回(content, usage_dict)
        """
        try:
            # 使用配置中的提供商（如果未指定）
            if provider is None:
                provider = self.llm_config.provider
            
            # 使用配置中的模型（如果未指定）
            if model is None:
                model = self.llm_config.get_model()
            
            return self._call_provider(prompt, model, provider, return_usage)
        
        except Exception as e:
            print(f"❌ LLM API调用失败: {e}")
            if return_usage:
                return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            return ""
    
    def _call_provider(self, prompt: str, model: str, provider: str, return_usage: bool = False):
        """调用指定提供商的API"""
        if provider not in self.clients:
            raise ValueError(f"提供商 {provider} 的客户端未初始化，请检查配置")
        
        client = self.clients[provider]
        provider_config = self.llm_config.config
        
        # 构建请求参数
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": provider_config.max_tokens or 2000
        }
        
        # 设置temperature（某些模型不支持自定义temperature，如gpt-5-mini）
        # 对于不支持temperature的模型，不设置该参数（使用默认值）
        if provider_config.temperature is not None:
            # 检查模型是否支持自定义temperature
            # gpt-5-mini等模型只支持默认temperature（1），不支持自定义值
            if not (model and ('gpt-5' in model.lower() or 'o3' in model.lower())):
                params["temperature"] = provider_config.temperature
            # 对于不支持自定义temperature的模型，不设置temperature参数
        else:
            # 如果没有配置temperature，对于支持自定义temperature的模型使用默认值
            if not (model and ('gpt-5' in model.lower() or 'o3' in model.lower())):
                params["temperature"] = 1
        
        # 添加额外参数（如Azure的api_version）
        if provider_config.additional_params:
            params.update(provider_config.additional_params)
        
        # 调用API
        response = client.chat.completions.create(**params)
        
        content = response.choices[0].message.content.strip()
        
        if return_usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            }
            return (content, usage)
        
        return content
    
    def batch_call_llm(self, prompts: List[str], model: str = None, provider: str = None) -> List[str]:
        """
        批量调用LLM API
        
        Args:
            prompts: 提示列表
            model: 模型名称
            provider: 提供商
        
        Returns:
            List[str]: 回复列表
        """
        results = []
        for prompt in prompts:
            result = self.call_llm(prompt, model, provider)
            results.append(result)
            # 添加小延迟避免API限制
            time.sleep(0.01)
        
        return results
    
    def test_connection(self, provider: str = None) -> bool:
        """
        测试连接是否正常
        
        Args:
            provider: 提供商，如果为None则使用配置中的提供商
        
        Returns:
            bool: 连接是否正常
        """
        try:
            test_prompt = "请回复'连接测试成功'"
            response = self.call_llm(test_prompt, provider=provider)
            return "成功" in response or "success" in response.lower()
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
            return False
