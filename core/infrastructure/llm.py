"""
LLM调用模块

支持OpenAI和DeepSeek两种模型调用
"""

import os
import time
from typing import Optional, List
from openai import OpenAI
from config import Config


class LLMClient:
    """统一的LLM调用客户端"""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        self.deepseek_client = None
        self._init_clients()
    
    def _init_clients(self):
        """初始化客户端"""
        # OpenAI客户端
        if self.config.openai_api_key:
            self.openai_client = OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                timeout=60.0
            )
        
        # DeepSeek客户端
        if self.config.deepseek_api_key:
            self.deepseek_client = OpenAI(
                api_key=self.config.deepseek_api_key,
                base_url=self.config.deepseek_base_url,
                timeout=60.0
            )
    
    def call_llm(self, prompt: str, model: str = None, provider: str = "deepseek") -> str:
        """
        调用LLM API
        
        Args:
            prompt: 输入提示
            model: 模型名称，如果为None则使用配置中的默认模型
            provider: 提供商 ("openai" 或 "deepseek")
        
        Returns:
            str: 模型回复内容
        """
        try:
            if provider == "openai":
                return self._call_openai(prompt, model)
            elif provider == "deepseek":
                return self._call_deepseek(prompt, model)
            else:
                raise ValueError(f"不支持的提供商: {provider}")
        
        except Exception as e:
            print(f"❌ LLM API调用失败: {e}")
            return ""
    
    def _call_openai(self, prompt: str, model: str = None) -> str:
        """调用OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI客户端未初始化，请检查API密钥")
        
        if not model:
            model = "gpt-4.1-mini"  # OpenAI默认模型
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    def _call_deepseek(self, prompt: str, model: str = None) -> str:
        """调用DeepSeek API"""
        if not self.deepseek_client:
            raise ValueError("DeepSeek客户端未初始化，请检查API密钥")
        
        if not model:
            model = self.config.llm_model  # 使用配置中的DeepSeek模型
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.deepseek_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    def batch_call_llm(self, prompts: List[str], model: str = None, provider: str = "deepseek") -> List[str]:
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
    
    def test_connection(self, provider: str = "deepseek") -> bool:
        """
        测试连接是否正常
        
        Args:
            provider: 提供商
        
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
