"""
Embedding管理模块

调用OpenAI embedding API生成向量
"""

import os
import time
from typing import List, Union
import numpy as np
from openai import OpenAI
from config import Config


class EmbeddingManager:
    """Embedding管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        if self.config.openai_api_key:
            self.client = OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                timeout=60.0
            )
        else:
            raise ValueError("OpenAI API密钥未配置")
    
    def get_embedding(self, text: str, model: str = None, max_retries: int = 3) -> List[float]:
        """
        获取单个文本的embedding（带重试）
        
        Args:
            text: 输入文本
            model: embedding模型，如果为None则使用配置中的默认模型
            max_retries: 最大重试次数
        
        Returns:
            List[float]: embedding向量，失败时抛出异常
        """
        if not self.client:
            raise ValueError("OpenAI客户端未初始化")
        
        if not model:
            model = self.config.embedding_model
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                
                # 验证embedding维度
                if len(embedding) != 1536:
                    raise ValueError(f"Embedding维度错误: {len(embedding)}, 期望1536")
                
                return embedding
            
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避：2s, 4s, 6s
                    print(f"⚠️  Embedding生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"   等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Embedding生成失败，已达最大重试次数: {e}")
        
        # 所有重试都失败，抛出异常
        raise RuntimeError(f"Embedding生成失败: {last_error}")
    
    def batch_get_embeddings(self, texts: List[str], model: str = None, max_retries: int = 3) -> List[List[float]]:
        """
        批量获取embeddings（带重试）
        
        Args:
            texts: 文本列表
            model: embedding模型
            max_retries: 最大重试次数
        
        Returns:
            List[List[float]]: embedding向量列表，失败时抛出异常
        """
        if not self.client:
            raise ValueError("OpenAI客户端未初始化")
        
        if not model:
            model = self.config.embedding_model
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 分批处理，避免API限制
                batch_size = self.config.embedding_batch_size
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    response = self.client.embeddings.create(
                        model=model,
                        input=batch_texts
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    
                    # 验证每个embedding的维度
                    for idx, emb in enumerate(batch_embeddings):
                        if len(emb) != 1536:
                            raise ValueError(f"Embedding[{idx}]维度错误: {len(emb)}, 期望1536")
                    
                    all_embeddings.extend(batch_embeddings)
                    
                    # 添加延迟避免API限制
                    if i + batch_size < len(texts):
                        time.sleep(0.1)
                
                return all_embeddings
            
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3  # 指数退避：3s, 6s, 9s
                    print(f"⚠️  批量Embedding生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"   等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 批量Embedding生成失败，已达最大重试次数: {e}")
        
        # 所有重试都失败，抛出异常
        raise RuntimeError(f"批量Embedding生成失败: {last_error}")
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个embedding的余弦相似度
        
        Args:
            embedding1: 第一个embedding
            embedding2: 第二个embedding
        
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            print(f"❌ 相似度计算失败: {e}")
            return 0.0
    
    def test_connection(self) -> bool:
        """
        测试embedding连接是否正常
        
        Returns:
            bool: 连接是否正常
        """
        try:
            test_text = "这是一个测试文本"
            embedding = self.get_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            print(f"❌ Embedding连接测试失败: {e}")
            return False
