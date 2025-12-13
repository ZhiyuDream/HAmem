

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有python-dotenv，手动加载.env文件
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


@dataclass
class Config:

    
    # API配置
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    openai_base_url: str = os.getenv('OPENAI_BASE_URL', '')
    
    # DeepSeek配置
    deepseek_api_key: str = os.getenv('DEEPSEEK_API_KEY', '')
    deepseek_base_url: str = os.getenv('DEEPSEEK_BASE_URL', '')
    
    # 模型配置
    llm_model: str = os.getenv('LLM_MODEL', 'deepseek-chat')  # DeepSeek模型
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')  # OpenAI模型
    
    # 性能配置
    max_retries: int = int(os.getenv('MAX_RETRIES', '3'))
    base_delay: float = float(os.getenv('BASE_DELAY', '1.0'))
    max_workers: int = int(os.getenv('MAX_WORKERS', '5'))
    embedding_batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
    
    # 缓存配置
    cache_dir: str = os.getenv('CACHE_DIR', 'cache')
    max_memory_cache_size: int = int(os.getenv('MAX_MEMORY_CACHE_SIZE', '1000'))
    
    # 存储配置
    storage_dir: str = os.getenv('STORAGE_DIR', 'storage')
    
    # Neo4j 配置
    neo4j_uri: str = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_username: str = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', '')
    neo4j_database: str = os.getenv('NEO4J_DATABASE', 'neo4j')
    use_neo4j: bool = os.getenv('USE_NEO4J', 'True').lower() == 'true'  # 是否使用 Neo4j（默认启用）
    use_hybrid_search: bool = os.getenv('USE_HYBRID_SEARCH', 'true').lower() == 'true'  # 是否使用混合检索（FAISS + Neo4j）
    
    # Neo4j 配置
    neo4j_uri: str = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_username: str = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', '')
    
    # 处理配置
    fragment_max_length: int = int(os.getenv('FRAGMENT_MAX_LENGTH', '6000'))
    fragment_overlap: int = int(os.getenv('FRAGMENT_OVERLAP', '200'))
    
    # 检索配置
    search_top_k: int = int(os.getenv('SEARCH_TOP_K', '10'))
    similarity_threshold: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    
    # 其他配置
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def from_env(cls) -> 'Config':

        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            'openai_api_key': self.openai_api_key,
            'openai_base_url': self.openai_base_url,
            'llm_model': self.llm_model,
            'embedding_model': self.embedding_model,
            'max_retries': self.max_retries,
            'base_delay': self.base_delay,
            'max_workers': self.max_workers,
            'embedding_batch_size': self.embedding_batch_size,
            'cache_dir': self.cache_dir,
            'max_memory_cache_size': self.max_memory_cache_size,
            'storage_dir': self.storage_dir,
            'use_neo4j': self.use_neo4j,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_username': self.neo4j_username,
            'fragment_max_length': self.fragment_max_length,
            'fragment_overlap': self.fragment_overlap,
            'search_top_k': self.search_top_k,
            'similarity_threshold': self.similarity_threshold,
            'debug': self.debug,
            'log_level': self.log_level,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_username': self.neo4j_username,
            'neo4j_database': self.neo4j_database,
            'use_neo4j': self.use_neo4j,
            'use_hybrid_search': self.use_hybrid_search
        }
    
    def validate(self) -> bool:
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.use_neo4j and not self.neo4j_password:
            raise ValueError("Neo4j password is required when USE_NEO4J is true")
        
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        
        if self.fragment_max_length < 1:
            raise ValueError("fragment_max_length must be at least 1")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        return True
