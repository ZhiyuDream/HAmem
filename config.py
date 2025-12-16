import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv     
    load_dotenv()
except ImportError:
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
    llm_config: Optional['LlmConfig'] = None
    embedding_config: Optional['EmbeddingConfig'] = None
    
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    openai_base_url: str = os.getenv('OPENAI_BASE_URL', '')
    deepseek_api_key: str = os.getenv('DEEPSEEK_API_KEY', '')
    deepseek_base_url: str = os.getenv('DEEPSEEK_BASE_URL', '')
    llm_model: str = os.getenv('LLM_MODEL', 'deepseek-chat')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # 性能
    max_retries: int = int(os.getenv('MAX_RETRIES', '3'))
    base_delay: float = float(os.getenv('BASE_DELAY', '1.0'))
    max_workers: int = int(os.getenv('MAX_WORKERS', '5'))
    embedding_batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
    
    # 缓存
    cache_dir: str = os.getenv('CACHE_DIR', 'cache')
    max_memory_cache_size: int = int(os.getenv('MAX_MEMORY_CACHE_SIZE', '1000'))
    
    # 存储
    storage_dir: str = os.getenv('STORAGE_DIR', 'storage')
    
    # Neo4j 配置
    neo4j_uri: str = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_username: str = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', '')
    neo4j_database: str = os.getenv('NEO4J_DATABASE', 'neo4j')
    use_neo4j: bool = os.getenv('USE_NEO4J', 'True').lower() == 'true'
    use_hybrid_search: bool = os.getenv('USE_HYBRID_SEARCH', 'true').lower() == 'true'
    
    # 处理配置
    fragment_max_length: int = int(os.getenv('FRAGMENT_MAX_LENGTH', '6000'))
    fragment_overlap: int = int(os.getenv('FRAGMENT_OVERLAP', '200'))
    
    # 检索配置
    search_top_k: int = int(os.getenv('SEARCH_TOP_K', '10'))
    similarity_threshold: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    
    # 其他配置
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def __post_init__(self):
        """初始化后处理：如果新配置未提供，从旧配置自动创建"""
        from InputConfig import LlmConfig, EmbeddingConfig, ProviderConfig
        
        # 如果未提供llm_config，从旧配置创建
        if self.llm_config is None:
            if self.deepseek_api_key:
                self.llm_config = LlmConfig(
                    provider="deepseek",
                    config=ProviderConfig(
                        api_key=self.deepseek_api_key,
                        base_url=self.deepseek_base_url or "https://api.deepseek.com",
                        model=self.llm_model
                    )
                )
            elif self.openai_api_key:
                self.llm_config = LlmConfig(
                    provider="openai",
                    config=ProviderConfig(
                        api_key=self.openai_api_key,
                        base_url=self.openai_base_url or "https://api.openai.com/v1",
                        model=self.llm_model or "gpt-4o-mini"
                    )
                )
            else:
                # 尝试从环境变量创建
                try:
                    self.llm_config = LlmConfig.from_env()
                except:
                    pass
        
        # 如果未提供embedding_config，从旧配置创建
        if self.embedding_config is None:
            if self.openai_api_key:
                self.embedding_config = EmbeddingConfig(
                    provider="openai",
                    config=ProviderConfig(
                        api_key=self.openai_api_key,
                        base_url=self.openai_base_url or "https://api.openai.com/v1",
                        model=self.embedding_model,
                        additional_params={'dimensions': 1536}
                    )
                )
            else:
                # 尝试从环境变量创建
                try:
                    self.embedding_config = EmbeddingConfig.from_env()
                except:
                    pass
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'llm_provider': self.llm_config.provider if self.llm_config else None,
            'llm_model': self.llm_config.get_model() if self.llm_config else self.llm_model,
            'embedding_provider': self.embedding_config.provider if self.embedding_config else None,
            'embedding_model': self.embedding_config.config.model if self.embedding_config else self.embedding_model,
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
            'neo4j_database': self.neo4j_database,
            'use_hybrid_search': self.use_hybrid_search,
            'fragment_max_length': self.fragment_max_length,
            'fragment_overlap': self.fragment_overlap,
            'search_top_k': self.search_top_k,
            'similarity_threshold': self.similarity_threshold,
            'debug': self.debug,
            'log_level': self.log_level
        }
        return result
    
    def validate(self) -> bool:
        """验证配置"""
        # 验证LLM配置
        if not self.llm_config:
            raise ValueError("LLM配置未提供，请设置llm_config或提供API密钥")
        
        if not self.llm_config.get_api_key():
            raise ValueError("LLM API密钥未配置")
        
        # 验证Embedding配置
        if not self.embedding_config:
            raise ValueError("Embedding配置未提供，请设置embedding_config或提供API密钥")
        
        if not self.embedding_config.config.api_key:
            raise ValueError("Embedding API密钥未配置")
        
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