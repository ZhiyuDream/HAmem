"""
LLM和Embedding配置类

支持自由配置多种LLM提供商和Embedding提供商
参考mem0的设计，但保持HAmem的代码风格（使用dataclass）
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ProviderConfig:
    """
    基础提供商配置类
    
    所有提供商的通用配置
    """
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """类型检查和默认值设置"""
        if self.timeout <= 0:
            self.timeout = 60
        if self.max_retries < 0:
            self.max_retries = 3


@dataclass
class LlmConfig:
    """
    LLM配置类
    
    支持多种LLM提供商，通过provider字段指定，config字段包含提供商特定配置
    
    示例:
        # 方式1：直接创建
        llm_config = LlmConfig(
            provider="openai",
            config={
                "api_key": "sk-...",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        
        # 方式2：从环境变量创建
        llm_config = LlmConfig.from_env("openai")
        
        # 方式3：从字典创建
        llm_config = LlmConfig.from_dict({
            "provider": "deepseek",
            "config": {
                "api_key": "sk-...", 
                "model": "deepseek-chat",
                "temperature": 0.7
            }
        })
        
        # 方式4：创建带有特定配置的实例
        llm_config = LlmConfig.create_openai(
            api_key="sk-...",
            model="gpt-4",
            temperature=0.8
        )
    """
    provider: str = "deepseek"  # 默认提供商
    config: ProviderConfig = field(default_factory=ProviderConfig)  # 提供商特定配置
    
    # 支持的LLM提供商列表和对应的环境变量前缀
    PROVIDER_CONFIGS = {
        "openai": {
            "env_prefix": "OPENAI",
            "default_model": "gpt-4o-mini",
            "required_fields": ["api_key"]
        },
        "deepseek": {
            "env_prefix": "DEEPSEEK",
            "default_model": "deepseek-chat",
            "required_fields": ["api_key"]
        },
        "anthropic": {
            "env_prefix": "ANTHROPIC",
            "default_model": "claude-3-haiku-20240307",
            "required_fields": ["api_key"]
        },
        "ollama": {
            "env_prefix": "OLLAMA",
            "default_model": "llama3.1",
            "default_base_url": "http://localhost:11434",
            "required_fields": []
        },
        "groq": {
            "env_prefix": "GROQ",
            "default_model": "llama3-70b-8192",
            "required_fields": ["api_key"]
        },
        "together": {
            "env_prefix": "TOGETHER",
            "default_model": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "required_fields": ["api_key"]
        },
        "azure_openai": {
            "env_prefix": "AZURE_OPENAI",
            "required_fields": ["api_key", "base_url", "api_version"]
        },
        "gemini": {
            "env_prefix": "GEMINI",
            "default_model": "gemini-pro",
            "required_fields": ["api_key"]
        },
        "xai": {
            "env_prefix": "XAI",
            "default_model": "grok-beta",
            "required_fields": ["api_key"]
        },
        "lmstudio": {
            "env_prefix": "LMSTUDIO",
            "default_base_url": "http://localhost:1234/v1",
            "required_fields": []
        },
        "litellm": {
            "env_prefix": "LITELLM",
            "required_fields": ["api_key"]
        }
    }
    
    # 支持的LLM提供商列表
    SUPPORTED_PROVIDERS = list(PROVIDER_CONFIGS.keys())
    
    def __post_init__(self):
        """验证provider是否支持并设置默认值"""
        # 验证提供商
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"不支持的LLM提供商: {self.provider}. "
                f"支持的提供商: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        
        # 获取提供商配置
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})
        
        # 设置默认值
        if not self.config.model and provider_config.get('default_model'):
            self.config.model = provider_config['default_model']
        
        if not self.config.base_url and provider_config.get('default_base_url'):
            self.config.base_url = provider_config['default_base_url']
        
        # 验证必填字段
        self._validate_required_fields()
    
    def _validate_required_fields(self):
        """验证必填字段"""
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})
        required_fields = provider_config.get('required_fields', [])
        
        for field_name in required_fields:
            # 先检查是否是ProviderConfig的直接属性
            if hasattr(self.config, field_name):
                field_value = getattr(self.config, field_name)
                if not field_value:
                    raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段")
            # 再检查是否在additional_params中
            elif field_name in self.config.additional_params:
                field_value = self.config.additional_params[field_name]
                if not field_value:
                    raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段")
            else:
                # 如果既不在直接属性中，也不在additional_params中，说明未提供
                raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段，但未提供")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LlmConfig':
        """从字典创建配置"""
        provider = data.get('provider', 'deepseek')
        config_data = data.get('config', {})
        
        # 将字典转换为ProviderConfig对象
        config_obj = ProviderConfig(**config_data)
        
        return cls(provider=provider, config=config_obj)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'provider': self.provider,
            'config': self.config.__dict__
        }
    
    @classmethod
    def from_env(cls, provider: str = None) -> 'LlmConfig':
        """
        从环境变量创建配置
        
        Args:
            provider: 提供商名称，如果为None则从环境变量LLM_PROVIDER读取
        
        环境变量命名规则：
        - {PREFIX}_API_KEY: API密钥
        - {PREFIX}_BASE_URL: API地址（可选）
        - {PREFIX}_MODEL: 模型名称（可选）
        - {PREFIX}_TEMPERATURE: 温度（可选）
        - {PREFIX}_MAX_TOKENS: 最大token数（可选）
        - LLM_PROVIDER: 默认提供商
        """
        # 如果没有指定provider，尝试从环境变量读取
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "deepseek")
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的LLM提供商: {provider}")
        
        # 获取提供商配置
        provider_config = cls.PROVIDER_CONFIGS.get(provider, {})
        env_prefix = provider_config.get('env_prefix', provider.upper())
        
        # 从环境变量读取配置
        config_data = {}
        
        # 通用字段映射（ProviderConfig直接支持的字段）
        env_mappings = {
            'api_key': f"{env_prefix}_API_KEY",
            'base_url': f"{env_prefix}_BASE_URL",
            'model': f"{env_prefix}_MODEL",
            'temperature': f"{env_prefix}_TEMPERATURE",
            'max_tokens': f"{env_prefix}_MAX_TOKENS",
            'timeout': f"{env_prefix}_TIMEOUT",
            'max_retries': f"{env_prefix}_MAX_RETRIES",
        }
        
        # 额外参数（需要放入additional_params的字段）
        additional_env_mappings = {
            'api_version': f"{env_prefix}_API_VERSION",  # Azure专用
            'organization': f"{env_prefix}_ORGANIZATION",  # OpenAI专用
        }
        
        # 读取ProviderConfig直接支持的字段
        for attr, env_var in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                if attr in ['timeout', 'max_tokens', 'max_retries']:
                    try:
                        config_data[attr] = int(value)
                    except ValueError:
                        pass  # 如果转换失败，跳过该字段
                elif attr in ['temperature']:
                    try:
                        config_data[attr] = float(value)
                    except ValueError:
                        pass  # 如果转换失败，跳过该字段
                else:
                    config_data[attr] = value
        
        # 读取额外参数（放入additional_params）
        additional_params = {}
        for attr, env_var in additional_env_mappings.items():
            value = os.getenv(env_var)
            if value:
                additional_params[attr] = value
        
        # 从额外的环境变量读取（例如 Azure 的特定配置）
        additional_env_vars = [
            f"{env_prefix}_DEPLOYMENT_NAME",  # Azure
            f"{env_prefix}_ENGINE",  # Azure
        ]
        
        for env_var in additional_env_vars:
            value = os.getenv(env_var)
            if value:
                param_name = env_var.replace(f"{env_prefix}_", "").lower()
                additional_params[param_name] = value
        
        if additional_params:
            config_data['additional_params'] = additional_params
        
        # 创建ProviderConfig对象
        config_obj = ProviderConfig(**config_data)
        
        return cls(provider=provider, config=config_obj)
    
    @classmethod
    def create_openai(
        cls,
        api_key: str,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> 'LlmConfig':
        """创建OpenAI配置"""
        config = ProviderConfig(
            api_key=api_key,
            model=model or cls.PROVIDER_CONFIGS['openai']['default_model'],
            base_url=base_url or "https://api.openai.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=kwargs
        )
        return cls(provider="openai", config=config)
    
    @classmethod
    def create_deepseek(
        cls,
        api_key: str,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> 'LlmConfig':
        """创建DeepSeek配置"""
        config = ProviderConfig(
            api_key=api_key,
            model=model or cls.PROVIDER_CONFIGS['deepseek']['default_model'],
            base_url=base_url or "https://api.deepseek.com",
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=kwargs
        )
        return cls(provider="deepseek", config=config)
    
    @classmethod
    def create_anthropic(
        cls,
        api_key: str,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> 'LlmConfig':
        """创建Anthropic配置"""
        config = ProviderConfig(
            api_key=api_key,
            model=model or cls.PROVIDER_CONFIGS['anthropic']['default_model'],
            base_url=base_url or "https://api.anthropic.com",
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=kwargs
        )
        return cls(provider="anthropic", config=config)
    
    @classmethod
    def create_ollama(
        cls,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> 'LlmConfig':
        """创建Ollama配置"""
        config = ProviderConfig(
            model=model or cls.PROVIDER_CONFIGS['ollama']['default_model'],
            base_url=base_url or cls.PROVIDER_CONFIGS['ollama']['default_base_url'],
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=kwargs
        )
        return cls(provider="ollama", config=config)
    
    def get_api_key(self) -> Optional[str]:
        return self.config.api_key
    
    def get_base_url(self) -> Optional[str]:
        return self.config.base_url
    
    def get_model(self) -> Optional[str]:
        return self.config.model
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.additional_params[key] = value
        # 重新验证
        self.__post_init__()
    
    def __str__(self):
        """字符串表示"""
        return f"LlmConfig(provider={self.provider}, model={self.config.model})"


@dataclass
class EmbeddingConfig:
    """
    Embedding配置类，支持多种Embedding提供商
    示例:
        # 方式1：直接创建
        embedding_config = EmbeddingConfig(
            provider="openai",
            config={
                "api_key": "sk-...",
                "model": "text-embedding-3-small",
                "dimensions": 1536
            }
        )
        
        # 方式2：从环境变量创建
        embedding_config = EmbeddingConfig.from_env("openai")
    """
    provider: str = "openai"  # 默认提供商
    config: ProviderConfig = field(default_factory=ProviderConfig)  # 提供商特定配置
    
    # 支持的Embedding提供商列表和对应的环境变量前缀
    PROVIDER_CONFIGS = {
        "openai": {
            "env_prefix": "OPENAI",
            "default_model": "text-embedding-3-small",
            "default_dimensions": 1536,
            "required_fields": ["api_key"]
        },
        "deepseek": {
            "env_prefix": "DEEPSEEK",
            "default_model": "embedding",
            "default_dimensions": 1024,
            "required_fields": ["api_key"]
        },
        "huggingface": {
            "env_prefix": "HUGGINGFACE",
            "required_fields": ["api_key"]
        },
        "cohere": {
            "env_prefix": "COHERE",
            "default_model": "embed-english-v3.0",
            "required_fields": ["api_key"]
        },
        "ollama": {
            "env_prefix": "OLLAMA",
            "default_model": "nomic-embed-text",
            "default_base_url": "http://localhost:11434",
            "required_fields": []
        },
        "jina": {
            "env_prefix": "JINA",
            "default_model": "jina-embeddings-v2-base-en",
            "required_fields": ["api_key"]
        },
        "voyage": {
            "env_prefix": "VOYAGE",
            "default_model": "voyage-2",
            "required_fields": ["api_key"]
        },
        "azure_openai": {
            "env_prefix": "AZURE_OPENAI",
            "required_fields": ["api_key", "base_url", "api_version"]
        },
        "together": {
            "env_prefix": "TOGETHER",
            "required_fields": ["api_key"]
        }
    }
    
    # 支持的Embedding提供商列表
    SUPPORTED_PROVIDERS = list(PROVIDER_CONFIGS.keys())
    
    def __post_init__(self):
        """验证provider是否支持并设置默认值"""
        # 验证提供商
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"不支持的Embedding提供商: {self.provider}. "
                f"支持的提供商: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        
        # 获取提供商配置
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})
        
        # 设置默认值
        if not self.config.model and provider_config.get('default_model'):
            self.config.model = provider_config['default_model']
        
        if not self.config.base_url and provider_config.get('default_base_url'):
            self.config.base_url = provider_config['default_base_url']
        
        # 验证必填字段
        self._validate_required_fields()
    
    def _validate_required_fields(self):
        """验证必填字段"""
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})
        required_fields = provider_config.get('required_fields', [])
        
        for field_name in required_fields:
            # 先检查是否是ProviderConfig的直接属性
            if hasattr(self.config, field_name):
                field_value = getattr(self.config, field_name)
                if not field_value:
                    raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段")
            # 再检查是否在additional_params中
            elif field_name in self.config.additional_params:
                field_value = self.config.additional_params[field_name]
                if not field_value:
                    raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段")
            else:
                # 如果既不在直接属性中，也不在additional_params中，说明未提供
                raise ValueError(f"对于 {self.provider} 提供商，{field_name} 是必填字段，但未提供")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """从字典创建配置"""
        provider = data.get('provider', 'openai')
        config_data = data.get('config', {})
        
        # 将字典转换为ProviderConfig对象
        config_obj = ProviderConfig(**config_data)
        
        return cls(provider=provider, config=config_obj)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'provider': self.provider,
            'config': self.config.__dict__
        }
    
    @classmethod
    def from_env(cls, provider: str = None) -> 'EmbeddingConfig':
        """
        从环境变量创建配置
        
        Args:
            provider: 提供商名称，如果为None则从环境变量EMBEDDING_PROVIDER读取
        """
        # 如果没有指定provider，尝试从环境变量读取
        if provider is None:
            provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的Embedding提供商: {provider}")
        
        # 获取提供商配置
        provider_config = cls.PROVIDER_CONFIGS.get(provider, {})
        env_prefix = provider_config.get('env_prefix', provider.upper())
        
        # 从环境变量读取配置
        config_data = {}
        
        # 环境变量映射
        env_mappings = {
            'api_key': f"{env_prefix}_API_KEY",
            'base_url': f"{env_prefix}_BASE_URL",
            'model': f"{env_prefix}_MODEL",
            'timeout': f"{env_prefix}_TIMEOUT",
            'max_retries': f"{env_prefix}_MAX_RETRIES",
            'dimensions': f"{env_prefix}_DIMENSIONS",
        }
        
        for attr, env_var in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                if attr in ['timeout', 'max_retries', 'dimensions']:
                    try:
                        config_data[attr] = int(value)
                    except ValueError:
                        pass  # 如果转换失败，跳过该字段
                else:
                    config_data[attr] = value
        
        # 创建ProviderConfig对象
        config_obj = ProviderConfig(**config_data)
        
        return cls(provider=provider, config=config_obj)
    
    @classmethod
    def create_openai(
        cls,
        api_key: str,
        model: str = None,
        dimensions: int = None,
        **kwargs
    ) -> 'EmbeddingConfig':
        """创建OpenAI Embedding配置"""
        config = ProviderConfig(
            api_key=api_key,
            model=model or cls.PROVIDER_CONFIGS['openai']['default_model'],
            base_url="https://api.openai.com/v1",
            additional_params={
                'dimensions': dimensions or cls.PROVIDER_CONFIGS['openai']['default_dimensions'],
                **kwargs
            }
        )
        return cls(provider="openai", config=config)
    
    def get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        return self.config.api_key
    
    def get_base_url(self) -> Optional[str]:
        """获取API基础URL"""
        return self.config.base_url
    
    def get_model(self) -> Optional[str]:
        """获取模型名称"""
        return self.config.model
    
    def get_dimensions(self) -> Optional[int]:
        """获取嵌入维度"""
        return self.config.additional_params.get('dimensions')
    
    def __str__(self):
        """字符串表示"""
        dimensions = self.get_dimensions()
        dim_str = f", dimensions={dimensions}" if dimensions else ""
        return f"EmbeddingConfig(provider={self.provider}, model={self.config.model}{dim_str})"


# 使用示例
if __name__ == "__main__":
    # 示例1: 从环境变量创建LLM配置
    # 假设设置了环境变量: LLM_PROVIDER=openai, OPENAI_API_KEY=sk-xxx, OPENAI_MODEL=gpt-4
    try:
        llm_config = LlmConfig.from_env()
        print(f"LLM配置: {llm_config}")
        print(f"API Key: {llm_config.get_api_key()}")
        print(f"Model: {llm_config.get_model()}")
    except ValueError as e:
        print(f"创建LLM配置失败: {e}")
    
    # 示例2: 手动创建DeepSeek配置
    deepseek_config = LlmConfig.create_deepseek(
        api_key="sk-deepseek-xxx",
        model="deepseek-chat",
        temperature=0.8
    )
    print(f"\nDeepSeek配置: {deepseek_config}")
    
    # 示例3: 创建Embedding配置
    embedding_config = EmbeddingConfig.create_openai(
        api_key="sk-openai-xxx",
        model="text-embedding-3-small",
        dimensions=1536
    )
    print(f"\nEmbedding配置: {embedding_config}")
    
    # 示例4: 从字典创建配置
    config_dict = {
        "provider": "ollama",
        "config": {
            "model": "llama3.1",
            "temperature": 0.5,
            "max_tokens": 1000
        }
    }
    ollama_config = LlmConfig.from_dict(config_dict)
    print(f"\nOllama配置: {ollama_config}")