# 配置指南

## 快速开始

### 方式1：使用统一配置接口（推荐，最简单）

在 `.env` 文件中设置：

```bash
# LLM配置（统一接口）
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_PROVIDER=deepseek

# Embedding配置（统一接口）
EMBEDDING_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai
```

然后在代码中：

```python
from config import Config
from main import HAmem

config = Config()
hamem = HAmem(config)
```

### 方式1.5：向后兼容（旧配置方式仍支持）

如果你使用旧的配置方式，系统会自动兼容：

```bash
# 旧配置方式（仍支持）
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
```

### 方式2：代码中直接配置（推荐）

```python
from config import Config
from InputConfig import LlmConfig, EmbeddingConfig
from main import HAmem

# 创建LLM配置
llm_config = LlmConfig.create_deepseek(
    api_key="your-deepseek-key",
    model="deepseek-chat"
)

# 创建Embedding配置
embedding_config = EmbeddingConfig.create_openai(
    api_key="your-openai-key",
    model="text-embedding-3-small"
)

# 创建完整配置
config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config
)

hamem = HAmem(config)
```

## 支持的LLM提供商

### OpenAI

```python
llm_config = LlmConfig.create_openai(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7
)
```

### DeepSeek

```python
llm_config = LlmConfig.create_deepseek(
    api_key="sk-...",
    model="deepseek-chat",
    temperature=0.7
)
```

### Anthropic

```python
llm_config = LlmConfig.create_anthropic(
    api_key="sk-ant-...",
    model="claude-3-haiku-20240307"
)
```

### Ollama（本地）

```python
llm_config = LlmConfig.create_ollama(
    model="llama3.1",
    base_url="http://localhost:11434"
)
```

### Groq

```python
llm_config = LlmConfig.create_groq(
    api_key="gsk_...",
    model="llama-3.1-70b-versatile"
)
```

## 支持的Embedding提供商

### OpenAI

```python
embedding_config = EmbeddingConfig.create_openai(
    api_key="sk-...",
    model="text-embedding-3-small",
    dimensions=1536
)
```

### DeepSeek

```python
embedding_config = EmbeddingConfig.create_deepseek(
    api_key="sk-...",
    model="embedding"
)
```

## 从字典创建配置

```python
# LLM配置
llm_dict = {
    "provider": "deepseek",
    "config": {
        "api_key": "sk-...",
        "model": "deepseek-chat",
        "temperature": 0.7
    }
}
llm_config = LlmConfig.from_dict(llm_dict)

# Embedding配置
embedding_dict = {
    "provider": "openai",
    "config": {
        "api_key": "sk-...",
        "model": "text-embedding-3-small"
    }
}
embedding_config = EmbeddingConfig.from_dict(embedding_dict)

config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config
)
```

## 从环境变量创建配置

```python
# 设置环境变量前缀
# DEEPSEEK_API_KEY=sk-...
# DEEPSEEK_MODEL=deepseek-chat

llm_config = LlmConfig.from_env("deepseek")

# OPENAI_API_KEY=sk-...
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small

embedding_config = EmbeddingConfig.from_env("openai")

config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config
)
```

## 混合配置

可以只配置LLM或Embedding，另一个从环境变量自动创建：

```python
# 只配置LLM，Embedding从环境变量创建
llm_config = LlmConfig.create_deepseek(api_key="sk-...")
config = Config(llm_config=llm_config)
# embedding_config会自动从OPENAI_API_KEY等环境变量创建
```

## 完整示例

```python
from config import Config
from InputConfig import LlmConfig, EmbeddingConfig
from main import HAmem

# 方式1：使用工厂方法
llm_config = LlmConfig.create_deepseek(
    api_key="your-key",
    model="deepseek-chat",
    temperature=0.7
)

embedding_config = EmbeddingConfig.create_openai(
    api_key="your-key",
    model="text-embedding-3-small",
    dimensions=1536
)

config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config,
    max_workers=10,
    embedding_batch_size=200
)

hamem = HAmem(config)

# 使用
result = hamem.build_memory(conversation_data)
answer = hamem.ask_question("你的问题")
```

## 注意事项

1. 如果同时提供新配置和环境变量，新配置优先
2. 如果只提供环境变量，系统会自动创建对应的配置对象
3. 必须至少配置LLM和Embedding中的一个API密钥
4. 更多提供商和选项请参考 `InputConfig.py` 中的 `PROVIDER_CONFIGS`

