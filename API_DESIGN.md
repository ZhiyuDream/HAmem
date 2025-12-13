# HAmem 开源接口设计方案

## 一、核心设计原则

1. **输入格式标准化**：统一对话输入格式，便于用户使用
2. **配置灵活性**：支持任意LLM提供商（OpenAI兼容API）
3. **记忆隔离**：通过namespace实现多用户/多场景隔离
4. **操作便捷性**：提供一键清除、查询等管理接口
5. **可扩展性**：预留扩展接口，便于后续功能增强

---

## 二、输入格式规范

### 2.1 标准对话格式

```json
{
  "messages": [
    {
      "speaker": "Alice",
      "timestamp": "2024-01-15T10:30:00",
      "content": "今天天气真好"
    },
    {
      "speaker": "Bob", 
      "timestamp": "2024-01-15T10:31:00",
      "content": "是啊，适合出去走走"
    }
  ],
  "metadata": {
    "conversation_id": "conv_001",
    "source": "user_input"
  }
}
```

### 2.2 字段说明

- **speaker** (必需): 说话者名称，字符串
- **timestamp** (必需): ISO 8601格式时间戳，字符串
- **content** (必需): 对话内容，字符串
- **metadata** (可选): 额外元数据，字典

### 2.3 批量输入格式

支持一次性输入多条对话：

```json
{
  "conversations": [
    {
      "messages": [...],
      "metadata": {...}
    },
    {
      "messages": [...],
      "metadata": {...}
    }
  ]
}
```

---

## 三、LLM配置方案

### 3.1 配置结构

```python
llm_config = {
    "default_provider": "openai",  # 默认提供商
    "providers": {
        "openai": {
            "api_key": "sk-...",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "timeout": 60.0
        },
        "deepseek": {
            "api_key": "sk-...",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "timeout": 60.0
        },
        "custom": {
            "api_key": "sk-...",
            "base_url": "https://api.custom.com/v1",
            "model": "custom-model",
            "timeout": 60.0
        }
    },
    "embedding": {
        "provider": "openai",
        "api_key": "sk-...",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small"
    }
}
```

### 3.2 配置方式

**方式1：通过代码配置**
```python
from hamem import HAmem, LLMConfig

llm_config = LLMConfig(
    default_provider="openai",
    providers={
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        }
    }
)

hamem = HAmem(llm_config=llm_config)
```

**方式2：通过环境变量配置**
```bash
# .env文件
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

DEFAULT_LLM_PROVIDER=openai
```

**方式3：通过配置文件（YAML/JSON）**
```yaml
# config.yaml
llm:
  default_provider: openai
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      base_url: https://api.openai.com/v1
      model: gpt-4o-mini
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
      base_url: https://api.deepseek.com/v1
      model: deepseek-chat
embedding:
  provider: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
```

---

## 四、核心API接口

### 4.1 初始化

```python
from hamem import HAmem

# 方式1：使用默认配置（从环境变量读取）
hamem = HAmem()

# 方式2：自定义配置
hamem = HAmem(
    llm_config=llm_config,
    cache_dir="./cache",
    storage_dir="./storage",
    neo4j_config={
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j"
    }
)
```

### 4.2 构建记忆

```python
# 输入标准格式的对话
conversation = {
    "messages": [
        {
            "speaker": "Alice",
            "timestamp": "2024-01-15T10:30:00",
            "content": "今天天气真好"
        }
    ]
}

# 构建记忆（指定namespace实现隔离）
result = hamem.build_memory(
    conversation=conversation,
    namespace="user_001"  # 用户隔离
)

# 返回结果
print(result)
# {
#     "namespace": "user_001",
#     "total_fragments": 10,
#     "total_entities": 25,
#     "total_events": 15,
#     "total_clusters": 3,
#     "time_stats": {...},
#     "token_stats": {...}
# }
```

### 4.3 问答接口

```python
# 提问（自动使用对应namespace的记忆）
answer = hamem.ask_question(
    question="Alice今天说了什么？",
    namespace="user_001"
)

print(answer)
# {
#     "answer": "Alice说：今天天气真好",
#     "confidence": 0.95,
#     "sources": [...],
#     "token_usage": {...}
# }
```

### 4.4 搜索记忆

```python
# 搜索记忆
results = hamem.search_memory(
    query="天气",
    top_k=5,
    namespace="user_001"
)

print(results)
# [
#     {
#         "content": "今天天气真好",
#         "speaker": "Alice",
#         "timestamp": "2024-01-15T10:30:00",
#         "similarity": 0.92
#     },
#     ...
# ]
```

### 4.5 记忆管理

```python
# 清除指定namespace的记忆（Neo4j + 缓存）
hamem.clear_memory(namespace="user_001")

# 列出所有namespace
namespaces = hamem.list_namespaces()
print(namespaces)
# ["user_001", "user_002", "default"]

# 获取记忆统计信息
stats = hamem.get_memory_stats(namespace="user_001")
print(stats)
# {
#     "namespace": "user_001",
#     "fragments": 10,
#     "entities": 25,
#     "events": 15,
#     "clusters": 3,
#     "cache_size": "50MB"
# }
```

---

## 五、需要补充的功能

### 5.1 输入验证

- **格式校验**：检查输入是否符合标准格式
- **时间格式校验**：验证timestamp是否为有效ISO 8601格式
- **必填字段检查**：确保speaker、timestamp、content都存在

### 5.2 错误处理

- **配置错误**：LLM配置缺失或无效时的提示
- **API调用失败**：重试机制和降级策略
- **数据格式错误**：友好的错误提示和建议

### 5.3 批量操作

- **批量构建记忆**：支持一次性处理多条对话
- **批量问答**：支持一次性回答多个问题
- **批量清除**：支持清除多个namespace

### 5.4 监控和统计

- **Token使用统计**：按namespace、按时间统计
- **性能监控**：各阶段耗时统计
- **错误日志**：记录错误信息便于排查

### 5.5 数据导出/导入

- **导出记忆**：将指定namespace的记忆导出为JSON
- **导入记忆**：从JSON文件导入记忆
- **备份/恢复**：支持记忆的备份和恢复

### 5.6 高级功能

- **记忆更新**：支持增量更新已有记忆
- **记忆合并**：支持合并多个namespace的记忆
- **记忆版本管理**：支持记忆的版本控制

---

## 六、实现建议

### 6.1 配置管理重构

创建 `LLMConfig` 类，支持：
- 多提供商配置
- 动态添加/删除提供商
- 配置验证
- 配置热更新（可选）

### 6.2 输入格式验证器

创建 `ConversationValidator` 类：
- 验证输入格式
- 自动修复常见错误（如时间格式）
- 提供详细的错误报告

### 6.3 记忆管理器

创建 `MemoryManager` 类：
- 统一管理所有namespace
- 提供清除、查询、统计接口
- 支持批量操作

### 6.4 CLI工具

提供命令行工具：
```bash
# 构建记忆
hamem build --input conversation.json --namespace user_001

# 问答
hamem ask --question "..." --namespace user_001

# 清除记忆
hamem clear --namespace user_001

# 列出所有namespace
hamem list

# 查看统计
hamem stats --namespace user_001
```

---

## 七、示例代码

### 7.1 完整示例

```python
from hamem import HAmem, LLMConfig

# 1. 配置LLM
llm_config = LLMConfig(
    default_provider="openai",
    providers={
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        }
    }
)

# 2. 初始化
hamem = HAmem(llm_config=llm_config)

# 3. 构建记忆
conversation = {
    "messages": [
        {
            "speaker": "Alice",
            "timestamp": "2024-01-15T10:30:00",
            "content": "今天天气真好"
        }
    ]
}
result = hamem.build_memory(conversation, namespace="user_001")

# 4. 问答
answer = hamem.ask_question("Alice说了什么？", namespace="user_001")
print(answer["answer"])

# 5. 清除记忆
hamem.clear_memory(namespace="user_001")
```

---

## 八、待讨论的问题

1. **是否需要支持流式输出**？问答时是否支持流式返回答案？
2. **是否需要异步接口**？是否提供async/await版本？
3. **是否需要REST API**？是否提供HTTP接口？
4. **配置优先级**：代码配置 > 配置文件 > 环境变量？
5. **错误恢复策略**：LLM调用失败时是否自动重试？重试几次？
6. **记忆更新策略**：增量更新时如何处理冲突？
7. **多语言支持**：是否需要支持多语言输入/输出？

