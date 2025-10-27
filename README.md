# HAmem - Hierarchical Memory System

一个高性能的分层记忆系统，专为对话记忆构建和检索而设计。

## 🎯 设计理念

### 极简主义
- **最少必要组件** - 只保留核心功能
- **清晰职责划分** - 每个模块单一职责  
- **简单数据流** - 线性处理流程

### 高性能
- **批量处理优先** - 减少API调用
- **智能缓存策略** - 避免重复计算
- **异步处理** - 提升并发性能

## 🏗️ 架构设计

```
HAmem/
├── 🧠 core/                    # 核心系统
│   ├── infrastructure/         # 基础设施
│   │   ├── embedding.py        # Embedding管理
│   │   ├── llm.py             # LLM客户端
│   │   └── cache.py           # 缓存系统
│   ├── search/                # 检索系统
│   │   ├── recall.py          # 召回引擎
│   │   ├── expansion.py       # 图扩展
│   │   ├── router.py          # 问题路由
│   │   ├── answer.py          # 答案生成
│   │   └── qa_system.py       # 问答系统
│   └── fragment/              # 片段处理
│       ├── buffer_manager.py  # 缓冲区管理
│       ├── fragment_processor.py # 片段处理器
│       └── fragment_storage.py   # 片段存储
├── ⚙️ config.py               # 配置管理
├── 🚀 main.py                 # 主入口
└── 📋 requirements.txt        # 依赖
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

#### 方式1: 使用.env文件 (推荐)
```bash
# 在HAmem目录下创建.env文件
echo "OPENAI_API_KEY=your-openai-api-key" > .env
echo "DEEPSEEK_API_KEY=your-deepseek-api-key" >> .env
```

#### 方式2: 环境变量
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. 基础用法

```python
from main import HAmem

# 初始化
hamem = HAmem()

# 构建记忆
conversation_data = {
    "sessions": [
        {
            "session_id": "session_1",
            "timestamp": "2024-01-01T10:00:00Z",
            "turns": [
                {
                    "speaker": "user",
                    "text": "Hello, I'm working on AI memory systems.",
                    "timestamp": "2024-01-01T10:00:00Z"
                },
                {
                    "speaker": "assistant",
                    "text": "That sounds interesting! What kind of system?",
                    "timestamp": "2024-01-01T10:00:01Z"
                }
            ]
        }
    ]
}

# 构建记忆
memory_result = hamem.build_memory(conversation_data)

# 搜索记忆
results = hamem.search_memory("What is the user working on?")

# 问答
answer = hamem.ask_question("What did we discuss?")
```

## 📊 核心功能

### 1. 记忆构建
- **智能分割** - 基于内容长度和语义的智能分割
- **分层处理** - Layer1(实体关系) → Layer2(事件状态) → Layer3(模式分析)
- **批量优化** - 批量embedding生成和LLM调用

### 2. 记忆检索
- **向量检索** - 高效的相似度搜索
- **智能排序** - 基于相关性和重要性的结果排序
- **上下文构建** - 自动构建问答上下文

### 3. 问答系统
- **智能问答** - 基于检索记忆的智能问答
- **置信度评估** - 答案质量评估
- **来源追踪** - 答案来源的可追溯性

## ⚡ 性能优化

### 批量处理
```python
# 批量embedding生成
embeddings = embedding_manager.batch_get_embeddings(texts)

# 批量LLM调用
results = llm_client.batch_generate(prompts)
```

### 智能缓存
```python
# 多级缓存策略
- 内存缓存 (最快)
- 磁盘缓存 (持久化)
- Redis缓存 (分布式，可选)
```

### 异步处理
```python
# 异步embedding生成
async def process_fragments_async(fragments):
    tasks = [process_fragment_async(f) for f in fragments]
    results = await asyncio.gather(*tasks)
    return results
```

## 🔧 配置选项

### 环境变量
```bash
# API配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 模型配置
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# 性能配置
MAX_RETRIES=3
BASE_DELAY=1.0
MAX_WORKERS=5
EMBEDDING_BATCH_SIZE=100

# 缓存配置
CACHE_DIR=cache
MAX_MEMORY_CACHE_SIZE=1000

# 存储配置
STORAGE_DIR=storage
```

### 代码配置
```python
from config import Config

config = Config(
    openai_api_key="your-key",
    llm_model="gpt-4",
    embedding_model="text-embedding-3-large",
    max_workers=10,
    embedding_batch_size=200
)

hamem = HAmem(config)
```

## 🧪 测试

### 运行基础测试
```bash
python main.py
```

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 许可证

MIT License

---

**HAmem - 让AI拥有更好的记忆** 🧠✨