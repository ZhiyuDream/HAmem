# HAmem - Hierarchical Memory System

A high-performance hierarchical memory system designed for conversational memory construction and retrieval, powered by **Neo4j graph database** and **hybrid search architecture**.

## 🎯 Design Philosophy

### Minimalism
- **Minimal Components** - Only essential core functionality
- **Clear Responsibility Division** - Single responsibility per module  
- **Simple Data Flow** - Linear processing pipeline

### High Performance
- **Batch Processing Priority** - Reduce API calls
- **Intelligent Caching Strategy** - Avoid redundant computations
- **Asynchronous Processing** - Improve concurrent performance
- **Hybrid Search** - FAISS vector search + Neo4j graph expansion

### Graph-Powered
- **Neo4j Integration** - Native graph database for relationship storage
- **Multi-tenant Support** - Namespace isolation for different projects/users
- **Graph Expansion** - Discover related memories through relationships

## 🏗️ Architecture

```
HAmem/
├── 🧠 core/                    # Core system
│   ├── infrastructure/         # Infrastructure
│   │   ├── embedding.py        # Embedding management
│   │   ├── llm.py             # LLM client
│   │   ├── cache.py           # Cache system (FAISS)
│   │   ├── neo4j_client.py    # Neo4j client
│   │   ├── neo4j_storage_base.py # Neo4j storage base
│   │   ├── neo4j_hybrid_search.py # Hybrid search (FAISS + Neo4j)
│   │   └── token_tracker.py   # Token usage tracking
│   ├── search/                # Retrieval system
│   │   ├── recall.py          # Recall engine
│   │   ├── expansion.py       # Graph expansion
│   │   ├── router.py          # Question routing
│   │   ├── answer.py          # Answer generation
│   │   ├── qa_system.py       # Q&A system
│   │   └── neo4j_hybrid_recall.py # Hybrid recall
│   ├── layer1/                 # Layer1: Entity Relations
│   │   ├── neo4j_storage.py   # Neo4j storage
│   │   └── ...
│   ├── layer2/                 # Layer2: Event States
│   │   ├── neo4j_storage.py   # Neo4j storage
│   │   └── ...
│   ├── layer3/                 # Layer3: Pattern Analysis
│   │   ├── neo4j_storage.py   # Neo4j storage
│   │   └── ...
│   └── fragment/              # Fragment processing
│       ├── buffer_manager.py  # Buffer management
│       ├── fragment_processor.py # Fragment processor
│       ├── fragment_storage.py   # Fragment storage
│       └── neo4j_storage.py   # Fragment Neo4j storage
├── ⚙️ config.py               # Configuration
├── 🚀 main.py                 # Entry point
├── 📋 requirements.txt        # Dependencies
└── 🛠️ tools/                  # Utility scripts
    ├── calculate_token_count.py # Token calculation (estimation)
    ├── calculate_token_and_time_real.py # Token calculation (real API calls)
    └── clear_neo4j.py         # Neo4j cleanup utility
```

### Hybrid Search Architecture

```
Query Flow:
1. FAISS Vector Search (milliseconds)
   → Find initial relevant nodes
2. Neo4j Graph Expansion
   → Discover related nodes through relationships
3. Return combined results
   → Initial nodes + expanded nodes
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Neo4j Database

HAmem uses Neo4j as the primary storage backend. Make sure Neo4j is running:

```bash
# Using Docker (recommended)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest

# Or install Neo4j locally
# Visit: https://neo4j.com/download/
```

### 3. Configure API Keys and Neo4j

#### Method 1: Using .env file (Recommended)
```bash
# Create .env file in HAmem directory
cat > .env << EOF
# API Keys
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
USE_NEO4J=true
USE_HYBRID_SEARCH=true
EOF
```

#### Method 2: Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
export USE_NEO4J="true"
export USE_HYBRID_SEARCH="true"
```

### 4. Basic Usage

```python
from main import HAmem

# Initialize
hamem = HAmem()

# Build memory with namespace (multi-tenant support)
conversation_data = {
    "sessions": [
        {
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

# Build memory (namespace for data isolation)
namespace = "project_1"
memory_result = hamem.build_memory(conversation_data, namespace=namespace)

# Search memory
results = hamem.search_memory("What is the user working on?", namespace=namespace)

# Q&A
answer = hamem.ask_question("What did we discuss?", namespace=namespace)
```

## 📊 Core Features

### 1. Memory Construction
- **Intelligent Segmentation** - Smart segmentation based on content length and semantics
- **Hierarchical Processing** - Layer1(Entity Relations) → Layer2(Event States) → Layer3(Pattern Analysis)
- **Batch Optimization** - Batch embedding generation and LLM calls
- **Neo4j Storage** - Native graph database for persistent storage
- **Multi-tenant Support** - Namespace isolation for different projects/users

### 2. Memory Retrieval
- **Hybrid Search** - FAISS vector search (milliseconds) + Neo4j graph expansion
- **Vector Retrieval** - Efficient similarity search using FAISS
- **Graph Expansion** - Discover related memories through relationships
- **Intelligent Ranking** - Results ranking based on relevance and importance
- **Context Construction** - Automatic Q&A context building

### 3. Q&A System
- **Intelligent Q&A** - Smart Q&A based on retrieved memories
- **Confidence Assessment** - Answer quality evaluation
- **Source Tracking** - Traceability of answer sources
- **Graph-aware Answers** - Leverage relationship context for better answers

### 4. Advanced Features
- **Token Tracking** - Monitor API token usage across all operations
- **Namespace Isolation** - Separate data for different projects/users
- **Graph Visualization** - Explore memory relationships in Neo4j Browser

## ⚡ Performance Optimization

### Hybrid Search Architecture
```python
# Hybrid search combines FAISS and Neo4j
# 1. FAISS: Millisecond-level vector search
# 2. Neo4j: Graph expansion for related nodes
results = hybrid_search.hybrid_search(
    query="user question",
    vector_top_k=10,
    max_hops=2,
    expand_limit=50
)
```

### Batch Processing
```python
# Batch embedding generation
embeddings = embedding_manager.batch_get_embeddings(texts)

# Batch LLM calls
results = llm_client.batch_generate(prompts)

# Batch Neo4j operations
neo4j_storage.batch_create_nodes(nodes)
```

### Intelligent Caching
```python
# Multi-level caching strategy
- FAISS index (in-memory, fastest)
- Disk cache (persistent embeddings)
- Neo4j (graph relationships)
```

### Asynchronous Processing
```python
# Asynchronous embedding generation
async def process_fragments_async(fragments):
    tasks = [process_fragment_async(f) for f in fragments]
    results = await asyncio.gather(*tasks)
    return results
```

### Performance Metrics
- **Vector Search**: < 10ms (FAISS)
- **Graph Expansion**: < 50ms (Neo4j, 2 hops)
- **Total Query Time**: < 100ms (typical)
- **Batch Processing**: 90% reduction in API calls

## 🔧 Configuration Options

### Environment Variables
```bash
# API Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Model Configuration
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small

# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
USE_NEO4J=true
USE_HYBRID_SEARCH=true

# Performance Configuration
MAX_RETRIES=3
BASE_DELAY=1.0
MAX_WORKERS=5
EMBEDDING_BATCH_SIZE=100

# Cache Configuration
CACHE_DIR=cache
MAX_MEMORY_CACHE_SIZE=1000

# Fragment Configuration
FRAGMENT_MAX_LENGTH=6000
FRAGMENT_OVERLAP=200

# Search Configuration
SEARCH_TOP_K=10
SIMILARITY_THRESHOLD=0.7
```

### Code Configuration

#### 方式1：使用新的灵活配置（推荐）
```python
from config import Config
from InputConfig import LlmConfig, EmbeddingConfig

# 创建LLM配置
llm_config = LlmConfig.create_deepseek(
    api_key="your-deepseek-key",
    model="deepseek-chat",
    temperature=0.7
)

# 创建Embedding配置
embedding_config = EmbeddingConfig.create_openai(
    api_key="your-openai-key",
    model="text-embedding-3-small",
    dimensions=1536
)

# 创建完整配置
config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config,
    max_workers=10,
    embedding_batch_size=200
)

hamem = HAmem(config)
```

#### 方式2：使用环境变量（向后兼容）
```python
from config import Config

# 自动从环境变量加载（支持旧配置方式）
config = Config()
hamem = HAmem(config)
```

#### 方式3：混合配置
```python
from config import Config
from InputConfig import LlmConfig

# 只自定义LLM配置，Embedding使用环境变量
llm_config = LlmConfig.create_openai(
    api_key="your-key",
    model="gpt-4o-mini"
)

config = Config(
    llm_config=llm_config,
    # embedding_config 会自动从环境变量创建
)

hamem = HAmem(config)
```

#### 支持的LLM提供商
- OpenAI: `LlmConfig.create_openai(...)`
- DeepSeek: `LlmConfig.create_deepseek(...)`
- Anthropic: `LlmConfig.create_anthropic(...)`
- Ollama: `LlmConfig.create_ollama(...)`
- Groq: `LlmConfig.create_groq(...)`
- 更多提供商请参考 `InputConfig.py`

#### 支持的Embedding提供商
- OpenAI: `EmbeddingConfig.create_openai(...)`
- DeepSeek: `EmbeddingConfig.create_deepseek(...)`
- 更多提供商请参考 `InputConfig.py`

## 🛠️ Utility Tools

### Token Calculation
```bash
# Estimate token usage (fast, no API calls)
python calculate_token_count.py <conversation_idx> [dataset_path]

# Real token calculation (accurate, uses API)
python calculate_token_and_time_real.py <conversation_idx> [llm_provider] [dataset_path]
```

### Neo4j Management
```bash
# Clear all data in Neo4j
python clear_neo4j.py

# Clear specific namespace
python clear_neo4j.py <namespace>
```

## 🧪 Testing

### Run Basic Tests
```bash
python main.py
```

### Verify Neo4j Connection
```bash
# Access Neo4j Browser
# http://localhost:7474
# Login with your Neo4j credentials
```

## 📚 Documentation

- **API Design**: See `API_DESIGN.md` for detailed API documentation
- **Project History**: See `PROJECT_HISTORY.md` for development history and features

## 🔍 Key Differences from Original HAmem

This open-source version includes:
- ✅ **Neo4j Integration** - Graph database for relationship storage
- ✅ **Hybrid Search** - FAISS + Neo4j for better retrieval
- ✅ **Multi-tenant Support** - Namespace isolation
- ✅ **Token Tracking** - Monitor API usage
- ✅ **Production Ready** - Optimized for deployment

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

MIT License

---

**HAmem - Give AI Better Memory** 🧠✨