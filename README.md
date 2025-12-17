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
├── 📚 InputConfig.py          # Flexible LLM/Embedding configuration
├── 📖 INPUT_FORMAT.md         # Input format specification
├── 📖 CONFIG_GUIDE.md         # Configuration guide
├── 🧪 experiment/             # Experiment scripts
│   ├── test_memory_building.py # Memory building test with token/time stats
│   └── test_qa.py             # QA system test with token/time stats
└── 📝 examples/               # Usage examples
    ├── chatbot_example.py     # Chatbot example using search_memory
    └── README.md              # Examples documentation
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

### 2. Create .env Configuration File

Create a `.env` file in the project root directory with the following environment variables:

```bash
# Create .env file in project root
cat > .env << EOF
# ============================================
# Required Configuration
# ============================================

# OpenAI API Configuration (for LLM and Embedding)
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
# If you need different embedding configuration, you can set it in config.py lines 30-31

# Neo4j Database Configuration (if using Neo4j)
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# ============================================
# Optional Configuration (with defaults)
# ============================================

# LLM Model Configuration
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai

# Embedding Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# Neo4j Feature Flags
USE_NEO4J=True
USE_HYBRID_SEARCH=true
NEO4J_DATABASE=neo4j

# Performance Configuration
MAX_RETRIES=3
BASE_DELAY=1.0
MAX_WORKERS=5
EMBEDDING_BATCH_SIZE=100

# Cache Configuration
CACHE_DIR=cache
MAX_MEMORY_CACHE_SIZE=1000

# Processing Configuration
FRAGMENT_MAX_LENGTH=6000
FRAGMENT_OVERLAP=200

# Search Configuration
SEARCH_TOP_K=10
SIMILARITY_THRESHOLD=0.7

# Other Configuration
DEBUG=false
LOG_LEVEL=INFO
EOF
```

**Important Notes:**
- `OPENAI_API_KEY` and `OPENAI_BASE_URL` are **required** for LLM and Embedding services
- If using Neo4j, `NEO4J_PASSWORD` is also **required**
- Other configuration items have reasonable defaults and can be modified as needed
- If using a custom API service, modify `OPENAI_BASE_URL` to the corresponding API address

### 3. Setup Neo4j Database

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

**Note**: If you don't want to use Neo4j, you can set `USE_NEO4J=False` in the `.env` file, and the system will use FAISS for vector search only.

### 4. Quick Start - Run Locomo Dataset Experiment

We provide a convenient parallel experiment script that can run memory building and QA testing for all conversations (0-9) at once:

```bash
# Navigate to the experiment directory
cd experiment

# Run memory building and QA testing for all conversations (0-9) (silent mode, suitable for background)
python run_locomo_experiment.py --dataset ../locomo/data/locomo10.json

# Run in background (recommended)
nohup python run_locomo_experiment.py --dataset ../locomo/data/locomo10.json > /dev/null 2>&1 &

# Specify model and parallelism
python run_locomo_experiment.py --dataset ../locomo/data/locomo10.json --model gpt-4o-mini --max-workers 5

# Only run memory building, skip QA testing
python run_locomo_experiment.py --dataset ../locomo/data/locomo10.json --skip-qa

# Run conversations in specified range
python run_locomo_experiment.py --dataset ../locomo/data/locomo10.json --start 0 --end 4
```

**Script Features:**
- ✅ **Silent Mode**: All output is saved to log files, not printed to terminal, suitable for background execution
- ✅ **Parallel Processing**: Supports multi-threaded parallel execution for improved efficiency
- ✅ **Isolated Namespaces**: Each conversation uses an independent namespace (`locomo_conv_0`, `locomo_conv_1`, etc.), ensuring no interference
- ✅ **Detailed Logs**: Each conversation's log is saved in `experiment/logs/conv_{idx}.log`
- ✅ **Main Log**: Overall progress and summary information is saved in `experiment/logs/main.log`
- ✅ **Summary Report**: After the experiment, generates `experiment/logs/summary.json` with all results
- ✅ **Token Statistics**: Automatically tracks token usage for all conversations

**Output Description:**
- **Main Log**: `experiment/logs/main.log` - Overall progress and summary information
- **Detailed Logs**: `experiment/logs/conv_{conversation_idx}.log` - Detailed logs for each conversation
- **Summary File**: `experiment/logs/summary.json` - Contains result statistics for all conversations

**View Progress:**
```bash
# View main log (overall progress)
tail -f experiment/logs/main.log

# View detailed log for a specific conversation
tail -f experiment/logs/conv_0.log

# View summary results
cat experiment/logs/summary.json
```

### 5. Basic Usage

#### Input Format

HAmem supports multiple input formats, and the system will automatically identify and convert them. **We recommend using the HAmem standard format**.

**HAmem Standard Format (Recommended)**:
```python
conversation_data = {
    "messages": [
        {
            "speaker": "user",
            "content": "User message content",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "speaker": "assistant",
            "content": "Assistant reply content",
            "timestamp": "2024-01-01T10:00:05"
        }
    ]
}
```

**Supported Formats**:
- ✅ HAmem Standard Format (Recommended)
- ✅ Messages Format (similar to OpenAI Chat API)
- ✅ Locomo Format
- ✅ Sessions Format
- ✅ Batch Format

For detailed format specifications, please refer to [INPUT_FORMAT.md](INPUT_FORMAT.md).

#### Usage Examples

**Basic Usage**:

```python
from core.main import HAmem

# Initialize
hamem = HAmem()

# Method 1: Read from file (automatic format detection)
result = hamem.build_memory_from_file("conversation.json", namespace="my_project")

# Method 2: Pass data directly (HAmem Standard Format)
conversation_data = {
    "messages": [
        {
            "speaker": "user",
            "content": "User message",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "speaker": "assistant",
            "content": "Assistant reply",
            "timestamp": "2024-01-01T10:00:05"
        }
    ]
}
result = hamem.build_memory(conversation_data, namespace="my_project")

# Method 3: Use other formats (system will automatically convert)
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
result = hamem.build_memory(conversation_data, namespace="my_project")

# Build memory (namespace for data isolation)
namespace = "project_1"
memory_result = hamem.build_memory(conversation_data, namespace=namespace)

# Search memory
results = hamem.search_memory("What is the user working on?", namespace=namespace)

# Q&A
answer = hamem.ask_question("What did we discuss?", namespace=namespace)
```

**Chatbot Example** (using search_memory to retrieve historical information and generate conversational responses):

```python
from core.main import HAmem
from examples.chatbot_example import ChatBot

# Initialize HAmem
hamem = HAmem()

# Optional: Build some memory first
hamem.build_memory_from_file("conversation.json")

# Create chatbot
chatbot = ChatBot(
    hamem=hamem,
    namespace="default",
    save_conversation=True  # Save new conversations to memory
)

# Start interactive chat
chatbot.interactive_chat()

# Or single conversation
response = chatbot.chat("Hello, what did we talk about before?")
print(response)
```

For detailed instructions, please refer to [examples/README.md](examples/README.md).
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

#### Unified Configuration Interface (Recommended)
```bash
# LLM Configuration (Unified Interface)
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_PROVIDER=deepseek

# Embedding Configuration (Unified Interface)
EMBEDDING_API_KEY=your-api-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai
```

#### Backward Compatible Configuration (Still Supported)
```bash
# Old configuration method (still supported, but unified interface is recommended)
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
```

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

#### Method 1: Use New Flexible Configuration (Recommended)
```python
from config import Config
from InputConfig import LlmConfig, EmbeddingConfig

# Create LLM configuration
llm_config = LlmConfig.create_deepseek(
    api_key="your-deepseek-key",
    model="deepseek-chat",
    temperature=0.7
)

# Create Embedding configuration
embedding_config = EmbeddingConfig.create_openai(
    api_key="your-openai-key",
    model="text-embedding-3-small",
    dimensions=1536
)

# Create complete configuration
config = Config(
    llm_config=llm_config,
    embedding_config=embedding_config,
    max_workers=10,
    embedding_batch_size=200
)

hamem = HAmem(config)
```

#### Method 2: Use Environment Variables (Recommended, Simplest)
```python
from config import Config

# Automatically load from environment variables (prioritizes unified interface, also supports old configuration)
config = Config()
hamem = HAmem(config)
```

**Environment Variable Priority**:
1. Unified configuration interface (`LLM_API_KEY`, `EMBEDDING_API_KEY`) - Recommended
2. Old configuration method (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) - Backward compatible

#### Method 3: Mixed Configuration
```python
from config import Config
from InputConfig import LlmConfig

# Only customize LLM configuration, Embedding uses environment variables
llm_config = LlmConfig.create_openai(
    api_key="your-key",
    model="gpt-4o-mini"
)

config = Config(
    llm_config=llm_config,
    # embedding_config will be automatically created from environment variables
)

hamem = HAmem(config)
```

#### Supported LLM Providers
- OpenAI: `LlmConfig.create_openai(...)`
- DeepSeek: `LlmConfig.create_deepseek(...)`
- Anthropic: `LlmConfig.create_anthropic(...)`
- Ollama: `LlmConfig.create_ollama(...)`
- Groq: `LlmConfig.create_groq(...)`
- For more providers, please refer to `InputConfig.py`

#### Supported Embedding Providers
- OpenAI: `EmbeddingConfig.create_openai(...)`
- DeepSeek: `EmbeddingConfig.create_deepseek(...)`
- For more providers, please refer to `InputConfig.py`

## 🛠️ Utility Tools

### Locomo Experiment (Recommended)
```bash
# One-click run locomo experiment (recommended)
python experiment/run_locomo.py <conversation_idx> [--provider {openai,deepseek}] [--dataset DATASET] [--skip-storage]

# Examples
python experiment/run_locomo.py 0
python experiment/run_locomo.py 0 --provider deepseek
python experiment/run_locomo.py 0 --dataset /path/to/locomo10.json
```

For detailed usage instructions, please refer to [experiment/README.md](experiment/README.md)

### Neo4j Management
```bash
# Clear all data in Neo4j and cache
python clear.py

# Clear specific namespace
python clear.py <namespace>
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