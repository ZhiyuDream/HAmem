# HAmem - Hierarchical Memory System

A high-performance hierarchical memory system designed for conversational memory construction and retrieval.

## 🎯 Design Philosophy

### Minimalism
- **Minimal Components** - Only essential core functionality
- **Clear Responsibility Division** - Single responsibility per module  
- **Simple Data Flow** - Linear processing pipeline

### High Performance
- **Batch Processing Priority** - Reduce API calls
- **Intelligent Caching Strategy** - Avoid redundant computations
- **Asynchronous Processing** - Improve concurrent performance

## 🏗️ Architecture

```
HAmem/
├── 🧠 core/                    # Core system
│   ├── infrastructure/         # Infrastructure
│   │   ├── embedding.py        # Embedding management
│   │   ├── llm.py             # LLM client
│   │   └── cache.py           # Cache system
│   ├── search/                # Retrieval system
│   │   ├── recall.py          # Recall engine
│   │   ├── expansion.py       # Graph expansion
│   │   ├── router.py          # Question routing
│   │   ├── answer.py          # Answer generation
│   │   └── qa_system.py       # Q&A system
│   └── fragment/              # Fragment processing
│       ├── buffer_manager.py  # Buffer management
│       ├── fragment_processor.py # Fragment processor
│       └── fragment_storage.py   # Fragment storage
├── ⚙️ config.py               # Configuration
├── 🚀 main.py                 # Entry point
└── 📋 requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

#### Method 1: Using .env file (Recommended)
```bash
# Create .env file in HAmem directory
echo "OPENAI_API_KEY=your-openai-api-key" > .env
echo "DEEPSEEK_API_KEY=your-deepseek-api-key" >> .env
```

#### Method 2: Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. Basic Usage

```python
from main import HAmem

# Initialize
hamem = HAmem()

# Build memory
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

# Build memory
memory_result = hamem.build_memory(conversation_data)

# Search memory
results = hamem.search_memory("What is the user working on?")

# Q&A
answer = hamem.ask_question("What did we discuss?")
```

## 📊 Core Features

### 1. Memory Construction
- **Intelligent Segmentation** - Smart segmentation based on content length and semantics
- **Hierarchical Processing** - Layer1(Entity Relations) → Layer2(Event States) → Layer3(Pattern Analysis)
- **Batch Optimization** - Batch embedding generation and LLM calls

### 2. Memory Retrieval
- **Vector Retrieval** - Efficient similarity search
- **Intelligent Ranking** - Results ranking based on relevance and importance
- **Context Construction** - Automatic Q&A context building

### 3. Q&A System
- **Intelligent Q&A** - Smart Q&A based on retrieved memories
- **Confidence Assessment** - Answer quality evaluation
- **Source Tracking** - Traceability of answer sources

## ⚡ Performance Optimization

### Batch Processing
```python
# Batch embedding generation
embeddings = embedding_manager.batch_get_embeddings(texts)

# Batch LLM calls
results = llm_client.batch_generate(prompts)
```

### Intelligent Caching
```python
# Multi-level caching strategy
- Memory cache (fastest)
- Disk cache (persistent)
- Redis cache (distributed, optional)
```

### Asynchronous Processing
```python
# Asynchronous embedding generation
async def process_fragments_async(fragments):
    tasks = [process_fragment_async(f) for f in fragments]
    results = await asyncio.gather(*tasks)
    return results
```

## 🔧 Configuration Options

### Environment Variables
```bash
# API Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Model Configuration
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# Performance Configuration
MAX_RETRIES=3
BASE_DELAY=1.0
MAX_WORKERS=5
EMBEDDING_BATCH_SIZE=100

# Cache Configuration
CACHE_DIR=cache
MAX_MEMORY_CACHE_SIZE=1000

# Storage Configuration
STORAGE_DIR=storage
```

### Code Configuration
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

## 🧪 Testing

### Run Basic Tests
```bash
python main.py
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

MIT License

---

**HAmem - Give AI Better Memory** 🧠✨