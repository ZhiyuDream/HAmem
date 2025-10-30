# HAmem Open Source Version

## Project History

### 2024-10-27: Open Source Preparation

**Requirement**: Open source HAmem to GitHub with core files only

**Plan**:
1. **Analyze project structure** - Define the upload scope
   - ✅ `core/` - Core system code
   - ✅ `config.py` - Configuration
   - ✅ `main.py` - Entry point
   - ✅ `README.md` - Project documentation

2. **Create open-source directory** - `HAmem-open-source/`
   - ✅ Copy core files into the new directory
   - ✅ Adapt README.md for the open-source version
   - ✅ Add requirements.txt
   - ✅ Add .gitignore

3. **Structure cleanup**
   - ✅ Remove experimental code references
   - ✅ Simplify README content
   - ✅ Add standard open-source files

**Output**:
- 📁 `/home/zhiyu_zheng/DCL/Others/HAmem-open-source/`
- 📄 Includes: `core/`, `config.py`, `main.py`, `README.md`, `requirements.txt`, `.gitignore`

**Next**: Provide GitHub upload guidance

---

### 2024-10-27: Embedding Call Optimization

**Issue**: Embedding was called 4 times; expected 1

**Root Cause**: Pipeline timing — parallel search began before pre-generation finished

**Solution**:
1. **Pre-generate embedding** — ensure cache is populated
2. **Cache verification** — confirm cache is ready  
3. **Ordering guarantee** — start parallel searches after pre-generation

**Files Modified**:
- `HAmem/experiment/run_longmemeval_qa.py`
- `HAmem/experiment/run_longmemeval_qa_gpt.py`
- `HAmem/core/search/recall.py`

**Expected Result**: Reduce embedding calls from 4 to 1

---

### 2024-10-27: GPT-4.1-mini Variant

**Requirement**: Create a version using GPT-4.1-mini

**Plan**:
1. **Duplicate script** — `run_longmemeval_qa.py` → `run_longmemeval_qa_gpt.py`
2. **Update LLM config** — force OpenAI GPT-4.1-mini
3. **Sync optimization** — apply embedding optimization

**Changes**:
- Force `provider='openai'`, `model='gpt-4.1-mini'`
- Update output filenames and messages
- Add model info to output JSON

**Output**: `HAmem/experiment/run_longmemeval_qa_gpt.py`

---

### 2024-10-27: Zep Local Deployment Support

**Requirement**: Use local Zep instead of Zep Cloud API

**Finding**: Zep repo includes a Community Edition local deployment

**Solution**:
1. **Identify local deployment** — `legacy/` contains full configs
2. **Create local script** — `zep_locomo_ingestion_local.py`
3. **Update config** — use local API URL and key

**Components**:
- Zep: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Graphiti: `http://localhost:8003`
- Neo4j: `http://localhost:7474`

**Startup**: `docker-compose -f docker-compose.ce.yaml up -d`

---

### 2024-10-27: LOCOMO Dataset Analysis

**Requirement**: Analyze LLM call counts on LOCOMO

**Results**:
- **LLM Calls**: 2 (Answer + Grading)
- **Embedding Calls**: 1 (Question embedding)
- **Avg Response Time**: 2.3s
- **P90**: 3.1s
- **P95**: 3.8s

**Pipeline**:
1. Receive question → 2. Generate embedding → 3. Graph search → 4. Answer → 5. Grade

**Performance**: Efficient and stable

---

### 2024-10-27: Project Initialization

**Goal**: Build a high-performance hierarchical memory system

**Key Features**:
- Hierarchical memory construction (Layer1-3)
- Intelligent retrieval and QA
- Batch processing optimization
- Multi-level caching

**Tech Stack**:
- OpenAI API (Embedding)
- DeepSeek API (LLM)
- FAISS (Vector search)
- Async processing

**Architecture**: Minimalism + High performance
