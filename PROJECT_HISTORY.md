# HAmem Open Source Version

## Project History

### 2024-12-XX: Token数量计算工具（真实调用版本）

**Requirement**: 在实际调用LLM时统计locomo数据集中一个conversation的token数量，确保统计准确性

**Implementation**:
1. **修改LLMClient支持token统计**
   - ✅ 修改 `core/infrastructure/llm.py` 添加 `return_usage` 参数
   - ✅ `call_llm` 方法现在可以返回token使用信息（prompt_tokens, completion_tokens, total_tokens）
   - ✅ 支持OpenAI和DeepSeek两种提供商的token统计

2. **创建真实调用版本的Token计算脚本**
   - ✅ 创建 `calculate_token_count_real.py` - 实际调用LLM并统计token
   - ✅ 实际调用LLM进行fragment splitting判断
   - ✅ 实际调用LLM进行Layer1提取
   - ✅ 实际调用LLM进行Layer2提取
   - ✅ 从API响应中获取真实的token使用情况

3. **功能特性**
   - ✅ 使用tiktoken计算原始对话内容的token数（估算）
   - ✅ 从API响应获取Fragment splitting的实际token数
   - ✅ 从API响应获取Layer1 extraction的实际token数（prompt + completion）
   - ✅ 从API响应获取Layer2 extraction的实际token数（prompt + completion）
   - ✅ 详细的token统计报告（分类统计、调用次数、总计）

4. **保留模拟版本**
   - ✅ `calculate_token_count.py` - 使用tiktoken估算的版本（不调用API，速度快）

**Files Created**:
- `calculate_token_count_real.py` - 真实调用版本

**Files Modified**:
- `core/infrastructure/llm.py` - 添加return_usage参数支持

**Usage**:
```bash
# 真实调用版本（准确但会消耗API调用）
python calculate_token_count_real.py <conversation_idx> [llm_provider] [dataset_path]
# 示例: python calculate_token_count_real.py 0 openai
#       python calculate_token_count_real.py 0 deepseek

# 模拟版本（快速估算，不调用API）
python calculate_token_count.py <conversation_idx> [dataset_path]
```

**Output (真实调用版本)**: 
- 原始对话内容token数（tiktoken估算）
- Fragment splitting prompts的实际token数（prompt + completion，含调用次数）
- Layer1 extraction prompts的实际token数（prompt + completion，含fragment数量）
- Layer2 extraction prompts的实际token数（prompt + completion，含fragment数量）
- 总计token数
- 平均每个fragment的token数

---

### 2024-12-XX: Neo4j 存储集成

**Requirement**: 将存储系统从本地文件（JSONL）迁移到 Neo4j 图数据库

**Implementation**:
1. **Neo4j 连接管理**
   - ✅ 创建 `core/infrastructure/neo4j_client.py` - Neo4j 客户端封装
   - ✅ 支持从环境变量读取配置（`.env` 文件）
   - ✅ 修复 `id()` 函数弃用警告，使用 `elementId()` 替代

2. **Neo4j 存储基类**
   - ✅ 创建 `core/infrastructure/neo4j_storage_base.py` - 通用 Neo4j 存储基类
   - ✅ 提供节点创建、更新、查询、删除等基础操作
   - ✅ 支持命名空间隔离

3. **各层 Neo4j Storage 实现**
   - ✅ `core/layer1/neo4j_storage.py` - Layer1 实体和关系存储
   - ✅ `core/layer2/neo4j_storage.py` - Layer2 事件、状态、上下文存储
   - ✅ `core/layer3/neo4j_storage.py` - Layer3 聚类、模式、偏好、规则存储

4. **配置更新**
   - ✅ 更新 `config.py` 添加 Neo4j 配置项
   - ✅ 更新 `requirements.txt` 添加 `neo4j>=5.0.0` 依赖
   - ✅ 创建 `.env.example` 配置模板
   - ✅ 默认启用 Neo4j 存储（`USE_NEO4J=true`）

5. **测试验证**
   - ✅ 创建 `test_neo4j_basic.py` 测试脚本
   - ✅ 验证节点创建、关系创建、查询等基本功能
   - ✅ 验证分层结构（Layer0 → Layer1 → Layer2 → Layer3）

**Files Created**:
- `core/infrastructure/neo4j_client.py`
- `core/infrastructure/neo4j_storage_base.py`
- `core/layer1/neo4j_storage.py`
- `core/layer2/neo4j_storage.py`
- `core/layer3/neo4j_storage.py`
- `test_neo4j_basic.py`
- `.env.example`

**Files Modified**:
- `config.py` - 添加 Neo4j 配置
- `requirements.txt` - 添加 neo4j 依赖

**Next Steps**:
- 集成到现有系统，根据 `USE_NEO4J` 配置选择存储后端
- 在真实数据集上验证性能

---

### 2024-12-XX: Neo4j 混合检索架构（FAISS + Neo4j）

**Requirement**: 实现混合检索架构，结合 FAISS 向量搜索和 Neo4j 图扩展的优势

**Implementation**:
1. **混合检索核心类**
   - ✅ 创建 `core/infrastructure/neo4j_hybrid_search.py` - 混合检索核心类
   - ✅ 结合 UnifiedCache 的 FAISS 索引进行快速向量搜索（毫秒级）
   - ✅ 使用 Neo4j 进行高效的图扩展和关系查询
   - ✅ 保留所有现有性能优化（批量处理、去重、缓存等）

2. **混合召回模块**
   - ✅ 创建 `core/search/neo4j_hybrid_recall.py` - 混合召回接口
   - ✅ 支持多层召回并扩展（Layer1/2/3）
   - ✅ 支持按类型召回并扩展

3. **数据同步机制**
   - ✅ 实现 `sync_cache_to_neo4j()` - 批量同步 UnifiedCache 数据到 Neo4j
   - ✅ 实现 `batch_sync_embeddings_to_neo4j()` - 批量同步 embedding
   - ✅ 自动处理节点、关系、embedding 的批量写入
   - ✅ 支持按层级分组同步（Layer1/2/3）

4. **测试验证**
   - ✅ 创建 `test_neo4j_hybrid_search.py` - 混合检索测试脚本
   - ✅ 验证 FAISS 向量搜索 + Neo4j 图扩展的完整流程
   - ✅ 验证批量同步机制
   - ✅ **测试结果**（2024-12-XX）:
     - 批量创建节点：5个节点，5个embedding（去重优化生效）
     - 批量同步到Neo4j：5个节点，3个边，5个embedding
     - 混合检索：FAISS找到3个初始节点（最高相似度0.7750），Neo4j扩展4个节点（跳数1-2）
     - 多层检索：Layer1/2/3均正常工作

**Architecture**:
```
查询流程：
1. 使用 UnifiedCache 的 FAISS 索引进行快速向量搜索（毫秒级）
   → 找到初始相关节点
2. 从初始节点通过 Neo4j 进行图扩展
   → 获取更多相关节点（通过关系连接）
3. 返回初始节点 + 扩展节点的完整结果
```

**优势**:
- ✅ **性能**: FAISS 毫秒级向量搜索，Neo4j 高效图扩展
- ✅ **兼容**: 完全保留 UnifiedCache 的所有优化（批量、去重、缓存）
- ✅ **灵活**: 支持多层、多类型、多跳扩展
- ✅ **可扩展**: 易于集成到现有检索系统

**Files Created**:
- `core/infrastructure/neo4j_hybrid_search.py`
- `core/search/neo4j_hybrid_recall.py`
- `test_neo4j_hybrid_search.py`

**Next Steps**:
- 集成到现有检索系统（`RetrievalEngine`）
- 在真实数据集上验证性能提升
- 优化批量同步性能

---

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
