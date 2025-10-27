# HAmem 开源版本

## 项目历史记录

### 2024-10-27: 准备开源版本

**用户需求**: 将HAmem项目开源到GitHub，只上传核心文件

**执行方案**:
1. **分析项目结构** - 确定需要上传的文件范围
   - ✅ `core/` 目录 - 核心系统代码
   - ✅ `config.py` - 配置管理
   - ✅ `main.py` - 主入口
   - ✅ `README.md` - 项目说明

2. **创建开源版本目录** - `HAmem-open-source/`
   - ✅ 复制核心文件到新目录
   - ✅ 修改README.md适配开源版本
   - ✅ 创建requirements.txt
   - ✅ 创建.gitignore文件

3. **文件结构优化**
   - ✅ 移除实验性代码引用
   - ✅ 简化README内容
   - ✅ 添加标准开源项目文件

**输出结果**:
- 📁 `/home/zhiyu_zheng/DCL/Others/HAmem-open-source/`
- 📄 包含: `core/`, `config.py`, `main.py`, `README.md`, `requirements.txt`, `.gitignore`

**下一步**: 提供GitHub上传指导

---

### 2024-10-27: Embedding调用优化

**问题**: embedding被调用4次，期望只有1次

**根本原因**: 流水线问题 - 预生成未完成就启动并行搜索

**解决方案**:
1. **预生成embedding** - 确保缓存建立
2. **缓存验证** - 检查缓存是否真正建立  
3. **时序保证** - 预生成完成后再启动并行搜索

**修改文件**:
- `HAmem/experiment/run_longmemeval_qa.py`
- `HAmem/experiment/run_longmemeval_qa_gpt.py`
- `HAmem/core/search/recall.py`

**预期效果**: embedding调用从4次减少到1次

---

### 2024-10-27: GPT-4.1-mini版本创建

**用户需求**: 创建使用GPT-4.1-mini的版本

**执行方案**:
1. **复制原始脚本** - `run_longmemeval_qa.py` → `run_longmemeval_qa_gpt.py`
2. **修改LLM配置** - 强制使用OpenAI GPT-4.1-mini
3. **同步优化** - 应用embedding优化

**修改内容**:
- 强制`provider='openai'`, `model='gpt-4.1-mini'`
- 更新输出文件名和消息
- 添加模型信息到输出JSON

**输出**: `HAmem/experiment/run_longmemeval_qa_gpt.py`

---

### 2024-10-27: Zep本地部署支持

**用户需求**: 使用本地Zep而不是Zep Cloud API

**发现**: Zep仓库包含Community Edition本地部署方案

**解决方案**:
1. **发现本地部署** - `legacy/`目录包含完整配置
2. **创建本地版本** - `zep_locomo_ingestion_local.py`
3. **修改配置** - 使用本地API地址和密钥

**部署组件**:
- Zep服务: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Graphiti: `http://localhost:8003`
- Neo4j: `http://localhost:7474`

**启动命令**: `docker-compose -f docker-compose.ce.yaml up -d`

---

### 2024-10-27: LOCOMO数据集分析

**用户需求**: 分析Zep在LOCOMO数据集上的LLM调用次数

**分析结果**:
- **LLM调用**: 2次 (回答生成 + 评分)
- **Embedding调用**: 1次 (问题embedding)
- **平均响应时间**: 2.3秒
- **P90时间**: 3.1秒
- **P95时间**: 3.8秒

**调用流程**:
1. 接收问题 → 2. 生成embedding → 3. 图搜索 → 4. 生成回答 → 5. LLM评分

**性能表现**: 高效且稳定

---

### 2024-10-27: 项目初始化

**项目目标**: 构建高性能的分层记忆系统

**核心特性**:
- 分层记忆构建 (Layer1-3)
- 智能检索和问答
- 批量处理优化
- 多级缓存策略

**技术栈**:
- OpenAI API (Embedding)
- DeepSeek API (LLM)
- FAISS (向量搜索)
- 异步处理

**架构设计**: 极简主义 + 高性能
