# HAmem-open-source 任务完成路线图

> **任务来源**：学长验收要求  
> **目标**：完成保底任务（Checkpoint 1），确保项目可一键运行

---

## 📋 一、任务分析

### 1.1 核心任务（保底任务 - Checkpoint 1）

根据学长要求，需要完成以下**保底任务**：

#### ✅ 任务1：用户能自由配置模型和Embedding
- **当前状态**：配置硬编码，只支持OpenAI和DeepSeek
- **目标**：支持任意LLM提供商和Embedding提供商（参考mem0设计）
- **优先级**：🔴 **高**

#### ✅ 任务2：对话格式要求规范化
- **当前状态**：
  - README示例使用 `sessions` 格式
  - 代码中 `ConversationData` 使用 `messages` 格式
  - 格式不统一，容易混淆
- **目标**：统一输入格式，明确规范，支持格式转换
- **优先级**：🔴 **高**

#### ✅ 任务3：一键运行测试
- **当前状态**：需要手动配置多个步骤
- **目标**：新环境git clone后，准备数据集，能直接一键运行
- **优先级**：🟡 **中**

### 1.2 参考标准

- **参考mem0的接口设计**：mem0使用了provider + config模式，支持19种LLM和10种Embedding提供商
- **接口设计文档**：`API_DESIGN.md` 中已有设计方案，但未完全实现

---

## 🎯 二、详细任务分解

### 任务1：自由配置模型和Embedding系统

#### 1.1 当前问题分析

**现有配置系统**（`config.py`）：
```python
# 硬编码的配置
openai_api_key: str
deepseek_api_key: str
llm_model: str = "deepseek-chat"  # 固定DeepSeek
embedding_model: str = "text-embedding-3-small"  # 固定OpenAI
```

**问题**：
- ❌ 只支持OpenAI和DeepSeek两种提供商
- ❌ 无法动态添加新的提供商
- ❌ 配置方式单一（只能环境变量）
- ❌ 不支持provider + config模式

#### 1.2 目标设计（参考mem0）

**mem0的设计模式**：
```python
# mem0的设计
class LlmConfig(BaseModel):
    provider: str = "openai"  # 提供商名称
    config: dict = {}         # 提供商特定配置

class EmbedderConfig(BaseModel):
    provider: str = "openai"
    config: dict = {}
```

**HAmem应该支持**：
```python
# 目标设计
llm_config = {
    "provider": "deepseek",  # 或 "openai", "anthropic", "ollama" 等
    "config": {
        "api_key": "...",
        "base_url": "...",
        "model": "deepseek-chat"
    }
}

embedding_config = {
    "provider": "openai",  # 或 "ollama", "huggingface" 等
    "config": {
        "api_key": "...",
        "base_url": "...",
        "model": "text-embedding-3-small"
    }
}
```

#### 1.3 实现步骤

**Step 1.1：创建LLM配置类**
- 文件：`core/infrastructure/llm_config.py`（新建）
- 功能：
  - 定义 `LlmConfig` 类（参考mem0）
  - 支持provider验证
  - 支持多种提供商（openai、deepseek、anthropic、ollama等）

**Step 1.2：创建Embedding配置类**
- 文件：`core/infrastructure/embedding_config.py`（新建）
- 功能：
  - 定义 `EmbeddingConfig` 类
  - 支持provider验证
  - 支持多种提供商

**Step 1.3：重构LLMClient**
- 文件：`core/infrastructure/llm.py`
- 修改：
  - 支持从 `LlmConfig` 初始化
  - 实现provider工厂模式
  - 支持动态添加提供商

**Step 1.4：重构EmbeddingManager**
- 文件：`core/infrastructure/embedding.py`
- 修改：
  - 支持从 `EmbeddingConfig` 初始化
  - 实现provider工厂模式

**Step 1.5：更新Config类**
- 文件：`config.py`
- 修改：
  - 添加 `llm_config` 和 `embedding_config` 字段
  - 支持从环境变量、代码、配置文件加载
  - 保持向后兼容

**Step 1.6：更新main.py**
- 文件：`main.py`
- 修改：
  - `HAmem.__init__()` 支持传入 `llm_config` 和 `embedding_config`
  - 向后兼容原有配置方式

---

### 任务2：对话格式规范化

#### 2.1 当前问题分析

**格式不一致**：
- README示例：`sessions` 格式
- 代码实现：`messages` 格式
- 用户困惑：不知道用哪种格式

**现有代码**（`memory.py`）：
```python
@dataclass
class ConversationData:
    messages: List[Dict[str, Any]]  # 使用messages格式
    metadata: Dict[str, Any] = None
```

**README示例**：
```python
conversation_data = {
    "sessions": [  # 使用sessions格式
        {
            "timestamp": "...",
            "turns": [...]
        }
    ]
}
```

#### 2.2 目标设计

**统一格式规范**：
- 支持两种格式输入（向后兼容）
- 内部统一转换为标准格式
- 明确文档说明

**标准格式**（推荐）：
```python
# 格式1：sessions格式（推荐，更符合对话场景）
{
    "sessions": [
        {
            "timestamp": "2024-01-01T10:00:00Z",
            "turns": [
                {
                    "speaker": "user",
                    "text": "...",
                    "timestamp": "2024-01-01T10:00:00Z"
                }
            ]
        }
    ]
}

# 格式2：messages格式（兼容）
{
    "messages": [
        {
            "role": "user",
            "content": "...",
            "timestamp": "2024-01-01T10:00:00Z"
        }
    ]
}
```

#### 2.3 实现步骤

**Step 2.1：创建输入格式验证器**
- 文件：`core/utils/input_validator.py`（新建）
- 功能：
  - 验证输入格式
  - 自动转换格式（sessions ↔ messages）
  - 提供详细错误提示

**Step 2.2：更新ConversationData类**
- 文件：`memory.py`
- 修改：
  - 支持从 `sessions` 格式创建
  - 支持从 `messages` 格式创建
  - 内部统一转换为标准格式

**Step 2.3：创建输入格式文档**
- 文件：`INPUT_FORMAT.md`（新建）
- 内容：
  - 详细格式说明
  - 字段要求
  - 示例代码
  - 格式转换说明

**Step 2.4：更新README**
- 文件：`README.md`
- 修改：
  - 明确推荐格式
  - 添加格式说明链接
  - 更新示例代码

---

### 任务3：一键运行支持

#### 3.1 当前问题

- ❌ 需要手动安装依赖
- ❌ 需要手动配置.env
- ❌ 需要手动启动Neo4j
- ❌ 没有一键运行脚本

#### 3.2 目标设计

**一键运行流程**：
```bash
# 1. git clone
git clone https://github.com/ZhiyuDream/HAmem.git
cd HAmem-open-source

# 2. 一键安装和配置
./setup.sh  # 或 python setup.py

# 3. 准备数据集（可选）
# 数据集放在 data/ 目录

# 4. 一键运行测试
python test_quick_start.py
```

#### 3.3 实现步骤

**Step 3.1：创建setup脚本**
- 文件：`setup.py` 或 `setup.sh`
- 功能：
  - 检查Python版本
  - 安装依赖（`pip install -r requirements.txt`）
  - 检查Neo4j是否运行
  - 创建.env.example模板
  - 验证配置

**Step 3.2：创建快速测试脚本**
- 文件：`test_quick_start.py`（新建）
- 功能：
  - 使用示例数据测试
  - 验证所有核心功能
  - 输出测试结果

**Step 3.3：完善README**
- 文件：`README.md`
- 添加：
  - 快速开始章节
  - 一键运行说明
  - 故障排查指南

**Step 3.4：创建.env.example**
- 文件：`.env.example`（已存在，需完善）
- 内容：
  - 所有配置项的示例
  - 详细注释说明

---

## 📅 三、完成路线图

### Phase 1：配置系统重构（2-3天）

#### Day 1：创建配置类
- [ ] 创建 `core/infrastructure/llm_config.py`
  - [ ] 实现 `LlmConfig` 类
  - [ ] 支持provider验证
  - [ ] 支持常见提供商（openai、deepseek、anthropic、ollama）
- [ ] 创建 `core/infrastructure/embedding_config.py`
  - [ ] 实现 `EmbeddingConfig` 类
  - [ ] 支持provider验证
- [ ] 编写单元测试

#### Day 2：重构LLM和Embedding客户端
- [ ] 重构 `LLMClient`
  - [ ] 支持从 `LlmConfig` 初始化
  - [ ] 实现provider工厂模式
  - [ ] 支持动态添加提供商
- [ ] 重构 `EmbeddingManager`
  - [ ] 支持从 `EmbeddingConfig` 初始化
  - [ ] 实现provider工厂模式
- [ ] 更新所有调用点

#### Day 3：更新Config和main.py
- [ ] 更新 `config.py`
  - [ ] 添加 `llm_config` 和 `embedding_config` 字段
  - [ ] 支持多种配置方式
  - [ ] 保持向后兼容
- [ ] 更新 `main.py`
  - [ ] 支持新配置方式
  - [ ] 更新文档字符串
- [ ] 测试和验证

**Git提交点**：`feat: 实现自由配置LLM和Embedding系统`

---

### Phase 2：对话格式规范化（1-2天）

#### Day 1：创建格式转换器
- [ ] 创建 `core/utils/input_validator.py`
  - [ ] 实现格式验证
  - [ ] 实现格式转换（sessions ↔ messages）
  - [ ] 错误处理和提示
- [ ] 更新 `ConversationData` 类
  - [ ] 支持sessions格式
  - [ ] 支持messages格式
  - [ ] 自动转换

#### Day 2：文档和测试
- [ ] 创建 `INPUT_FORMAT.md`
  - [ ] 详细格式说明
  - [ ] 字段要求
  - [ ] 示例代码
- [ ] 更新 `README.md`
  - [ ] 明确推荐格式
  - [ ] 添加格式说明链接
- [ ] 编写格式转换测试

**Git提交点**：`feat: 统一对话输入格式规范`

---

### Phase 3：一键运行支持（1天）

#### Day 1：创建脚本和文档
- [ ] 创建 `setup.py`
  - [ ] 依赖检查
  - [ ] 自动安装
  - [ ] Neo4j检查
  - [ ] 配置验证
- [ ] 创建 `test_quick_start.py`
  - [ ] 快速功能测试
  - [ ] 输出测试报告
- [ ] 完善 `.env.example`
- [ ] 更新 `README.md`
  - [ ] 快速开始章节
  - [ ] 一键运行说明

**Git提交点**：`feat: 添加一键运行支持`

---

### Phase 4：测试和验收（1天）

#### Day 1：完整测试
- [ ] 在新环境测试
  - [ ] git clone
  - [ ] 运行setup脚本
  - [ ] 准备数据集
  - [ ] 一键运行测试
- [ ] 修复发现的问题
- [ ] 更新文档
- [ ] 最终验收

**Git提交点**：`test: 完成验收测试`

---

## 🔧 四、技术实现细节

### 4.1 LLM配置系统设计

**文件结构**：
```
core/infrastructure/
├── llm_config.py          # LLM配置类（新建）
├── embedding_config.py    # Embedding配置类（新建）
├── llm.py                 # LLM客户端（重构）
└── embedding.py           # Embedding管理器（重构）
```

**LlmConfig设计**：
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class LlmConfig(BaseModel):
    """LLM配置"""
    provider: str = Field(
        default="deepseek",
        description="LLM提供商名称"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="提供商特定配置"
    )
    
    @field_validator("provider")
    def validate_provider(cls, v):
        supported = ["openai", "deepseek", "anthropic", "ollama", "groq", "together"]
        if v not in supported:
            raise ValueError(f"不支持的LLM提供商: {v}")
        return v
```

**LLMClient重构**：
```python
class LLMClient:
    def __init__(self, llm_config: LlmConfig):
        self.config = llm_config
        self.client = self._create_client()
    
    def _create_client(self):
        """根据provider创建客户端"""
        provider = self.config.provider
        config = self.config.config
        
        if provider == "openai":
            return OpenAI(**config)
        elif provider == "deepseek":
            return OpenAI(base_url="https://api.deepseek.com/v1", **config)
        elif provider == "anthropic":
            return Anthropic(**config)
        # ... 其他提供商
```

### 4.2 对话格式转换设计

**格式转换器**：
```python
class InputFormatConverter:
    @staticmethod
    def sessions_to_messages(sessions_data: Dict) -> List[Dict]:
        """将sessions格式转换为messages格式"""
        messages = []
        for session in sessions_data.get("sessions", []):
            for turn in session.get("turns", []):
                messages.append({
                    "role": turn.get("speaker", "user"),
                    "content": turn.get("text", ""),
                    "timestamp": turn.get("timestamp", session.get("timestamp", "")),
                    "metadata": turn.get("metadata", {})
                })
        return messages
    
    @staticmethod
    def messages_to_sessions(messages_data: Dict) -> Dict:
        """将messages格式转换为sessions格式"""
        # 按timestamp分组
        sessions = {}
        for msg in messages_data.get("messages", []):
            timestamp = msg.get("timestamp", "unknown")
            if timestamp not in sessions:
                sessions[timestamp] = []
            sessions[timestamp].append({
                "speaker": msg.get("role", "user"),
                "text": msg.get("content", ""),
                "timestamp": timestamp
            })
        
        return {
            "sessions": [
                {
                    "timestamp": ts,
                    "turns": turns
                }
                for ts, turns in sessions.items()
            ]
        }
```

### 4.3 一键运行脚本设计

**setup.py**：
```python
#!/usr/bin/env python3
"""一键安装和配置脚本"""

import os
import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version}")

def install_dependencies():
    """安装依赖"""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_neo4j():
    """检查Neo4j"""
    # 检查Neo4j是否运行
    pass

def create_env_example():
    """创建.env.example"""
    pass

if __name__ == "__main__":
    check_python_version()
    install_dependencies()
    check_neo4j()
    create_env_example()
    print("✅ Setup completed!")
```

---

## 📝 五、验收标准

### 5.1 功能验收

- [x] **自由配置模型**：
  - [ ] 支持至少5种LLM提供商（openai、deepseek、anthropic、ollama、groq）
  - [ ] 支持至少3种Embedding提供商（openai、ollama、huggingface）
  - [ ] 可以通过代码、环境变量、配置文件配置
  - [ ] 向后兼容原有配置方式

- [x] **对话格式规范**：
  - [ ] 支持sessions格式输入
  - [ ] 支持messages格式输入
  - [ ] 自动格式转换
  - [ ] 有详细的格式文档

- [x] **一键运行**：
  - [ ] 新环境git clone后能运行setup脚本
  - [ ] 自动安装依赖
  - [ ] 自动检查Neo4j
  - [ ] 有快速测试脚本

### 5.2 测试验收

**测试步骤**：
1. 在新目录git clone项目
2. 运行 `python setup.py`
3. 配置.env文件
4. 准备测试数据集
5. 运行 `python test_quick_start.py`
6. 验证所有功能正常

---

## 🚀 六、开始实施

### 立即行动项

1. **创建任务分支**
   ```bash
   git checkout -b feat/config-and-format
   ```

2. **开始Phase 1：配置系统重构**
   - 先创建 `llm_config.py` 和 `embedding_config.py`
   - 参考mem0的设计
   - 保持代码风格一致

3. **频繁提交**
   - 每完成一个小功能就提交
   - 提交信息清晰：`feat: 添加LlmConfig类`

---

## 📚 七、参考资源

1. **mem0配置设计**：
   - `mem0-main/mem0/llms/configs.py`
   - `mem0-main/mem0/embeddings/configs.py`

2. **API设计文档**：
   - `HAmem-open-source/API_DESIGN.md`

3. **项目历史**：
   - `HAmem-open-source/PROJECT_HISTORY.md`

---

**预计完成时间**：4-5天  
**优先级**：🔴 高（保底任务）  
**验收标准**：新环境一键运行，支持自由配置模型和格式

