# HAmem 示例

本目录包含 HAmem 的使用示例。

## 记忆召回测试示例

### `test_recall.py` - 测试记忆召回功能

展示如何使用 HAmem 的 `search_memory` 接口，用一句话召回相关记忆。

#### 功能特点

1. **简单易用**
   - 通过命令行参数指定查询语句和命名空间
   - 支持自定义返回结果数量

2. **完整展示**
   - 显示召回到的所有节点信息
   - 包括节点ID、类型、层级和内容

#### 使用方法

```bash
# 基本用法（使用默认命名空间）
python examples/test_recall.py "你的查询语句"

# 指定命名空间
python examples/test_recall.py "记忆机制" --namespace "locomo_conv_0"

# 指定返回数量
python examples/test_recall.py "记忆机制" --namespace "default" --top-k 5
```

#### 示例输出

```
======================================================================
🔍 HAmem 记忆召回测试
======================================================================
查询: 我研究的是大模型记忆方向
命名空间: default
Top-K: 10
======================================================================

📦 初始化 HAmem...
✅ HAmem 初始化成功

🔍 开始召回记忆...
✅ 召回完成，共找到 3 条结果
======================================================================

📋 召回结果:

[1] fragment_1
  - Type: fragment
  - Layer: 0
  - Content: user: 我研究的是大模型记忆方向...

[2] entity_1
  - Type: entity
  - Layer: 1
  - Content: 大模型记忆方向...

[3] event_1
  - Type: event
  - Layer: 2
  - Content: 研究大模型记忆方向...
======================================================================
```

## 聊天机器人示例

### `chatbot_example.py` - 基于 HAmem 的聊天机器人

展示如何使用 HAmem 的 `search_memory` 接口检索历史信息，并将检索结果作为上下文生成对话式回答。

#### 功能特点

1. **自动检索历史信息**
   - 使用 `hamem.search_memory()` 接口检索相关历史信息
   - 自动将检索结果格式化为上下文

2. **对话式回答生成**
   - 不是简单的 QA，而是基于历史上下文的自然对话
   - 使用 LLM 生成友好、自然的回答

3. **交互式对话**
   - 支持多轮对话
   - 维护对话历史上下文

4. **自动保存对话**
   - 可选：将新对话自动保存到记忆中
   - 下次对话时可以检索到之前的内容

#### 使用方法

```bash
# 1. 确保已配置 API keys（在 .env 文件中）
#    - LLM_API_KEY
#    - LLM_PROVIDER
#    - EMBEDDING_API_KEY
#    - EMBEDDING_PROVIDER

# 2. 运行聊天机器人
python examples/chatbot_example.py
```

#### 使用流程

1. **首次使用**（可选）：先构建一些记忆
   ```python
   from core.main import HAmem
   
   hamem = HAmem()
   # 从文件构建记忆
   hamem.build_memory_from_file('your_conversation.json')
   # 或者直接构建
   hamem.build_memory(conversation_data)
   ```

2. **启动聊天机器人**
   ```bash
   python examples/chatbot_example.py
   ```

3. **开始对话**
   - 输入你的问题或消息
   - 机器人会自动检索相关历史信息
   - 基于历史信息生成回答

#### 命令

- `quit` 或 `exit`: 退出程序
- `clear`: 清空对话历史（不影响已保存的记忆）

#### 代码示例

```python
from core.main import HAmem
from examples.chatbot_example import ChatBot

# 初始化 HAmem
hamem = HAmem()

# 创建聊天机器人
chatbot = ChatBot(
    hamem=hamem,
    namespace="default",
    save_conversation=True  # 保存新对话到记忆
)

# 单次对话
response = chatbot.chat("你好，我们之前聊过什么？")
print(response)

# 或者启动交互式对话
chatbot.interactive_chat()
```

#### 工作原理

1. **检索阶段**：使用 `search_memory()` 检索与用户输入相关的历史信息
2. **上下文构建**：将检索结果格式化为上下文文本
3. **提示构建**：结合历史上下文、最近对话历史和当前用户输入构建提示
4. **回答生成**：调用 LLM 生成对话式回答
5. **保存对话**（可选）：将新对话保存到记忆中，供后续检索

#### 与 QA 系统的区别

- **QA 系统** (`ask_question`): 专注于回答问题，返回结构化的答案
- **聊天机器人** (`chatbot_example`): 专注于自然对话，生成对话式的回答，更注重上下文连贯性

---

### `chatbot_web.py` - 网页版聊天机器人

提供基于 Web 的聊天界面，使用 HAmem 进行记忆检索和对话生成。默认使用 **gpt-4.1-mini** 生成回答。

#### 功能特点

1. **Web 界面**
   - 现代化的聊天界面设计
   - 实时对话，无需刷新页面
   - 响应式设计，支持移动端

2. **自动检索历史信息**
   - 每次对话自动检索相关历史信息
   - 显示检索到的历史信息数量

3. **对话式回答**
   - 使用 gpt-4.1-mini 生成自然、友好的回答
   - 基于历史上下文和对话历史生成回答

4. **自动保存对话**
   - 自动将新对话保存到记忆中
   - 下次对话时可以检索到之前的内容

#### 使用方法

```bash
# 1. 安装依赖（如果还没有安装）
pip install flask flask-cors

# 2. 确保已配置 API keys（在 .env 文件中）
#    - OPENAI_API_KEY
#    - OPENAI_BASE_URL
#    - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD（如果使用Neo4j）

# 3. 启动 Web 服务器
python examples/chatbot_web.py
```

#### 访问界面

启动后，在浏览器中访问：
```
http://localhost:5000
```

#### API 接口

- `POST /api/chat`: 发送消息，获取回答
  ```json
  {
    "message": "你好，我们之前聊过什么？",
    "top_k": 5
  }
  ```

- `GET /api/history`: 获取对话历史

- `POST /api/clear`: 清空对话历史

- `GET /api/status`: 获取服务状态

#### 界面功能

- **发送消息**：在输入框中输入消息，点击"发送"或按 Enter 键
- **清空历史**：点击右上角"清空历史"按钮
- **实时状态**：显示连接状态和检索到的历史信息数量

#### 与终端版的区别

- **终端版** (`chatbot_example.py`): 在终端中交互，适合开发和调试
- **网页版** (`chatbot_web.py`): 提供 Web 界面，更适合用户使用，支持多用户访问


