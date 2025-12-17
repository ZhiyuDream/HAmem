# HAmem 示例

本目录包含 HAmem 的使用示例。

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
   from main import HAmem
   
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
from main import HAmem
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


