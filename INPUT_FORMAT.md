# HAmem 输入格式规范

本文档详细说明 HAmem 系统支持的所有输入格式，以及如何将数据转换为 HAmem 标准格式。

## 📋 目录

- [HAmem 标准格式（推荐）](#hamem-标准格式推荐)
- [支持的输入格式](#支持的输入格式)
  - [Messages 格式](#messages-格式)
  - [Locomo 格式](#locomo-格式)
  - [Sessions 格式](#sessions-格式)
  - [Batch 格式](#batch-格式)
- [格式转换示例](#格式转换示例)
- [使用方式](#使用方式)

---

## HAmem 标准格式（推荐）

**这是 HAmem 系统的标准格式，推荐所有用户使用此格式。**

### 格式结构

```json
{
  "messages": [
    {
      "speaker": "user",
      "content": "用户消息内容",
      "timestamp": "2024-01-01T10:00:00",
      "metadata": {
        "source": "web",
        "session_id": "session_1"
      }
    },
    {
      "speaker": "assistant",
      "content": "助手回复内容",
      "timestamp": "2024-01-01T10:00:05",
      "metadata": {}
    }
  ],
  "metadata": {
    "conversation_id": "conv_001",
    "source": "custom"
  }
}
```

### 字段说明

#### messages（必需）

消息列表，每个消息包含：

- **speaker** (string, 必需): 发言者标识
  - 推荐值：`"user"` 或 `"assistant"`
  - 也可以是其他自定义标识符
  
- **content** (string, 必需): 消息内容
  
- **timestamp** (string, 必需): 时间戳
  - 格式：ISO 8601 格式（如 `"2024-01-01T10:00:00"`）
  - 如果未提供，系统会自动生成当前时间
  
- **metadata** (object, 可选): 额外元数据
  - 可以包含任意自定义字段
  - 例如：`source`, `session_id`, `user_id` 等

#### metadata（可选）

对话级别的元数据，可以包含任意自定义字段。

---

## 支持的输入格式

HAmem 系统支持以下输入格式，并会自动识别和转换：

### Messages 格式

类似 OpenAI Chat API 的格式，使用 `role` 字段。

```json
{
  "messages": [
    {
      "role": "user",
      "content": "用户消息内容",
      "timestamp": "2024-01-01T10:00:00"
    },
    {
      "role": "assistant",
      "content": "助手回复内容",
      "timestamp": "2024-01-01T10:00:05"
    }
  ]
}
```

**识别规则**：包含 `messages` 数组，且消息包含 `role` 和 `content` 字段。

**转换规则**：
- `role` → `speaker`
- `content` 保持不变
- `timestamp` 如果缺失，使用当前时间

---

### Locomo 格式

Locomo 数据集的格式。

```json
{
  "conversation": {
    "speaker_a": "User",
    "speaker_b": "Assistant",
    "session_1": [
      {
        "speaker": "User",
        "text": "用户消息",
        "dia_id": "dia_001"
      },
      {
        "speaker": "Assistant",
        "text": "助手回复",
        "dia_id": "dia_002"
      }
    ],
    "session_1_date_time": "2024-01-01T10:00:00",
    "session_2": [
      {
        "speaker": "User",
        "text": "另一条消息",
        "dia_id": "dia_003"
      }
    ],
    "session_2_date_time": "2024-01-01T11:00:00"
  }
}
```

**识别规则**：包含 `conversation` 字段。

**转换规则**：
- 遍历所有 `session_*` 字段（排除 `*_date_time` 和 `*_summary`）
- 将 `speaker` 映射为 `user` 或 `assistant`（根据 `speaker_a` 和 `speaker_b`）
- 使用 `session_*_date_time` 作为时间戳
- 保留原始信息到 `metadata` 中

---

### Sessions 格式

按会话组织的格式。

```json
{
  "sessions": [
    {
      "timestamp": "2024-01-01T10:00:00",
      "turns": [
        {
          "speaker": "user",
          "text": "用户消息",
          "timestamp": "2024-01-01T10:00:00"
        },
        {
          "speaker": "assistant",
          "text": "助手回复",
          "timestamp": "2024-01-01T10:00:05"
        }
      ]
    },
    {
      "timestamp": "2024-01-01T11:00:00",
      "turns": [
        {
          "speaker": "user",
          "text": "另一条消息",
          "timestamp": "2024-01-01T11:00:00"
        }
      ]
    }
  ],
  "metadata": {
    "conversation_id": "conv_001"
  }
}
```

**识别规则**：包含 `sessions` 数组。

**转换规则**：
- 遍历所有会话
- 将会话中的 `turns` 展开为消息列表
- 使用会话的 `timestamp` 作为默认时间戳（如果 turn 没有单独的时间戳）

---

### Batch 格式

包含多个对话的批量格式。

```json
{
  "conversations": [
    {
      "messages": [
        {
          "speaker": "user",
          "content": "消息1",
          "timestamp": "2024-01-01T10:00:00"
        }
      ]
    },
    {
      "messages": [
        {
          "speaker": "user",
          "content": "消息2",
          "timestamp": "2024-01-01T11:00:00"
        }
      ]
    }
  ]
}
```

**识别规则**：包含 `conversations` 数组。

**转换规则**：
- 取第一个 conversation 进行转换
- 如果需要对多个 conversation 处理，需要分别调用

---

## 格式转换示例

### 示例 1：从 Messages 格式转换

**输入（Messages 格式）**：
```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
  ]
}
```

**输出（HAmem 标准格式）**：
```json
{
  "messages": [
    {
      "speaker": "user",
      "content": "你好",
      "timestamp": "2024-01-01T10:00:00",
      "metadata": {}
    },
    {
      "speaker": "assistant",
      "content": "你好！有什么可以帮助你的吗？",
      "timestamp": "2024-01-01T10:00:05",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

### 示例 2：从 Locomo 格式转换

**输入（Locomo 格式）**：
```json
{
  "conversation": {
    "speaker_a": "User",
    "speaker_b": "Assistant",
    "session_1": [
      {"speaker": "User", "text": "你好"},
      {"speaker": "Assistant", "text": "你好！"}
    ],
    "session_1_date_time": "2024-01-01T10:00:00"
  }
}
```

**输出（HAmem 标准格式）**：
```json
{
  "messages": [
    {
      "speaker": "user",
      "content": "你好",
      "timestamp": "2024-01-01T10:00:00",
      "metadata": {
        "original_speaker": "User",
        "session": "session_1",
        "session_time": "2024-01-01T10:00:00"
      }
    },
    {
      "speaker": "assistant",
      "content": "你好！",
      "timestamp": "2024-01-01T10:00:00",
      "metadata": {
        "original_speaker": "Assistant",
        "session": "session_1",
        "session_time": "2024-01-01T10:00:00"
      }
    }
  ],
  "metadata": {
    "speaker_a": "User",
    "speaker_b": "Assistant",
    "source": "locomo"
  }
}
```

---

## 使用方式

### 方式 1：从文件读取（推荐）

```python
from core.main import HAmem

hamem = HAmem()

# 自动识别格式并转换
result = hamem.build_memory_from_file("conversation.json")
```

系统会自动：
1. 读取 JSON 文件
2. 检测输入格式
3. 转换为 HAmem 标准格式
4. 验证格式正确性
5. 构建记忆

### 方式 2：直接传入数据

```python
from core.main import HAmem

hamem = HAmem()

# 使用 HAmem 标准格式
conversation_data = {
    "messages": [
        {
            "speaker": "user",
            "content": "你好",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "speaker": "assistant",
            "content": "你好！",
            "timestamp": "2024-01-01T10:00:05"
        }
    ]
}

result = hamem.build_memory(conversation_data)
```

### 方式 3：手动转换格式

```python
from core.utils.input_processor import process_input_file

# 从文件读取并转换
standard_data = process_input_file("conversation.json")

# 或者手动转换
from core.utils.input_processor import convert_to_hamem_format, detect_format

data = {...}  # 你的数据
format_type = detect_format(data)
standard_data = convert_to_hamem_format(data, format_type)
```

---

## 格式验证

系统会自动验证转换后的数据是否符合 HAmem 标准格式：

- ✅ 必须包含 `messages` 字段
- ✅ `messages` 必须是列表
- ✅ 每个消息必须包含 `speaker`、`content`、`timestamp` 字段
- ✅ `messages` 列表不能为空

如果验证失败，系统会抛出 `ValueError` 异常，并说明具体的错误原因。

---

## 最佳实践

1. **推荐使用 HAmem 标准格式**：这是最直接、最高效的方式
2. **提供准确的时间戳**：有助于系统更好地理解对话的时间顺序
3. **合理使用 metadata**：可以存储额外的上下文信息，但不要过度使用
4. **保持格式一致性**：在同一对话中，尽量保持字段命名和结构的一致性

---

## 常见问题

### Q: 我的数据格式不在支持列表中怎么办？

A: 你可以：
1. 先将数据转换为 HAmem 标准格式
2. 或者转换为 Messages 格式（最简单）
3. 或者联系开发者添加新的格式支持

### Q: 时间戳格式有什么要求？

A: 推荐使用 ISO 8601 格式（如 `"2024-01-01T10:00:00"`），但系统也接受其他格式。如果未提供时间戳，系统会自动使用当前时间。

### Q: speaker 字段可以使用中文吗？

A: 可以，但推荐使用英文标识符（如 `"user"`、`"assistant"`），这样更便于系统处理。

### Q: 如何处理多轮对话？

A: 将所有消息按时间顺序放入 `messages` 数组中即可。系统会自动处理多轮对话。

---

## 相关文档

- [README.md](README.md) - 项目总览
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 配置指南

