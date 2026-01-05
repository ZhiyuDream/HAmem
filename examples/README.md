# H-SEAM Examples

This directory contains usage examples for H-SEAM.

## Memory Recall Test Example

### `test_recall.py` - Test Memory Recall Functionality

Demonstrates how to use H-SEAM's `search_memory` interface to recall relevant memories with a single query.

#### Features

1. **Simple and Easy to Use**
   - Specify query statement and namespace via command-line arguments
   - Support custom number of returned results

2. **Complete Display**
   - Show all recalled node information
   - Include node ID, type, layer, and content

#### Usage

```bash
# Basic usage (using default namespace)
python examples/test_recall.py "your query statement"

# Specify namespace
python examples/test_recall.py "memory mechanism" --namespace "locomo_conv_0"

# Specify number of results
python examples/test_recall.py "memory mechanism" --namespace "default" --top-k 5
```

#### Example Output

```
======================================================================
🔍 H-SEAM Memory Recall Test
======================================================================
Query: I'm researching large model memory systems
Namespace: default
Top-K: 10
======================================================================

📦 Initializing H-SEAM...
✅ H-SEAM initialized successfully

🔍 Starting memory recall...
✅ Recall completed, found 3 results
======================================================================

📋 Recall Results:

[1] fragment_1
  - Type: fragment
  - Layer: 0
  - Content: user: I'm researching large model memory systems...

[2] entity_1
  - Type: entity
  - Layer: 1
  - Content: Large model memory systems...

[3] event_1
  - Type: event
  - Layer: 2
  - Content: Researching large model memory systems...
======================================================================
```

## Chatbot Example

### `chatbot_example.py` - H-SEAM-based Chatbot

Demonstrates how to use H-SEAM's `search_memory` interface to retrieve historical information and generate conversational responses based on the retrieved results.

#### Features

1. **Automatic Historical Information Retrieval**
   - Use `h_seam.search_memory()` interface to retrieve relevant historical information
   - Automatically format retrieved results as context

2. **Conversational Response Generation**
   - Not simple QA, but natural conversation based on historical context
   - Use LLM to generate friendly, natural responses

3. **Interactive Conversation**
   - Support multi-turn conversations
   - Maintain conversation history context

4. **Automatic Conversation Saving**
   - Optional: Automatically save new conversations to memory
   - Can retrieve previous content in next conversation

#### Usage

```bash
# 1. Ensure API keys are configured (in .env file)
#    - LLM_API_KEY
#    - LLM_PROVIDER
#    - EMBEDDING_API_KEY
#    - EMBEDDING_PROVIDER

# 2. Run the chatbot
python examples/chatbot_example.py
```

#### Usage Flow

1. **First Use** (optional): Build some memory first
   ```python
   from core.main import H_SEAM
   
   h_seam = H_SEAM()
   # Build memory from file
   h_seam.build_memory_from_file('your_conversation.json')
   # Or build directly
   h_seam.build_memory(conversation_data)
   ```

2. **Start Chatbot**
   ```bash
   python examples/chatbot_example.py
   ```

3. **Start Conversation**
   - Enter your question or message
   - The bot will automatically retrieve relevant historical information
   - Generate responses based on historical information

#### Commands

- `quit` or `exit`: Exit the program
- `clear`: Clear conversation history (does not affect saved memories)

#### Code Example

```python
from core.main import H_SEAM
from examples.chatbot_example import ChatBot

# Initialize H-SEAM
h_seam = H_SEAM()

# Create chatbot
chatbot = ChatBot(
    h_seam=h_seam,
    namespace="default",
    save_conversation=True  # Save new conversations to memory
)

# Single conversation
response = chatbot.chat("Hello, what did we talk about before?")
print(response)

# Or start interactive conversation
chatbot.interactive_chat()
```

#### How It Works

1. **Retrieval Phase**: Use `search_memory()` to retrieve historical information related to user input
2. **Context Construction**: Format retrieved results as context text
3. **Prompt Construction**: Combine historical context, recent conversation history, and current user input to build prompts
4. **Response Generation**: Call LLM to generate conversational responses
5. **Save Conversation** (optional): Save new conversations to memory for subsequent retrieval

#### Differences from QA System

- **QA System** (`ask_question`): Focuses on answering questions, returns structured answers
- **Chatbot** (`chatbot_example`): Focuses on natural conversation, generates conversational responses, emphasizes context coherence

---

### `chatbot_web.py` - Web-based Chatbot

Provides a web-based chat interface using H-SEAM for memory retrieval and conversation generation. Default uses **gpt-4.1-mini** to generate responses.

#### Features

1. **Web Interface**
   - Modern chat interface design
   - Real-time conversation, no page refresh needed
   - Responsive design, supports mobile devices

2. **Automatic Historical Information Retrieval**
   - Automatically retrieve relevant historical information for each conversation
   - Display the number of retrieved historical information

3. **Conversational Responses**
   - Use gpt-4.1-mini to generate natural, friendly responses
   - Generate responses based on historical context and conversation history

4. **Automatic Conversation Saving**
   - Automatically save new conversations to memory
   - Can retrieve previous content in next conversation

#### Usage

```bash
# 1. Install dependencies (if not already installed)
pip install flask flask-cors

# 2. Ensure API keys are configured (in .env file)
#    - OPENAI_API_KEY
#    - OPENAI_BASE_URL
#    - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (if using Neo4j)

# 3. Start web server
python examples/chatbot_web.py
```

#### Access Interface

After starting, access in browser:
```
http://localhost:5000
```

#### API Endpoints

- `POST /api/chat`: Send message, get response
  ```json
  {
    "message": "Hello, what did we talk about before?",
    "top_k": 5
  }
  ```

- `GET /api/history`: Get conversation history

- `POST /api/clear`: Clear conversation history

- `GET /api/status`: Get service status

#### Interface Features

- **Send Message**: Enter message in input box, click "Send" or press Enter
- **Clear History**: Click "Clear History" button in top right corner
- **Real-time Status**: Display connection status and number of retrieved historical information

#### Differences from Terminal Version

- **Terminal Version** (`chatbot_example.py`): Interactive in terminal, suitable for development and debugging
- **Web Version** (`chatbot_web.py`): Provides web interface, more suitable for users, supports multi-user access
