"""
HAmem 网页版聊天机器人

使用 Flask 提供 Web 界面，支持基于记忆的对话式聊天。

使用方法:
    python examples/chatbot_web.py

访问:
    http://localhost:5000
"""

import os
import sys
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


def get_hour_timestamp() -> str:
    
    now = datetime.now()
    # 只保留到小时，分钟和秒都设为0
    hour_timestamp = now.replace(minute=0, second=0, microsecond=0)
    return hour_timestamp.isoformat()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.main import HAmem
from config import Config
from core.infrastructure.llm import LLMClient
from core.fragment.buffer_manager import BufferManager

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)

# 全局变量
hamem_instance: Optional[HAmem] = None
llm_client: Optional[LLMClient] = None
conversation_history: List[Dict[str, str]] = []
buffer_manager: Optional[BufferManager] = None
BUFFER_MAX_LENGTH = 2000  # 缓冲区最大长度
MAX_HISTORY_ROUNDS = 10  # 最大历史轮数（最近N轮对话）
turn_count = 0  # 对话轮数计数器
SUMMARY_INTERVAL = 10  # 每10轮总结一次记忆


def init_hamem():
    """初始化 HAmem 实例"""
    global hamem_instance, llm_client, buffer_manager
    
    try:
        config = Config()
        config.validate()
        hamem_instance = HAmem(config)
        llm_client = LLMClient(config)
        buffer_manager = BufferManager(max_length=BUFFER_MAX_LENGTH)
        return True
    except Exception as e:
        print(f"❌ HAmem 初始化失败: {e}")
        return False


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """格式化检索结果为上下文文本"""
    if not results:
        return "暂无相关历史信息。"
    
    context_parts = ["【相关历史信息】"]
    for i, result in enumerate(results, 1):
        content = result.get('content', '')
        layer = result.get('layer', 'unknown')
        score = result.get('score', 0.0)
        context_parts.append(
            f"{i}. [{layer}] (相关度: {score:.2f})\n   {content[:200]}..."
        )
    
    return "\n".join(context_parts)


def build_conversation_prompt(user_input: str, context: str, history: List[Dict[str, str]]) -> str:
    """构建对话提示"""
    # 构建最近对话历史（仅保留最近几轮，减少到2轮）
    recent_history = history[-MAX_HISTORY_ROUNDS:]  # 最近N轮对话
    history_text = ""
    if recent_history:
        history_text = "【最近对话历史】\n"
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content']}\n"
    
    # 如果没有检索到记忆，使用默认提示
    if not context or "【相关历史信息】" not in context:
        context = "暂无相关历史信息。"
    
    # 构建完整提示
    if context and "【相关历史信息】" in context:
        # 有检索到的记忆
        prompt = f"""你是一个友好的聊天助手，能够基于历史信息进行自然对话。

{context}

{history_text}

【当前对话】
用户: {user_input}

请基于上述历史信息和对话上下文，生成一个自然、友好的回答。注意：
1. 如果历史信息与当前问题相关，请自然地引用这些信息
2. 回答应该是对话式的，而不是问答式的
3. 保持回答简洁、友好、自然

助手:"""
    else:
        # 只有缓冲区上下文，简化提示
        prompt = f"""你是一个友好的聊天助手。

{context}

【当前对话】
用户: {user_input}

请基于上述对话上下文，生成一个自然、友好的回答。保持回答简洁、友好、自然。

助手:"""
    
    return prompt


@app.route('/')
def index():
    """主页"""
    return render_template('chatbot.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天API"""
    global conversation_history, buffer_manager
    
    if not hamem_instance or not llm_client or not buffer_manager:
        return jsonify({
            'success': False,
            'error': 'HAmem 未初始化，请检查配置'
        }), 500
    
    data = request.json
    user_input = data.get('message', '').strip()
    namespace = data.get('namespace', 'default')
    top_k = data.get('top_k', 3)  # 减少默认检索数量
    save_conversation = data.get('save_conversation', True)
    
    if not user_input:
        return jsonify({
            'success': False,
            'error': '消息不能为空'
        }), 400
    
    try:
        # 添加用户输入到缓冲区
        user_turn = {"role": "user", "content": user_input}
        fragment, needs_llm = buffer_manager.add_turn(user_turn)
        
        # 每次用户查询都进行记忆检索（使用相似度阈值过滤）
        context = ""
        search_results = []
        
        try:
            # 使用 search_memory 接口检索，默认相似度阈值为 0.5
            search_results = hamem_instance.search_memory(
                query=user_input,
                top_k=top_k,
                namespace=namespace,
                similarity_threshold=0.4  # 过滤相似度低于0.5的结果
            )
            context = format_search_results(search_results)
        except Exception as e:
            print(f"⚠️  检索历史信息失败: {e}")
            context = ""
        
        # 2. 构建对话提示
        prompt = build_conversation_prompt(user_input, context, conversation_history)
        
        # 3. 调用 LLM 生成回答（使用配置中的模型，如果没有指定则使用gpt-4.1-mini）
        # 优先使用配置中的模型，如果没有则使用gpt-4.1-mini
        model = hamem_instance.config.llm_config.get_model() or "gpt-4.1-mini"
        response = llm_client.call_llm(
            prompt=prompt,
            model=model,
            provider=None  # 使用配置中的provider
        )
        
        # 添加助手回答到缓冲区
        assistant_turn = {"role": "assistant", "content": response}
        buffer_manager.add_turn(assistant_turn)
        
        # 4. 保存对话历史（仅保留最近N轮）
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # 限制历史长度（只保留最近 max_history_rounds * 2 轮）
        max_history_items = MAX_HISTORY_ROUNDS * 2
        if len(conversation_history) > max_history_items:
            conversation_history = conversation_history[-max_history_items:]
        
        # 5. 每10轮对话总结一次记忆（最简单的方法）
        global turn_count
        turn_count += 1
        if save_conversation and turn_count % SUMMARY_INTERVAL == 0:
            def save_memory_async():
                """异步保存记忆，不阻塞主流程"""
                try:
                    print(f"\n💾 达到 {SUMMARY_INTERVAL} 轮对话，开始总结记忆...")
                    # 获取最近10轮对话（5轮用户+5轮助手）
                    recent_turns = conversation_history[-SUMMARY_INTERVAL * 2:]
                    if recent_turns:
                        # 构建对话数据格式
                        hour_timestamp = get_hour_timestamp()
                        conversation_data = {
                            "messages": [
                                {
                                    "speaker": msg["role"],
                                    "content": msg["content"],
                                    "timestamp": hour_timestamp
                                }
                                for msg in recent_turns
                            ]
                        }
                        # 使用配置中的LLM provider
                        llm_provider = hamem_instance.config.llm_config.provider
                        hamem_instance.build_memory(conversation_data, namespace=namespace, llm_provider=llm_provider)
                        print(f"   ✅ {SUMMARY_INTERVAL} 轮对话已总结并保存到记忆")
                except Exception as e:
                    print(f"⚠️  总结记忆失败: {e}")
            
            # 在后台线程中异步执行，不阻塞响应
            thread = threading.Thread(target=save_memory_async, daemon=True)
            thread.start()
        
        return jsonify({
            'success': True,
            'response': response,
            'context_count': len(search_results),
            'history_length': len(conversation_history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """清空对话历史"""
    global conversation_history, buffer_manager, turn_count
    conversation_history = []
    turn_count = 0  # 重置计数器
    if buffer_manager:
        buffer_manager.clear()
    return jsonify({
        'success': True,
        'message': '对话历史已清空'
    })


@app.route('/api/status', methods=['GET'])
def status():
    """获取状态"""
    return jsonify({
        'success': True,
        'initialized': hamem_instance is not None,
        'history_length': len(conversation_history)
    })


def main():
    """主函数"""
    print("🚀 启动 HAmem 网页版聊天机器人...")
    
    # 初始化 HAmem
    if not init_hamem():
        print("❌ 初始化失败，请检查配置")
        return
    
    print("✅ HAmem 初始化成功")
    print("🌐 访问 http://localhost:5000 开始聊天")
    print("按 Ctrl+C 停止服务器\n")
    
    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
