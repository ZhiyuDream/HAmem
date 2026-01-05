"""
聊天机器人示例

展示如何使用 H-SEAM 的 search_memory 接口检索历史信息，
并将检索结果作为上下文生成对话式回答。

使用方法:
    python examples/chatbot_example.py

功能:
    1. 自动调用 search_memory 检索相关历史信息
    2. 将检索结果作为上下文，生成对话式回答（不是QA）
    3. 支持交互式对话
    4. 可选：将新对话保存到记忆中
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any


def get_hour_timestamp() -> str:
    """
    获取精确到小时的时间戳（格式：YYYY-MM-DDTHH:00:00）
    同一小时内的对话使用相同的时间戳，避免频繁触发记忆处理
    """
    now = datetime.now()
    # 只保留到小时，分钟和秒都设为0
    hour_timestamp = now.replace(minute=0, second=0, microsecond=0)
    return hour_timestamp.isoformat()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.main import H_SEAM
from config import Config
from core.infrastructure import LLMClient
from core.fragment.buffer_manager import BufferManager


class ChatBot:
    """基于 H-SEAM 的聊天机器人"""
    
    def __init__(self, h_seam: H_SEAM, namespace: str = "default", save_conversation: bool = True, 
                 buffer_max_length: int = 2000, max_history_rounds: int = 4):
        """
        初始化聊天机器人
        
        Args:
            h_seam: H_SEAM 实例
            namespace: 命名空间
            save_conversation: 是否将新对话保存到记忆中
            buffer_max_length: 缓冲区最大长度（字符数），超过此长度才检索记忆
            max_history_rounds: 最大历史轮数（最近N轮对话）
        """
        self.h_seam = h_seam
        self.namespace = namespace
        self.save_conversation = save_conversation
        self.buffer_max_length = buffer_max_length
        self.max_history_rounds = max_history_rounds
        
        # 初始化 LLM 客户端用于生成对话
        self.llm_client = LLMClient(self.h_seam.config)
        
        # 对话历史（用于维护上下文）
        self.conversation_history: List[Dict[str, str]] = []
        
        # 缓冲区管理器（用于判断是否需要检索记忆）
        self.buffer_manager = BufferManager(max_length=buffer_max_length)
        
        # 对话轮数计数器（用于每10轮总结一次记忆）
        self.turn_count = 0
        self.summary_interval = 10  # 每10轮总结一次
        
        print("🤖 聊天机器人已初始化")
        print(f"   - 命名空间: {namespace}")
        print(f"   - 保存对话: {'是' if save_conversation else '否'}")
        print(f"   - 缓冲区长度: {buffer_max_length} 字符")
        print(f"   - 最大历史轮数: {max_history_rounds} 轮")
    
    def chat(self, user_input: str, top_k: int = 3) -> str:
        """
        与用户聊天，自动检索历史信息并生成回答
        
        Args:
            user_input: 用户输入
            top_k: 检索的历史信息数量（仅在需要检索时使用）
            
        Returns:
            机器人的回答
        """
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")
        
        # 每次用户查询都进行记忆检索（使用相似度阈值过滤）
        context = ""
        search_results = []
        
        print(f"\n🔍 检索历史信息...")
        try:
            # 使用 search_memory 接口检索，默认相似度阈值为 0.5
            search_results = self.h_seam.search_memory(
                query=user_input,
                top_k=top_k,
                namespace=self.namespace,
                similarity_threshold=0.5  # 过滤相似度低于0.5的结果
            )
            
            # 格式化检索结果作为上下文
            context = self._format_search_results(search_results)
            print(f"   ✅ 检索到 {len(search_results)} 条相关历史信息")
            
        except Exception as e:
            print(f"   ⚠️  检索历史信息失败: {e}")
            context = ""
        
        # 添加用户输入到缓冲区（用于后续的10轮总结）
        user_turn = {"role": "user", "content": user_input}
        self.buffer_manager.add_turn(user_turn)
        
        # 2. 构建对话提示（包含历史上下文和当前对话）
        prompt = self._build_conversation_prompt(user_input, context)
        
        # 3. 调用 LLM 生成对话式回答（使用配置中的模型，如果没有指定则使用gpt-4.1-mini）
        print(f"\n💬 生成回答...")
        try:
            # 优先使用配置中的模型，如果没有则使用gpt-4.1-mini
            model = self.h_seam.config.llm_config.get_model() or "gpt-4.1-mini"
            response = self.llm_client.call_llm(
                prompt=prompt,
                model=model,
                provider=None  # 使用配置中的provider
            )
            print(f"   ✅ 回答生成完成")
        except Exception as e:
            print(f"   ❌ 生成回答失败: {e}")
            response = "抱歉，我遇到了一些问题，无法生成回答。"
        
        # 添加助手回答到缓冲区
        assistant_turn = {"role": "assistant", "content": response}
        self.buffer_manager.add_turn(assistant_turn)
        
        # 4. 保存对话历史（仅保留最近N轮）
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # 限制历史长度（只保留最近 max_history_rounds * 2 轮）
        max_history_items = self.max_history_rounds * 2
        if len(self.conversation_history) > max_history_items:
            self.conversation_history = self.conversation_history[-max_history_items:]
        
        # 5. 每10轮对话总结一次记忆（最简单的方法）
        self.turn_count += 1
        if self.save_conversation and self.turn_count % self.summary_interval == 0:
            def save_memory_async():
                """异步保存记忆，不阻塞主流程"""
                try:
                    print(f"\n💾 达到 {self.summary_interval} 轮对话，开始总结记忆...")
                    # 获取最近10轮对话（5轮用户+5轮助手）
                    recent_turns = self.conversation_history[-self.summary_interval * 2:]
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
                        llm_provider = self.h_seam.config.llm_config.provider
                        self.h_seam.build_memory(conversation_data, namespace=self.namespace, llm_provider=llm_provider)
                        print(f"   ✅ {self.summary_interval} 轮对话已总结并保存到记忆")
                except Exception as e:
                    print(f"⚠️  总结记忆失败: {e}")
            
            # 在后台线程中异步执行，不阻塞响应
            thread = threading.Thread(target=save_memory_async, daemon=True)
            thread.start()
        
        return response
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化检索结果为上下文文本
        
        Args:
            results: 检索结果列表
            
        Returns:
            格式化后的上下文文本
        """
        if not results:
            return "暂无相关历史信息。"
        
        context_parts = ["【相关历史信息】"]
        for i, result in enumerate(results, 1):
            # 提取关键信息
            content = result.get('content', '')
            layer = result.get('layer', 'unknown')
            score = result.get('score', 0.0)
            
            # 格式化
            context_parts.append(
                f"{i}. [{layer}] (相关度: {score:.2f})\n   {content[:200]}..."  # 限制长度
            )
        
        return "\n".join(context_parts)
    
    def _build_conversation_prompt(self, user_input: str, context: str) -> str:
        """
        构建对话提示，包含历史上下文和当前对话
        
        Args:
            user_input: 用户输入
            context: 检索到的历史上下文
            
        Returns:
            完整的提示文本
        """
        # 如果没有检索到记忆，使用缓冲区上下文
        if not context or context == "暂无相关历史信息。":
            buffer_content = self.buffer_manager.get_content()
            if buffer_content:
                context = f"【当前对话上下文】\n{buffer_content}"
            else:
                context = "暂无相关历史信息。"
        
        # 构建最近对话历史（仅保留最近几轮，减少到2轮）
        recent_history = self.conversation_history[-self.max_history_rounds:]  # 最近N轮对话
        history_text = ""
        if recent_history:
            history_text = "【最近对话历史】\n"
            for msg in recent_history:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"
        
        # 构建完整提示（简化版，减少冗余）
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
    
    def _save_to_memory(self, user_input: str, assistant_response: str):
        """
        将新对话保存到记忆中（仅在需要时调用）
        
        Args:
            user_input: 用户输入
            assistant_response: 助手回答
        """
        try:
            # 构建标准格式的对话数据（使用小时级别的时间戳）
            hour_timestamp = get_hour_timestamp()
            conversation_data = {
                "messages": [
                    {
                        "speaker": "user",
                        "content": user_input,
                        "timestamp": hour_timestamp
                    },
                    {
                        "speaker": "assistant",
                        "content": assistant_response,
                        "timestamp": hour_timestamp
                    }
                ]
            }
            
            # 保存到记忆（增量更新）
            print(f"\n💾 保存对话到记忆...")
            # 使用配置中的LLM provider
            llm_provider = self.h_seam.config.llm_config.provider
            self.h_seam.build_memory(conversation_data, namespace=self.namespace, llm_provider=llm_provider)
            print(f"   ✅ 对话已保存")
            
        except Exception as e:
            print(f"   ⚠️  保存对话失败: {e}")
    
    def interactive_chat(self):
        """启动交互式对话"""
        print("\n" + "="*60)
        print("🤖 H-SEAM 聊天机器人")
        print("="*60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 再见！")
                    break
                
                if user_input.lower() in ['clear', '清空']:
                    self.conversation_history = []
                    self.turn_count = 0  # 重置计数器
                    print("✅ 对话历史已清空\n")
                    continue
                
                # 生成回答
                response = self.chat(user_input)
                print(f"\n🤖 助手: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")


def main():
    """主函数"""
    print("🚀 初始化 H-SEAM 聊天机器人示例...")
    
    # 1. 检查配置
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("\n💡 请先配置 API keys（在 .env 文件中）:")
        print("   - LLM_API_KEY")
        print("   - LLM_PROVIDER")
        print("   - EMBEDDING_API_KEY")
        print("   - EMBEDDING_PROVIDER")
        return
    
    # 2. 初始化 H-SEAM
    try:
        h_seam = H_SEAM(config)
    except Exception as e:
        print(f"❌ H-SEAM 初始化失败: {e}")
        return
    
    # 3. 检查是否有已构建的记忆
    print("\n💡 提示:")
    print("   如果这是第一次运行，建议先构建一些记忆:")
    print("   h_seam.build_memory_from_file('your_conversation.json')")
    print("   或者使用 h_seam.build_memory(conversation_data)")
    print()
    
    # 4. 创建聊天机器人
    chatbot = ChatBot(
        h_seam=h_seam,
        namespace="default",
        save_conversation=True  # 将新对话保存到记忆中
    )
    
    # 5. 启动交互式对话
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()





