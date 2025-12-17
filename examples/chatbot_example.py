"""
聊天机器人示例

展示如何使用 HAmem 的 search_memory 接口检索历史信息，
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HAmem
from config import Config
from core.infrastructure import LLMClient


class ChatBot:
    """基于 HAmem 的聊天机器人"""
    
    def __init__(self, hamem: HAmem, namespace: str = "default", save_conversation: bool = True):
        """
        初始化聊天机器人
        
        Args:
            hamem: HAmem 实例
            namespace: 命名空间
            save_conversation: 是否将新对话保存到记忆中
        """
        self.hamem = hamem
        self.namespace = namespace
        self.save_conversation = save_conversation
        
        # 初始化 LLM 客户端用于生成对话
        self.llm_client = LLMClient(hamem.config)
        
        # 对话历史（用于维护上下文）
        self.conversation_history: List[Dict[str, str]] = []
        
        print("🤖 聊天机器人已初始化")
        print(f"   - 命名空间: {namespace}")
        print(f"   - 保存对话: {'是' if save_conversation else '否'}")
    
    def chat(self, user_input: str, top_k: int = 5) -> str:
        """
        与用户聊天，自动检索历史信息并生成回答
        
        Args:
            user_input: 用户输入
            top_k: 检索的历史信息数量
            
        Returns:
            机器人的回答
        """
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")
        
        # 1. 检索相关历史信息（使用 search_memory 接口）
        print(f"\n🔍 检索历史信息...")
        try:
            search_results = self.hamem.search_memory(
                query=user_input,
                top_k=top_k,
                namespace=self.namespace
            )
            
            # 格式化检索结果作为上下文
            context = self._format_search_results(search_results)
            print(f"   ✅ 检索到 {len(search_results)} 条相关历史信息")
            
        except Exception as e:
            print(f"   ⚠️  检索历史信息失败: {e}")
            context = "暂无相关历史信息。"
        
        # 2. 构建对话提示（包含历史上下文和当前对话）
        prompt = self._build_conversation_prompt(user_input, context)
        
        # 3. 调用 LLM 生成对话式回答
        print(f"\n💬 生成回答...")
        try:
            response = self.llm_client.call_llm(
                prompt=prompt,
                provider=self.hamem.config.llm_config.provider
            )
            print(f"   ✅ 回答生成完成")
        except Exception as e:
            print(f"   ❌ 生成回答失败: {e}")
            response = "抱歉，我遇到了一些问题，无法生成回答。"
        
        # 4. 保存对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # 5. 可选：将新对话保存到记忆中
        if self.save_conversation:
            self._save_to_memory(user_input, response)
        
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
        # 构建最近对话历史（仅保留最近几轮）
        recent_history = self.conversation_history[-6:]  # 最近3轮对话
        history_text = ""
        if recent_history:
            history_text = "【最近对话历史】\n"
            for msg in recent_history:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"
        
        # 构建完整提示
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
        
        return prompt
    
    def _save_to_memory(self, user_input: str, assistant_response: str):
        """
        将新对话保存到记忆中
        
        Args:
            user_input: 用户输入
            assistant_response: 助手回答
        """
        try:
            # 构建标准格式的对话数据
            conversation_data = {
                "messages": [
                    {
                        "speaker": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "speaker": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            # 保存到记忆（增量更新）
            print(f"\n💾 保存对话到记忆...")
            self.hamem.build_memory(conversation_data, namespace=self.namespace)
            print(f"   ✅ 对话已保存")
            
        except Exception as e:
            print(f"   ⚠️  保存对话失败: {e}")
    
    def interactive_chat(self):
        """启动交互式对话"""
        print("\n" + "="*60)
        print("🤖 HAmem 聊天机器人")
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
    print("🚀 初始化 HAmem 聊天机器人示例...")
    
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
    
    # 2. 初始化 HAmem
    try:
        hamem = HAmem(config)
    except Exception as e:
        print(f"❌ HAmem 初始化失败: {e}")
        return
    
    # 3. 检查是否有已构建的记忆
    print("\n💡 提示:")
    print("   如果这是第一次运行，建议先构建一些记忆:")
    print("   hamem.build_memory_from_file('your_conversation.json')")
    print("   或者使用 hamem.build_memory(conversation_data)")
    print()
    
    # 4. 创建聊天机器人
    chatbot = ChatBot(
        hamem=hamem,
        namespace="default",
        save_conversation=True  # 将新对话保存到记忆中
    )
    
    # 5. 启动交互式对话
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()


