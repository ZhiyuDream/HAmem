"""
测试记忆构建流程并统计token数量和时延

使用真实的MemoryBuilder.build_memory()流程，包括：
1. Fragment splitting（LLM调用）
2. Layer1 extraction（LLM调用）
3. Layer1 recall（向量搜索，无LLM）
4. Layer1 conflict resolution（LLM调用）
5. Layer2 extraction（LLM调用）
6. Layer3 pattern extraction（LLM调用，周期性触发）
7. Embedding生成和Neo4j写入

统计内容：
- 原始对话内容的token数（tiktoken估算）
- 所有LLM调用的实际token数（从API响应获取）
- 各阶段的时延统计
"""

import sys
import os
import json
import time
import tiktoken
import argparse
from typing import Dict, Any, List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config
from core.main import HAmem
from core.infrastructure.token_tracker import TokenTracker


def load_locomo_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载locomo数据集"""
    print(f"📂 加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成，共 {len(data)} 个conversation")
    return data


def convert_conversation_to_hamem_format(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将locomo格式的conversation转换为HAmem格式
    """
    conversation = conversation_data.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "User")
    speaker_b = conversation.get("speaker_b", "Assistant")
    
    messages = []
    
    # 遍历所有session
    session_keys = [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time") and not k.endswith("_summary")]
    session_keys.sort()
    
    for session_key in session_keys:
        session = conversation.get(session_key, [])
        if not isinstance(session, list):
            continue
        
        date_time_key = f"{session_key}_date_time"
        session_time = conversation.get(date_time_key, "")
        
        if not session_time:
            session_time = "unknown"
        
        for turn in session:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            
            if not text:
                continue
            
            if speaker == speaker_a:
                role = "user"
            elif speaker == speaker_b:
                role = "assistant"
            else:
                role = "user"
            
            messages.append({
                "role": role,
                "content": text,
                "timestamp": session_time,
                "metadata": {
                    "speaker": speaker,
                    "dia_id": turn.get("dia_id", ""),
                    "session": session_key,
                    "session_time": session_time
                }
            })
    
    return {
        "messages": messages,
        "metadata": {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b
        }
    }


def count_tokens(text: str, encoding) -> int:
    """使用tiktoken计算token数量"""
    return len(encoding.encode(text))


def create_token_tracking_llm_client(config: Config, token_tracker: TokenTracker, default_provider: str = "openai"):
    """
    创建支持token追踪的LLMClient包装器
    
    由于需要拦截所有LLM调用，我们需要修改各个Extractor/Processor来记录token。
    这里我们创建一个包装器，但更好的方式是在各个组件中直接记录。
    
    实际上，我们需要修改：
    1. FragmentProcessor.should_split() - 记录fragment_splitting
    2. Layer1Extractor.extract_from_fragment() - 记录layer1_extraction
    3. Layer1ConflictResolver.resolve_conflicts_batch() - 记录layer1_conflict
    4. Layer2Extractor.extract_from_fragment() - 记录layer2_extraction
    5. Layer3Extractor.extract_patterns_from_cluster() - 记录layer3_pattern
    
    但这样需要修改很多文件。更好的方式是创建一个包装的LLMClient。
    """
    from core.infrastructure.llm import LLMClient
    
    class TokenTrackingLLMClient(LLMClient):
        """支持token追踪的LLMClient包装器"""
        
        def __init__(self, config, token_tracker, default_provider):
            super().__init__(config)
            self.token_tracker = token_tracker
            self.default_provider = default_provider
            self._current_call_type = None  # 当前调用类型（由调用者设置）
        
        def call_llm(self, prompt: str, model: str = None, provider: str = "deepseek", return_usage: bool = False, call_type: str = None):
            """
            调用LLM并记录token
            
            Args:
                call_type: 调用类型（用于统计分类）
            """
            # 如果没有指定provider，使用默认值
            if provider == "deepseek" and self.default_provider != "deepseek":
                provider = self.default_provider
            
            # 如果没有指定call_type，尝试从调用栈推断（简化处理，使用传入的call_type）
            if call_type:
                self._current_call_type = call_type
            
            # 调用父类方法，但强制返回usage
            result = super().call_llm(prompt, model, provider, return_usage=True)
            
            if isinstance(result, tuple):
                content, usage = result
                
                # 记录token使用情况
                if self.token_tracker and self._current_call_type:
                    self.token_tracker.record_llm_call(
                        self._current_call_type,
                        usage,
                        provider=provider
                    )
                
                # 如果调用者不需要usage，只返回content
                if not return_usage:
                    return content
                else:
                    return (content, usage)
            else:
                # 如果父类没有返回usage（不应该发生），尝试从响应中获取
                # 这里我们无法获取，所以返回空usage
                if return_usage:
                    return (result, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                return result
    
    return TokenTrackingLLMClient(config, token_tracker, default_provider)


def test_memory_building(conversation_idx: int, dataset_path: str = None, model: str = None, skip_storage: bool = False, namespace: str = None):
    """
    测试记忆构建流程并统计token和时延
    
    Args:
        conversation_idx: conversation索引（从0开始）
        dataset_path: 数据集路径
        model: LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）
        skip_storage: 是否跳过Neo4j存储（仅用于测试，不影响token统计）
        namespace: 命名空间（如果为None，使用默认值 locomo_conv_{conversation_idx}）
    """
    if dataset_path is None:
        # 默认路径（相对于项目根目录）
        dataset_path = os.path.join(
            project_root,
            "locomo", "data", "locomo10.json"
        )
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {dataset_path}")
        print(f"   请使用 --dataset 参数指定正确的数据集路径")
        return None
    
    # 加载数据集
    dataset = load_locomo_dataset(dataset_path)
    
    if conversation_idx >= len(dataset):
        print(f"❌ 错误: conversation索引 {conversation_idx} 超出范围（共 {len(dataset)} 个conversation）")
        return None
    
    # 获取conversation
    conversation_data = dataset[conversation_idx]
    
    # 转换为HAmem格式
    hamem_data = convert_conversation_to_hamem_format(conversation_data)
    messages = hamem_data["messages"]
    
    print(f"\n📊 Conversation {conversation_idx} 统计:")
    print(f"  - 消息数量: {len(messages)}")
    print(f"  - 数据集路径: {dataset_path}")
    if skip_storage:
        print(f"  - ⚠️  跳过Neo4j存储（仅测试模式）")
    
    # 初始化tiktoken编码器
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # 计算原始对话内容的token数
    conversation_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
    ])
    original_tokens = count_tokens(conversation_text, encoding)
    
    # 创建token追踪器
    token_tracker = TokenTracker()
    
    # 初始化Config
    config = Config()
    
    # 如果指定了模型，设置模型
    if model:
        config.set_llm_model(model)
        print(f"  - 使用模型: {model}")
    else:
        actual_model = config.llm_model
        print(f"  - 使用模型: {actual_model} (来自配置)")
    
    print(f"\n🔄 开始实际记忆构建流程...")
    
    # 初始化HAmem
    hamem = HAmem(config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 构建记忆（传入token_tracker）
    # 如果没有指定namespace，使用默认值
    if namespace is None:
        namespace = f"locomo_conv_{conversation_idx}"
    
    print(f"  - Namespace: {namespace}")
    
    try:
        from core.memory import MemoryBuilder, ConversationData
        memory_builder = MemoryBuilder(config)
        conversation = ConversationData.from_dict(hamem_data)
        # 使用config中的provider（从模型推断或配置中获取）
        llm_provider = config.llm_provider
        result = memory_builder.build_memory(conversation, namespace=namespace, token_tracker=token_tracker, llm_provider=llm_provider)
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 获取时延统计和token统计
        if hasattr(result, 'time_stats'):
            time_stats = result.time_stats
        else:
            time_stats = result.get('time_stats', {}) if isinstance(result, dict) else {}
        
        if hasattr(result, 'token_stats'):
            token_stats = result.token_stats
        else:
            token_stats = result.get('token_stats', {}) if isinstance(result, dict) else token_tracker.get_stats()
        
        print(f"\n✅ 记忆构建完成!")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        
        # 显示详细时延统计
        print(f"\n📈 时延统计:")
        if time_stats and time_stats.get('fragment_processing'):
            frag_times = time_stats['fragment_processing']
            print(f"  - Fragment处理:")
            print(f"    * 总时间: {sum(frag_times):.2f}秒")
            print(f"    * 平均: {sum(frag_times)/len(frag_times):.3f}秒/个")
            print(f"    * 最大: {max(frag_times):.3f}秒")
            print(f"    * 最小: {min(frag_times):.3f}秒")
            print(f"    * 数量: {len(frag_times)}")
        
        if time_stats and time_stats.get('layer1_processing'):
            layer1_times = time_stats['layer1_processing']
            print(f"  - Layer1处理:")
            print(f"    * 总时间: {sum(layer1_times):.2f}秒")
            print(f"    * 平均: {sum(layer1_times)/len(layer1_times):.3f}秒/个")
            print(f"    * 数量: {len(layer1_times)}")
        
        if time_stats and time_stats.get('layer2_processing'):
            layer2_times = time_stats['layer2_processing']
            print(f"  - Layer2处理:")
            print(f"    * 总时间: {sum(layer2_times):.2f}秒")
            print(f"    * 平均: {sum(layer2_times)/len(layer2_times):.3f}秒/个")
            print(f"    * 数量: {len(layer2_times)}")
        
        if time_stats and time_stats.get('layer3_processing'):
            layer3_times = time_stats['layer3_processing']
            print(f"  - Layer3处理:")
            print(f"    * 总时间: {sum(layer3_times):.2f}秒")
            print(f"    * 平均: {sum(layer3_times)/len(layer3_times):.3f}秒/次")
            print(f"    * 数量: {len(layer3_times)}")
        
        # 显示详细token统计
        print(f"\n📊 Token统计（实际API调用）:")
        print(f"  {'='*70}")
        print(f"  1. 原始对话内容: {original_tokens:,} tokens (tiktoken估算)")
        print(f"  {'='*70}")
        
        if token_stats:
            total_llm_tokens = 0
            for call_type, stats in token_stats.items():
                if call_type == "by_provider":
                    continue
                prompt_tokens = stats.get("prompt_tokens", 0)
                completion_tokens = stats.get("completion_tokens", 0)
                total_tokens = stats.get("total_tokens", 0)
                calls = stats.get("calls", 0)
                total_llm_tokens += total_tokens
                
                print(f"  2. {call_type.replace('_', ' ').title()}:")
                print(f"     - Prompt tokens: {prompt_tokens:,}")
                print(f"     - Completion tokens: {completion_tokens:,}")
                print(f"     - Total tokens: {total_tokens:,}")
                print(f"     - LLM调用次数: {calls}")
                
                # 显示按提供商分类的统计
                by_provider = stats.get("by_provider", {})
                if by_provider:
                    for provider, provider_stats in by_provider.items():
                        print(f"       * {provider}: {provider_stats.get('total_tokens', 0):,} tokens ({provider_stats.get('calls', 0)} calls)")
            
            print(f"  {'='*70}")
            print(f"  📊 LLM调用总计: {total_llm_tokens:,} tokens")
            print(f"  📊 总计（原始+LLM）: {original_tokens + total_llm_tokens:,} tokens")
            print(f"  {'='*70}")
        else:
            print(f"  ⚠️  未获取到token统计信息")
        
        return {
            "original_tokens": original_tokens,
            "total_time": total_time,
            "time_stats": time_stats,
            "token_stats": token_stats,
            "result": result
        }
        
    except Exception as e:
        print(f"❌ 记忆构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="测试记忆构建流程并统计token数量和时延",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型（从Config读取）
  python test_memory_building.py 0
  
  # 指定模型
  python test_memory_building.py 0 --model gpt-4o-mini
  
  # 指定数据集路径和模型
  python test_memory_building.py 0 --dataset /path/to/locomo10.json --model deepseek-chat
  
  # 跳过Neo4j存储（仅测试模式）
  python test_memory_building.py 0 --skip-storage
        """
    )
    
    parser.add_argument(
        "conversation_idx",
        type=int,
        help="conversation索引（从0开始）"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/zhiyu_zheng/DCL/Others/locomo/data/locomo10.json",
        help="数据集路径（默认: <project_root>/locomo/data/locomo10.json）"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）"
    )
    
    parser.add_argument(
        "--skip-storage",
        action="store_true",
        help="跳过Neo4j存储（仅用于测试，不影响token统计）"
    )
    
    args = parser.parse_args()
    
    test_memory_building(
        conversation_idx=args.conversation_idx,
        dataset_path=args.dataset,
        model=args.model,
        skip_storage=args.skip_storage
    )


if __name__ == "__main__":
    main()

