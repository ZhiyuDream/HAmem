"""
计算locomo数据集中一个conversation的token数量

包括：
1. 原始对话内容的token数
2. Fragment splitting prompt的token数
3. Layer1 extraction prompt的token数（每个fragment）
4. Layer2 extraction prompt的token数（每个fragment）
"""

import sys
import os
import json
import tiktoken
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fragment.buffer_manager import BufferManager
from core.fragment.prompt import build_batch_split_fragment_prompt
from core.layer1.prompt.extraction_prompt import build_layer1_extraction_prompt
from core.layer2.prompt.extraction_prompt import build_layer2_extraction_prompt


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


def simulate_fragment_processing(messages: List[Dict[str, Any]], encoding) -> Dict[str, Any]:
    """
    模拟fragment处理流程，计算所有prompt的token数
    
    Returns:
        包含token统计的字典
    """
    buffer_manager = BufferManager(max_length=5000)
    
    stats = {
        "original_conversation_tokens": 0,
        "fragment_splitting_tokens": 0,
        "layer1_extraction_tokens": 0,
        "layer2_extraction_tokens": 0,
        "fragments_created": 0,
        "llm_splitting_calls": 0,
        "fragments": []
    }
    
    # 计算原始对话内容的token数
    conversation_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
    ])
    stats["original_conversation_tokens"] = count_tokens(conversation_text, encoding)
    
    # 模拟处理流程
    for i, msg in enumerate(messages, 1):
        turn = {
            'role': msg.get('role', 'user'),
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', ''),
            'metadata': msg.get('metadata', {})
        }
        
        # 检查是否需要LLM判断分片
        fragment, needs_llm = buffer_manager.add_turn(turn, timestamp=msg.get('timestamp'))
        
        if fragment:
            stats["fragments_created"] += 1
            stats["fragments"].append(fragment)
        
        if needs_llm:
            # 需要LLM判断分片
            turns = buffer_manager.get_turns_for_llm()
            if turns:
                # 构建fragment splitting prompt
                formatted_turns = []
                for t in turns:
                    formatted_turns.append({
                        'speaker': t.get('role', 'unknown'),
                        'text': t.get('content', '')
                    })
                
                prompt = build_batch_split_fragment_prompt(formatted_turns)
                tokens = count_tokens(prompt, encoding)
                stats["fragment_splitting_tokens"] += tokens
                stats["llm_splitting_calls"] += 1
                
                # 模拟分片（假设在中间位置分片）
                # 实际中这里会调用LLM，但我们现在只计算token
                split_point = len(turns) // 2 if len(turns) > 1 else len(turns)
                
                # 提取fragment
                fragment = buffer_manager.extract_fragment(split_point)
                if fragment:
                    stats["fragments_created"] += 1
                    stats["fragments"].append(fragment)
                
                # 保留剩余部分
                buffer_manager.keep_remaining(split_point)
    
    # 处理剩余的buffer（如果有）
    if not buffer_manager.is_empty():
        fragment = buffer_manager._save_timestamp_fragment()
        if fragment:
            stats["fragments_created"] += 1
            stats["fragments"].append(fragment)
    
    # 对每个fragment计算Layer1和Layer2的token数
    for fragment in stats["fragments"]:
        fragment_text = fragment.get('content', '')
        
        # Layer1 extraction prompt
        layer1_prompt = build_layer1_extraction_prompt(fragment_text)
        layer1_tokens = count_tokens(layer1_prompt, encoding)
        stats["layer1_extraction_tokens"] += layer1_tokens
        
        # 估算Layer1提取的实体（用于Layer2 prompt）
        # 基于fragment长度估算实体数量：每500字符约1-2个实体
        fragment_length = len(fragment_text)
        estimated_entity_count = max(2, min(10, fragment_length // 500))
        
        # 构建估算的实体列表（使用典型的实体描述格式）
        # 典型的实体描述约30-50词，包含身份、特征等信息
        typical_entity_descriptions = [
            "A supportive person who values family and relationships",
            "A creative individual with a passion for art and self-expression",
            "A professional working in healthcare with empathy and understanding",
            "A close friend who provides emotional support and encouragement",
            "An organization focused on community support and advocacy",
            "A specific location that holds personal significance",
            "A concept or activity that represents personal values or preferences",
            "A family member with strong emotional connections",
            "A mentor or role model who provides guidance",
            "A place of personal importance or emotional attachment"
        ]
        
        layer1_entities = []
        for i in range(estimated_entity_count):
            desc = typical_entity_descriptions[i % len(typical_entity_descriptions)]
            layer1_entities.append({
                "name": f"Entity{i+1}",
                "content": desc
            })
        
        # Layer2 extraction prompt
        session_time = fragment.get('time', 'unknown')
        layer2_prompt = build_layer2_extraction_prompt(
            fragment_text,
            session_time,
            layer1_entities
        )
        layer2_tokens = count_tokens(layer2_prompt, encoding)
        stats["layer2_extraction_tokens"] += layer2_tokens
    
    return stats


def calculate_token_count(conversation_idx: int, dataset_path: str = None):
    """
    计算指定conversation的token数量
    
    Args:
        conversation_idx: conversation索引（从0开始）
        dataset_path: 数据集路径
    """
    if dataset_path is None:
        # 默认路径
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "locomo", "data", "locomo10.json"
        )
    
    # 加载数据集
    dataset = load_locomo_dataset(dataset_path)
    
    if conversation_idx >= len(dataset):
        print(f"❌ 错误: conversation索引 {conversation_idx} 超出范围（共 {len(dataset)} 个conversation）")
        return
    
    # 获取conversation
    conversation_data = dataset[conversation_idx]
    
    # 转换为HAmem格式
    hamem_data = convert_conversation_to_hamem_format(conversation_data)
    messages = hamem_data["messages"]
    
    print(f"\n📊 Conversation {conversation_idx} 统计:")
    print(f"  - 消息数量: {len(messages)}")
    
    # 初始化tiktoken编码器（使用gpt-4o-mini的编码器，即cl100k_base）
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # 模拟处理流程并计算token
    print(f"\n🔄 模拟记忆处理流程...")
    stats = simulate_fragment_processing(messages, encoding)
    
    # 汇总结果
    total_tokens = (
        stats["original_conversation_tokens"] +
        stats["fragment_splitting_tokens"] +
        stats["layer1_extraction_tokens"] +
        stats["layer2_extraction_tokens"]
    )
    
    print(f"\n📈 Token统计结果:")
    print(f"  {'='*60}")
    print(f"  1. 原始对话内容: {stats['original_conversation_tokens']:,} tokens")
    print(f"  2. Fragment splitting prompts: {stats['fragment_splitting_tokens']:,} tokens")
    print(f"     - LLM分片判断调用次数: {stats['llm_splitting_calls']}")
    print(f"  3. Layer1 extraction prompts: {stats['layer1_extraction_tokens']:,} tokens")
    print(f"     - Fragment数量: {stats['fragments_created']}")
    print(f"  4. Layer2 extraction prompts: {stats['layer2_extraction_tokens']:,} tokens")
    print(f"     - Fragment数量: {stats['fragments_created']}")
    print(f"  {'='*60}")
    print(f"  📊 总计: {total_tokens:,} tokens")
    print(f"  {'='*60}")
    
    # 计算平均每个fragment的token数
    if stats['fragments_created'] > 0:
        avg_layer1 = stats['layer1_extraction_tokens'] / stats['fragments_created']
        avg_layer2 = stats['layer2_extraction_tokens'] / stats['fragments_created']
        print(f"\n  📊 平均每个Fragment:")
        print(f"     - Layer1 prompt: {avg_layer1:.0f} tokens")
        print(f"     - Layer2 prompt: {avg_layer2:.0f} tokens")
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calculate_token_count.py <conversation_idx> [dataset_path]")
        print("示例: python calculate_token_count.py 0")
        sys.exit(1)
    
    conversation_idx = int(sys.argv[1])
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    calculate_token_count(conversation_idx, dataset_path)

