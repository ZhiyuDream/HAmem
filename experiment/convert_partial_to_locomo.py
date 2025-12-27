"""
将 partial.json 格式转换为 locomo 格式

partial.json 格式:
{
  "character": "...",
  "assistant_character": "...",
  "dialogue_sessions": [
    {
      "timestamp": "...",
      "location": "...",
      "utterances": [
        {"speaker": "user", "text": "..."},
        {"speaker": "assistant", "text": "..."}
      ]
    }
  ],
  "questions": [
    {
      "question_id": "...",
      "user_query": "...",
      "query_timestamp": "...",
      "reasoning_type": "single-hop|temporal|adversarial|multi-hop",
      "evidence_timestamps": ["..."],
      "constraint_for_response": "..."
    }
  ]
}

locomo 格式:
{
  "conversation": {
    "speaker_a": "...",
    "speaker_b": "...",
    "session_0": [
      {"speaker": "...", "text": "...", "dia_id": "..."}
    ],
    "session_0_date_time": "...",
    "session_1": [...],
    "session_1_date_time": "..."
  },
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "evidence": ["D1:3"],
      "category": 1|2|3|4
    }
  ]
}

转换说明:
- questions.user_query -> qa.question
- questions.constraint_for_response -> qa.answer (注意：constraint不是直接答案，而是约束条件)
- questions.reasoning_type -> qa.category (映射: single-hop->1, temporal->2, multi-hop->3, adversarial->1)
- questions.evidence_timestamps -> qa.evidence (转换为 "D1:3" 格式)
"""

import json
import os
import sys
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime


def map_reasoning_type_to_category(reasoning_type: str) -> int:
    """
    将 reasoning_type 映射到 category
    
    Args:
        reasoning_type: single-hop, temporal, adversarial, multi-hop
        
    Returns:
        category: 1 (事实性), 2 (时间相关), 3 (推理性), 4 (其他)
    """
    mapping = {
        "single-hop": 1,      # 事实性问题
        "temporal": 2,        # 时间相关问题
        "adversarial": 1,     # 对抗性问题，通常也是事实性
        "multi-hop": 3        # 多跳推理问题
    }
    return mapping.get(reasoning_type, 4)  # 默认为4（其他）


def find_evidence_references(
    evidence_timestamps: List[str],
    dialogue_sessions: List[Dict[str, Any]]
) -> List[str]:
    """
    根据时间戳找到对应的evidence引用（格式：D1:3）
    
    Args:
        evidence_timestamps: 证据时间戳列表
        dialogue_sessions: 对话session列表
        
    Returns:
        evidence引用列表，格式如 ["D1:3", "D2:5"]
    """
    evidence_refs = []
    
    for evidence_ts in evidence_timestamps:
        # 尝试解析时间戳
        try:
            evidence_dt = datetime.fromisoformat(evidence_ts.replace('Z', '+00:00'))
        except:
            evidence_dt = None
        
        # 找到对应的session（精确匹配或最接近的）
        best_match = None
        best_diff = None
        
        for session_idx, session in enumerate(dialogue_sessions):
            session_ts = session.get("timestamp", "")
            if session_ts == evidence_ts:
                # 精确匹配，使用第一个turn
                doc_ref = f"D{session_idx + 1}:1"
                evidence_refs.append(doc_ref)
                best_match = None  # 已找到，不需要继续
                break
            elif evidence_dt and session_ts:
                # 尝试时间比较（找到最接近的session）
                try:
                    session_dt = datetime.fromisoformat(session_ts.replace('Z', '+00:00'))
                    diff = abs((evidence_dt - session_dt).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_match = (session_idx, session)
                except:
                    pass
        
        # 如果没有精确匹配，使用最接近的session
        if best_match is not None:
            session_idx, session = best_match
            # 使用该session的第一个turn
            doc_ref = f"D{session_idx + 1}:1"
            evidence_refs.append(doc_ref)
        elif not evidence_refs:  # 如果还没有添加任何引用
            # 如果找不到匹配，使用第一个session的第一个turn作为默认
            evidence_refs.append("D1:1")
    
    # 去重并保持顺序
    seen = set()
    unique_refs = []
    for ref in evidence_refs:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    
    return unique_refs if unique_refs else ["D1:1"]


def extract_answer_from_constraint(constraint: str) -> str:
    """
    从 constraint_for_response 中提取答案信息
    
    Args:
        constraint: constraint_for_response 的内容
        
    Returns:
        提取的答案文本
    
    注意：constraint_for_response 不是直接的答案，而是对答案的约束条件。
    这里我们尝试提取关键信息，但可能不够准确。实际使用时可能需要人工调整。
    """
    if not constraint:
        return ""
    
    # 尝试提取关键信息
    # 1. 如果包含日期信息，尝试提取
    # 2. 如果包含具体事实，尝试提取
    # 3. 否则使用整个constraint
    
    # 简单处理：尝试提取关键短语
    # 例如："Sunday, November 28st" -> "Sunday, November 28st"
    # 或者："dairy" -> "no dairy"
    
    # 为了保持信息的完整性，我们直接使用constraint
    # 虽然这不是完美的答案，但至少包含了期望的答案信息
    return constraint


def convert_questions_to_qa(
    questions: List[Dict[str, Any]],
    dialogue_sessions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    将 partial.json 的 questions 转换为 locomo 的 qa 格式
    
    Args:
        questions: partial.json 格式的问题列表
        dialogue_sessions: 对话session列表（用于查找evidence）
        
    Returns:
        locomo 格式的 qa 列表
    """
    qa_list = []
    
    for question_data in questions:
        user_query = question_data.get("user_query", "")
        constraint = question_data.get("constraint_for_response", "")
        reasoning_type = question_data.get("reasoning_type", "")
        evidence_timestamps = question_data.get("evidence_timestamps", [])
        
        if not user_query:
            continue
        
        # 转换答案：使用constraint作为答案（或从中提取）
        answer = extract_answer_from_constraint(constraint)
        
        # 转换category
        category = map_reasoning_type_to_category(reasoning_type)
        
        # 转换evidence
        evidence = find_evidence_references(evidence_timestamps, dialogue_sessions)
        
        qa_item = {
            "question": user_query,
            "answer": answer,
            "evidence": evidence,
            "category": category
        }
        
        qa_list.append(qa_item)
    
    return qa_list


def convert_partial_to_locomo(partial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 partial.json 格式的单个对话转换为 locomo 格式
    
    Args:
        partial_data: partial.json 格式的数据
        
    Returns:
        locomo 格式的数据
    """
    # 提取基本信息
    character = partial_data.get("character", "User")
    assistant_character = partial_data.get("assistant_character", "Assistant")
    dialogue_sessions = partial_data.get("dialogue_sessions", [])
    questions = partial_data.get("questions", [])
    
    # 构建 conversation 对象
    conversation = {
        "speaker_a": character,
        "speaker_b": assistant_character
    }
    
    # 转换 dialogue_sessions 为 session_X 格式
    for idx, session in enumerate(dialogue_sessions):
        session_key = f"session_{idx}"
        timestamp = session.get("timestamp", "")
        utterances = session.get("utterances", [])
        
        # 转换 utterances
        session_turns = []
        for turn_idx, utterance in enumerate(utterances):
            speaker = utterance.get("speaker", "")
            text = utterance.get("text", "")
            
            if not text:
                continue
            
            # 将 "user" 转换为 speaker_a，将 "assistant" 转换为 speaker_b
            if speaker == "user":
                actual_speaker = character
            elif speaker == "assistant":
                actual_speaker = assistant_character
            else:
                # 如果speaker不是user或assistant，保持原样
                actual_speaker = speaker
            
            session_turns.append({
                "speaker": actual_speaker,
                "text": text,
                "dia_id": f"{session_key}_turn_{turn_idx}"  # 生成dia_id
            })
        
        # 添加session数据
        conversation[session_key] = session_turns
        
        # 添加时间戳
        if timestamp:
            conversation[f"{session_key}_date_time"] = timestamp
    
    # 转换 questions 为 qa 格式
    qa_list = convert_questions_to_qa(questions, dialogue_sessions)
    
    # 构建最终的locomo格式
    locomo_data = {
        "conversation": conversation,
        "qa": qa_list
    }
    
    return locomo_data


def convert_file(input_path: str, output_path: str = None):
    """
    转换整个文件
    
    Args:
        input_path: 输入的 partial.json 文件路径
        output_path: 输出的 locomo 格式文件路径（如果为None，自动生成）
    """
    print(f"📂 读取文件: {input_path}")
    
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        partial_data_list = json.load(f)
    
    if not isinstance(partial_data_list, list):
        print(f"❌ 错误: 输入文件应该是一个JSON数组")
        return
    
    print(f"✅ 读取完成，共 {len(partial_data_list)} 个对话")
    
    # 转换每个对话
    locomo_data_list = []
    for idx, partial_data in enumerate(partial_data_list):
        try:
            locomo_data = convert_partial_to_locomo(partial_data)
            locomo_data_list.append(locomo_data)
            
            # 显示进度
            if (idx + 1) % 10 == 0:
                print(f"  已转换: {idx + 1}/{len(partial_data_list)}")
        except Exception as e:
            print(f"⚠️  转换第 {idx} 个对话时出错: {e}")
            continue
    
    print(f"✅ 转换完成，共 {len(locomo_data_list)} 个对话")
    
    # 确定输出路径
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_basename = os.path.basename(input_path)
        input_name, _ = os.path.splitext(input_basename)
        output_path = os.path.join(input_dir, f"{input_name}_locomo.json")
    
    # 保存输出文件
    print(f"💾 保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(locomo_data_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 保存完成")
    
    # 显示统计信息
    print(f"\n📊 转换统计:")
    print(f"  - 输入对话数: {len(partial_data_list)}")
    print(f"  - 成功转换: {len(locomo_data_list)}")
    print(f"  - 失败: {len(partial_data_list) - len(locomo_data_list)}")
    
    # 显示第一个对话的示例信息
    if locomo_data_list:
        first_conv = locomo_data_list[0]["conversation"]
        first_qa = locomo_data_list[0].get("qa", [])
        session_count = len([k for k in first_conv.keys() if k.startswith("session_") and not k.endswith("_date_time")])
        print(f"\n📝 示例对话信息:")
        print(f"  - Speaker A: {first_conv.get('speaker_a')}")
        print(f"  - Speaker B: {first_conv.get('speaker_b')}")
        print(f"  - Session数量: {session_count}")
        print(f"  - QA数量: {len(first_qa)}")
        
        # 统计所有对话的QA数量
        total_qa = sum(len(item.get("qa", [])) for item in locomo_data_list)
        print(f"  - 总QA数量: {total_qa}")


def main():
    parser = argparse.ArgumentParser(
        description="将 partial.json 格式转换为 locomo 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认输出路径（在同一目录下生成 _locomo.json 文件）
  python convert_partial_to_locomo.py /path/to/partial.json
  
  # 指定输出路径
  python convert_partial_to_locomo.py /path/to/partial.json -o /path/to/output.json
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="输入的 partial.json 文件路径"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出的 locomo 格式文件路径（默认: <input_dir>/<input_name>_locomo.json）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 执行转换
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()

