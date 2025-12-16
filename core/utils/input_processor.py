"""
输入格式处理和转换工具

支持从文件读取对话数据，自动识别格式并转换为HAmem标准格式
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


def load_conversation_file(file_path: str) -> Dict[str, Any]:
    """
    从文件加载对话数据
    
    Args:
        file_path: 文件路径（支持 .json 格式）
    
    Returns:
        对话数据字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if not file_path.endswith('.json'):
        raise ValueError(f"目前只支持 JSON 格式文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def detect_format(data: Dict[str, Any]) -> str:
    """
    自动检测输入数据格式
    
    Args:
        data: 输入数据字典
    
    Returns:
        格式类型: 'hamem_standard', 'locomo', 'sessions', 'messages', 'unknown'
    """
    # 检查是否是HAmem标准格式
    if 'messages' in data and isinstance(data['messages'], list):
        if len(data['messages']) > 0:
            msg = data['messages'][0]
            # 标准格式：包含 speaker, timestamp, content
            if 'speaker' in msg and 'timestamp' in msg and 'content' in msg:
                return 'hamem_standard'
            # messages格式：包含 role, content
            elif 'role' in msg and 'content' in msg:
                return 'messages'
    
    # 检查是否是locomo格式
    if 'conversation' in data:
        return 'locomo'
    
    # 检查是否是sessions格式
    if 'sessions' in data and isinstance(data['sessions'], list):
        return 'sessions'
    
    # 检查是否是批量格式
    if 'conversations' in data and isinstance(data['conversations'], list):
        return 'batch'
    
    return 'unknown'


def convert_to_hamem_format(data: Dict[str, Any], format_type: str = None) -> Dict[str, Any]:
    """
    将各种格式转换为HAmem标准格式
    
    Args:
        data: 输入数据
        format_type: 格式类型（如果为None，则自动检测）
    
    Returns:
        HAmem标准格式数据: {"messages": [...], "metadata": {...}}
    """
    if format_type is None:
        format_type = detect_format(data)
    
    if format_type == 'hamem_standard':
        # 已经是标准格式，直接返回
        return data
    
    elif format_type == 'messages':
        # messages格式转换为标准格式
        messages = []
        for msg in data.get('messages', []):
            # 将 role 转换为 speaker
            speaker = msg.get('speaker', msg.get('role', 'unknown'))
            content = msg.get('content', msg.get('text', ''))
            timestamp = msg.get('timestamp', datetime.now().isoformat())
            
            messages.append({
                'speaker': speaker,
                'content': content,
                'timestamp': timestamp,
                'metadata': msg.get('metadata', {})
            })
        
        return {
            'messages': messages,
            'metadata': data.get('metadata', {})
        }
    
    elif format_type == 'locomo':
        # locomo格式转换
        return convert_locomo_to_hamem(data)
    
    elif format_type == 'sessions':
        # sessions格式转换
        messages = []
        for session in data.get('sessions', []):
            session_timestamp = session.get('timestamp', datetime.now().isoformat())
            for turn in session.get('turns', []):
                speaker = turn.get('speaker', 'unknown')
                content = turn.get('text', turn.get('content', ''))
                turn_timestamp = turn.get('timestamp', session_timestamp)
                
                messages.append({
                    'speaker': speaker,
                    'content': content,
                    'timestamp': turn_timestamp,
                    'metadata': turn.get('metadata', {})
                })
        
        return {
            'messages': messages,
            'metadata': data.get('metadata', {})
        }
    
    elif format_type == 'batch':
        # 批量格式：取第一个conversation
        if len(data.get('conversations', [])) > 0:
            return convert_to_hamem_format(data['conversations'][0])
        else:
            raise ValueError("批量格式中没有conversation数据")
    
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")


def convert_locomo_to_hamem(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将locomo格式转换为HAmem标准格式
    
    Args:
        conversation_data: locomo格式数据
    
    Returns:
        HAmem标准格式数据
    """
    conversation = conversation_data.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "User")
    speaker_b = conversation.get("speaker_b", "Assistant")
    
    messages = []
    
    # 遍历所有session
    session_keys = [k for k in conversation.keys() 
                   if k.startswith("session_") 
                   and not k.endswith("_date_time") 
                   and not k.endswith("_summary")]
    session_keys.sort()
    
    for session_key in session_keys:
        session = conversation.get(session_key, [])
        if not isinstance(session, list):
            continue
        
        date_time_key = f"{session_key}_date_time"
        session_time = conversation.get(date_time_key, "")
        
        if not session_time:
            session_time = datetime.now().isoformat()
        
        for turn in session:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            
            if not text:
                continue
            
            # 将speaker转换为标准格式
            if speaker == speaker_a:
                speaker_name = "user"
            elif speaker == speaker_b:
                speaker_name = "assistant"
            else:
                speaker_name = speaker or "unknown"
            
            messages.append({
                "speaker": speaker_name,
                "content": text,
                "timestamp": session_time,
                "metadata": {
                    "original_speaker": speaker,
                    "dia_id": turn.get("dia_id", ""),
                    "session": session_key,
                    "session_time": session_time
                }
            })
    
    return {
        "messages": messages,
        "metadata": {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "source": "locomo"
        }
    }


def validate_hamem_format(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    验证数据是否符合HAmem标准格式
    
    Args:
        data: 待验证的数据
    
    Returns:
        (是否有效, 错误信息)
    """
    if not isinstance(data, dict):
        return False, "数据必须是字典类型"
    
    if 'messages' not in data:
        return False, "缺少 'messages' 字段"
    
    if not isinstance(data['messages'], list):
        return False, "'messages' 必须是列表类型"
    
    if len(data['messages']) == 0:
        return False, "'messages' 列表不能为空"
    
    # 验证每个message
    for i, msg in enumerate(data['messages']):
        if not isinstance(msg, dict):
            return False, f"messages[{i}] 必须是字典类型"
        
        if 'speaker' not in msg:
            return False, f"messages[{i}] 缺少 'speaker' 字段"
        
        if 'content' not in msg:
            return False, f"messages[{i}] 缺少 'content' 字段"
        
        if 'timestamp' not in msg:
            return False, f"messages[{i}] 缺少 'timestamp' 字段"
    
    return True, None


def process_input_file(file_path: str) -> Dict[str, Any]:
    """
    从文件读取并处理输入数据（一站式处理）
    
    Args:
        file_path: 输入文件路径
    
    Returns:
        HAmem标准格式数据
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式不支持或转换失败
    """
    # 1. 读取文件
    data = load_conversation_file(file_path)
    
    # 2. 检测格式
    format_type = detect_format(data)
    if format_type == 'unknown':
        raise ValueError(f"无法识别文件格式: {file_path}")
    
    # 3. 转换为标准格式
    standard_data = convert_to_hamem_format(data, format_type)
    
    # 4. 验证格式
    is_valid, error_msg = validate_hamem_format(standard_data)
    if not is_valid:
        raise ValueError(f"格式验证失败: {error_msg}")
    
    return standard_data