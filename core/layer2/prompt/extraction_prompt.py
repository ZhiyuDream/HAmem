"""
Layer2 Extraction Prompt

从fragment中提取时间线信息（事件、状态、上下文）
"""


def build_layer2_extraction_prompt(fragment_text, session_time, layer1_entities):
    """
    构建Layer2提取prompt
    
    Args:
        fragment_text: fragment文本内容
        session_time: 会话时间信息
        layer1_entities: Layer1提取的实体列表（用于参考）
    
    Returns:
        str: 构建的prompt
    """
    # 格式化实体列表
    entities_str = "\n".join([
        f"- {entity.get('name', '')}: {entity.get('content', '')}" 
        for entity in layer1_entities
    ]) if layer1_entities else "No entities available"
    
    return f"""
Extract timeline information (events, states, contexts) from the conversation fragment.

Fragment: {fragment_text}
Session time: {session_time}

Available entities for reference:
{entities_str}

Classify each piece of information as:
- EVENT: Dynamic activities, plans, temporary states
- STATE: Ongoing situations, conditions
- CONTEXT: Environmental/background information

Output JSON format:
{{
    "events": [
        {{
            "content": "comprehensive event description including time, location, participants",
            "participants": ["entity names involved"],
            "location": "location if mentioned",
            "conversation_time": "{session_time}",
            "relative_time": "relative time expression from text"
        }}
    ],
    "states": [
        {{
            "content": "ongoing situation description",
            "participants": ["entity names involved"],
            "conversation_time": "{session_time}",
            "relative_time": "relative time expression from text",
            "duration": "duration if mentioned"
        }}
    ],
    "contexts": [
        {{
            "content": "background/environmental information",
            "conversation_time": "{session_time}",
            "relative_time": "relative time expression from text",
            "impact": "impact on entities",
            "affected_entities": ["entity names affected"]
        }}
    ]
}}
"""

