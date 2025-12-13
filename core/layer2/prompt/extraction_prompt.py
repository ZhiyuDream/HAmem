"""
Layer2 Extraction Prompt

从fragment中提取时间线信息（事件、状态、上下文）
"""

from typing import List, Dict, Any


def build_layer2_extraction_prompt(fragment_text, session_time, layer1_entities, existing_layer2_nodes: List[Dict[str, Any]] = None):
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
    
    # 格式化已有Layer2节点信息
    existing_nodes_section = ""
    if existing_layer2_nodes:
        existing_nodes_list = []
        for node in existing_layer2_nodes:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            node_content = node.get('content', '')
            existing_nodes_list.append(f"- ID: {node_id}, Type: {node_type}, Content: {node_content}")
        existing_nodes_section = f"""
# Existing Layer2 Nodes (recalled from previous fragments):
{chr(10).join(existing_nodes_list)}

IMPORTANT: When extracting timeline information, you should:
1. If a new event/state/context is mentioned that should be linked to existing nodes, use "link_to_existing" to establish connections
2. Only use "create_new" for completely new timeline information
3. Events/States/Contexts do NOT support "update_existing" - always create new nodes
"""
    
    return f"""
Extract timeline information (events, states, contexts) from the conversation fragment.

Fragment: {fragment_text}
Session time: {session_time}

Available entities for reference:
{entities_str}
{existing_nodes_section}

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
            "relative_time": "relative time expression from text",
            "action": "create_new"
        }},
        {{
            "content": "another event description",
            "participants": ["entity names involved"],
            "action": "create_new",
            "link_to_existing": ["event_1", "event_2"]
        }}
    ],
    "states": [
        {{
            "content": "ongoing situation description",
            "participants": ["entity names involved"],
            "conversation_time": "{session_time}",
            "relative_time": "relative time expression from text",
            "duration": "duration if mentioned",
            "action": "create_new"
        }},
        {{
            "content": "another state description",
            "participants": ["entity names involved"],
            "action": "create_new",
            "link_to_existing": ["state_1"]
        }}
    ],
    "contexts": [
        {{
            "content": "background/environmental information",
            "conversation_time": "{session_time}",
            "relative_time": "relative time expression from text",
            "impact": "impact on entities",
            "affected_entities": ["entity names affected"],
            "action": "create_new"
        }},
        {{
            "content": "another context description",
            "action": "create_new",
            "link_to_existing": ["context_1"]
        }}
    ]
}}

# Action Types:
- "create_new": Create a new event/state/context node
- "link_to_existing": For new nodes that should be linked to existing nodes (specify existing node IDs in the list)
- Note: Events/States/Contexts do NOT support "update_existing" - always create new nodes
"""

