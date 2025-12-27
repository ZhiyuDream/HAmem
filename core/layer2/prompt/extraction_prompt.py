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
"""
    
    return f"""Extract timeline information (events, states, contexts) from the conversation fragment.

IMPORTANT COMPLETENESS REQUIREMENT:
You must extract ALL timeline-relevant information explicitly mentioned in the fragment.
Do NOT omit events simply because they seem minor, casual, emotional, or low-importance.
If an explicit action occurred, it MUST be extracted as an EVENT.

Fragment: {fragment_text}
Session time: {session_time}

Available entities for reference:
{entities_str}
{existing_nodes_section}
IMPORTANT extraction rules:
1. If a new event/state/context should be related to existing nodes, use "link_to_existing".
2. Only use "create_new" for new timeline information.
3. Events/States/Contexts do NOT support "update_existing" — always create new nodes.
4. Do NOT summarize multiple distinct actions into one event if they happened separately.
5. Casual or one-time actions (e.g., purchases, gifts, remarks about buying something)
   are valid EVENTS and must be extracted.

Classification guidelines:

EVENT:
- Any explicit action, occurrence, or completed activity mentioned in the text.
- Includes but is not limited to:
  • purchases or gifts (e.g., "bought a figurine")
  • celebrations, interviews, meetings
  • plans that were carried out
  • one-time or casual actions with a time reference
- If the question "Did something happen?" can be answered with yes → EVENT.

STATE:
- Ongoing emotions, motivations, beliefs, or conditions.
- Typically persistent rather than momentary.
- If the question "Is this an ongoing condition?" is yes → STATE.

CONTEXT:
- Background, framing, or situational information that explains
  why events or states matter.
- Often abstract or thematic rather than action-based.

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

FINAL CHECK BEFORE OUTPUT:
- Re-scan the fragment and confirm that every explicit action
  (including purchases or casual activities) has been extracted as an EVENT.
"""

