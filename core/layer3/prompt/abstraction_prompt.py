"""
递归高层抽象 Prompt

把同一 cluster 的 Layer N 节点摘要概括为 Layer N+1 的高层抽象（N>=3）
"""


def build_abstraction_prompt(source_layer: int, target_layer: int, nodes) -> str:
    """
    构建高层抽象prompt

    Args:
        source_layer: 来源层（N）
        target_layer: 目标层（N+1）
        nodes: 待抽象的 Layer N 节点列表

    Returns:
        str: 构建的prompt
    """
    node_summaries = []
    for node in nodes[:30]:  # 限制数量
        node_summaries.append(f"- {node.get('content', 'Unknown')}")

    nodes_text = "\n".join(node_summaries) if node_summaries else "No nodes"

    return f"""
The following are memory items from Layer {source_layer} of a hierarchical memory system.
They have been found to form a coherent group (cluster) by semantic similarity.

Layer {source_layer} items:
{nodes_text}

Please synthesize a higher-level abstraction at Layer {target_layer} that captures the
common theme of this group. Layer {target_layer} is more abstract than Layer {source_layer}:
it should read like a summary of summaries, not a list of individual details.

Output JSON format:
{{
    "abstraction": {{
        "summary": "1-3 sentence high-level summary of the common theme",
        "themes": ["key theme 1", "key theme 2"],
        "significance": "high/medium/low"
    }}
}}

RULES:
- summary must be abstract: describe the overarching theme, NOT individual items
- themes: 1-5 short keyword-style themes
- significance reflects how central this theme is to the person's life/story
- Output the JSON object only, no extra text

EXAMPLES:
Good summary: "User's ongoing career transition from finance to data science, driven by
a desire for more meaningful work"
Bad summary: "User asked about Python courses, then updated resume, then interviewed
at a data company..."
"""
