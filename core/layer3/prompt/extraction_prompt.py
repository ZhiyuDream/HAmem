"""
Layer3 Pattern Analysis Prompt

从聚类的事件中提取模式、偏好和行为规则
"""


def build_pattern_analysis_prompt(cluster_events, related_states, related_contexts):
    """
    构建模式分析prompt
    
    Args:
        cluster_events: 聚类的事件列表
        related_states: 相关的状态列表
        related_contexts: 相关的上下文列表
    
    Returns:
        str: 构建的prompt
    """
    # 格式化事件
    event_summaries = []
    for event in cluster_events:
        event_summaries.append(f"- {event.get('content', 'Unknown event')}")
    
    # 格式化状态
    state_summaries = []
    for state in related_states[:10]:  # 限制数量
        state_summaries.append(f"- {state.get('content', 'Unknown state')}")
    
    # 格式化上下文
    context_summaries = []
    for context in related_contexts[:10]:  # 限制数量
        context_summaries.append(f"- {context.get('content', 'Unknown context')}")
    
    events_text = "\n".join(event_summaries) if event_summaries else "No events"
    states_text = "\n".join(state_summaries) if state_summaries else "No states"
    contexts_text = "\n".join(context_summaries) if context_summaries else "No contexts"
    
    return f"""
Analyze the following clustered events, states, and contexts to identify high-level patterns.

Clustered Events (related events):
{events_text}

Related States:
{states_text}

Related Contexts:
{contexts_text}

Please analyze and extract:
1. Event cluster summary
2. Patterns (recurring behaviors)
3. Preferences (likes/dislikes)
4. Behavior rules (decision-making patterns)

Output JSON format:
{{
    "event_cluster": {{
        "description": "BRIEF summary of the larger activity (1-2 sentences max)",
        "cluster_type": "category like shopping/social/learning/work",
        "participants": ["main participants"],
        "time_span": "overall time period if identifiable",
        "significance": "high/medium/low"
    }},
    "patterns": [
        {{
            "person": "person name",
            "pattern_type": "brief pattern category",
            "description": "Brief description of the pattern"
        }}
    ],
    "preferences": [
        {{
            "person": "person name",
            "category": "preference category",
            "description": "Brief description of the preference"
        }}
    ],
    "behavior_rules": [
        {{
            "person": "person name",
            "rule_type": "rule category",
            "description": "Brief description of the behavior rule"
        }}
    ]
}}

RULES:
- Keep cluster description BRIEF (under 2 sentences)
- Separate patterns/preferences/rules by person
- Focus on significant, recurring patterns
- Use specific person names from participants
- event_cluster is REQUIRED, others are optional (can be empty arrays)

EXAMPLES:
Good cluster description: "User explored Ethiopian cuisine options and dining customs"
Bad cluster description: "User asked about Ethiopian food, then asked about restaurants, then asked about how to eat injera..."

Good pattern: {{"person": "user", "pattern_type": "food_exploration", "description": "Actively researches cuisines before trying them"}}
Good preference: {{"person": "user", "category": "dietary", "description": "Prefers fruity desserts without nuts"}}
Good rule: {{"person": "user", "rule_type": "planning", "description": "Considers guest preferences when hosting events"}}
"""

