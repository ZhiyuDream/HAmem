"""
Layer1 Conflict Resolution Prompt
"""

from typing import Dict, List, Any


def build_conflict_resolution_prompt(
    fragment_text: str,
    new_entities: List[Dict[str, str]],
    new_relationships: List[Dict[str, str]],
    entity_candidates: Dict[str, List[Dict[str, Any]]],
    relation_candidates: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Build batch conflict resolution prompt
    
    Args:
        fragment_text: Original fragment text
        new_entities: List of newly extracted entities
        new_relationships: List of newly extracted relationships
        entity_candidates: Entity candidate set {entity_name: [candidates]}
        relation_candidates: Relationship candidate set {relation_key: [candidates]}
    
    Returns:
        Conflict resolution prompt string
    """
    
    # Build entity candidate information
    entity_candidates_str = ""
    for entity_name, candidates in entity_candidates.items():
        if candidates:
            entity_candidates_str += f"\n### Historical Candidates for '{entity_name}':\n"
            for i, candidate in enumerate(candidates[:5], 1):
                node = candidate.get('node', candidate)
                entity_candidates_str += f"  {i}. ID: {node.get('id')}, Name: {node.get('name')}, Description: {node.get('content', node.get('description', ''))}\n"
                entity_candidates_str += f"     Similarity: {candidate.get('similarity', 'N/A'):.3f}\n"
        else:
            entity_candidates_str += f"\n### Historical Candidates for '{entity_name}': None\n"
    
    # Build relationship candidate information
    relation_candidates_str = ""
    for relation_key, candidates in relation_candidates.items():
        if candidates and candidates != "new_relation":
            relation_candidates_str += f"\n### Historical Candidates for '{relation_key}':\n"
            for i, candidate in enumerate(candidates[:3], 1):
                edge = candidate.get('edge', candidate)
                relation_candidates_str += f"  {i}. ID: {edge.get('id')}, Source: {edge.get('source')}, Target: {edge.get('target')}\n"
                relation_candidates_str += f"     Description: {edge.get('content', edge.get('description', ''))}\n"
                relation_candidates_str += f"     Similarity: {candidate.get('similarity', 'N/A'):.3f}\n"
        else:
            relation_candidates_str += f"\n### Historical Candidates for '{relation_key}': None\n"
    
    # Build new entity list
    new_entities_str = ""
    for i, entity in enumerate(new_entities, 1):
        name = entity.get('name', 'Unknown')
        content = entity.get('content', 'No content')
        new_entities_str += f"  {i}. Name: {name}, Content: {content}\n"
    
    # Build new relationship list
    new_relationships_str = ""
    for i, rel in enumerate(new_relationships, 1):
        source = rel.get('source', 'Unknown')
        target = rel.get('target', 'Unknown')
        content = rel.get('content', 'No content')
        new_relationships_str += f"  {i}. {source} -> {target}: {content}\n"
    
    prompt = f"""
You are an expert in knowledge graph construction. Your task is to resolve conflicts between newly extracted information and existing knowledge.

# Original Conversation Fragment:
```
{fragment_text}
```

# Newly Extracted Information:

## New Entities:
{new_entities_str}

## New Relationships:
{new_relationships_str}

# Historical Candidates:

## Entity Candidates:
{entity_candidates_str}

## Relationship Candidates:
{relation_candidates_str}

# Your Task:

For each NEW ENTITY, decide:
1. **create_new**: Create a new entity (when no similar entity exists, or historical entities refer to different things)
2. **update_existing**: Update an existing entity (when a historical entity refers to the same thing - merge/update the content even if they're similar)

For each NEW RELATIONSHIP, decide:
1. **create_new**: Create a new relationship
2. **update_existing**: Update an existing relationship (when it refers to the same entity pair - merge/update the content)

# Decision Criteria:

## For Entities:
- **Same person/thing?** Check if the entity refers to the same real-world object
  - If YES → use **update_existing** (merge the old and new content, keeping all useful information)
  - If NO → use **create_new**
- **Name match?** Exact name match is a strong signal but not absolute
- **Context match?** Consider the conversation context

## For Relationships:
- **Same entity pair?** Check if source and target refer to the same entities
- **Same relationship type?** "is friends with" vs "works with" are different
  - If same pair AND same type → use **update_existing** (merge old and new content)
  - If different → use **create_new**

# Output Format:

Return a JSON object with the following structure:

```json
{{
  "entity_decisions": [
    {{
      "new_entity_name": "Entity Name",
      "action": "create_new" | "update_existing",
      "target_entity_id": "entity_id" (only if action is update_existing),
      "updated_content": "merged/updated content" (only if action is update_existing),
      "reason": "brief explanation"
    }}
  ],
  "relation_decisions": [
    {{
      "new_relation": "source_target",
      "action": "create_new" | "update_existing",
      "target_relation_id": "edge_id" (only if action is update_existing),
      "updated_content": "merged/updated content" (only if action is update_existing),
      "reason": "brief explanation"
    }}
  ]
}}
```

# Important Notes:
1. For update_existing: ALWAYS merge old and new content to create a comprehensive description
2. Prefer create_new when uncertain about whether entities refer to the same thing
3. Consider the conversation context when making decisions
4. Provide clear, concise reasons for each decision

Now analyze the information and make your decisions:
"""
    
    return prompt

