"""
Layer1 Entity and Relationship Extraction Prompt
"""

from typing import List, Dict, Any

def build_layer1_extraction_prompt(fragment_text: str, existing_entities: List[Dict[str, Any]] = None) -> str:
    """
    Build Layer1 extraction prompt (supports existing entity recall and linking)
    
    Args:
        fragment_text: Conversation content of the fragment
        existing_entities: List of recalled existing entities (for linking and supplementing)
    
    Returns:
        Extraction prompt string
    """
    # Format existing entity information
    existing_entities_section = ""
    if existing_entities:
        existing_entities_list = []
        for entity in existing_entities:
            entity_id = entity.get('id', 'unknown')
            entity_name = entity.get('name', '')
            entity_content = entity.get('content', '')
            existing_entities_list.append(f"- ID: {entity_id}, Name: {entity_name}, Content: {entity_content}")
        existing_entities_section = f"""
# Existing Entities (recalled from previous fragments):
{chr(10).join(existing_entities_list)}

IMPORTANT: When extracting entities, you should:
1. If a new entity is mentioned that matches an existing entity (same person/organization/place), use "update_existing" action to supplement the existing entity's information
2. If a new entity is mentioned that should be linked to existing entities, use "link_to_existing" to establish relationships
3. Only use "create_new" for completely new entities that don't match any existing ones
    """
    
    prompt = f"""
You are an expert in information extraction. Extract entities and relationships from the conversation fragment below.

# Input Fragment:
{fragment_text}
{existing_entities_section}

# Extraction Requirements:

## Entities:
- Extract people, organizations, places, and concepts that have PERSONALIZED significance
- For each entity provide:
  - name: The exact name/term used
  - description: HIGH-LEVEL static characteristics ONLY (max 50 words)

### CRITICAL: Entity Extraction Rules:
- ✅ ALWAYS EXTRACT: People, organizations, specific places, proper nouns
- ✅ EXTRACT IF PERSONALIZED: Concepts/activities with expressed attitudes, preferences, or special significance
- ❌ SKIP GENERIC CONCEPTS: Common activities, objects, or concepts without personal context
- ❌ FORBIDDEN WORDS in descriptions: "recently", "currently", "is doing", "is planning", "just", "now", "today", "this week"

### DESCRIPTION LENGTH & CONTENT RULES:
- 📏 MAX 50 WORDS: Keep descriptions concise and high-level
- 🎯 ENTITY ESSENCE ONLY: Who they ARE, not what they DO
- ❌ NO ACTIVITY LISTS: Don't list hobbies, activities, or specific actions
- ❌ NO EVENT/STATE INFO: Activities belong in events, not entity descriptions
- ❌ NO TEMPORARY SITUATIONS: Current states, ongoing processes, recent actions
- ✅ CORE IDENTITY: Personality, values, role, background, fundamental traits

### CRITICAL: Focus on BEING vs DOING:
- ✅ BEING: "supportive person", "creative individual", "family-oriented"
- ❌ DOING: "enjoys running", "plays violin", "supports Caroline's adoption"
- ✅ TRAITS: "values self-care", "emphasizes family"
- ❌ ACTIVITIES: "swimming with kids", "painting for relaxation"

### Examples:
- ✅ "Caroline" (person) → Extract
- ✅ "Central Park" (specific place) → Extract  
- ✅ "swimming" mentioned as "Caroline hates swimming" → Extract with personal context
- ❌ "swimming" mentioned as "going swimming" → Skip (use events instead)
- ❌ "coffee" mentioned casually → Skip (generic concept)

## Relationships:
- Extract STABLE, PERSISTENT relationships between entities
- For each relationship provide:
  - source: Source entity name (exactly as extracted above)
  - target: Target entity name (exactly as extracted above)  
  - description: Natural language description of the STABLE relationship

### CRITICAL: Relationship Rules:
- ✅ STABLE RELATIONSHIPS: friendships, family ties, work relationships, affiliations, preferences, skills
- ✅ EXAMPLES: "X is friends with Y", "X works at Y", "X loves/hates Y", "X is skilled in Y"
- ❌ TEMPORARY ACTIVITIES: "X is going to Y", "X is about to do Y", "X is currently doing Y"
- ❌ ONE-TIME EVENTS: Activities that happen at specific times belong in events, NOT relationships

### Relationship vs Event Guide:
- Relationship: "Caroline loves art" (stable preference)
- Event: "Caroline is painting" (temporary activity)
- Relationship: "Melanie works at the clinic" (stable employment)  
- Event: "Melanie is going to work" (temporary action)

# Output Format:
Return a JSON object with the following structure:

```json
{{
  "entities": [
    {{
      "name": "New Entity Name",
      "content": "Static description of the entity",
      "action": "create_new"
    }},
    {{
      "name": "Existing Entity Name",
      "content": "Updated description with new information from this fragment",
      "action": "update_existing",
      "existing_entity_id": "entity_1"
    }},
    {{
      "name": "Another New Entity",
      "content": "Description",
      "action": "create_new",
      "link_to_existing": [
        {{
          "existing_entity_id": "entity_2",
          "relation_type": "works_at",
          "relation_content": "works at the company"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "source": "Entity A",
      "target": "Entity B",
      "content": "Relationship description",
      "action": "create_new"
    }},
    {{
      "source": "Entity A",
      "target": "Entity B",
      "content": "Updated relationship description",
      "action": "update_existing",
      "existing_relation_id": "edge_1"
    }}
  ]
}}
```

# Action Types:
- "create_new": Create a new entity/relationship
- "update_existing": Update an existing entity/relationship with new information from this fragment (use existing_entity_id or existing_relation_id)
- "link_to_existing": For new entities that should be linked to existing entities (specify relation_type and relation_content)

# Important Rules:
1. Entity names in relationships MUST exactly match entity names in the entities list
2. Focus on extracting factual, stable information
3. Avoid extracting temporary states or emotions as entity descriptions
4. Be precise and concise in descriptions
5. Extract all meaningful entities and relationships, but avoid over-segmentation
6. When an entity matches an existing one, use "update_existing" to supplement information
7. When a new entity should be linked to existing entities, use "link_to_existing" with proper relation_type
8. Only create fragment-to-entity connections when there is a meaningful relationship (LLM should judge)

Extract the information now:
"""
    return prompt

