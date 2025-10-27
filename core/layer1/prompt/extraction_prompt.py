"""
Layer1实体和关系提取Prompt
"""

def build_layer1_extraction_prompt(fragment_text: str) -> str:
    """
    构建Layer1提取prompt（优化版v2）
    
    Args:
        fragment_text: Fragment的对话内容
    
    Returns:
        提取prompt
    """
    
    prompt = f"""
You are an expert in information extraction. Extract entities and relationships from the conversation fragment below.

# Input Fragment:
{fragment_text}

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
      "name": "Entity Name", 
      "content": "Static description of the entity"
    }}
  ],
  "relationships": [
    {{
      "source": "Source Entity Name", 
      "target": "Target Entity Name", 
      "content": "Description of the relationship"
    }}
  ]
}}
```

# Important Rules:
1. Entity names in relationships MUST exactly match entity names in the entities list
2. Focus on extracting factual, stable information
3. Avoid extracting temporary states or emotions as entity descriptions
4. Be precise and concise in descriptions
5. Extract all meaningful entities and relationships, but avoid over-segmentation

Extract the information now:
"""
    return prompt

