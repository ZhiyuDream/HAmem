"""
Answer Generator module

Generate answers based on recalled context and specialized modules
"""

import json
import os
from typing import List, Dict, Any
from core.infrastructure import LLMClient, parse_llm_json


class AnswerGenerator:
    """
    Combine base_answer prompt with specialized modules to generate the final answer
    """
    
    def __init__(self, llm_client: LLMClient, prompt_dir: str = None, default_provider: str = "deepseek"):
        """
        Args:
            llm_client: LLM client
            prompt_dir: prompt directory (defaults to core/search/prompt)
            default_provider: 默认LLM提供商 ("openai" 或 "deepseek")
        """
        self.llm_client = llm_client
        self.default_provider = default_provider
        
        if prompt_dir is None:
            # Default prompt directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompt_dir = os.path.join(current_dir, 'prompt')
        else:
            self.prompt_dir = prompt_dir
    
    def try_answer(
        self,
        question: str,
        context: Dict[str, Any],
        selected_modules: List[str] = None
    ) -> Dict[str, str]:
        """
        Try to answer (used when can_answer_early=true)
        
        Args:
            question: Question text
            context: Context dictionary
            selected_modules: Pre-selected specialized modules (from expansion decision)
                            If None, uses default module
        """
        print(f"\n💬 Trying to answer...")
        
        # 使用传入的模块（来自扩展决策），不再进行路由
        if selected_modules is None:
            selected_modules = ['detail_extraction']
            print(f"  🎯 Using default modules: {selected_modules}")
        else:
            print(f"  🎯 Using selected modules: {selected_modules}")
        
        # 直接生成答案
        return self.generate(question, context, selected_modules)
    
    def generate(
        self,
        question: str,
        context: Dict[str, Any],
        selected_modules: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate an answer
        """
        print(f"\n💬 Generating answer...")
        
        # 如果没有指定modules，默认为空列表
        if selected_modules is None:
            selected_modules = []
        
        # Build answer prompt
        prompt = self._build_answer_prompt(question, context, selected_modules)
        
        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider=self.default_provider)
        
        # Parse result
        result = self._parse_answer_response(response)
        
        answer = result.get('answer', 'No answer generated')
        reason = result.get('reason', 'No reasoning provided')
        
        print(f"  ✅ Answer generation completed")
        print(f"  📝 Answer length: {len(answer)} chars")
        
        return {
            'answer': answer,
            'reason': reason,
            'prompt_used': prompt  # 返回完整prompt供调试
        }
    
    def _build_answer_prompt(
        self,
        question: str,
        context: Dict[str, Any],
        selected_modules: List[str]
    ) -> str:
        """
        Build the answer-generation prompt
        """
        # Load base_answer prompt
        base_prompt = self._load_prompt('base_answer.txt')
        
        # Load specialized modules
        specialized_prompts = []
        for module in selected_modules:
            module_prompt = self._load_prompt(f'modules/{module}.txt')
            if module_prompt:
                specialized_prompts.append(module_prompt)
        
        # Format context
        context_str = self._format_context(context)
        
        # Debug: show context chunk passed to LLM
        print(f"  🔍 Context passed to LLM:")
        print(f"    {context_str[:500]}...")
        
        # Compose prompt
        specialized_section = "\n\n".join(specialized_prompts) if specialized_prompts else ""
        
        # Note: prompt may use {query} or {question}
        full_prompt = base_prompt.format(
            query=question,  # 兼容 {query}
            question=question,  # 兼容 {question}
            context=context_str,
            specialized_modules=specialized_section if specialized_section else ""
        )
        
        return full_prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context to string
        """
        sections = []
        
        # Layer1: Entities & Relationships
        layer1_nodes = context.get('layer1', [])
        if layer1_nodes:
            sections.append("### Entities and Relationships\n")
            for node in layer1_nodes:
                node_type = node.get('type', 'unknown')
                node_id = node.get('id', 'unknown')
                content = node.get('content', '')
                name = node.get('name', '')
                
                if name:
                    sections.append(f"- {node_id} ({node_type}): {name} - {content}\n")
                else:
                    sections.append(f"- {node_id} ({node_type}): {content}\n")
        
        # Layer2: Events/States/Context
        layer2_nodes = context.get('layer2', [])
        if layer2_nodes:
            sections.append("\n### Timeline Information\n")
            for node in layer2_nodes:
                node_type = node.get('type', 'unknown').upper()
                content = node.get('content', '')
                conversation_time = node.get('conversation_time', '')
                relative_time = node.get('relative_time', '')
                participants = node.get('participants', [])
                location = node.get('location', '')
                
                # Build time info
                time_parts = []
                if conversation_time:
                    time_parts.append(f"time: {conversation_time}")
                if relative_time:
                    time_parts.append(f"relative: {relative_time}")
                
                time_str = f" ({', '.join(time_parts)})" if time_parts else ""
                
                # Participants
                participants_str = f" [participants: {', '.join(participants)}]" if participants else ""
                
                # Location
                location_str = f" [location: {location}]" if location and location != "not specified" else ""
                
                sections.append(f"- {node_type}{time_str}{participants_str}{location_str}: {content}\n")
        
        # Layer3: Patterns/Rules
        layer3_nodes = context.get('layer3', [])
        if layer3_nodes:
            sections.append("\n### Patterns and Rules\n")
            for node in layer3_nodes:
                node_type = node.get('type', 'unknown').upper()
                content = node.get('content', '')
                sections.append(f"- {node_type}: {content}\n")
        
        # Edges
        edges = context.get('edges', [])
        if edges:
            sections.append("\n### Relationships\n")
            for edge in edges[:20]:  # show up to 20 edges
                source = edge.get('source', 'unknown')
                target = edge.get('target', 'unknown')
                content = edge.get('content', '')
                edge_type = edge.get('type', 'relationship')
                
                if content:
                    sections.append(f"- {source} --[{edge_type}]--> {target}: {content}\n")
                else:
                    sections.append(f"- {source} --[{edge_type}]--> {target}\n")
        
        # Fragments
        fragments = context.get('fragments', [])
        if fragments:
            sections.append("\n### Original Conversations\n")
            for frag in fragments[:3]:  # up to 3 fragments
                frag_id = frag.get('id', 'unknown')
                content = frag.get('content', '')[:300]  # 限制长度
                sections.append(f"- {frag_id}: {content}...\n")
        
        return "".join(sections)
    
    def _load_prompt(self, filename: str) -> str:
        """
        Load prompt file; return '' if missing
        """
        filepath = os.path.join(self.prompt_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  Prompt file not found: {filepath}")
            return ""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️  Failed to read prompt file {filepath}: {e}")
            return ""
    
    def _parse_answer_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM answer response
        """
        default_result = {
            'answer': 'Unable to generate answer',
            'reason': 'Failed to parse LLM response'
        }
        
        result = parse_llm_json(
            response,
            expected_keys=['answer', 'reason'],
            default=default_result
        )
        
        if result is None:
            return default_result
        
        return result

