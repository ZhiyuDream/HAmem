"""
Answer Generator模块

基于召回的上下文和specialized modules生成答案
"""

import json
import os
from typing import List, Dict, Any
from core.infrastructure import LLMClient, parse_llm_json


class AnswerGenerator:
    """
    答案生成模块
    
    组合base_answer prompt和specialized modules生成最终答案
    """
    
    def __init__(self, llm_client: LLMClient, prompt_dir: str = None):
        """
        Args:
            llm_client: LLM客户端
            prompt_dir: prompt文件目录（默认为core/search/prompt）
        """
        self.llm_client = llm_client
        
        if prompt_dir is None:
            # 默认prompt目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompt_dir = os.path.join(current_dir, 'prompt')
        else:
            self.prompt_dir = prompt_dir
    
    def try_answer(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        尝试回答问题（在can_answer_early=true时调用）
        
        使用路由选择specialized modules，然后尝试生成答案
        
        Args:
            question: 用户问题
            context: 完整上下文（包含nodes, edges, fragments）
        
        Returns:
            {
                'answer': '答案内容',
                'reason': '推理过程'
            }
        """
        print(f"\n💬 尝试回答...")
        
        # 1. 路由问题，选择specialized modules
        from .router import QuestionRouter
        from core.infrastructure import LLMClient
        from config import Config
        
        
        
        # 创建临时router（如果没有的话）
        # 注意：实际使用时应该复用已有的router
        router = QuestionRouter(LLMClient(Config()))
        selected_modules = router.route(question)
        
        print(f"  ✅ 选择模块: {selected_modules}")
        
        # 2. 生成答案
        return self.generate(question, context, selected_modules)
    
    def generate(
        self,
        question: str,
        context: Dict[str, Any],
        selected_modules: List[str] = None
    ) -> Dict[str, str]:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 完整上下文（包含nodes, edges, fragments）
            selected_modules: 选择的specialized modules（可选，如果为None则不使用modules）
        
        Returns:
            {
                'answer': '答案内容',
                'reason': '推理过程'
            }
        """
        print(f"\n💬 生成答案...")
        
        # 如果没有指定modules，默认为空列表
        if selected_modules is None:
            selected_modules = []
        
        # 构建答案prompt
        prompt = self._build_answer_prompt(question, context, selected_modules)
        
        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider="deepseek")
        
        # 解析结果
        result = self._parse_answer_response(response)
        
        answer = result.get('answer', 'No answer generated')
        reason = result.get('reason', 'No reasoning provided')
        
        print(f"  ✅ 答案生成完成")
        print(f"  📝 答案长度: {len(answer)} 字符")
        
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
        构建答案生成prompt
        
        Args:
            question: 用户问题
            context: 上下文
            selected_modules: 选择的模块
        
        Returns:
            完整的prompt
        """
        # 加载base_answer prompt
        base_prompt = self._load_prompt('base_answer.txt')
        
        # 加载specialized modules
        specialized_prompts = []
        for module in selected_modules:
            module_prompt = self._load_prompt(f'modules/{module}.txt')
            if module_prompt:
                specialized_prompts.append(module_prompt)
        
        # 格式化上下文
        context_str = self._format_context(context)
        
        # 调试：显示传递给LLM的上下文
        print(f"  🔍 传递给LLM的上下文:")
        print(f"    {context_str[:500]}...")
        
        # 组合prompt
        specialized_section = "\n\n".join(specialized_prompts) if specialized_prompts else ""
        
        # 注意：prompt中可能使用 {query} 或 {question}
        full_prompt = base_prompt.format(
            query=question,  # 兼容 {query}
            question=question,  # 兼容 {question}
            context=context_str,
            specialized_modules=specialized_section if specialized_section else ""
        )
        
        return full_prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        格式化上下文为字符串
        
        Args:
            context: {
                'layer1': [nodes],
                'layer2': [nodes],
                'layer3': [nodes],
                'edges': [edges],
                'fragments': [fragments]
            }
        
        Returns:
            格式化后的上下文字符串
        """
        sections = []
        
        # Layer1: 实体和关系
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
        
        # Layer2: 事件/状态/上下文
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
                
                # 构建时间信息
                time_parts = []
                if conversation_time:
                    time_parts.append(f"time: {conversation_time}")
                if relative_time:
                    time_parts.append(f"relative: {relative_time}")
                
                time_str = f" ({', '.join(time_parts)})" if time_parts else ""
                
                # 构建参与者信息
                participants_str = f" [participants: {', '.join(participants)}]" if participants else ""
                
                # 构建位置信息
                location_str = f" [location: {location}]" if location and location != "not specified" else ""
                
                sections.append(f"- {node_type}{time_str}{participants_str}{location_str}: {content}\n")
        
        # Layer3: 模式/规则
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
            for edge in edges[:20]:  # 最多显示20条边
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
            for frag in fragments[:3]:  # 最多3个fragment
                frag_id = frag.get('id', 'unknown')
                content = frag.get('content', '')[:300]  # 限制长度
                sections.append(f"- {frag_id}: {content}...\n")
        
        return "".join(sections)
    
    def _load_prompt(self, filename: str) -> str:
        """
        加载prompt文件
        
        Args:
            filename: prompt文件名
        
        Returns:
            prompt内容，如果文件不存在返回''
        """
        filepath = os.path.join(self.prompt_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  Prompt文件不存在: {filepath}")
            return ""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️  读取prompt文件失败 {filepath}: {e}")
            return ""
    
    def _parse_answer_response(self, response: str) -> Dict[str, str]:
        """
        解析LLM的答案响应
        
        Args:
            response: LLM响应
        
        Returns:
            解析后的结果
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

