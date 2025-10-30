"""
QA System main class

Coordinates recall, expansion, routing, and answer generation
"""

import os
from typing import Dict, Any, List
from core.infrastructure import LLMClient, UnifiedCache, parse_llm_json
from .recall import SearchRecall
from .expansion import GraphExpansion
from .router import QuestionRouter
from .answer import AnswerGenerator


class QASystem:
    """
    End-to-end QA pipeline:
    1. Initial recall (FAISS)
    2. Intelligent expansion (LLM decision + graph traversal)
    3. Question routing (LLM)
    4. Answer generation (LLM)
    """
    
    def __init__(
        self,
        cache: UnifiedCache,
        storage,
        llm_client: LLMClient,
        namespace: str,
        max_hops: int = 2
    ):
        """
        Args:
            cache: UnifiedCache (embeddings + FAISS indexes)
            storage: Storage instance
            llm_client: LLM client
            namespace: Namespace
            max_hops: Max expansion hops
        """
        self.cache = cache
        self.storage = storage
        self.llm_client = llm_client
        self.namespace = namespace
        self.max_hops = max_hops
        
        # Initialize submodules
        self.recall = SearchRecall(cache, storage)
        self.expansion = GraphExpansion(storage, cache)
        self.router = QuestionRouter(llm_client)
        self.answer_generator = AnswerGenerator(llm_client)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question via full pipeline.
        Flow: Recall → Decide → Try answer → (Return or Expand) → Loop
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        stats = {
            'llm_calls': 0,
            'recalled_nodes': 0,
            'expanded_nodes': 0,
            'hops': 0
        }
        
        # Stage 1: Initial recall
        print(f"\n📍 Stage 1: Initial recall")
        recalled = self.recall.multi_layer_recall(question)
        
        current_nodes = recalled['layer1'] + recalled['layer2'] + recalled['layer3']
        stats['recalled_nodes'] = len(current_nodes)
        
        # Show recalled node IDs
        print(f"\n📋 Recalled node IDs:")
        print(f"  Layer1: {[n['id'] for n in recalled['layer1'][:10]]}" + 
              (f" ... (+{len(recalled['layer1'])-10})" if len(recalled['layer1']) > 10 else ""))
        print(f"  Layer2: {[n['id'] for n in recalled['layer2'][:10]]}" + 
              (f" ... (+{len(recalled['layer2'])-10})" if len(recalled['layer2']) > 10 else ""))
        print(f"  Layer3: {[n['id'] for n in recalled['layer3'][:10]]}" + 
              (f" ... (+{len(recalled['layer3'])-10})" if len(recalled['layer3']) > 10 else ""))
        
        all_edges = []
        all_fragments = []
        
        # Stage 2: Multi-hop expansion loop
        print(f"\n📍 Stage 2: Decision & Expansion")
        
        # 存储选定的模块（在第一次决策时确定）
        selected_modules = None
        
        for hop in range(self.max_hops):
            print(f"\n🔄 Hop {hop + 1}/{self.max_hops}")
            
            # 2.1: Decision (sufficiency, need to expand)
            candidates, connecting_edges = self.expansion.find_candidates(
                current_node_ids=[n['id'] for n in current_nodes],
                max_candidates=50
            )
            
            decision = self._make_expansion_decision(
                question,
                current_nodes,
                candidates,
                connecting_edges
            )
            stats['llm_calls'] += 1
            
            # Select modules on first decision
            if selected_modules is None:
                selected_modules = [decision.get('selected_module', 'detail_extraction')]
                print(f"  🎯 选择模块: {selected_modules}")
            
            # Use all recalled nodes without filtering
            can_answer_early = decision.get('can_answer_early', False)
            information_sufficiency = decision.get('information_sufficiency', 'insufficient')
            
            print(f"  📊 Information sufficiency: {information_sufficiency}")
            print(f"  📊 Nodes used: {len(current_nodes)} (no filtering)")
            print(f"  📊 can_answer_early: {can_answer_early}")
            
            # 2.2: If sufficient, try to answer
            if can_answer_early:
                print(f"  🚀 Attempting to generate answer...")
                
                # Get fragments
                all_fragments = self.recall.get_fragments_by_nodes(
                    [n['id'] for n in current_nodes]
                )
                
                # Build context
                context = {
                    'layer1': [n for n in current_nodes if n.get('layer') == 1],
                    'layer2': [n for n in current_nodes if n.get('layer') == 2],
                    'layer3': [n for n in current_nodes if n.get('layer') == 3],
                    'edges': all_edges + connecting_edges,
                    'fragments': all_fragments
                }
                
                # Try to answer (specialized modules)
                answer_result = self.answer_generator.try_answer(question, context)
                stats['llm_calls'] += 1
                
                # Return LLM answer directly
                print(f"  ✅ Answer generated successfully!")
                
                stats['hops'] = hop + 1
                
                print(f"\n{'='*60}")
                print(f"✅ QA completed (early return at hop {hop + 1})")
                print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
                      f"{stats['recalled_nodes']} recalled, "
                      f"{stats['expanded_nodes']} expanded, "
                      f"{stats['hops']} hops")
                print(f"{'='*60}")
                
                return {
                    'question': question,
                    'answer': answer_result['answer'],
                    'reason': answer_result['reason'],
                    'prompt_used': answer_result.get('prompt_used', ''),
                    'stats': stats
                }
            
            # 2.3: Insufficient → expand
            should_expand = decision.get('should_expand', False)
            nodes_to_expand = decision.get('nodes_to_expand', [])
            
            if not should_expand or not nodes_to_expand:
                print(f"  ⚠️  No expansion needed, stop loop")
                break
            
            if not candidates:
                print(f"  ⚠️  No candidates to expand, stop")
                break
            
            # Execute expansion
            print(f"  🔗 Expanding {len(nodes_to_expand)} nodes...")
            expanded_nodes, expanded_edges = self.expansion.expand_nodes(
                nodes_to_expand,
                max_neighbors=20
            )
            
            if not expanded_nodes:
                print(f"  ⚠️  Expansion yielded no new nodes, stop")
                break
            
            # Update node set
            old_node_ids = {n['id'] for n in current_nodes}
            new_nodes = [n for n in expanded_nodes if n['id'] not in old_node_ids]
            
            current_nodes.extend(new_nodes)
            all_edges.extend(expanded_edges)
            stats['expanded_nodes'] += len(new_nodes)
            stats['hops'] = hop + 1
            
            print(f"  ✅ Expanded {len(new_nodes)} new nodes")
            if new_nodes:
                new_node_ids = [n['id'] for n in new_nodes[:10]]
                print(f"     Expanded node IDs: {new_node_ids}" + 
                      (f" ... (+{len(new_nodes)-10})" if len(new_nodes) > 10 else ""))
        
        # Stage 3: Final answer generation
        print(f"\n📍 Stage 3: Final answer generation")
        
        all_fragments = self.recall.get_fragments_by_nodes(
            [n['id'] for n in current_nodes]
        )
        
        context = {
            'layer1': [n for n in current_nodes if n.get('layer') == 1],
            'layer2': [n for n in current_nodes if n.get('layer') == 2],
            'layer3': [n for n in current_nodes if n.get('layer') == 3],
            'edges': all_edges,
            'fragments': all_fragments
        }
        
        # Final generation (specialized modules)
        print(f"\n💬 Generating answer...")
        
        # 使用之前选定的模块（如果没有则使用默认模块）
        if selected_modules is None:
            selected_modules = ['detail_extraction']
            print(f"  🎯 Using default modules: {selected_modules}")
        else:
            print(f"  🎯 Using selected modules: {selected_modules}")
        
        answer_result = self.answer_generator.generate(question, context, selected_modules)
        stats['llm_calls'] += 1
        
        print(f"\n{'='*60}")
        print(f"✅ QA completed (final generation)")
        print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
              f"{stats['recalled_nodes']} recalled, "
              f"{stats['expanded_nodes']} expanded, "
              f"{stats['hops']} hops")
        print(f"{'='*60}")
        
        return {
            'question': question,
            'answer': answer_result['answer'],
            'reason': answer_result['reason'],
            'prompt_used': answer_result.get('prompt_used', ''),
            'stats': stats
        }
    
    def _make_expansion_decision(
        self,
        question: str,
        current_nodes: List[Dict],
        candidates: List[Dict],
        edges: List[Dict]
    ) -> Dict[str, Any]:
        """
        LLM decision: node filtering, whether to expand, and targets
        """
        print(f"  🤖 调用LLM进行扩展决策...")
        
        # 构建决策prompt
        prompt = self._build_decision_prompt(
            question,
            current_nodes,
            candidates,
            edges
        )
        
        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider="deepseek")
        
        # 解析决策
        decision = self._parse_decision_response(response, current_nodes, candidates)
        
        print(f"    选择模块: {decision.get('selected_module', 'unknown')}")
        print(f"    扩展: {decision['should_expand']}")
        print(f"    提前回答: {decision['can_answer_early']}")
        
        return decision
    
    def _build_decision_prompt(
        self,
        question: str,
        current_nodes: List[Dict],
        candidates: List[Dict],
        edges: List[Dict]
    ) -> str:
        """
        Build decision prompt (using base_decision.txt)
        """
        # 加载base_decision prompt
        prompt_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'prompt',
            'base_decision.txt'
        )
        
        if not os.path.exists(prompt_file):
            print(f"  ⚠️  Decision prompt not found, using simplified version")
            return self._build_simple_decision_prompt(question, current_nodes, candidates)
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # 格式化recalled context
        recalled_context = self._format_nodes_for_prompt(current_nodes, max_nodes=30)
        
        # 格式化candidate context
        candidate_context = self._format_nodes_for_prompt(candidates, max_nodes=30)
        
        # Fragments信息（暂时留空，可以后续扩展）
        fragment_info = ""
        
        return template.format(
            query=question,
            recalled_context=recalled_context,
            candidate_context=candidate_context,
            fragment_info=fragment_info
        )
    
    def _format_nodes_for_prompt(self, nodes: List[Dict], max_nodes: int = 30) -> str:
        """Format nodes as prompt string"""
        if not nodes:
            return "  (no nodes)"
        
        result = ""
        for node in nodes[:max_nodes]:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            content = node.get('content', '')[:150]
            result += f"  - {node_id} ({node_type}): {content}...\n"
        
        if len(nodes) > max_nodes:
            result += f"  ... ({len(nodes) - max_nodes} more nodes not shown)\n"
        
        return result
    
    def _build_simple_decision_prompt(
        self,
        question: str,
        current_nodes: List[Dict],
        candidates: List[Dict]
    ) -> str:
        """Simplified decision prompt (fallback)"""
        current_str = self._format_nodes_for_prompt(current_nodes, 20)
        candidates_str = self._format_nodes_for_prompt(candidates, 30)
        
        return f"""
Analyze question and decide on information retrieval strategy.

QUESTION: {question}

RECALLED NODES:
{current_str}

CANDIDATES:
{candidates_str}

TASKS:
1. Classify question type
2. Identify relevant nodes (node filtering)
3. Assess if information is sufficient
4. Decide if expansion is needed
5. Decide if can answer now

JSON Response:
{{
    "question_type": "...",
    "retained_node_ids": ["node_id1", ...],
    "information_sufficiency": "sufficient|partial|insufficient", 
    "should_expand": true/false,
    "nodes_to_expand": ["node_id1", ...],
    "can_answer_early": true/false,
    "reasoning": "..."
}}
"""
    
    def _parse_decision_response(
        self,
        response: str,
        current_nodes: List[Dict],
        candidates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Parse LLM decision JSON
        """
        default_result = {
            'selected_module': 'detail_extraction',
            'information_sufficiency': 'insufficient',
            'should_expand': False,
            'nodes_to_expand': [],
            'can_answer_early': False,
            'reasoning': 'Default: insufficient information'
        }
        
        result = parse_llm_json(
            response,
            expected_keys=['selected_module', 'information_sufficiency', 'should_expand', 'nodes_to_expand', 'can_answer_early'],
            default=default_result
        )
        
        if result is None:
            return default_result
        
        # 验证扩展节点IDs
        current_ids = {n['id'] for n in current_nodes}
        candidate_ids = {n['id'] for n in candidates}
        
        nodes_to_expand = result.get('nodes_to_expand', [])
        if not isinstance(nodes_to_expand, list):
            nodes_to_expand = []
        result['nodes_to_expand'] = [nid for nid in nodes_to_expand if nid in current_ids]
        
        return result

