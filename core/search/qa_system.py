"""
QA System主类

协调召回、扩展、路由和答案生成的完整流程
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
    问答系统主类
    
    完整的QA流程：
    1. 初始召回（FAISS）
    2. 智能扩展（LLM决策 + 图遍历）
    3. 问题路由（LLM）
    4. 答案生成（LLM）
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
            cache: UnifiedCache实例（包含所有embedding和FAISS索引）
            storage: Storage实例
            llm_client: LLM客户端
            namespace: 命名空间
            max_hops: 最多扩展hop数
        """
        self.cache = cache
        self.storage = storage
        self.llm_client = llm_client
        self.namespace = namespace
        self.max_hops = max_hops
        
        # 初始化各个子模块
        self.recall = SearchRecall(cache, storage)
        self.expansion = GraphExpansion(storage, cache)
        self.router = QuestionRouter(llm_client)
        self.answer_generator = AnswerGenerator(llm_client)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        回答问题（完整流程）
        
        正确流程：召回 → 判断 → 尝试回答 → (成功返回 / 失败扩展) → 循环
        
        Args:
            question: 用户问题
        
        Returns:
            {
                'question': question,
                'answer': answer,
                'reason': reason,
                'stats': {统计信息}
            }
        """
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"{'='*60}")
        
        stats = {
            'llm_calls': 0,
            'recalled_nodes': 0,
            'expanded_nodes': 0,
            'hops': 0
        }
        
        # 阶段1: 初始召回
        print(f"\n📍 阶段1: 初始召回")
        recalled = self.recall.multi_layer_recall(question)
        
        current_nodes = recalled['layer1'] + recalled['layer2'] + recalled['layer3']
        stats['recalled_nodes'] = len(current_nodes)
        
        # 输出召回节点的ID列表
        print(f"\n📋 召回节点列表:")
        print(f"  Layer1: {[n['id'] for n in recalled['layer1'][:10]]}" + 
              (f" ... (+{len(recalled['layer1'])-10})" if len(recalled['layer1']) > 10 else ""))
        print(f"  Layer2: {[n['id'] for n in recalled['layer2'][:10]]}" + 
              (f" ... (+{len(recalled['layer2'])-10})" if len(recalled['layer2']) > 10 else ""))
        print(f"  Layer3: {[n['id'] for n in recalled['layer3'][:10]]}" + 
              (f" ... (+{len(recalled['layer3'])-10})" if len(recalled['layer3']) > 10 else ""))
        
        all_edges = []
        all_fragments = []
        
        # 阶段2: 多跳扩展循环
        print(f"\n📍 阶段2: 智能决策与扩展")
        
        # 存储选定的模块（在第一次决策时确定）
        selected_modules = None
        
        for hop in range(self.max_hops):
            print(f"\n🔄 Hop {hop + 1}/{self.max_hops}")
            
            # 2.1: 智能决策（判断信息充分度、是否需要扩展）
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
            
            # 在第一次决策时获取选定的模块
            if selected_modules is None:
                selected_modules = [decision.get('selected_module', 'detail_extraction')]
                print(f"  🎯 选择模块: {selected_modules}")
            
            # 不进行节点筛选，使用所有召回的节点
            can_answer_early = decision.get('can_answer_early', False)
            information_sufficiency = decision.get('information_sufficiency', 'insufficient')
            
            print(f"  📊 信息充分度: {information_sufficiency}")
            print(f"  📊 使用节点数: {len(current_nodes)}（无筛选）")
            print(f"  📊 can_answer_early: {can_answer_early}")
            
            # 2.2: 如果判断可以回答，尝试生成答案
            if can_answer_early:
                print(f"  🚀 尝试生成答案...")
                
                # 获取fragments
                all_fragments = self.recall.get_fragments_by_nodes(
                    [n['id'] for n in current_nodes]
                )
                
                # 构建context
                context = {
                    'layer1': [n for n in current_nodes if n.get('layer') == 1],
                    'layer2': [n for n in current_nodes if n.get('layer') == 2],
                    'layer3': [n for n in current_nodes if n.get('layer') == 3],
                    'edges': all_edges + connecting_edges,
                    'fragments': all_fragments
                }
                
                # 尝试回答（使用specialized modules）
                answer_result = self.answer_generator.try_answer(question, context)
                stats['llm_calls'] += 1
                
                # 直接返回LLM的答案（信任LLM的判断）
                print(f"  ✅ 答案生成成功！")
                
                stats['hops'] = hop + 1
                
                print(f"\n{'='*60}")
                print(f"✅ 问答完成（Hop {hop + 1}提前返回）")
                print(f"📊 统计: {stats['llm_calls']}次LLM调用, "
                      f"{stats['recalled_nodes']}个召回, "
                      f"{stats['expanded_nodes']}个扩展, "
                      f"{stats['hops']}次hop")
                print(f"{'='*60}")
                
                return {
                    'question': question,
                    'answer': answer_result['answer'],
                    'reason': answer_result['reason'],
                    'prompt_used': answer_result.get('prompt_used', ''),
                    'stats': stats
                }
            
            # 2.3: 信息不足，需要扩展
            should_expand = decision.get('should_expand', False)
            nodes_to_expand = decision.get('nodes_to_expand', [])
            
            if not should_expand or not nodes_to_expand:
                print(f"  ⚠️  不需要扩展，停止循环")
                break
            
            if not candidates:
                print(f"  ⚠️  无候选节点可扩展，停止")
                break
            
            # 执行扩展
            print(f"  🔗 扩展 {len(nodes_to_expand)} 个节点...")
            expanded_nodes, expanded_edges = self.expansion.expand_nodes(
                nodes_to_expand,
                max_neighbors=20
            )
            
            if not expanded_nodes:
                print(f"  ⚠️  扩展无新节点，停止")
                break
            
            # 更新节点集合
            old_node_ids = {n['id'] for n in current_nodes}
            new_nodes = [n for n in expanded_nodes if n['id'] not in old_node_ids]
            
            current_nodes.extend(new_nodes)
            all_edges.extend(expanded_edges)
            stats['expanded_nodes'] += len(new_nodes)
            stats['hops'] = hop + 1
            
            print(f"  ✅ 扩展了 {len(new_nodes)} 个新节点")
            if new_nodes:
                new_node_ids = [n['id'] for n in new_nodes[:10]]
                print(f"     扩展节点: {new_node_ids}" + 
                      (f" ... (+{len(new_nodes)-10})" if len(new_nodes) > 10 else ""))
        
        # 阶段3: 最终答案生成（如果循环结束仍未返回）
        print(f"\n📍 阶段3: 最终答案生成")
        
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
        
        # 最终生成答案（使用specialized modules）
        print(f"\n💬 生成答案...")
        
        # 使用之前选定的模块（如果没有则使用默认模块）
        if selected_modules is None:
            selected_modules = ['detail_extraction']
            print(f"  🎯 使用默认模块: {selected_modules}")
        else:
            print(f"  🎯 使用选定模块: {selected_modules}")
        
        answer_result = self.answer_generator.generate(question, context, selected_modules)
        stats['llm_calls'] += 1
        
        print(f"\n{'='*60}")
        print(f"✅ 问答完成（最终生成）")
        print(f"📊 统计: {stats['llm_calls']}次LLM调用, "
              f"{stats['recalled_nodes']}个召回, "
              f"{stats['expanded_nodes']}个扩展, "
              f"{stats['hops']}次hop")
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
        LLM决策：筛选节点、是否扩展、扩展目标
        
        Args:
            question: 用户问题
            current_nodes: 当前已有节点
            candidates: 候选节点
            edges: 连接边
        
        Returns:
            决策结果
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
        构建决策prompt（使用base_decision.txt）
        """
        # 加载base_decision prompt
        prompt_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'prompt',
            'base_decision.txt'
        )
        
        if not os.path.exists(prompt_file):
            print(f"  ⚠️  决策prompt不存在，使用简化版")
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
        """格式化节点为prompt字符串"""
        if not nodes:
            return "  (无节点)"
        
        result = ""
        for node in nodes[:max_nodes]:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            content = node.get('content', '')[:150]
            result += f"  - {node_id} ({node_type}): {content}...\n"
        
        if len(nodes) > max_nodes:
            result += f"  ... (还有 {len(nodes) - max_nodes} 个节点未显示)\n"
        
        return result
    
    def _build_simple_decision_prompt(
        self,
        question: str,
        current_nodes: List[Dict],
        candidates: List[Dict]
    ) -> str:
        """简化版决策prompt（作为后备）"""
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
        解析LLM的决策结果
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

