"""
QA System main class

Coordinates recall, expansion, routing, and answer generation
"""

import os
from typing import Dict, Any, List, Optional
from core.infrastructure import LLMClient, UnifiedCache, parse_llm_json
from .recall import SearchRecall
from .expansion import GraphExpansion
from .router import QuestionRouter
from .answer import AnswerGenerator


def select_modules_by_rules(question: str) -> List[str]:
    """
    基于规则快速选择相关模块（无需LLM调用）
    
    Args:
        question: 问题文本
    
    Returns:
        List[str]: 选中的模块列表（最多3个）
    """
    question_lower = question.lower()
    modules = []
    
    # 时间相关
    time_keywords = ['when', 'what time', 'how long', 'since when', 'date', 'year', 'month', 
                     'day', 'which day', 'which date', 'how long ago', 'how many days', 
                     'how many years', 'how many months', 'start', 'begin', 'end', 'finish',
                     'occur', 'happen', 'took place', 'yesterday', 'last week', 'last month', 
                     'last year', 'ago', 'since']
    if any(kw in question_lower for kw in time_keywords):
        modules.append('time_handling')
    
    # 状态相关
    state_keywords = ['currently', 'now', 'recent status', 'is doing', 'was doing', 
                      'current', 'recent', 'status', 'relationship status', 'living', 
                      'working', 'studying']
    if any(kw in question_lower for kw in state_keywords):
        modules.append('state_analysis')
    
    # 情感/观点相关
    opinion_keywords = ['feel', 'think', 'like', 'dislike', 'opinion', 'emotion', 
                        'feeling', 'attitude', 'preference', 'sentiment', 'reaction']
    if any(kw in question_lower for kw in opinion_keywords):
        modules.append('opinion_sentiment')
    
    # 预测/推理相关
    inference_keywords = ['will', 'would', 'might', 'predict', 'future', 'likely', 
                          'probably', 'probably', 'chance', 'possibility']
    if any(kw in question_lower for kw in inference_keywords):
        modules.append('inference_prediction')
    
    # 细节提取（大多数问题都需要）
    detail_keywords = ['what did', 'how did', 'explain', 'describe', 'what', 'how', 
                       'why', 'which', 'detail', 'specific']
    if any(kw in question_lower for kw in detail_keywords):
        modules.append('detail_extraction')
    
    # 事实查询（简单短问题）
    if len(question.split()) <= 5 and not modules:
        modules.append('factual_lookup')
    
    # 如果没有匹配到，默认使用time_handling
    if not modules:
        modules = ['time_handling']
    
    # 去重并限制最多3个模块
    unique_modules = []
    for m in modules:
        if m not in unique_modules:
            unique_modules.append(m)
    
    return unique_modules[:3]


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
        max_hops: int = 2,
        use_hybrid_search: bool = False,
        neo4j_client = None,
        default_provider: str = "deepseek"
    ):
        """
        Args:
            cache: UnifiedCache (embeddings + FAISS indexes)
            storage: Storage instance
            llm_client: LLM client
            namespace: Namespace
            max_hops: Max expansion hops
            use_hybrid_search: 是否使用混合检索（FAISS + Neo4j）
            neo4j_client: Neo4j 客户端（如果使用混合检索）
            default_provider: 默认LLM提供商 ("openai" 或 "deepseek")
        """
        self.cache = cache
        self.storage = storage
        self.llm_client = llm_client
        self.namespace = namespace
        self.max_hops = max_hops
        self.use_hybrid_search = use_hybrid_search
        self.default_provider = default_provider
        
        # Initialize submodules
        if use_hybrid_search and neo4j_client:
            # 使用混合检索
            from .neo4j_hybrid_recall import Neo4jHybridRecall
            self.recall = Neo4jHybridRecall(cache, neo4j_client, namespace)
            self.hybrid_recall = self.recall  # 保存引用以便访问混合检索功能
            # 扩展仍然使用原有的 GraphExpansion（基于 UnifiedCache）
            self.expansion = GraphExpansion(storage, cache)
        else:
            # 使用原有检索
            self.recall = SearchRecall(cache, storage)
            self.expansion = GraphExpansion(storage, cache)
        
        self.router = QuestionRouter(llm_client, default_provider=default_provider)
        self.answer_generator = AnswerGenerator(llm_client, default_provider=default_provider)
    
    def _graph_expand_from_seeds(
        self,
        seed_node_ids: List[str],
        max_hops: int
    ) -> List[Dict[str, Any]]:
        """
        从种子节点开始进行图扩散（纯图遍历）
        
        Args:
            seed_node_ids: 种子节点ID列表
            max_hops: 最大扩散跳数
        
        Returns:
            List[Dict]: 所有扩散得到的节点（包括种子节点和所有跳的邻居节点）
        """
        if not seed_node_ids:
            return []
        
        print(f"  🔗 从 {len(seed_node_ids)} 个种子节点开始图扩散 (max_hops={max_hops})...")
        print(f"  📋 种子节点IDs: {seed_node_ids}")
        
        # 使用Neo4j进行图扩散
        if self.use_hybrid_search and hasattr(self.recall, 'hybrid_search'):
            # 过滤掉fragment节点（layer=0），不允许从fragment开始扩展
            filtered_seed_ids = []
            for seed_id in seed_node_ids:
                node = self.cache.cache['nodes'].get(seed_id)
                if node and node.get('layer', -1) != 0:  # 排除layer=0的节点
                    filtered_seed_ids.append(seed_id)
                elif node:
                    print(f"  ⚠️  跳过fragment节点作为扩展起点: {seed_id}")
            
            if not filtered_seed_ids:
                print(f"  ⚠️  所有种子节点都是fragment，无法扩展")
                return []
            
            print(f"  🔗 从 {len(filtered_seed_ids)} 个非fragment节点开始扩展（已过滤 {len(seed_node_ids) - len(filtered_seed_ids)} 个fragment节点）")
            
            # 使用Neo4j的expand_from_nodes方法
            expanded_nodes = self.recall.hybrid_search.vector_search.expand_from_nodes(
                node_ids=filtered_seed_ids,
                max_hops=max_hops,
                limit=100  # 限制节点数量
            )
            
            # 统计每跳的节点数（expand_from_nodes返回的节点包含hops信息）
            hop_counts = {}
            seed_nodes_count = 0
            expanded_nodes_count = 0
            
            for node in expanded_nodes:
                hops = node.get('hops', 0)
                is_initial = node.get('is_initial', False)
                
                if is_initial or hops == 0:
                    seed_nodes_count += 1
                else:
                    expanded_nodes_count += 1
                
                if hops not in hop_counts:
                    hop_counts[hops] = 0
                hop_counts[hops] += 1
            
            print(f"  ✅ 图扩散完成，获得 {len(expanded_nodes)} 个节点")
            print(f"  📊 统计: 种子节点 {seed_nodes_count} 个, 扩展节点 {expanded_nodes_count} 个")
            if hop_counts:
                print(f"  📊 按跳数统计: {dict(sorted(hop_counts.items()))}")
                # 验证max_hops是否正确应用
                max_hop_found = max(hop_counts.keys()) if hop_counts else 0
                if max_hop_found > max_hops:
                    print(f"  ⚠️  警告: 发现跳数 {max_hop_found} > max_hops {max_hops}，可能有问题！")
                else:
                    print(f"  ✅ 最大跳数验证: {max_hop_found} <= {max_hops}")
            
            return expanded_nodes
        else:
            # 降级：使用原有的expansion方法（不支持多跳，只能1跳）
            print(f"  ⚠️  未使用混合检索，降级到单跳扩展")
            all_expanded_nodes = []
            all_expanded_edges = []
            max_total_nodes = 100  # 总节点数限制
            
            # 逐跳扩展
            current_seeds = seed_node_ids
            for hop in range(max_hops):
                # 过滤掉 layer=0 (fragment) 的种子节点，不允许从fragment开始扩展
                # 但允许扩展到fragment节点
                filtered_seeds = []
                for seed_id in current_seeds:
                    node = self.cache.cache['nodes'].get(seed_id)
                    if node and node.get('layer', -1) != 0:  # 排除layer=0的节点
                        filtered_seeds.append(seed_id)
                    elif node:
                        print(f"  ⚠️  跳过fragment节点作为扩展起点: {seed_id}")
                
                if not filtered_seeds:
                    print(f"  ⚠️  第{hop+1}跳：所有种子节点都是fragment，停止扩展")
                    break
                
                print(f"  🔗 第{hop+1}跳：从 {len(filtered_seeds)} 个非fragment节点开始扩展（已过滤 {len(current_seeds) - len(filtered_seeds)} 个fragment节点）")
                
                expanded_nodes, expanded_edges = self.expansion.expand_nodes(
                    filtered_seeds,
                    max_neighbors=10  # 从50改为10
                )
                
                if not expanded_nodes:
                    break
                
                # 获取新节点（去重）
                existing_ids = {n['id'] for n in all_expanded_nodes}
                new_nodes = [n for n in expanded_nodes if n['id'] not in existing_ids]
                
                # 检查总节点数限制
                if len(all_expanded_nodes) + len(new_nodes) > max_total_nodes:
                    remaining = max_total_nodes - len(all_expanded_nodes)
                    if remaining > 0:
                        new_nodes = new_nodes[:remaining]
                        print(f"  ⚠️  达到总节点数限制({max_total_nodes})，截断到 {len(all_expanded_nodes) + len(new_nodes)} 个节点")
                    else:
                        print(f"  ⚠️  已达到总节点数限制({max_total_nodes})，停止扩展")
                        break
                
                all_expanded_nodes.extend(new_nodes)
                all_expanded_edges.extend(expanded_edges)
                
                # 下一跳的种子是当前跳的所有新节点（但会过滤掉fragment）
                current_seeds = [n['id'] for n in new_nodes]
                
                if not current_seeds:
                    break
                
                # 如果已达到总节点数限制，停止扩展
                if len(all_expanded_nodes) >= max_total_nodes:
                    print(f"  ⚠️  已达到总节点数限制({max_total_nodes})，停止扩展")
                    break
            
            print(f"  ✅ 图扩散完成，获得 {len(all_expanded_nodes)} 个节点（限制: {max_total_nodes}）")
            return all_expanded_nodes
    
    def _is_answer_sufficient(self, answer: str) -> bool:
        """
        判断答案是否足够
        
        Args:
            answer: 生成的答案
        
        Returns:
            bool: True表示答案足够，False表示信息不足
        """
        answer_lower = answer.lower()
        is_insufficient = (
            'insufficient information' in answer_lower or
            '信息不足' in answer_lower or
            len(answer.strip()) < 10  # 答案太短也可能表示信息不足
        )
        return not is_insufficient
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question via new pipeline.
        Flow: 
        1. Initial recall (embedding similarity)
        2. Try answer
        3. If insufficient, LLM selects seed nodes → graph expansion (max_hops) → try answer again
        4. Max 2 iterations
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        stats = {
            'llm_calls': 0,
            'recalled_nodes': 0,
            'expanded_nodes': 0,
            'hops': 0,
            'iterations': 0
        }
        
        # Stage 1: Initial recall (embedding similarity)
        print(f"\n📍 Stage 1: Initial recall (embedding similarity)")
        
        # 检查 cache 是否为空
        if self.cache.faiss_index is None or self.cache.faiss_index.ntotal == 0:
            print(f"⚠️  警告: FAISS索引为空 (namespace: {self.namespace})")
            return {
                'question': question,
                'answer': '抱歉，无法找到相关记忆。请检查是否使用了正确的 namespace，或先调用 build_memory() 构建记忆。',
                'reason': f'FAISS索引为空 (namespace: {self.namespace})',
                'stats': stats
            }
        
        # 初始召回（向量相似度，不进行图扩展）
        if self.use_hybrid_search and hasattr(self.recall, 'multi_layer_recall_with_expansion'):
            # 使用混合检索，但初始召回时不扩展（max_hops=0）
            # 方案1：减少初始召回数量（从37个减少到约19个）
            recalled = self.recall.multi_layer_recall_with_expansion(
                question,
                layer0_top_k=1,   # 2 → 1
                layer1_top_k=3,   # 10 → 5
                layer2_top_k=3,  # 20 → 10
                layer3_top_k=3,   # 5 → 3
                max_hops=0,  # 初始召回不扩展
                expand_limit=0
            )
            current_nodes = []
            for layer_key in ['layer0', 'layer1', 'layer2', 'layer3']:
                if layer_key in recalled:
                    layer_data = recalled[layer_key]
                    current_nodes.extend(layer_data.get('all_nodes', []))
        else:
            # 使用原有检索
            # 方案1：减少初始召回数量（从37个减少到约19个）
            recalled = self.recall.multi_layer_recall(
                question,
                layer0_top_k=1,   # 2 → 1
                layer1_top_k=3,   # 10 → 5
                layer2_top_k=3,  # 20 → 10
                layer3_top_k=3   # 5 → 3
            )
            current_nodes = (
                recalled.get('layer0', []) + 
                recalled.get('layer1', []) + 
                recalled.get('layer2', []) + 
                recalled.get('layer3', [])
            )
        
        stats['recalled_nodes'] = len(current_nodes)
        print(f"  ✅ 初始召回: {len(current_nodes)} 个节点")
        
        # 显示召回的节点
        node_ids = [n.get('id', 'unknown') for n in current_nodes]
        print(f"  📋 节点IDs: {node_ids[:10]}... (共 {len(node_ids)} 个)")
        
        # 获取fragments
        all_fragments = self.recall.get_fragments_by_nodes(node_ids)
        
        # 构建context
        context = {
            'layer1': [n for n in current_nodes if n.get('layer') == 1],
            'layer2': [n for n in current_nodes if n.get('layer') == 2],
            'layer3': [n for n in current_nodes if n.get('layer') == 3],
            'edges': [],
            'fragments': all_fragments
        }
        
        # 尝试生成答案（基于规则快速选择模块）
        selected_modules = select_modules_by_rules(question)
        print(f"  🎯 基于规则选择的模块: {selected_modules}")
        answer_result = self.answer_generator.try_answer(
            question, 
            context, 
            selected_modules=selected_modules
        )
        stats['llm_calls'] += 1
        stats['iterations'] += 1
        
        answer = answer_result.get('answer', '')
        reason = answer_result.get('reason', '')
        seed_node_ids = answer_result.get('seed_node_ids', [])  # 从答案结果中获取选中的节点
        
        # 判断答案是否足够
        if self._is_answer_sufficient(answer):
            print(f"  ✅ 答案足够，直接返回")
            print(f"\n{'='*60}")
            print(f"✅ QA completed (iteration {stats['iterations']})")
            print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
                  f"{stats['recalled_nodes']} recalled, "
                  f"{stats['expanded_nodes']} expanded, "
                  f"{stats['hops']} hops, "
                  f"{stats['iterations']} iterations")
            print(f"{'='*60}")
            
            return {
                'question': question,
                'answer': answer,
                'reason': reason,
                'prompt_used': answer_result.get('prompt_used', ''),
                'stats': stats
            }
        
        # 答案不足，进入扩展循环（最多2次迭代）
        print(f"  ⚠️  答案不足，进入扩展循环 (最多2次迭代)")
        
        max_iterations = 2
        for iteration in range(max_iterations):
            print(f"\n🔄 扩展迭代 {iteration + 1}/{max_iterations}")
            
            # Step 1: 使用上次答案生成时LLM选中的种子节点
            # 验证节点ID是否存在于当前节点中，并过滤掉fragment节点（layer=0）
            valid_node_ids = {n.get('id') for n in current_nodes}
            seed_node_ids = [nid for nid in seed_node_ids if nid in valid_node_ids]
            
            # 过滤掉fragment节点（layer=0），不允许从fragment开始扩展
            filtered_seed_ids = []
            for nid in seed_node_ids:
                node = next((n for n in current_nodes if n.get('id') == nid), None)
                if node and node.get('layer', -1) != 0:  # 排除layer=0的节点
                    filtered_seed_ids.append(nid)
                elif node:
                    print(f"  ⚠️  过滤掉fragment节点作为种子: {nid}")
            
            seed_node_ids = filtered_seed_ids
            
            # 如果LLM没有返回有效的节点，选择前3个非fragment节点作为默认值
            if not seed_node_ids:
                print(f"  ⚠️  LLM未返回有效节点，从前3个非fragment节点中选择种子")
                for node in current_nodes:
                    if node.get('layer', -1) != 0:  # 排除fragment
                        seed_node_ids.append(node.get('id'))
                        if len(seed_node_ids) >= 3:
                            break
            
            if not seed_node_ids:
                print(f"  ⚠️  未选中种子节点，停止扩展")
                break
            
            print(f"  🎯 使用种子节点: {seed_node_ids[:5]}")
            
            # Step 2: 从种子节点开始图扩散（按max_hops）
            expanded_nodes = self._graph_expand_from_seeds(seed_node_ids, self.max_hops)
            
            if not expanded_nodes:
                print(f"  ⚠️  图扩散未获得新节点，停止扩展")
                break
            
            # Step 3: 合并节点（去重）
            existing_node_ids = {n.get('id') for n in current_nodes}
            new_nodes = [n for n in expanded_nodes if n.get('id') not in existing_node_ids]
            
            if not new_nodes:
                print(f"  ⚠️  未获得新节点，停止扩展")
                break
            
            current_nodes.extend(new_nodes)
            stats['expanded_nodes'] += len(new_nodes)
            stats['hops'] = self.max_hops
            
            print(f"  ✅ 扩展完成，新增 {len(new_nodes)} 个节点")
            
            # Step 4: 更新fragments和context
            all_node_ids = [n.get('id') for n in current_nodes]
            all_fragments = self.recall.get_fragments_by_nodes(all_node_ids)
            
            context = {
                'layer1': [n for n in current_nodes if n.get('layer') == 1],
                'layer2': [n for n in current_nodes if n.get('layer') == 2],
                'layer3': [n for n in current_nodes if n.get('layer') == 3],
                'edges': [],
                'fragments': all_fragments
            }
            
            # Step 5: 再次尝试生成答案（基于规则快速选择模块）
            selected_modules = select_modules_by_rules(question)
            answer_result = self.answer_generator.try_answer(
                question, 
                context, 
                selected_modules=selected_modules
            )
            stats['llm_calls'] += 1
            stats['iterations'] += 1
            
            answer = answer_result.get('answer', '')
            reason = answer_result.get('reason', '')
            seed_node_ids = answer_result.get('seed_node_ids', [])  # 更新种子节点（用于下次迭代）
            
            # 判断答案是否足够
            if self._is_answer_sufficient(answer):
                print(f"  ✅ 答案足够，返回结果")
                print(f"\n{'='*60}")
                print(f"✅ QA completed (iteration {stats['iterations']})")
                print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
                      f"{stats['recalled_nodes']} recalled, "
                      f"{stats['expanded_nodes']} expanded, "
                      f"{stats['hops']} hops, "
                      f"{stats['iterations']} iterations")
                print(f"{'='*60}")
                
                return {
                    'question': question,
                    'answer': answer,
                    'reason': reason,
                    'prompt_used': answer_result.get('prompt_used', ''),
                    'stats': stats
                }
            else:
                print(f"  ⚠️  答案仍不足，继续扩展...")
        
        # 所有迭代完成，返回最终答案（即使不足）
        print(f"\n{'='*60}")
        print(f"✅ QA completed (final, {stats['iterations']} iterations)")
        print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
              f"{stats['recalled_nodes']} recalled, "
              f"{stats['expanded_nodes']} expanded, "
              f"{stats['hops']} hops, "
              f"{stats['iterations']} iterations")
        print(f"{'='*60}")
        
        return {
            'question': question,
            'answer': answer,
            'reason': reason,
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
        response = self.llm_client.call_llm(prompt, provider=self.default_provider)
        
        # 解析决策（传入question用于辅助检查）
        decision = self._parse_decision_response(response, current_nodes, candidates, question)
        
        print(f"    选择模块: {decision.get('selected_module', 'unknown')}")
        print(f"    提前回答: {decision.get('can_answer_early', False)}")
        
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

"""
    
    def _parse_decision_response(
        self,
        response: str,
        current_nodes: List[Dict],
        candidates: List[Dict],
        question: str = ""
    ) -> Dict[str, Any]:
        """
        Parse LLM decision JSON
        """
        default_result = {
            'selected_module': 'detail_extraction',
            'can_answer_early': False,
            'reasoning': 'Default: insufficient information'
        }
        
        result = parse_llm_json(
            response,
            expected_keys=['selected_module', 'can_answer_early'],
            default=default_result
        )
        
        if result is None:
            result = default_result
        
        # 辅助检查：如果问题明显是时间相关的，但LLM没有选择time_handling，强制选择
        if question:
            time_keywords = [
                'when', 'what time', 'which day', 'which date', 'which year', 'which month',
                'how long', 'how long ago', 'how many days', 'how many years', 'how many months',
                'since when', 'since', 'ago', 'yesterday', 'last week', 'last month', 'last year',
                'start', 'begin', 'end', 'finish', 'occur', 'happen', 'took place'
            ]
            question_lower = question.lower()
            is_time_question = any(keyword in question_lower for keyword in time_keywords)
            
            if is_time_question and result.get('selected_module') != 'time_handling':
                # 如果明显是时间问题但LLM没有选择time_handling，强制选择
                result['selected_module'] = 'time_handling'
                if result.get('reasoning'):
                    result['reasoning'] = f"Time-related question detected. {result['reasoning']}"
        
        return result

