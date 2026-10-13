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


def _sorted_layer_keys(d):
    """返回字典中 layerN 键按层数排序后的列表（layer0, layer1, layer2, ...）"""
    keys = [k for k in d if k.startswith('layer') and k[5:].isdigit()]
    return sorted(keys, key=lambda k: int(k[5:]))


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
        default_provider: str = "openai"
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
        
        # 检查 cache 是否为空
        if self.cache.faiss_index is None or self.cache.faiss_index.ntotal == 0:
            print(f"⚠️  警告: FAISS索引为空 (namespace: {self.namespace})")
            print(f"   💡 提示: 请确保已使用正确的 namespace 调用 build_memory()")
            print(f"   💡 如果数据在其他 namespace，请在调用 ask_question() 时指定正确的 namespace 参数")
            return {
                'question': question,
                'answer': '抱歉，无法找到相关记忆。请检查是否使用了正确的 namespace，或先调用 build_memory() 构建记忆。',
                'reason': f'FAISS索引为空 (namespace: {self.namespace})',
                'stats': stats
            }
        
        if self.use_hybrid_search and hasattr(self.recall, 'multi_layer_recall_with_expansion'):
            # 使用混合检索（FAISS + Neo4j 图扩展）
            print(f"  🔍 使用混合检索模式（FAISS + Neo4j）")
            recalled = self.recall.multi_layer_recall_with_expansion(
                question,
                layer0_top_k=2,  # Fragment召回top2
                layer1_top_k=10,
                layer2_top_k=20,
                layer3_top_k=5,
                max_hops=0,  # 初始召回纯向量，不带图扩展
                expand_limit=0
            )
            # 提取初始节点和扩展节点（包括Layer0，动态支持Layer4+）
            current_nodes = []
            for layer_key in _sorted_layer_keys(recalled):
                layer_data = recalled[layer_key]
                current_nodes.extend(layer_data.get('all_nodes', []))
        else:
            # 使用原有检索（包括Layer0）
            recalled = self.recall.multi_layer_recall(
                question,
                layer0_top_k=2,  # Fragment召回top2
                layer1_top_k=10,
                layer2_top_k=20,
                layer3_top_k=5
            )
            current_nodes = []
            for layer_key in _sorted_layer_keys(recalled):
                current_nodes.extend(recalled.get(layer_key, []))
        
        stats['recalled_nodes'] = len(current_nodes)
        
        # Show recalled node IDs (完整列表)
        print(f"\n📋 Recalled node IDs:")
        if self.use_hybrid_search and hasattr(self.recall, 'multi_layer_recall_with_expansion'):
            # 混合检索格式（包括Layer0，动态支持Layer4+）
            for layer_key in _sorted_layer_keys(recalled):
                layer_data = recalled[layer_key]
                layer_nodes = layer_data.get('all_nodes', [])
                layer_name = layer_key.upper()
                node_ids = [n.get('id', 'unknown') for n in layer_nodes]
                print(f"  {layer_name}: {node_ids} (共 {len(node_ids)} 个)")
        else:
            # 原有检索格式（动态支持Layer4+）
            for layer_key in _sorted_layer_keys(recalled):
                layer_nodes = recalled.get(layer_key, [])
                layer_name = layer_key.upper()
                node_ids = [n.get('id', 'unknown') for n in layer_nodes]
                print(f"  {layer_name}: {node_ids} (共 {len(node_ids)} 个)")
        
        all_edges = []
        all_fragments = []
        
        # Stage 2: LLM决策（选module + 选扩展seed）→ 试答 → 不足则沿seed扩展
        print(f"\n📍 Stage 2: Decision + Answer Generation")
        
        # 2.0: LLM决策：选specialized module + 选扩展seed
        candidates, connecting_edges = self.expansion.find_candidates(
            current_node_ids=[n['id'] for n in current_nodes],
            max_candidates=50
        )
        # 语义候选：FAISS全库top中与问题相关但未进入召回集的节点（图断裂节点的发现通道）
        semantic_candidates = []
        if self.use_hybrid_search and hasattr(self.recall, 'hybrid_search'):
            try:
                query_embedding = self.recall.hybrid_search._get_query_embedding(question)
                recalled_ids = {n['id'] for n in current_nodes}
                for cand in self.recall.hybrid_search.cache.filter_and_search(
                    query_embedding, filters=None, top_k=40
                ):
                    node = cand.get('node', {})
                    if node and node.get('id') not in recalled_ids and node.get('layer', 0) >= 1:
                        semantic_candidates.append(node)
            except Exception as ex:
                print(f"  ⚠️  Failed to collect semantic candidates: {ex}")
        decision = self._make_expansion_decision(
            question, current_nodes, candidates, all_edges, semantic_candidates
        )
        selected_modules = [decision.get('selected_module', 'detail_extraction')]
        pending_seeds = decision.get('expansion_seeds', [])
        print(f"  🎯 Using selected module: {selected_modules}")
        print(f"  🌱 Expansion seeds from decision: {pending_seeds}")
        
        for hop in range(self.max_hops):
            print(f"\n🔄 Hop {hop + 1}/{self.max_hops}")
            
            print(f"  📊 Nodes used: {len(current_nodes)}")
            
            # 2.2: Try to answer directly (with decision-selected module)
            print(f"  🚀 Attempting to generate answer directly...")
            
            # Get fragments
            all_fragments = self.recall.get_fragments_by_nodes(
                [n['id'] for n in current_nodes]
            )
            
            # Build context（动态按层分组，支持Layer4+）
            context = {
                'edges': all_edges + connecting_edges,
                'fragments': all_fragments
            }
            for layer_num in sorted({n.get('layer') for n in current_nodes
                                     if isinstance(n.get('layer'), int) and n.get('layer') >= 1}):
                context[f'layer{layer_num}'] = [n for n in current_nodes if n.get('layer') == layer_num]
                
            answer_result = self.answer_generator.try_answer(
                question,
                context,
                selected_modules=selected_modules  # 决策LLM选中的module
            )
            stats['llm_calls'] += 1
            
            answer = answer_result.get('answer', '')
            reason = answer_result.get('reason', '')
            
            # Check if answer is sufficient (not "Insufficient information")
            is_insufficient = (
                'insufficient information' in answer.lower() or
                '信息不足' in answer.lower() or
                len(answer.strip()) < 10  # 答案太短也可能表示信息不足
            )
            
            if not is_insufficient:
                # Answer is sufficient, return it
                print(f"  ✅ Answer generated successfully!")
                stats['hops'] = hop + 1
                
                print(f"\n{'='*60}")
                print(f"✅ QA completed (hop {hop + 1})")
                print(f"📊 Stats: {stats['llm_calls']} LLM calls, "
                      f"{stats['recalled_nodes']} recalled, "
                      f"{stats['expanded_nodes']} expanded, "
                      f"{stats['hops']} hops")
                print(f"{'='*60}")
                
                return {
                    'question': question,
                    'answer': answer,
                    'reason': reason,
                    'prompt_used': answer_result.get('prompt_used', ''),
                    'stats': stats
                }
            else:
                # Answer is insufficient: expand along LLM-selected seeds from recalled nodes
                print(f"  ⚠️  Answer insufficient (detected 'Insufficient information'), attempting Neo4j expansion...")
                
                if hop == 0:
                    nodes_to_expand = list(pending_seeds)
                    if not nodes_to_expand:
                        # 决策未给出seed时的兜底：按向量相似度取前10个非Fragment召回节点
                        nodes_to_expand = [n['id'] for n in current_nodes
                                           if n.get('layer', 0) >= 1][:10]
                        print(f"  🌱 Decision gave no seeds, fallback to top-10 recalled non-fragment nodes")
                else:
                    # 基于扩展后的节点集重新决策选seed
                    candidates, connecting_edges = self.expansion.find_candidates(
                        current_node_ids=[n['id'] for n in current_nodes],
                        max_candidates=50
                    )
                    decision = self._make_expansion_decision(
                        question, current_nodes, candidates, all_edges, semantic_candidates
                    )
                    selected_modules = [decision.get('selected_module', 'detail_extraction')]
                    nodes_to_expand = decision.get('expansion_seeds', [])
                    if not nodes_to_expand:
                        print(f"  ⚠️  No seeds selected on re-decision, stop")
                        break
                
                if not nodes_to_expand:
                    print(f"  ⚠️  No seeds to expand, stop")
                    break
                
                print(f"  🔗 Expanding from {len(nodes_to_expand)} LLM-selected seeds via Neo4j...")
                
                # 使用Neo4j扩展（如果可用）
                if self.use_hybrid_search and hasattr(self.recall, 'hybrid_search'):
                    # 使用Neo4j的expand_from_nodes方法
                    # recall.hybrid_search 是 Neo4jHybridSearch 实例
                    # recall.hybrid_search.vector_search 是 Neo4jVectorSearch 实例
                    expanded_nodes_list = self.recall.hybrid_search.vector_search.expand_from_nodes(
                        node_ids=nodes_to_expand,
                        max_hops=2,  # 扩展2跳
                        limit=50  # 最多50个节点
                    )
                    
                    # 分离Layer1/Layer2/Layer3节点和Fragment节点（layer=0）
                    old_node_ids = {n['id'] for n in current_nodes}
                    
                    # Layer1/Layer2/Layer3节点：过滤掉已存在的节点
                    new_nodes = [
                        n for n in expanded_nodes_list 
                        if n.get('id') not in old_node_ids and n.get('layer') != 0
                    ]
                    
                    # Fragment节点（layer=0）：单独处理，使用向量相似度召回top2
                    fragment_nodes = [
                        n for n in expanded_nodes_list 
                        if n.get('layer') == 0 and n.get('id') not in old_node_ids
                    ]
                    
                    # 对Fragment节点进行向量相似度搜索，取top2
                    top_fragments = []
                    if fragment_nodes and hasattr(self.recall, 'hybrid_search'):
                        try:
                            # 获取问题的embedding
                            query_embedding = self.recall.hybrid_search._get_query_embedding(question)
                            
                            # 使用cache的filter_and_search对Fragment进行向量搜索
                            fragment_candidates = self.recall.hybrid_search.cache.filter_and_search(
                                query_embedding,
                                filters={'layer': 0},  # 只搜索Fragment
                                top_k=2  # 取top2
                            )
                            
                            # 提取Fragment节点
                            for candidate in fragment_candidates:
                                frag_node = candidate.get('node', {})
                                if frag_node and frag_node.get('id') not in old_node_ids:
                                    top_fragments.append(frag_node)
                            
                            print(f"  📄 Recalled {len(top_fragments)} top Fragment nodes via vector similarity")
                        except Exception as e:
                            print(f"  ⚠️  Failed to recall Fragment nodes: {e}")
                            # 如果向量搜索失败，使用扩展得到的Fragment节点（最多2个）
                            top_fragments = fragment_nodes[:2]
                    
                    # 合并所有新节点
                    all_new_nodes = new_nodes + top_fragments
                    
                    if all_new_nodes:
                        current_nodes.extend(all_new_nodes)
                        stats['expanded_nodes'] += len(all_new_nodes)
                        stats['hops'] = hop + 1
                        print(f"  ✅ Expanded {len(new_nodes)} nodes + {len(top_fragments)} fragments via Neo4j")
                        if all_new_nodes:
                            new_node_ids = [n['id'] for n in all_new_nodes]
                            print(f"     Expanded node IDs: {new_node_ids[:10]}... (共 {len(new_node_ids)} 个)")
                        # 继续下一轮循环，重新尝试生成答案
                        continue
                    else:
                        print(f"  ⚠️  Neo4j expansion yielded no new nodes, stop")
                        break
                else:
                    # 降级到原有的expansion方法
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
                        new_node_ids = [n['id'] for n in new_nodes]
                        print(f"     Expanded node IDs: {new_node_ids} (共 {len(new_node_ids)} 个)")
                        # 继续下一轮循环，重新尝试生成答案
                        continue
                    else:
                        print(f"  ⚠️  No new nodes after expansion, stop")
                        break
        
        # Stage 3: Final answer generation
        print(f"\n📍 Stage 3: Final answer generation")
        
        all_fragments = self.recall.get_fragments_by_nodes(
            [n['id'] for n in current_nodes]
        )
        
        context = {
            'edges': all_edges,
            'fragments': all_fragments
        }
        for layer_num in sorted({n.get('layer') for n in current_nodes
                                 if isinstance(n.get('layer'), int) and n.get('layer') >= 1}):
            context[f'layer{layer_num}'] = [n for n in current_nodes if n.get('layer') == layer_num]
        
        # Final generation (with all modules available)
        print(f"\n💬 Generating answer...")
        
        # 使用所有模块，让LLM自动选择和应用
        all_modules = ['time_handling', 'state_analysis', 'opinion_sentiment', 
                      'inference_prediction', 'detail_extraction', 'factual_lookup']
        print(f"  🎯 Using all modules (LLM will auto-select): {all_modules}")
        
        answer_result = self.answer_generator.generate(question, context, selected_modules=all_modules)
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
        edges: List[Dict],
        semantic_candidates: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        LLM decision: node filtering, whether to expand, and targets
        """
        print(f"  🤖 调用LLM进行扩展决策...")
        if semantic_candidates:
            print(f"    语义候选节点: {len(semantic_candidates)}个")

        # 构建决策prompt
        prompt = self._build_decision_prompt(
            question,
            current_nodes,
            candidates,
            edges,
            semantic_candidates
        )

        # 调用LLM
        response = self.llm_client.call_llm(prompt, provider=self.default_provider)

        # 解析决策（传入question用于辅助检查；seed校验池=召回节点∪语义候选）
        seed_pool = list(current_nodes) + list(semantic_candidates or [])
        decision = self._parse_decision_response(response, current_nodes, candidates, question, seed_pool)
        
        print(f"    选择模块: {decision.get('selected_module', 'unknown')}")
        print(f"    提前回答: {decision.get('can_answer_early', False)}")
        
        return decision
    
    def _build_decision_prompt(
        self,
        question: str,
        current_nodes: List[Dict],
        candidates: List[Dict],
        edges: List[Dict],
        semantic_candidates: List[Dict] = None
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

        # 语义相关但未召回的节点（向量top，图断裂节点的发现通道）
        semantic_context = self._format_nodes_for_prompt(semantic_candidates or [], max_nodes=30)

        # Fragments信息（暂时留空，可以后续扩展）
        fragment_info = ""

        return template.format(
            query=question,
            recalled_context=recalled_context,
            candidate_context=candidate_context,
            semantic_context=semantic_context,
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
        question: str = "",
        seed_pool: List[Dict] = None
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
        
        # 校验 expansion_seeds：必须是seed池（召回节点∪语义候选）中的 id，最多10个
        pool = seed_pool if seed_pool is not None else current_nodes
        valid_ids = {n.get('id') for n in pool}
        raw_seeds = result.get('expansion_seeds', [])
        if not isinstance(raw_seeds, list):
            raw_seeds = []
        result['expansion_seeds'] = [s for s in raw_seeds if s in valid_ids][:10]
        
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

