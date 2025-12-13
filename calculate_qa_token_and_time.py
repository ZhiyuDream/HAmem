"""
测试QA系统并统计token数量和时延

使用真实的QASystem.answer_question()流程，包括：
1. Initial recall（向量搜索，无LLM）
2. Expansion decision（LLM调用）
3. Answer generation（LLM调用）

统计内容：
- 问题的token数（tiktoken估算）
- 所有LLM调用的实际token数（从API响应获取）
- 各阶段的时延统计
"""

import sys
import os
import json
import time
import tiktoken
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import HAmem
from core.infrastructure.token_tracker import TokenTracker
from core.infrastructure.llm import LLMClient


def load_locomo_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载locomo数据集"""
    print(f"📂 加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成，共 {len(data)} 个conversation")
    return data


def count_tokens(text: str, encoding) -> int:
    """使用tiktoken计算token数量"""
    return len(encoding.encode(text))


class TokenTrackingLLMClient(LLMClient):
    """支持token追踪的LLMClient包装器"""
    
    def __init__(self, config, token_tracker, default_provider="deepseek"):
        super().__init__(config)
        self.token_tracker = token_tracker
        self.default_provider = default_provider
        self._current_call_type = None
    
    def call_llm(self, prompt: str, model: str = None, provider: str = "deepseek", return_usage: bool = False, call_type: str = None):
        """
        调用LLM并记录token
        
        Args:
            call_type: 调用类型（用于统计分类）
        """
        # 如果没有指定provider，使用默认值
        if provider == "deepseek" and self.default_provider != "deepseek":
            provider = self.default_provider
        
        # 设置调用类型
        if call_type:
            self._current_call_type = call_type
        
        # 调用父类方法，强制返回usage
        result = super().call_llm(prompt, model, provider, return_usage=True)
        
        if isinstance(result, tuple):
            content, usage = result
            
            # 记录token使用情况
            if self.token_tracker and self._current_call_type:
                self.token_tracker.record_llm_call(
                    self._current_call_type,
                    usage,
                    provider=provider
                )
            
            # 如果调用者不需要usage，只返回content
            if not return_usage:
                return content
            else:
                return (content, usage)
        else:
            # 如果父类没有返回usage，返回空usage
            if return_usage:
                return (result, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            return result


def patch_qa_system_with_token_tracking(qa_system, token_tracker, llm_provider):
    """
    为QA系统打补丁，使其使用支持token追踪的LLMClient
    
    由于QASystem的llm_client是内部属性，我们需要替换它
    """
    from core.infrastructure.llm import LLMClient
    from config import Config
    
    # 创建支持token追踪的LLMClient
    config = Config()
    tracking_llm_client = TokenTrackingLLMClient(config, token_tracker, default_provider=llm_provider)
    
    # 替换QA系统的LLMClient
    qa_system.llm_client = tracking_llm_client
    
    # 同时替换子模块的LLMClient
    if hasattr(qa_system, 'router'):
        qa_system.router.llm_client = tracking_llm_client
    if hasattr(qa_system, 'answer_generator'):
        qa_system.answer_generator.llm_client = tracking_llm_client
    
    return qa_system


def patch_qa_system_call_types(qa_system):
    """
    为QA系统的LLM调用添加call_type标记
    
    我们需要修改QASystem的方法来传递call_type
    """
    original_make_decision = qa_system._make_expansion_decision
    original_try_answer = qa_system.answer_generator.try_answer
    original_generate = qa_system.answer_generator.generate
    original_route = qa_system.router.route if hasattr(qa_system, 'router') else None
    
    def make_decision_with_tracking(question, current_nodes, candidates, edges):
        """包装扩展决策方法，添加call_type"""
        # 临时设置call_type（通过llm_client）
        if hasattr(qa_system.llm_client, '_current_call_type'):
            qa_system.llm_client._current_call_type = 'qa_expansion_decision'
            # 同时设置router和answer_generator的llm_client
            if hasattr(qa_system, 'router') and hasattr(qa_system.router, 'llm_client'):
                qa_system.router.llm_client._current_call_type = 'qa_expansion_decision'
        return original_make_decision(question, current_nodes, candidates, edges)
    
    def try_answer_with_tracking(question, context, selected_modules=None):
        """包装try_answer方法，添加call_type（不再需要路由，直接生成答案）"""
        if hasattr(qa_system.llm_client, '_current_call_type'):
            # 只设置answer_generation的call_type（不再有路由调用）
            if hasattr(qa_system.answer_generator, 'llm_client'):
                qa_system.answer_generator.llm_client._current_call_type = 'qa_answer_generation'
        return original_try_answer(question, context, selected_modules=selected_modules)
    
    def generate_with_tracking(question, context, selected_modules):
        """包装generate方法，添加call_type"""
        if hasattr(qa_system.llm_client, '_current_call_type'):
            qa_system.llm_client._current_call_type = 'qa_answer_generation'
            # 同时设置answer_generator的llm_client
            if hasattr(qa_system.answer_generator, 'llm_client'):
                qa_system.answer_generator.llm_client._current_call_type = 'qa_answer_generation'
        return original_generate(question, context, selected_modules)
    
    def route_with_tracking(question):
        """包装route方法，添加call_type"""
        if hasattr(qa_system.llm_client, '_current_call_type'):
            qa_system.llm_client._current_call_type = 'qa_question_routing'
            # 同时设置router的llm_client
            if hasattr(qa_system, 'router') and hasattr(qa_system.router, 'llm_client'):
                qa_system.router.llm_client._current_call_type = 'qa_question_routing'
        if original_route:
            return original_route(question)
        return ['detail_extraction']  # 默认模块
    
    # 替换方法
    qa_system._make_expansion_decision = make_decision_with_tracking
    qa_system.answer_generator.try_answer = try_answer_with_tracking
    qa_system.answer_generator.generate = generate_with_tracking
    if original_route:
        qa_system.router.route = route_with_tracking
    
    return qa_system


def test_qa_with_token_tracking(
    conversation_idx: int,
    question_idx: Optional[int] = None,
    dataset_path: str = None,
    llm_provider: str = "deepseek",
    namespace: str = None
):
    """
    测试QA系统并统计token和时延
    
    Args:
        conversation_idx: conversation索引（从0开始）
        question_idx: 问题索引（如果为None，测试所有问题）
        dataset_path: 数据集路径
        llm_provider: LLM提供商 ("openai" 或 "deepseek")
        namespace: 命名空间（如果为None，使用默认值）
    """
    if dataset_path is None:
        # 默认路径
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "locomo", "data", "locomo10.json"
        )
    
    # 加载数据集
    dataset = load_locomo_dataset(dataset_path)
    
    if conversation_idx >= len(dataset):
        print(f"❌ 错误: conversation索引 {conversation_idx} 超出范围（共 {len(dataset)} 个conversation）")
        return
    
    # 获取conversation和QA对
    conversation_data = dataset[conversation_idx]
    qa_pairs = conversation_data.get("qa", [])
    
    if not qa_pairs:
        print(f"❌ 错误: conversation {conversation_idx} 没有QA对")
        return
    
    # 确定要测试的问题
    if question_idx is not None:
        if question_idx >= len(qa_pairs):
            print(f"❌ 错误: 问题索引 {question_idx} 超出范围（共 {len(qa_pairs)} 个问题）")
            return
        test_questions = [qa_pairs[question_idx]]
    else:
        test_questions = qa_pairs
    
    # 确定namespace
    if namespace is None:
        namespace = f"locomo{conversation_idx}_test"
    
    print(f"\n📊 QA测试配置:")
    print(f"  - Conversation索引: {conversation_idx}")
    print(f"  - 问题数量: {len(test_questions)}")
    print(f"  - LLM提供商: {llm_provider}")
    print(f"  - Namespace: {namespace}")
    
    # 初始化tiktoken编码器
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # 创建token追踪器
    token_tracker = TokenTracker()
    
    # 初始化HAmem
    config = Config()
    hamem = HAmem(config)
    
    # 获取QA系统并打补丁
    # 使用hamem.ask_question来触发QA系统初始化，然后替换其LLMClient
    # 但更好的方式是直接创建QA系统
    from retrieval import create_qa_system
    from core.infrastructure import UnifiedCache, EmbeddingManager
    from core.infrastructure.neo4j_client import Neo4jClient
    
    # 创建必要的组件
    embedding_manager = EmbeddingManager(config)
    cache = UnifiedCache(
        cache_dir=config.cache_dir,
        namespace=namespace,
        embedding_manager=embedding_manager
    )
    
    # 创建QA系统（使用create_qa_system，它会处理Neo4j连接）
    # 但我们需要先创建支持token追踪的LLMClient
    tracking_llm_client = TokenTrackingLLMClient(config, token_tracker, default_provider=llm_provider)
    
    # 手动创建QA系统（模仿create_qa_system的逻辑）
    from core.search.qa_system import QASystem
    
    use_hybrid_search = config.use_hybrid_search and config.use_neo4j
    neo4j_client = None
    if use_hybrid_search:
        try:
            neo4j_client = Neo4jClient(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            )
            if not neo4j_client.connect():
                print("⚠️  Neo4j connection failed, falling back to standard search")
                use_hybrid_search = False
                neo4j_client = None
            else:
                print("✅ Neo4j connected, using hybrid search mode")
        except Exception as e:
            print(f"⚠️  Failed to initialize Neo4j: {e}, falling back to standard search")
            use_hybrid_search = False
            neo4j_client = None
    
    # 创建QA系统（使用支持token追踪的LLMClient）
    qa_system = QASystem(
        cache=cache,
        storage=None,  # QA系统可能不需要storage（使用Neo4j）
        llm_client=tracking_llm_client,
        namespace=namespace,
        max_hops=2,
        use_hybrid_search=use_hybrid_search,
        neo4j_client=neo4j_client,
        default_provider=llm_provider
    )
    
    # 确保所有子模块都使用支持token追踪的LLMClient
    if hasattr(qa_system, 'router'):
        qa_system.router.llm_client = tracking_llm_client
    if hasattr(qa_system, 'answer_generator'):
        qa_system.answer_generator.llm_client = tracking_llm_client
    
    # 注意：try_answer已经优化，不再调用路由，直接使用决策返回的模块
    # 所以不再需要修复版本
    
    # 打补丁：添加call_type追踪
    qa_system = patch_qa_system_call_types(qa_system)
    
    print(f"\n🔄 开始QA测试...")
    
    all_results = []
    total_qa_time = 0
    
    for i, qa_pair in enumerate(test_questions, 1):
        question = qa_pair.get("question", "")
        expected_answer = qa_pair.get("answer", "")
        
        print(f"\n{'='*70}")
        print(f"问题 {i}/{len(test_questions)}: {question}")
        if expected_answer:
            print(f"期望答案: {expected_answer[:100]}..." if len(expected_answer) > 100 else f"期望答案: {expected_answer}")
        print(f"{'='*70}")
        
        # 计算问题的token数
        question_tokens = count_tokens(question, encoding)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用QA系统
            answer_result = qa_system.answer_question(question)
            
            # 记录结束时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_qa_time += elapsed_time
            
            answer = answer_result.get('answer', 'N/A')
            reason = answer_result.get('reason', 'N/A')
            stats = answer_result.get('stats', {})
            
            print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
            print(f"📝 生成答案: {answer[:200]}..." if len(answer) > 200 else f"📝 生成答案: {answer}")
            
            # 显示统计信息
            if stats:
                print(f"\n📊 QA统计信息:")
                print(f"  - LLM调用次数: {stats.get('llm_calls', 0)}")
                print(f"  - 召回节点数: {stats.get('recalled_nodes', 0)}")
                print(f"  - 扩展节点数: {stats.get('expanded_nodes', 0)}")
                print(f"  - 扩展跳数: {stats.get('hops', 0)}")
            
            all_results.append({
                'question': question,
                'question_tokens': question_tokens,
                'answer': answer,
                'reason': reason,
                'elapsed_time': elapsed_time,
                'stats': stats
            })
            
        except Exception as e:
            print(f"❌ QA失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'question': question,
                'question_tokens': question_tokens,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            })
    
    # 显示总体统计
    print(f"\n{'='*70}")
    print(f"📊 总体统计")
    print(f"{'='*70}")
    print(f"  - 测试问题数: {len(test_questions)}")
    print(f"  - 总耗时: {total_qa_time:.2f} 秒")
    print(f"  - 平均耗时: {total_qa_time/len(test_questions):.2f} 秒/问题")
    
    # 显示详细token统计
    print(f"\n📊 Token统计（实际API调用）:")
    print(f"  {'='*70}")
    
    # 计算问题的总token数
    total_question_tokens = sum(r.get('question_tokens', 0) for r in all_results)
    print(f"  1. 问题内容总计: {total_question_tokens:,} tokens (tiktoken估算)")
    print(f"  {'='*70}")
    
    # 获取token统计
    token_stats = token_tracker.get_stats()
    
    if token_stats:
        total_llm_tokens = 0
        for call_type, stats in token_stats.items():
            if call_type == "by_provider":
                continue
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("completion_tokens", 0)
            total_tokens = stats.get("total_tokens", 0)
            calls = stats.get("calls", 0)
            total_llm_tokens += total_tokens
            
            print(f"  2. {call_type.replace('_', ' ').title()}:")
            print(f"     - Prompt tokens: {prompt_tokens:,}")
            print(f"     - Completion tokens: {completion_tokens:,}")
            print(f"     - Total tokens: {total_tokens:,}")
            print(f"     - LLM调用次数: {calls}")
            
            # 显示按提供商分类的统计
            by_provider = stats.get("by_provider", {})
            if by_provider:
                for provider, provider_stats in by_provider.items():
                    print(f"       * {provider}: {provider_stats.get('total_tokens', 0):,} tokens ({provider_stats.get('calls', 0)} calls)")
        
        print(f"  {'='*70}")
        print(f"  📊 LLM调用总计: {total_llm_tokens:,} tokens")
        print(f"  📊 总计（问题+LLM）: {total_question_tokens + total_llm_tokens:,} tokens")
        print(f"  {'='*70}")
    else:
        print(f"  ⚠️  未获取到token统计信息")
    
    return {
        'results': all_results,
        'total_time': total_qa_time,
        'token_stats': token_stats
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calculate_qa_token_and_time.py <conversation_idx> [question_idx] [llm_provider] [namespace]")
        print("示例: python calculate_qa_token_and_time.py 1")
        print("      python calculate_qa_token_and_time.py 1 0 openai")
        print("      python calculate_qa_token_and_time.py 1 None deepseek locomo1_test")
        sys.exit(1)
    
    conversation_idx = int(sys.argv[1])
    question_idx = None if len(sys.argv) > 2 and sys.argv[2].lower() == 'none' else (int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None)
    llm_provider = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in ["openai", "deepseek"] else "deepseek"
    namespace = sys.argv[4] if len(sys.argv) > 4 else None
    
    if llm_provider not in ["openai", "deepseek"]:
        print(f"❌ 错误: 不支持的LLM提供商 '{llm_provider}'，请使用 'openai' 或 'deepseek'")
        sys.exit(1)
    
    test_qa_with_token_tracking(conversation_idx, question_idx, None, llm_provider, namespace)

