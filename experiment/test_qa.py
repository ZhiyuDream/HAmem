"""
测试QA系统并统计token数量和时延

使用真实的QASystem.answer_question()流程，包括：
1. Initial recall（向量搜索，无LLM）
2. Answer generation（LLM调用，可能包含扩展）

统计内容：
- 问题的token数（tiktoken估算）
- 所有LLM调用的实际token数（从API响应获取）
- 各阶段的时延统计
- LLM Judge评估（使用gpt-4o-mini评估答案正确性）
"""

import sys
import os
import json
import time
import tiktoken
import argparse
import re
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config
from core.main import HAmem
from core.infrastructure.token_tracker import TokenTracker
from core.infrastructure.llm import LLMClient

# LLM Judge评估的Prompt
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def extract_json(text: str) -> str:
    """从文本中提取JSON内容"""
    # 尝试找到JSON对象
    json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    # 如果没有找到，尝试提取整个JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text


def evaluate_answer_with_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    llm_client: LLMClient,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    使用LLM Judge评估答案
    
    Args:
        question: 问题
        gold_answer: 期望答案（ground truth）
        generated_answer: 生成的答案
        llm_client: LLM客户端
        model: 评估模型（默认gpt-4o-mini）
    
    Returns:
        Dict包含: label (CORRECT/WRONG), score (1/0), reasoning
    """
    try:
        prompt = ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        # 调用LLM进行评估
        response = llm_client.call_llm(
            prompt=prompt,
            model=model,
            provider=None  # 使用配置中的provider
        )
        
        # 解析JSON响应
        try:
            json_text = extract_json(response)
            result = json.loads(json_text)
            label = result.get("label", "WRONG").upper()
        except (json.JSONDecodeError, KeyError):
            # 如果JSON解析失败，尝试从文本中提取
            if "CORRECT" in response.upper():
                label = "CORRECT"
            else:
                label = "WRONG"
        
        score = 1 if label == "CORRECT" else 0
        reasoning = response.strip()
        
        return {
            "label": label,
            "score": score,
            "reasoning": reasoning
        }
    except Exception as e:
        print(f"⚠️  LLM Judge评估失败: {e}")
        return {
            "label": "WRONG",
            "score": 0,
            "reasoning": f"评估失败: {str(e)}"
        }


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


def patch_qa_system_call_types(qa_system):
    """
    为QA系统的LLM调用添加call_type标记
    
    我们需要修改QASystem的方法来传递call_type
    """
    original_try_answer = qa_system.answer_generator.try_answer
    original_generate = qa_system.answer_generator.generate
    
    def try_answer_with_tracking(question, context, selected_modules=None):
        """包装try_answer方法，添加call_type"""
        if hasattr(qa_system.llm_client, '_current_call_type'):
            # 只设置answer_generation的call_type
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
    
    # 替换方法
    qa_system.answer_generator.try_answer = try_answer_with_tracking
    qa_system.answer_generator.generate = generate_with_tracking
    
    return qa_system


def process_single_qa(
    qa_pair: Dict[str, Any],
    qa_system,
    tracking_llm_client,
    encoding,
    index: int,
    total: int
) -> Dict[str, Any]:
    """
    处理单个QA对（用于并行处理）
    """
    question = qa_pair.get("question", "")
    expected_answer = qa_pair.get("answer", "")
    category = qa_pair.get("category", None)
    
    # 确保expected_answer是字符串类型
    if expected_answer is not None:
        if not isinstance(expected_answer, str):
            expected_answer = str(expected_answer)
    else:
        expected_answer = ""
    
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
        
        answer = answer_result.get('answer', 'N/A')
        reason = answer_result.get('reason', 'N/A')
        stats = answer_result.get('stats', {})
        
        # LLM Judge评估（如果有期望答案）
        evaluation_result = None
        if expected_answer:
            evaluation_result = evaluate_answer_with_llm_judge(
                question=question,
                gold_answer=expected_answer,
                generated_answer=answer,
                llm_client=tracking_llm_client,
                model="gpt-4o-mini"
            )
        
        return {
            'index': index,
            'question': question,
            'question_tokens': question_tokens,
            'answer': answer,
            'expected_answer': expected_answer,
            'category': category,
            'reason': reason,
            'elapsed_time': elapsed_time,
            'stats': stats,
            'evaluation': evaluation_result
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return {
            'index': index,
            'question': question,
            'question_tokens': question_tokens,
            'category': category,
            'error': error_msg,
            'elapsed_time': time.time() - start_time
        }


def test_qa(
    conversation_idx: Optional[int] = None,
    question_idx: Optional[int] = None,
    dataset_path: str = None,
    model: str = None,
    namespace: str = None,
    parallel: int = 50,
    all_conversations: bool = False
):
    """
    测试QA系统并统计token和时延
    
    Args:
        conversation_idx: conversation索引（从0开始），如果all_conversations=True则忽略
        question_idx: 问题索引（如果为None，测试所有问题）
        dataset_path: 数据集路径
        model: LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）
        namespace: 命名空间（如果为None，使用默认值）
        parallel: 并行处理数量
        all_conversations: 是否测试所有conversation（如果为True，会测试所有conversation的QA）
    """
    if dataset_path is None:
        # 默认路径（相对于项目根目录）
        dataset_path = os.path.join(
            project_root,
            "locomo", "data", "locomo10.json"
        )
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集文件不存在: {dataset_path}")
        print(f"   请使用 --dataset 参数指定正确的数据集路径")
        return None
    
    # 加载数据集
    dataset = load_locomo_dataset(dataset_path)
    
    # 如果测试所有conversation
    if all_conversations:
        print(f"\n🔄 测试所有conversation的QA（共 {len(dataset)} 个conversation）...")
        all_conversation_results = []
        
        for conv_idx in range(len(dataset)):
            print(f"\n{'='*70}")
            print(f"📊 处理 Conversation {conv_idx}/{len(dataset)-1}")
            print(f"{'='*70}")
            
            # 递归调用test_qa处理单个conversation
            result = test_qa(
                conversation_idx=conv_idx,
                question_idx=question_idx,
                dataset_path=dataset_path,
                model=model,
                namespace=None,  # 使用默认namespace
                parallel=parallel,
                all_conversations=False  # 避免无限递归
            )
            
            if result:
                all_conversation_results.append({
                    'conversation_idx': conv_idx,
                    'result': result
                })
        
        # 汇总所有conversation的结果
        print(f"\n{'='*70}")
        print(f"📊 所有Conversation汇总统计")
        print(f"{'='*70}")
        
        total_questions = 0
        total_correct = 0
        total_evaluated = 0
        category_stats_all = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for conv_result in all_conversation_results:
            conv_idx = conv_result['conversation_idx']
            results = conv_result['result'].get('results', [])
            
            evaluated_results = [r for r in results if r.get('evaluation') is not None]
            correct_count = sum(1 for r in evaluated_results if r['evaluation'].get('score', 0) == 1)
            
            total_questions += len(results)
            total_evaluated += len(evaluated_results)
            total_correct += correct_count
            
            # 按类别统计
            for result in evaluated_results:
                category = result.get('category')
                if category is not None:
                    category_stats_all[category]['total'] += 1
                    if result['evaluation'].get('score', 0) == 1:
                        category_stats_all[category]['correct'] += 1
        
        overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0
        
        print(f"  - 总问题数: {total_questions}")
        print(f"  - 评估问题数: {total_evaluated}")
        print(f"  - 正确答案数: {total_correct}")
        print(f"  - 总体准确率: {overall_accuracy:.4f} ({total_correct}/{total_evaluated})")
        
        if category_stats_all:
            print(f"\n📊 所有Conversation按类别统计:")
            print(f"  {'类别':<10} {'正确数':<10} {'总数':<10} {'准确率':<10}")
            print(f"  {'-'*40}")
            for category in sorted(category_stats_all.keys()):
                stats = category_stats_all[category]
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                print(f"  {category:<10} {stats['correct']:<10} {stats['total']:<10} {accuracy:.4f}")
        
        # 准备返回结果
        result_data = {
            'all_conversations': True,
            'model': model or Config().llm_model,
            'overall_summary': {
                'total_questions': total_questions,
                'total_evaluated': total_evaluated,
                'total_correct': total_correct,
                'overall_accuracy': overall_accuracy,
                'category_stats': dict(category_stats_all)
            },
            'conversation_results': all_conversation_results
        }
        
        # 保存结果到JSON文件
        output_dir = os.path.join(project_root, "experiment", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qa_evaluation_all_conversations_{int(time.time())}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 评估结果已保存到: {output_file}")
        except Exception as e:
            print(f"⚠️  保存结果失败: {e}")
        
        return result_data
    
    # 单个conversation的处理逻辑
    if conversation_idx is None:
        print(f"❌ 错误: 必须指定conversation_idx或使用--all-conversations选项")
        return None
    
    if conversation_idx >= len(dataset):
        print(f"❌ 错误: conversation索引 {conversation_idx} 超出范围（共 {len(dataset)} 个conversation）")
        return None
    
    # 获取conversation和QA对
    conversation_data = dataset[conversation_idx]
    qa_pairs = conversation_data.get("qa", [])
    
    if not qa_pairs:
        print(f"❌ 错误: conversation {conversation_idx} 没有QA对")
        return None
    
    # 过滤掉 category=5 的问题
    filtered_qa_pairs = [qa for qa in qa_pairs if qa.get("category") != 5]
    filtered_count = len(qa_pairs) - len(filtered_qa_pairs)
    if filtered_count > 0:
        print(f"ℹ️  过滤掉 {filtered_count} 个 category=5 的问题")
    
    # 确定要测试的问题
    if question_idx is not None:
        if question_idx >= len(filtered_qa_pairs):
            print(f"❌ 错误: 问题索引 {question_idx} 超出范围（共 {len(filtered_qa_pairs)} 个问题，已过滤category=5）")
            return None
        test_questions = [filtered_qa_pairs[question_idx]]
    else:
        test_questions = filtered_qa_pairs
    
    # 确定namespace（与构建记忆时保持一致）
    if namespace is None:
        namespace = f"locomo_conv_{conversation_idx}"
    
    print(f"\n📊 QA测试配置:")
    print(f"  - Conversation索引: {conversation_idx}")
    print(f"  - 问题数量: {len(test_questions)}")
    print(f"  - Namespace: {namespace}")
    print(f"  - 数据集路径: {dataset_path}")
    
    # 初始化tiktoken编码器
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # 创建token追踪器
    token_tracker = TokenTracker()
    
    # 初始化Config
    config = Config()
    
    # 如果指定了模型，设置模型
    if model:
        config.set_llm_model(model)
        print(f"  - 使用模型: {model}")
    else:
        actual_model = config.llm_model
        print(f"  - 使用模型: {actual_model} (来自配置)")
    
    # 初始化HAmem
    hamem = HAmem(config)
    
    # 获取QA系统并打补丁
    from core.infrastructure import UnifiedCache, EmbeddingManager
    from core.infrastructure.neo4j_client import Neo4jClient
    
    # 创建必要的组件
    embedding_manager = EmbeddingManager(config)
    cache = UnifiedCache(
        cache_dir=config.cache_dir,
        namespace=namespace,
        embedding_manager=embedding_manager
    )
    
    # 获取provider（从config中获取）
    llm_provider = config.llm_provider
    
    # 创建支持token追踪的LLMClient
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
    
    # 打补丁：添加call_type追踪
    qa_system = patch_qa_system_call_types(qa_system)
    
    print(f"\n🔄 开始QA测试（并行数: {parallel}）...")
    
    all_results = []
    total_qa_time = 0
    start_total_time = time.time()
    
    # 并行处理QA对
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # 提交所有任务
        future_to_qa = {
            executor.submit(
                process_single_qa,
                qa_pair,
                qa_system,
                tracking_llm_client,
                encoding,
                i + 1,
                len(test_questions)
            ): (i, qa_pair) for i, qa_pair in enumerate(test_questions)
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_qa):
            completed += 1
            result = future.result()
            all_results.append(result)
            
            # 显示进度
            i = result.get('index', completed)
            question = result.get('question', '')
            elapsed = result.get('elapsed_time', 0)
            total_qa_time += elapsed
            
            print(f"[{completed}/{len(test_questions)}] 问题 {i}: {question[:60]}... (耗时: {elapsed:.2f}s)")
            
            if result.get('evaluation'):
                label = result['evaluation'].get("label", "WRONG")
                score = result['evaluation'].get("score", 0)
                print(f"  → 评估: {label} (得分: {score})")
    
    # 按index排序结果
    all_results.sort(key=lambda x: x.get('index', 0))
    
    total_wall_time = time.time() - start_total_time
    
    # 显示总体统计
    print(f"\n{'='*70}")
    print(f"📊 总体统计")
    print(f"{'='*70}")
    print(f"  - 测试问题数: {len(test_questions)}")
    print(f"  - 总CPU时间: {total_qa_time:.2f} 秒")
    print(f"  - 总墙钟时间: {total_wall_time:.2f} 秒")
    if len(test_questions) > 0:
        print(f"  - 平均CPU时间: {total_qa_time/len(test_questions):.2f} 秒/问题")
        print(f"  - 平均墙钟时间: {total_wall_time/len(test_questions):.2f} 秒/问题")
        print(f"  - 加速比: {total_qa_time/total_wall_time:.2f}x")
    
    # 显示LLM Judge评估统计（总体）
    evaluated_results = [r for r in all_results if r.get('evaluation') is not None]
    if evaluated_results:
        correct_count = sum(1 for r in evaluated_results if r['evaluation'].get('score', 0) == 1)
        total_evaluated = len(evaluated_results)
        accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0
        
        print(f"\n📊 LLM Judge评估统计（总体）:")
        print(f"  - 评估问题数: {total_evaluated}")
        print(f"  - 正确答案数: {correct_count}")
        print(f"  - 准确率: {accuracy:.4f} ({correct_count}/{total_evaluated})")
    
    # 按类别统计正确率
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for result in evaluated_results:
        category = result.get('category')
        if category is not None:
            category_stats[category]['total'] += 1
            if result['evaluation'].get('score', 0) == 1:
                category_stats[category]['correct'] += 1
    
    if category_stats:
        print(f"\n📊 LLM Judge评估统计（按类别）:")
        print(f"  {'类别':<10} {'正确数':<10} {'总数':<10} {'准确率':<10}")
        print(f"  {'-'*40}")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"  {category:<10} {stats['correct']:<10} {stats['total']:<10} {accuracy:.4f}")
    
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
    
    # 准备返回结果
    result_data = {
        'conversation_idx': conversation_idx,
        'namespace': namespace,
        'model': model or config.llm_model,
        'total_questions': len(test_questions),
        'total_time': total_qa_time,
        'total_wall_time': total_wall_time,
        'results': all_results,
        'token_stats': token_stats,
        'evaluation_summary': {
            'total_evaluated': len(evaluated_results),
            'correct_count': correct_count if evaluated_results else 0,
            'accuracy': accuracy if evaluated_results else 0.0,
            'category_stats': dict(category_stats) if category_stats else {}
        }
    }
    
    # 保存结果到JSON文件
    output_dir = os.path.join(project_root, "experiment", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"qa_evaluation_conv_{conversation_idx}_{int(time.time())}.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 评估结果已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️  保存结果失败: {e}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="测试QA系统并统计token数量和时延",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个conversation的所有问题（使用默认模型）
  python test_qa.py 1
  
  # 测试指定问题，使用指定模型
  python test_qa.py 1 --question-idx 0 --model gpt-4o-mini
  
  # 测试所有conversation的QA
  python test_qa.py --all-conversations --model gpt-4o-mini
  
  # 指定数据集路径和模型
  python test_qa.py 1 --dataset /path/to/locomo10.json --model deepseek-chat
  
  # 指定namespace
  python test_qa.py 1 --namespace custom_namespace
  
  # 测试所有conversation，指定并行数
  python test_qa.py --all-conversations --parallel 30
        """
    )
    
    parser.add_argument(
        "conversation_idx",
        type=int,
        nargs='?',
        default=None,
        help="conversation索引（从0开始），如果使用--all-conversations则不需要指定"
    )
    
    parser.add_argument(
        "--question-idx", "-q",
        type=int,
        default=None,
        help="问题索引（如果未指定，测试所有问题）"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="数据集路径（默认: <project_root>/locomo/data/locomo10.json）"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM模型名称，如 gpt-4o-mini, deepseek-chat 等（如果未指定，使用Config中的默认值）"
    )
    
    parser.add_argument(
        "--namespace", "-n",
        type=str,
        default=None,
        help="命名空间（默认: locomo_conv_<conversation_idx>）"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=50,
        help="并行处理数量（默认: 50）"
    )
    
    parser.add_argument(
        "--all-conversations", "-a",
        action="store_true",
        help="测试所有conversation的QA（如果指定此选项，不需要指定conversation_idx）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.all_conversations and args.conversation_idx is None:
        parser.error("必须指定conversation_idx或使用--all-conversations选项")
    
    if args.all_conversations and args.conversation_idx is not None:
        print("⚠️  警告: 指定了--all-conversations，将忽略conversation_idx参数")
    
    test_qa(
        conversation_idx=args.conversation_idx,
        question_idx=args.question_idx,
        dataset_path=args.dataset,
        model=args.model,
        namespace=args.namespace,
        parallel=args.parallel,
        all_conversations=args.all_conversations
    )


if __name__ == "__main__":
    main()

