"""
并行运行locomo数据集的记忆构建和QA测试

支持并行处理多个conversation（0-9），每个conversation使用独立的namespace，互不干扰。
"""

import sys
import os
import json
import time
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import StringIO

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# 线程本地存储，用于存储每个线程的日志记录器
_thread_local = threading.local()


def setup_logging(log_dir: str, conversation_idx: int = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        conversation_idx: conversation索引，如果为None则创建主日志
    
    Returns:
        Logger对象
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if conversation_idx is not None:
        log_file = os.path.join(log_dir, f"conv_{conversation_idx}.log")
        logger_name = f"conv_{conversation_idx}"
    else:
        log_file = os.path.join(log_dir, "main.log")
        logger_name = "main"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 创建一个自定义的handler，每次写入后立即flush，确保日志及时保存
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # 只使用文件handler，不输出到控制台
    flushing_handler = FlushingFileHandler(log_file, encoding='utf-8')
    flushing_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    flushing_handler.setFormatter(file_formatter)
    logger.addHandler(flushing_handler)
    
    return logger


class ThreadLocalStdout:
    """线程本地stdout包装器，每个线程有独立的输出缓冲区"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_logger(self, logger: logging.Logger):
        """为当前线程设置日志记录器"""
        self._local.logger = logger
        self._local.buffer = StringIO()
    
    def write(self, text: str):
        """写入到当前线程的缓冲区"""
        if hasattr(self._local, 'buffer'):
            self._local.buffer.write(text)
        else:
            # 如果没有设置logger，回退到原始stdout
            sys.__stdout__.write(text)
    
    def flush(self):
        """刷新当前线程的缓冲区到日志"""
        if hasattr(self._local, 'buffer') and hasattr(self._local, 'logger'):
            buffer_content = self._local.buffer.getvalue()
            if buffer_content:
                self._local.logger.info(buffer_content.rstrip())
            self._local.buffer = StringIO()  # 重置缓冲区
    
    def get_buffer(self) -> str:
        """获取当前线程的缓冲区内容"""
        if hasattr(self._local, 'buffer'):
            return self._local.buffer.getvalue()
        return ""
    
    def clear_buffer(self):
        """清空当前线程的缓冲区"""
        if hasattr(self._local, 'buffer'):
            self._local.buffer = StringIO()


# 全局的线程本地stdout实例
_thread_stdout = ThreadLocalStdout()


def run_single_conversation(
    conversation_idx: int,
    dataset_path: str,
    model: Optional[str],
    log_dir: str,
    skip_qa: bool = False
) -> Dict[str, Any]:
    """
    运行单个conversation的记忆构建和QA测试
    
    Args:
        conversation_idx: conversation索引
        dataset_path: 数据集路径
        model: LLM模型名称
        log_dir: 日志目录
        skip_qa: 是否跳过QA测试
    
    Returns:
        包含结果的字典
    """
    # 延迟导入，避免在子线程中导入问题
    from experiment.test_memory_building import test_memory_building
    from experiment.test_qa import test_qa
    
    logger = setup_logging(log_dir, conversation_idx)
    namespace = f"locomo_conv_{conversation_idx}"
    
    # 为当前线程设置独立的stdout重定向
    _thread_stdout.set_logger(logger)
    old_stdout = sys.stdout
    sys.stdout = _thread_stdout
    
    result = {
        'conversation_idx': conversation_idx,
        'namespace': namespace,
        'memory_building': None,
        'qa': None,
        'success': False,
        'error': None
    }
    
    try:
        logger.info(f"{'='*70}")
        logger.info(f"开始处理 Conversation {conversation_idx} (Namespace: {namespace})")
        logger.info(f"{'='*70}")
        
        # 步骤1: 记忆构建
        logger.info(f"\n📝 步骤1: 记忆构建")
        logger.info(f"Namespace: {namespace}")
        
        memory_start_time = time.time()
        memory_result = test_memory_building(
            conversation_idx=conversation_idx,
            dataset_path=dataset_path,
            model=model,
            skip_storage=False
        )
        memory_end_time = time.time()
        memory_time = memory_end_time - memory_start_time
        
        # 刷新当前线程的stdout缓冲区到日志
        _thread_stdout.flush()
        
        if memory_result:
            result['memory_building'] = {
                'success': True,
                'time': memory_time,
                'original_tokens': memory_result.get('original_tokens', 0),
                'total_time': memory_result.get('total_time', 0),
                'token_stats': memory_result.get('token_stats', {})
            }
            logger.info(f"✅ 记忆构建完成，耗时: {memory_time:.2f}秒")
        else:
            result['memory_building'] = {
                'success': False,
                'time': memory_time
            }
            logger.error(f"❌ 记忆构建失败")
            result['error'] = "记忆构建失败"
            return result
        
        # 步骤2: QA测试（如果未跳过）
        if not skip_qa:
            logger.info(f"\n❓ 步骤2: QA测试")
            logger.info(f"Namespace: {namespace}")
            
            qa_start_time = time.time()
            qa_result = test_qa(
                conversation_idx=conversation_idx,
                question_idx=None,  # 测试所有问题
                dataset_path=dataset_path,
                model=model,
                namespace=namespace
            )
            qa_end_time = time.time()
            qa_time = qa_end_time - qa_start_time
            
            # 刷新当前线程的stdout缓冲区到日志
            _thread_stdout.flush()
            
            if qa_result:
                result['qa'] = {
                    'success': True,
                    'time': qa_time,
                    'total_time': qa_result.get('total_time', 0),
                    'token_stats': qa_result.get('token_stats', {}),
                    'results_count': len(qa_result.get('results', []))
                }
                logger.info(f"✅ QA测试完成，耗时: {qa_time:.2f}秒")
            else:
                result['qa'] = {
                    'success': False,
                    'time': qa_time
                }
                logger.error(f"❌ QA测试失败")
        
        result['success'] = True
        total_time = memory_time + (qa_time if not skip_qa else 0)
        logger.info(f"\n{'='*70}")
        logger.info(f"Conversation {conversation_idx} 处理完成，总耗时: {total_time:.2f}秒")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"❌ 处理Conversation {conversation_idx}时发生错误: {e}", exc_info=True)
        result['error'] = str(e)
        result['success'] = False
    finally:
        # 恢复原始stdout
        sys.stdout = old_stdout
    
    return result


def run_parallel_experiment(
    conversation_indices: List[int],
    dataset_path: str,
    model: Optional[str],
    log_dir: str,
    max_workers: int = 3,
    skip_qa: bool = False
) -> Dict[str, Any]:
    """
    并行运行多个conversation的实验
    
    Args:
        conversation_indices: conversation索引列表
        dataset_path: 数据集路径
        model: LLM模型名称
        log_dir: 日志目录
        max_workers: 最大并行数
        skip_qa: 是否跳过QA测试
    
    Returns:
        包含所有结果的字典
    """
    # 设置主日志记录器（不输出到控制台）
    main_logger = setup_logging(log_dir)
    
    main_logger.info(f"\n{'='*70}")
    main_logger.info(f"🚀 开始并行实验")
    main_logger.info(f"{'='*70}")
    main_logger.info(f"  - Conversation数量: {len(conversation_indices)}")
    main_logger.info(f"  - Conversation索引: {conversation_indices}")
    main_logger.info(f"  - 最大并行数: {max_workers}")
    main_logger.info(f"  - 数据集路径: {dataset_path}")
    main_logger.info(f"  - 模型: {model or '使用Config默认值'}")
    main_logger.info(f"  - 日志目录: {log_dir}")
    main_logger.info(f"  - 跳过QA: {skip_qa}")
    main_logger.info(f"{'='*70}\n")
    
    start_time = time.time()
    results = {}
    
    # 使用ThreadPoolExecutor进行并行处理（IO密集型任务）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                run_single_conversation,
                idx,
                dataset_path,
                model,
                log_dir,
                skip_qa
            ): idx
            for idx in conversation_indices
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                completed += 1
                
                status = "✅" if result['success'] else "❌"
                main_logger.info(f"{status} Conversation {idx} 完成 ({completed}/{len(conversation_indices)})")
                
            except Exception as e:
                main_logger.error(f"❌ Conversation {idx} 执行失败: {e}", exc_info=True)
                results[idx] = {
                    'conversation_idx': idx,
                    'success': False,
                    'error': str(e)
                }
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 生成汇总报告
    main_logger.info(f"\n{'='*70}")
    main_logger.info(f"📊 实验汇总")
    main_logger.info(f"{'='*70}")
    main_logger.info(f"  - 总耗时: {total_time:.2f}秒")
    main_logger.info(f"  - 平均耗时: {total_time/len(conversation_indices):.2f}秒/conversation")
    
    success_count = sum(1 for r in results.values() if r.get('success', False))
    main_logger.info(f"  - 成功: {success_count}/{len(conversation_indices)}")
    main_logger.info(f"  - 失败: {len(conversation_indices) - success_count}/{len(conversation_indices)}")
    
    if not skip_qa:
        # 统计token使用
        total_memory_tokens = 0
        total_qa_tokens = 0
        
        for idx, result in results.items():
            if result.get('memory_building', {}).get('token_stats'):
                for stats in result['memory_building']['token_stats'].values():
                    if isinstance(stats, dict):
                        total_memory_tokens += stats.get('total_tokens', 0)
            
            if result.get('qa', {}).get('token_stats'):
                for stats in result['qa']['token_stats'].values():
                    if isinstance(stats, dict):
                        total_qa_tokens += stats.get('total_tokens', 0)
        
        main_logger.info(f"\n📊 Token统计:")
        main_logger.info(f"  - 记忆构建总tokens: {total_memory_tokens:,}")
        main_logger.info(f"  - QA测试总tokens: {total_qa_tokens:,}")
        main_logger.info(f"  - 总计: {total_memory_tokens + total_qa_tokens:,}")
    
    main_logger.info(f"{'='*70}\n")
    
    # 保存汇总结果到JSON
    summary_file = os.path.join(log_dir, 'summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time': total_time,
        'conversation_indices': conversation_indices,
        'model': model,
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    main_logger.info(f"📄 汇总结果已保存到: {summary_file}")
    main_logger.info(f"📁 详细日志请查看: {log_dir}\n")
    
    return {
        'total_time': total_time,
        'results': results,
        'summary_file': summary_file
    }


def main():
    parser = argparse.ArgumentParser(
        description="并行运行locomo数据集的记忆构建和QA测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有conversation (0-9)
  python experiment/run_locomo_experiment.py
  
  # 运行指定范围的conversation
  python experiment/run_locomo_experiment.py --start 0 --end 4
  
  # 指定模型和并行数
  python experiment/run_locomo_experiment.py --model gpt-4o-mini --max-workers 5
  
  # 只运行记忆构建，跳过QA
  python experiment/run_locomo_experiment.py --skip-qa
        """
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="起始conversation索引（默认: 0）"
    )
    
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=9,
        help="结束conversation索引（默认: 9）"
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
        "--max-workers", "-w",
        type=int,
        default=3,
        help="最大并行数（默认: 3）"
    )
    
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="跳过QA测试，只运行记忆构建"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="日志目录（默认: experiment/logs）"
    )
    
    args = parser.parse_args()
    
    # 确定数据集路径
    if args.dataset is None:
        args.dataset = os.path.join(
            project_root,
            "locomo", "data", "locomo10.json"
        )
    
    # 检查文件是否存在
    if not os.path.exists(args.dataset):
        # 只在错误时输出到stderr（这样nohup可以捕获）
        sys.stderr.write(f"❌ 错误: 数据集文件不存在: {args.dataset}\n")
        sys.stderr.write(f"   请使用 --dataset 参数指定正确的数据集路径\n")
        sys.exit(1)
    
    # 确定日志目录
    if args.log_dir is None:
        args.log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs"
        )
    
    # 生成conversation索引列表
    conversation_indices = list(range(args.start, args.end + 1))
    
    # 运行实验
    run_parallel_experiment(
        conversation_indices=conversation_indices,
        dataset_path=args.dataset,
        model=args.model,
        log_dir=args.log_dir,
        max_workers=args.max_workers,
        skip_qa=args.skip_qa
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 捕获Ctrl+C，写入日志后退出
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        main_logger = setup_logging(log_dir)
        main_logger.warning("用户中断实验")
        sys.exit(130)
    except Exception as e:
        # 捕获其他异常，写入日志
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        main_logger = setup_logging(log_dir)
        main_logger.error(f"实验执行失败: {e}", exc_info=True)
        # 错误信息也输出到stderr，方便nohup捕获
        sys.stderr.write(f"❌ 实验执行失败: {e}\n")
        sys.stderr.write(f"📁 详细错误信息请查看日志: {log_dir}/main.log\n")
        sys.exit(1)

