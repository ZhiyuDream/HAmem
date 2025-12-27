"""
测试记忆召回功能

使用 HAmem 的 search_memory 接口，用一句话召回相关记忆。

使用方法:
    python examples/test_recall.py "你的查询语句" --namespace "your_namespace"

示例:
    python examples/test_recall.py "我研究的是大模型记忆方向" --namespace "default"
    python examples/test_recall.py "记忆机制" --namespace "locomo_conv_0"
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.main import HAmem


def format_search_result(result: Dict[str, Any], index: int) -> str:
    """格式化单个搜索结果"""
    node_id = result.get('id', 'unknown')
    node_type = result.get('type', 'unknown')
    layer = result.get('layer', 'unknown')
    content = result.get('content', '')
    
    # 截断过长的内容
    if len(content) > 200:
        content = content[:200] + "..."
    
    return f"""
[{index}] {node_id}
  - Type: {node_type}
  - Layer: {layer}
  - Content: {content}
"""


def test_recall(query: str, namespace: str = "default", top_k: int = 10):
    """
    测试记忆召回
    
    Args:
        query: 查询语句
        namespace: 命名空间
        top_k: 返回结果数量
    """
    print("=" * 70)
    print("🔍 HAmem 记忆召回测试")
    print("=" * 70)
    print(f"查询: {query}")
    print(f"命名空间: {namespace}")
    print(f"Top-K: {top_k}")
    print("=" * 70)
    
    try:
        # 初始化配置
        config = Config()
        config.validate()
        
        # 初始化 HAmem
        print("\n📦 初始化 HAmem...")
        hamem = HAmem(config)
        print("✅ HAmem 初始化成功")
        
        # 执行召回
        print(f"\n🔍 开始召回记忆...")
        results = hamem.search_memory(query, top_k=top_k, namespace=namespace)
        
        # 显示结果
        print(f"\n✅ 召回完成，共找到 {len(results)} 条结果")
        print("=" * 70)
        
        if results:
            print("\n📋 召回结果:")
            for i, result in enumerate(results, 1):
                print(format_search_result(result, i))
        else:
            print("\n⚠️  未找到相关记忆")
            print(f"   提示: 请确认命名空间 '{namespace}' 中是否有已构建的记忆")
        
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="测试 HAmem 记忆召回功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认命名空间
  python examples/test_recall.py "我研究的是大模型记忆方向"
  
  # 指定命名空间
  python examples/test_recall.py "记忆机制" --namespace "locomo_conv_0"
  
  # 指定返回数量
  python examples/test_recall.py "记忆机制" --namespace "default" --top-k 5
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="查询语句"
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="命名空间（默认: default）"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        help="返回结果数量（默认: 10）"
    )
    
    args = parser.parse_args()
    
    # 执行召回测试
    test_recall(args.query, args.namespace, args.top_k)


if __name__ == "__main__":
    main()

