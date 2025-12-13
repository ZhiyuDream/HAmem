"""
测试 locomo10.json 数据集

1. 加载数据集
2. 选择一个conversation进行测试
3. 构建记忆并分析时延
4. 验证数据是否存入Neo4j
5. 测试检索功能
6. 测试QA功能
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import HAmem
from core.infrastructure.neo4j_client import Neo4jClient


def load_locomo_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载locomo数据集"""
    print(f"📂 加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成，共 {len(data)} 个conversation")
    return data


def convert_conversation_to_hamem_format(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将locomo格式的conversation转换为HAmem格式
    
    locomo格式:
    {
        "conversation": {
            "speaker_a": "...",
            "speaker_b": "...",
            "session_1_date_time": "...",
            "session_1": [
                {"speaker": "...", "dia_id": "...", "text": "..."},
                ...
            ],
            ...
        },
        "qa": [...]
    }
    
    HAmem格式需要:
    {
        "messages": [
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "..."},
            ...
        ],
        "metadata": {...}
    }
    """
    conversation = conversation_data.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "User")
    speaker_b = conversation.get("speaker_b", "Assistant")
    
    messages = []
    
    # 遍历所有session
    session_keys = [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time") and not k.endswith("_summary")]
    session_keys.sort()  # 按顺序处理
    
    for session_key in session_keys:
        session = conversation.get(session_key, [])
        if not isinstance(session, list):
            continue
        
        # 获取session的时间信息（这是数据集提供的真实时间）
        date_time_key = f"{session_key}_date_time"
        session_time = conversation.get(date_time_key, "")
        
        # 如果没有session_time，尝试使用默认值
        if not session_time:
            # 尝试从其他session获取时间作为fallback
            session_time = "unknown"
        
        for turn in session:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            
            if not text:
                continue
            
            # 确定role
            if speaker == speaker_a:
                role = "user"
            elif speaker == speaker_b:
                role = "assistant"
            else:
                role = "user"  # 默认
            
            # 使用session_time作为timestamp，这样session变化时会自动切分fragment
            # 如果session_time相同，则使用递增的数值来区分同一session内的消息
            messages.append({
                "role": role,
                "content": text,
                "timestamp": session_time,  # 使用session_time作为timestamp，确保session变化时自动切分
                "metadata": {
                    "speaker": speaker,
                    "dia_id": dia_id,
                    "session": session_key,
                    "session_time": session_time
                }
            })
    
    return {
        "messages": messages,
        "metadata": {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "source": "locomo",
            "total_sessions": len(session_keys)
        }
    }


def test_memory_building(hamem: HAmem, conversation_data: Dict[str, Any], conversation_idx: int):
    """测试记忆构建并分析时延"""
    print("\n" + "="*60)
    print(f"🧠 测试记忆构建 - Conversation {conversation_idx}")
    print("="*60)
    
    # 转换数据格式
    print("\n📝 转换数据格式...")
    hamem_format = convert_conversation_to_hamem_format(conversation_data)
    print(f"✅ 转换完成: {len(hamem_format['messages'])} 条消息")
    
    # 记录开始时间
    start_time = time.time()
    
    # 构建记忆（使用conversation索引作为namespace，对应不同的Neo4j database）
    namespace = f"locomo{conversation_idx}"
    print(f"\n🔨 开始构建记忆 (namespace: {namespace}, database: {namespace})...")
    try:
        result = hamem.build_memory(hamem_format, namespace=namespace)
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n✅ 记忆构建完成!")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        print(f"📊 处理了 {result.get('total_fragments', 0)} 个fragments")
        
        # 分析时延
        print(f"\n📈 性能分析:")
        print(f"  - 平均每条消息耗时: {elapsed_time / len(hamem_format['messages']):.3f} 秒")
        print(f"  - 平均每个fragment耗时: {elapsed_time / max(result.get('total_fragments', 1), 1):.3f} 秒")
        
        # 显示详细的时间统计
        time_stats = result.get('time_stats', {})
        if time_stats:
            print(f"\n⏱️  详细时间统计:")
            if time_stats.get('fragment_processing'):
                frag_times = time_stats['fragment_processing']
                print(f"  - Fragment处理时间:")
                print(f"    * 总时间: {sum(frag_times):.2f}秒")
                print(f"    * 平均: {sum(frag_times)/len(frag_times):.3f}秒/个")
                print(f"    * 最大: {max(frag_times):.3f}秒")
                print(f"    * 最小: {min(frag_times):.3f}秒")
            
            if time_stats.get('layer1_processing'):
                layer1_times = time_stats['layer1_processing']
                print(f"  - Layer1处理时间:")
                print(f"    * 总时间: {sum(layer1_times):.2f}秒")
                print(f"    * 平均: {sum(layer1_times)/len(layer1_times):.3f}秒/个")
            
            if time_stats.get('layer2_processing'):
                layer2_times = time_stats['layer2_processing']
                print(f"  - Layer2处理时间:")
                print(f"    * 总时间: {sum(layer2_times):.2f}秒")
                print(f"    * 平均: {sum(layer2_times)/len(layer2_times):.3f}秒/个")
            
            if time_stats.get('layer3_processing'):
                layer3_times = time_stats['layer3_processing']
                print(f"  - Layer3处理时间:")
                print(f"    * 总时间: {sum(layer3_times):.2f}秒")
                print(f"    * 平均: {sum(layer3_times)/len(layer3_times):.3f}秒/个")
        
        return result, elapsed_time
        
    except Exception as e:
        print(f"❌ 记忆构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def verify_neo4j_storage(config: Config, namespace: str = "default"):
    """验证数据是否存入Neo4j"""
    print("\n" + "="*60)
    print(f"🔍 验证Neo4j存储 (namespace: {namespace})")
    print("="*60)
    
    if not config.use_neo4j:
        print("⚠️  Neo4j未启用，跳过验证")
        return
    
    try:
        neo4j_client = Neo4jClient(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database  # 使用默认数据库
        )
        
        if not neo4j_client.connect():
            print("❌ Neo4j连接失败")
            return
        
        # 查询统计信息
        stats = neo4j_client.get_stats()
        
        print(f"\n📊 Neo4j存储统计 (namespace: {namespace}):")
        print(f"  - 节点总数: {stats.get('node_count', 0)}")
        print(f"  - 关系总数: {stats.get('relationship_count', 0)}")
        
        # 按namespace和类型统计节点
        node_type_query = """
        MATCH (n)
        WHERE n.namespace = $namespace
        RETURN labels(n) as labels, count(*) as count
        ORDER BY count DESC
        """
        node_type_results = neo4j_client.execute_read(node_type_query, {'namespace': namespace})
        if node_type_results:
            print(f"\n📋 节点类型分布 (namespace: {namespace}):")
            for record in node_type_results:
                labels = record.get('labels', [])
                label_str = ':'.join(labels) if labels else 'Unknown'
                count = record.get('count', 0)
                print(f"  - {label_str}: {count}")
        
        # 按namespace和类型统计关系
        rel_type_query = """
        MATCH ()-[r]->()
        WHERE r.namespace = $namespace
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        """
        rel_type_results = neo4j_client.execute_read(rel_type_query, {'namespace': namespace})
        if rel_type_results:
            print(f"\n🔗 关系类型分布 (namespace: {namespace}):")
            for record in rel_type_results:
                rel_type = record.get('rel_type', 'Unknown')
                count = record.get('count', 0)
                print(f"  - {rel_type}: {count}")
        
        neo4j_client.close()
        print("\n✅ Neo4j验证完成")
        
    except Exception as e:
        print(f"❌ Neo4j验证失败: {e}")
        import traceback
        traceback.print_exc()


def test_retrieval(hamem: HAmem, test_queries: List[str], namespace: str = "default"):
    """测试检索功能"""
    print("\n" + "="*60)
    print("🔍 测试检索功能")
    print("="*60)
    
    for query in test_queries:
        print(f"\n❓ 查询: {query}")
        start_time = time.time()
        
        try:
            results = hamem.search_memory(query, top_k=5, namespace=namespace)
            elapsed_time = time.time() - start_time
            
            print(f"⏱️  耗时: {elapsed_time:.3f} 秒")
            print(f"📊 找到 {len(results)} 个结果")
            
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result.get('id', 'unknown')}: {str(result.get('content', ''))[:100]}...")
                
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            import traceback
            traceback.print_exc()


def test_qa(hamem: HAmem, qa_pairs: List[Dict[str, Any]], namespace: str = "default", max_questions: int = 3):
    """
    测试QA功能
    
    Args:
        hamem: HAmem实例
        qa_pairs: QA对列表
        namespace: 命名空间（必须与构建记忆时使用的namespace一致）
        max_questions: 最大测试问题数
    """
    print("\n" + "="*60)
    print(f"❓ 测试QA功能 (namespace: {namespace})")
    print("="*60)
    
    test_qa = qa_pairs[:max_questions]
    
    for i, qa in enumerate(test_qa, 1):
        question = qa.get("question", "")
        expected_answer = qa.get("answer", "")
        
        print(f"\n{'='*60}")
        print(f"问题 {i}/{len(test_qa)}: {question}")
        print(f"期望答案: {expected_answer}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            answer_result = hamem.ask_question(question, namespace=namespace)
            elapsed_time = time.time() - start_time
            
            print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
            print(f"📝 生成答案: {answer_result.get('answer', 'N/A')}")
            print(f"💭 推理过程: {answer_result.get('reason', 'N/A')}")
            
            # 显示统计信息
            stats = answer_result.get('stats', {})
            if stats:
                print(f"\n📊 统计信息:")
                print(f"  - LLM调用次数: {stats.get('llm_calls', 0)}")
                print(f"  - 召回节点数: {stats.get('recalled_nodes', 0)}")
                print(f"  - 扩展节点数: {stats.get('expanded_nodes', 0)}")
                print(f"  - 扩展跳数: {stats.get('hops', 0)}")
            
        except Exception as e:
            print(f"❌ QA失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("="*60)
    print("🚀 Locomo数据集测试")
    print("="*60)
    
    # 解析命令行参数
    # 用法: python test_locomo_dataset.py [conversation_idx] [mode]
    # mode: build (只构建), qa (只测试QA), all (全部，默认)
    conversation_idx = 0
    mode = "all"
    
    if len(sys.argv) > 1:
        conversation_idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
    
    # 1. 加载配置
    print("\n📝 加载配置...")
    config = Config.from_env()
    config.validate()
    print("✅ 配置加载完成")
    
    # 2. 加载数据集
    dataset_path = "/home/zhiyu_zheng/DCL/Others/locomo/data/locomo10.json"
    dataset = load_locomo_dataset(dataset_path)
    
    if conversation_idx >= len(dataset):
        print(f"❌ Conversation索引 {conversation_idx} 超出范围（共 {len(dataset)} 个）")
        return
    
    conversation_data = dataset[conversation_idx]
    namespace = f"locomo{conversation_idx}"
    print(f"\n✅ 选择Conversation {conversation_idx} (namespace: {namespace})")
    print(f"📋 测试模式: {mode}")
    
    # 4. 初始化HAmem系统
    print("\n📝 初始化HAmem系统...")
    try:
        hamem = HAmem(config)
        print("✅ HAmem初始化完成")
    except Exception as e:
        print(f"❌ HAmem初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 根据模式执行不同的测试
    if mode in ["build", "all"]:
        # 5. 测试记忆构建
        print("\n" + "="*60)
        print("🧠 模式: 构建记忆")
        print("="*60)
        result, elapsed_time = test_memory_building(hamem, conversation_data, conversation_idx)
        
        if result is None:
            print("❌ 记忆构建失败，终止测试")
            return
        
        # 6. 验证Neo4j存储
        verify_neo4j_storage(config, namespace=namespace)
    
    if mode in ["qa", "all"]:
        # 7. 测试检索功能
        print("\n" + "="*60)
        print("🔍 模式: 测试检索和QA")
        print("="*60)
        test_queries = [
            "Caroline",
            "Melanie",
            "LGBTQ support group",
            "painting"
        ]
        test_retrieval(hamem, test_queries, namespace=namespace)
        
        # 8. 测试QA功能（使用正确的namespace）
        qa_pairs = conversation_data.get("qa", [])
        if qa_pairs:
            print(f"\n📋 找到 {len(qa_pairs)} 个QA对，测试前3个")
            test_qa(hamem, qa_pairs, namespace=namespace, max_questions=3)
        else:
            print("\n⚠️  未找到QA对，跳过QA测试")
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()

