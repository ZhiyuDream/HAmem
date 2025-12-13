"""
测试 Neo4j 混合检索（FAISS + Neo4j）

验证结合现有优化和 Neo4j 图扩展的混合架构
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(Path(__file__).parent, '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from config import Config
from core.infrastructure.neo4j_client import Neo4jClient
from core.infrastructure.cache import UnifiedCache
from core.infrastructure.embedding import EmbeddingManager
from core.infrastructure.neo4j_hybrid_search import Neo4jHybridSearch
from core.search.neo4j_hybrid_recall import Neo4jHybridRecall


def test_hybrid_search():
    """测试混合检索"""
    print("=" * 60)
    print("🧪 测试 Neo4j 混合检索（FAISS + Neo4j）")
    print("=" * 60)
    
    # 初始化
    config = Config()
    if not config.openai_api_key:
        print("❌ OpenAI API Key 未配置")
        return False
    
    # 创建客户端
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_hybrid"
    
    # 创建 UnifiedCache（利用现有优化）
    embedding_manager = EmbeddingManager(config)
    cache = UnifiedCache(
        cache_dir="cache",
        namespace=namespace,
        embedding_manager=embedding_manager
    )
    
    print("\n📝 步骤 1: 使用 UnifiedCache 批量创建节点（利用现有优化）")
    print("-" * 60)
    
    # 批量创建节点（利用 UnifiedCache 的批量优化）
    test_nodes = [
        {
            'id': 'node_ai',
            'type': 'entity',
            'name': 'AI Research',
            'content': 'Artificial Intelligence and machine learning research',
            'layer': 1,
            'active': True
        },
        {
            'id': 'node_nlp',
            'type': 'entity',
            'name': 'NLP',
            'content': 'Natural Language Processing and text understanding',
            'layer': 1,
            'active': True
        },
        {
            'id': 'node_graph',
            'type': 'entity',
            'name': 'Graph Database',
            'content': 'Graph databases and knowledge graphs for data storage',
            'layer': 1,
            'active': True
        },
        {
            'id': 'event_1',
            'type': 'event',
            'content': 'AI research team published a paper on transformer models',
            'layer': 2,
            'active': True
        },
        {
            'id': 'event_2',
            'type': 'event',
            'content': 'NLP team developed a new language model',
            'layer': 2,
            'active': True
        }
    ]
    
    # 批量添加节点（自动批量生成 embedding，利用去重）
    cache.batch_add_nodes(test_nodes)
    print(f"✅ 批量添加了 {len(test_nodes)} 个节点")
    print(f"   实际生成 embedding 数: {len(cache.embeddings)} (去重优化)")
    
    # 创建关系
    test_edges = [
        {
            'id': 'edge_1',
            'source': 'node_ai',
            'target': 'event_1',
            'type': 'RELATED_TO',
            'content': 'AI research leads to paper publication',
            'layer': 1,
            'active': True
        },
        {
            'id': 'edge_2',
            'source': 'node_nlp',
            'target': 'event_2',
            'type': 'RELATED_TO',
            'content': 'NLP team creates language model',
            'layer': 1,
            'active': True
        },
        {
            'id': 'edge_3',
            'source': 'node_ai',
            'target': 'node_nlp',
            'type': 'RELATED_TO',
            'content': 'AI and NLP are related fields',
            'layer': 1,
            'active': True
        }
    ]
    
    cache.batch_add_edges(test_edges)
    print(f"✅ 批量添加了 {len(test_edges)} 个关系")
    
    print("\n🔄 步骤 2: 同步数据到 Neo4j")
    print("-" * 60)
    
    # 创建混合搜索实例
    hybrid_search = Neo4jHybridSearch(cache, client, namespace)
    
    # 同步数据到 Neo4j（批量操作）
    sync_stats = hybrid_search.sync_cache_to_neo4j(
        batch_size=50,
        update_embeddings=True
    )
    
    print(f"✅ 同步完成:")
    print(f"   - 节点数: {sync_stats['nodes_synced']}")
    print(f"   - 边数: {sync_stats['edges_synced']}")
    print(f"   - Embedding 数: {sync_stats['embeddings_synced']}")
    if sync_stats['errors']:
        print(f"   - 错误数: {len(sync_stats['errors'])}")
    
    print("\n🔍 步骤 3: 混合检索（FAISS 向量搜索 + Neo4j 图扩展）")
    print("-" * 60)
    
    query = "machine learning and artificial intelligence"
    print(f"查询: '{query}'")
    
    # 混合搜索
    result = hybrid_search.hybrid_search(
        query=query,
        vector_top_k=3,
        max_hops=2,
        expand_limit=20,
        layer=1,  # 只搜索 Layer1
        similarity_threshold=0.0
    )
    
    print(f"\n✅ 混合搜索结果:")
    print(f"   - 搜索方法: {result['search_method']}")
    print(f"   - 初始节点数: {len(result['initial_nodes'])}")
    print(f"   - 扩展节点数: {len(result['expanded_nodes'])}")
    print(f"   - 总节点数: {result['total_nodes']}")
    
    print(f"\n初始节点（FAISS 找到）:")
    for i, node in enumerate(result['initial_nodes'][:5], 1):
        name = node.get('name', node.get('id', 'Unknown'))
        score = node.get('similarity_score', 0.0)
        print(f"   {i}. {name} (相似度: {score:.4f})")
    
    print(f"\n扩展节点（Neo4j 图扩展）:")
    for i, node in enumerate(result['expanded_nodes'][:5], 1):
        name = node.get('name', node.get('id', 'Unknown'))
        hops = node.get('hops', 0)
        print(f"   {i}. {name} (跳数: {hops})")
    
    print("\n🔍 步骤 4: 多层混合检索")
    print("-" * 60)
    
    # 创建混合召回实例
    hybrid_recall = Neo4jHybridRecall(cache, client, namespace)
    
    multi_result = hybrid_recall.multi_layer_recall_with_expansion(
        query=query,
        layer1_top_k=5,
        layer2_top_k=5,
        layer3_top_k=3,
        max_hops=1,
        expand_limit=10
    )
    
    for layer, layer_data in multi_result.items():
        print(f"\n{layer.upper()}:")
        print(f"   - 初始节点: {len(layer_data['initial_nodes'])}")
        print(f"   - 扩展节点: {len(layer_data['expanded_nodes'])}")
        print(f"   - 总节点: {layer_data['total_nodes']}")
    
    # 保存缓存
    cache.save()
    
    client.close()
    print("\n✅ 混合检索测试完成！\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 开始测试 Neo4j 混合检索（FAISS + Neo4j）")
    print("=" * 60 + "\n")
    
    success = test_hybrid_search()
    
    if success:
        print("=" * 60)
        print("✅ 测试通过！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 测试失败")
        print("=" * 60)

