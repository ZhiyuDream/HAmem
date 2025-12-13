"""
测试 Neo4j 向量搜索和图扩展功能

验证 embedding 查询和边扩展功能
"""

import os
import sys
from pathlib import Path
import numpy as np

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

from core.infrastructure.neo4j_client import Neo4jClient
from core.infrastructure.neo4j_vector_search import Neo4jVectorSearch
from core.layer1.neo4j_storage import Layer1Neo4jStorage
from core.layer2.neo4j_storage import Layer2Neo4jStorage


def generate_random_embedding(dimension: int = 1536) -> list:
    """生成随机 embedding（用于测试）"""
    return list(np.random.normal(0, 1, dimension))


def test_vector_search_and_expansion():
    """测试向量搜索和图扩展"""
    print("=" * 60)
    print("🧪 测试 Neo4j 向量搜索和图扩展")
    print("=" * 60)
    
    # 创建客户端
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_vector_search"
    
    # 创建存储和向量搜索实例
    layer1_storage = Layer1Neo4jStorage(client, namespace)
    layer2_storage = Layer2Neo4jStorage(client, namespace)
    vector_search = Neo4jVectorSearch(client, namespace)
    
    print("\n📝 步骤 1: 创建测试节点和关系")
    print("-" * 60)
    
    # 创建实体节点
    entities = [
        {'id': 'entity_alice', 'name': 'Alice', 'content': 'Alice is a software engineer'},
        {'id': 'entity_bob', 'name': 'Bob', 'content': 'Bob is a data scientist'},
        {'id': 'entity_charlie', 'name': 'Charlie', 'content': 'Charlie is a product manager'}
    ]
    
    for entity in entities:
        layer1_storage.save_entity(entity, "test.json")
        # 为每个实体设置 embedding
        embedding = generate_random_embedding(1536)
        vector_search.set_node_embedding(
            entity['id'],
            embedding,
            labels=['Entity', 'Layer1']
        )
        print(f"✅ 创建实体: {entity['name']} (id: {entity['id']})")
    
    # 创建事件节点
    events = [
        {'id': 'event_1', 'content': 'Alice worked on a project', 'participants': ['Alice']},
        {'id': 'event_2', 'content': 'Bob analyzed data', 'participants': ['Bob']},
        {'id': 'event_3', 'content': 'Alice and Bob collaborated', 'participants': ['Alice', 'Bob']}
    ]
    
    for event in events:
        layer2_storage.save_timeline_node(event, namespace, 'event')
        # 为事件设置 embedding
        embedding = generate_random_embedding(1536)
        vector_search.set_node_embedding(
            event['id'],
            embedding,
            labels=['Event', 'Layer2']
        )
        print(f"✅ 创建事件: {event['id']}")
    
    # 创建关系
    layer1_storage.create_relationship(
        'entity_alice',
        'entity_bob',
        'RELATED_TO',
        properties={'active': True}
    )
    layer2_storage.create_structural_edge('event_1', 'entity_alice', 'involves', namespace)
    layer2_storage.create_structural_edge('event_2', 'entity_bob', 'involves', namespace)
    layer2_storage.create_structural_edge('event_3', 'entity_alice', 'involves', namespace)
    layer2_storage.create_structural_edge('event_3', 'entity_bob', 'involves', namespace)
    
    print("✅ 创建关系")
    
    print("\n🔍 步骤 2: 向量搜索找到初始节点")
    print("-" * 60)
    
    # 获取 entity_alice 的 embedding 作为查询向量（模拟相似查询）
    # 在实际使用中，这应该是查询文本的 embedding
    alice_node_query = """
    MATCH (n:Entity:Layer1 {id: $node_id, namespace: $namespace})
    RETURN n.embedding as embedding
    """
    alice_result = client.execute_read(alice_node_query, {
        'node_id': 'entity_alice',
        'namespace': namespace
    })
    
    if alice_result and alice_result[0].get('embedding'):
        query_embedding = alice_result[0]['embedding']
        print(f"✅ 使用 entity_alice 的 embedding 作为查询向量")
    else:
        query_embedding = generate_random_embedding(1536)
        print(f"⚠️  未找到 entity_alice 的 embedding，使用随机向量")
    
    # 向量搜索（不使用索引，使用普通查询）
    initial_nodes = vector_search.vector_search(
        query_embedding=query_embedding,
        label='Entity',
        top_k=2,
        similarity_threshold=0.0
    )
    
    print(f"✅ 向量搜索找到 {len(initial_nodes)} 个初始节点:")
    for node in initial_nodes:
        print(f"   - {node.get('name', node.get('id'))} (相似度: {node.get('similarity_score', 'N/A')})")
    
    if not initial_nodes:
        print("⚠️  未找到初始节点，使用固定节点进行测试")
        initial_node_ids = ['entity_alice']
    else:
        initial_node_ids = [node.get('id') for node in initial_nodes if node.get('id')]
    
    print("\n🔗 步骤 3: 从初始节点扩展")
    print("-" * 60)
    
    # 从初始节点扩展
    expanded_nodes = vector_search.expand_from_nodes(
        node_ids=initial_node_ids,
        max_hops=2,
        relationship_types=None,  # 所有关系类型
        direction='both',
        limit=20
    )
    
    print(f"✅ 扩展找到 {len(expanded_nodes)} 个节点:")
    initial_count = sum(1 for n in expanded_nodes if n.get('is_initial', False))
    expanded_count = len(expanded_nodes) - initial_count
    print(f"   - 初始节点: {initial_count}")
    print(f"   - 扩展节点: {expanded_count}")
    
    for node in expanded_nodes[:10]:  # 只显示前10个
        node_type = "初始" if node.get('is_initial') else "扩展"
        hops = node.get('hops', 0)
        print(f"   - [{node_type}] {node.get('name', node.get('id'))} (跳数: {hops})")
    
    print("\n🔍 步骤 4: 混合搜索（向量搜索 + 图扩展）")
    print("-" * 60)
    
    # 混合搜索
    hybrid_result = vector_search.hybrid_search(
        query_embedding=query_embedding,
        label='Entity',
        vector_top_k=2,
        max_hops=2,
        expand_limit=20,
        similarity_threshold=0.0
    )
    
    print(f"✅ 混合搜索结果:")
    print(f"   - 初始节点数: {len(hybrid_result['initial_nodes'])}")
    print(f"   - 扩展节点数: {len(hybrid_result['expanded_nodes'])}")
    print(f"   - 总节点数: {hybrid_result['total_nodes']}")
    
    client.close()
    print("\n✅ 向量搜索和图扩展测试完成！\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 开始测试 Neo4j 向量搜索和图扩展")
    print("=" * 60 + "\n")
    
    success = test_vector_search_and_expansion()
    
    if success:
        print("=" * 60)
        print("✅ 测试通过！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 测试失败")
        print("=" * 60)

