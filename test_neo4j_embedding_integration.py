"""
测试 Neo4j 与 OpenAI Embedding API 集成

验证自动生成 embedding 和向量搜索功能
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
from core.infrastructure.neo4j_vector_search import Neo4jVectorSearch
from core.infrastructure.embedding import EmbeddingManager
from core.layer1.neo4j_storage import Layer1Neo4jStorage


def test_openai_embedding_integration():
    """测试 OpenAI Embedding API 集成"""
    print("=" * 60)
    print("🧪 测试 Neo4j + OpenAI Embedding API 集成")
    print("=" * 60)
    
    # 检查配置
    config = Config()
    if not config.openai_api_key:
        print("❌ OpenAI API Key 未配置，请检查 .env 文件")
        return False
    
    # 创建客户端
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_openai_embedding"
    
    # 创建 EmbeddingManager
    try:
        embedding_manager = EmbeddingManager(config)
        print("✅ EmbeddingManager 初始化成功")
    except Exception as e:
        print(f"❌ EmbeddingManager 初始化失败: {e}")
        return False
    
    # 创建存储实例（使用 test.json 作为文件名，命名空间会是 "test"）
    layer1_storage = Layer1Neo4jStorage(client, "test")  # 从 test.json 提取的命名空间
    
    # 创建向量搜索实例（带 EmbeddingManager，使用相同的命名空间）
    # 注意：layer1_storage.save_entity() 会从文件名提取命名空间，所以这里需要匹配
    import os
    base_name = os.path.splitext("test.json")[0]  # "test"
    vector_search = Neo4jVectorSearch(
        client, 
        base_name,  # 使用 "test" 而不是 namespace
        embedding_manager=embedding_manager
    )
    
    print("\n📝 步骤 1: 创建节点并使用 OpenAI API 生成 embedding")
    print("-" * 60)
    
    # 创建实体节点
    entities = [
        {
            'id': 'entity_ai',
            'name': 'AI Research',
            'content': 'Artificial Intelligence and machine learning research',
            'layer': 1,
            'active': True
        },
        {
            'id': 'entity_nlp',
            'name': 'NLP',
            'content': 'Natural Language Processing and text understanding',
            'layer': 1,
            'active': True
        },
        {
            'id': 'entity_graph',
            'name': 'Graph Database',
            'content': 'Graph databases and knowledge graphs for data storage',
            'layer': 1,
            'active': True
        }
    ]
    
    for entity in entities:
        # 保存实体（会自动设置命名空间为 "test"）
        layer1_storage.save_entity(entity, "test.json")
        print(f"✅ 已保存实体: {entity['name']} (id: {entity['id']})")
        
        # 确保命名空间一致（layer1_storage 会从文件名提取命名空间）
        # vector_search 的命名空间已经在初始化时设置为 "test"
        
        # 使用 OpenAI API 生成 embedding 并设置到节点
        text = f"{entity['name']}: {entity['content']}"
        try:
            success = vector_search.generate_and_set_embedding(
                entity['id'],
                text,
                labels=['Entity', 'Layer1']
            )
            
            if success:
                print(f"✅ {entity['name']}: 已生成并设置 embedding")
            else:
                print(f"❌ {entity['name']}: embedding 设置失败（可能节点不存在或命名空间不匹配）")
                # 尝试直接查询节点验证
                node = layer1_storage.get_node(entity['id'], labels=['Entity', 'Layer1'])
                if node:
                    print(f"   节点存在，命名空间: {node.get('namespace')}, 期望: {vector_search.namespace}")
                else:
                    print(f"   节点不存在: {entity['id']}")
        except Exception as e:
            print(f"❌ {entity['name']}: embedding 生成失败 - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🔍 步骤 2: 使用查询文本进行向量搜索")
    print("-" * 60)
    
    # 使用查询文本生成 embedding
    query_text = "machine learning and artificial intelligence"
    print(f"查询文本: '{query_text}'")
    
    try:
        query_embedding = embedding_manager.get_embedding(query_text)
        print(f"✅ 查询 embedding 生成成功 (维度: {len(query_embedding)})")
    except Exception as e:
        print(f"❌ 查询 embedding 生成失败: {e}")
        return False
    
    # 向量搜索（确保命名空间正确）
    # vector_search 的命名空间应该是 "test"
    similar_nodes = vector_search.vector_search(
        query_embedding=query_embedding,
        label='Entity',
        top_k=3,
        similarity_threshold=0.0
    )
    
    print(f"\n✅ 找到 {len(similar_nodes)} 个相似节点:")
    for i, node in enumerate(similar_nodes, 1):
        name = node.get('name', node.get('id', 'Unknown'))
        score = node.get('similarity_score', 0.0)
        print(f"   {i}. {name} (相似度: {score:.4f})")
    
    print("\n🔗 步骤 3: 从相似节点进行图扩展")
    print("-" * 60)
    
    if similar_nodes:
        initial_node_ids = [node.get('id') for node in similar_nodes if node.get('id')]
        
        # 创建一些关系用于测试扩展
        if len(initial_node_ids) >= 2:
            layer1_storage.create_relationship(
                initial_node_ids[0],
                initial_node_ids[1],
                'RELATED_TO',
                properties={'active': True}
            )
            print("✅ 创建关系用于测试扩展")
        
        # 图扩展
        expanded_nodes = vector_search.expand_from_nodes(
            node_ids=initial_node_ids[:1],  # 只从第一个节点扩展
            max_hops=1,
            limit=10
        )
        
        print(f"✅ 扩展找到 {len(expanded_nodes)} 个节点:")
        for node in expanded_nodes:
            node_type = "初始" if node.get('is_initial') else "扩展"
            name = node.get('name', node.get('id', 'Unknown'))
            hops = node.get('hops', 0)
            print(f"   - [{node_type}] {name} (跳数: {hops})")
    
    print("\n🔍 步骤 4: 混合搜索（向量搜索 + 图扩展）")
    print("-" * 60)
    
    hybrid_result = vector_search.hybrid_search(
        query_embedding=query_embedding,
        label='Entity',
        vector_top_k=2,
        max_hops=1,
        expand_limit=10,
        similarity_threshold=0.0
    )
    
    print(f"✅ 混合搜索结果:")
    print(f"   - 初始节点数: {len(hybrid_result['initial_nodes'])}")
    print(f"   - 扩展节点数: {len(hybrid_result['expanded_nodes'])}")
    print(f"   - 总节点数: {hybrid_result['total_nodes']}")
    
    client.close()
    print("\n✅ OpenAI Embedding API 集成测试完成！\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 开始测试 Neo4j + OpenAI Embedding API 集成")
    print("=" * 60 + "\n")
    
    success = test_openai_embedding_integration()
    
    if success:
        print("=" * 60)
        print("✅ 测试通过！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 测试失败")
        print("=" * 60)

