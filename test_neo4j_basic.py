"""
Neo4j 基本功能测试脚本

测试插入和查询功能
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
    # 如果没有 python-dotenv，手动加载 .env 文件
    env_file = os.path.join(Path(__file__).parent, '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from core.infrastructure.neo4j_client import Neo4jClient


def test_basic_operations():
    """测试基本操作"""
    print("=" * 60)
    print("🧪 Neo4j 基本功能测试")
    print("=" * 60)
    
    # 创建客户端（会自动从环境变量读取配置）
    client = Neo4jClient()
    
    # 连接
    if not client.connect():
        print("❌ 无法连接到 Neo4j，请检查服务是否启动")
        return
    
    print("\n✅ 连接成功")
    
    # 测试 1: 创建节点
    print("\n📝 测试 1: 创建节点")
    print("-" * 60)
    
    # 创建实体节点
    entity_id = client.create_node(
        labels=['Entity', 'Person'],
        properties={
            'name': 'Alice',
            'content': 'Alice is a software engineer',
            'type': 'entity',
            'layer': 1,
            'active': True
        },
        node_id='entity_1'
    )
    print(f"✅ 创建实体节点: {entity_id}")
    
    # 创建事件节点
    event_id = client.create_node(
        labels=['Event'],
        properties={
            'content': 'Alice attended a conference',
            'type': 'event',
            'layer': 2,
            'participants': ['Alice'],
            'conversation_time': '2024-01-01T10:00:00Z',
            'active': True
        },
        node_id='event_1'
    )
    print(f"✅ 创建事件节点: {event_id}")
    
    # 创建片段节点
    fragment_id = client.create_node(
        labels=['Fragment'],
        properties={
            'content': 'This is a conversation fragment',
            'type': 'fragment',
            'layer': 0,
            'active': True
        },
        node_id='fragment_1'
    )
    print(f"✅ 创建片段节点: {fragment_id}")
    
    # 测试 2: 创建关系
    print("\n🔗 测试 2: 创建关系")
    print("-" * 60)
    
    # 创建 fragment -> event 关系
    success = client.create_relationship(
        source_id='fragment_1',
        target_id='event_1',
        rel_type='CONTAINS',
        properties={'active': True}
    )
    print(f"✅ 创建关系 fragment_1 -> event_1: {success}")
    
    # 创建 event -> entity 关系
    success = client.create_relationship(
        source_id='event_1',
        target_id='entity_1',
        rel_type='INVOLVES',
        properties={'active': True}
    )
    print(f"✅ 创建关系 event_1 -> entity_1: {success}")
    
    # 测试 3: 查询节点
    print("\n🔍 测试 3: 查询节点")
    print("-" * 60)
    
    # 根据ID查询
    node = client.get_node('entity_1')
    if node:
        print(f"✅ 查询到节点 entity_1: {node.get('name')}")
    else:
        print("❌ 未找到节点 entity_1")
    
    # 根据类型查询
    entities = client.get_nodes_by_type('Entity', limit=10)
    print(f"✅ 查询到 {len(entities)} 个 Entity 节点")
    for entity in entities:
        print(f"   - {entity.get('name')} (id: {entity.get('id')})")
    
    # 测试 4: 复杂查询
    print("\n🔍 测试 4: 复杂查询")
    print("-" * 60)
    
    # 查询 fragment 及其关联的事件和实体
    query = """
    MATCH (f:Fragment {id: $fragment_id})
    OPTIONAL MATCH (f)-[:CONTAINS]->(e:Event)
    OPTIONAL MATCH (e)-[:INVOLVES]->(ent:Entity)
    RETURN f, collect(DISTINCT e) as events, collect(DISTINCT ent) as entities
    """
    result = client.execute_read(query, {'fragment_id': 'fragment_1'})
    if result:
        data = result[0]
        fragment = dict(data['f'])
        events = [dict(e) for e in data['events'] if e]
        entities = [dict(e) for e in data['entities'] if e]
        print(f"✅ Fragment: {fragment.get('id')}")
        print(f"   关联事件数: {len(events)}")
        print(f"   关联实体数: {len(entities)}")
        for event in events:
            print(f"   - 事件: {event.get('content', '')[:50]}...")
        for entity in entities:
            print(f"   - 实体: {entity.get('name', '')}")
    
    # 测试 5: 更新节点
    print("\n✏️  测试 5: 更新节点")
    print("-" * 60)
    
    success = client.update_node('entity_1', {'content': 'Updated: Alice is a senior software engineer'})
    if success:
        updated_node = client.get_node('entity_1')
        print(f"✅ 更新成功: {updated_node.get('content')}")
    
    # 测试 6: 统计信息
    print("\n📊 测试 6: 数据库统计")
    print("-" * 60)
    
    stats = client.get_stats()
    print(f"节点总数: {stats['node_count']}")
    print(f"关系总数: {stats['relationship_count']}")
    print(f"数据库: {stats['database']}")
    
    # 关闭连接
    client.close()
    print("\n✅ 测试完成！")


def test_layer_structure():
    """测试分层结构（模拟实际使用场景）"""
    print("\n" + "=" * 60)
    print("🧪 分层结构测试（Layer1 -> Layer2 -> Layer3）")
    print("=" * 60)
    
    # 创建客户端（会自动从环境变量读取配置）
    client = Neo4jClient()
    
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return
    
    # Layer 0: Fragment
    fragment_id = client.create_node(
        labels=['Fragment'],
        properties={
            'id': 'frag_001',
            'content': 'User: I love Python programming. Assistant: That\'s great!',
            'type': 'fragment',
            'layer': 0,
            'active': True
        }
    )
    print(f"✅ Layer 0 - Fragment: {fragment_id}")
    
    # Layer 1: Entity
    person_id = client.create_node(
        labels=['Entity', 'Person'],
        properties={
            'id': 'person_001',
            'name': 'User',
            'content': 'A person who loves Python programming',
            'type': 'entity',
            'layer': 1,
            'active': True
        }
    )
    print(f"✅ Layer 1 - Entity: {person_id}")
    
    # Layer 2: Event
    event_id = client.create_node(
        labels=['Event'],
        properties={
            'id': 'event_001',
            'content': 'User expressed interest in Python programming',
            'type': 'event',
            'layer': 2,
            'participants': ['User'],
            'conversation_time': '2024-01-01T10:00:00Z',
            'active': True
        }
    )
    print(f"✅ Layer 2 - Event: {event_id}")
    
    # Layer 3: Pattern
    pattern_id = client.create_node(
        labels=['Pattern'],
        properties={
            'id': 'pattern_001',
            'content': 'User has interest in programming languages',
            'type': 'pattern',
            'pattern_type': 'interest',
            'layer': 3,
            'active': True
        }
    )
    print(f"✅ Layer 3 - Pattern: {pattern_id}")
    
    # 创建层级关系
    print("\n🔗 创建层级关系:")
    print("-" * 60)
    
    client.create_relationship(fragment_id, event_id, 'CONTAINS')
    print(f"✅ Fragment -> Event")
    
    client.create_relationship(event_id, person_id, 'INVOLVES')
    print(f"✅ Event -> Entity")
    
    client.create_relationship(pattern_id, person_id, 'DESCRIBES')
    print(f"✅ Pattern -> Entity")
    
    # 查询完整路径
    print("\n🔍 查询完整路径:")
    print("-" * 60)
    
    query = """
    MATCH path = (f:Fragment {id: $frag_id})
    -[:CONTAINS]->(e:Event)
    -[:INVOLVES]->(p:Entity)
    OPTIONAL MATCH (pat:Pattern)-[:DESCRIBES]->(p)
    RETURN f, e, p, collect(pat) as patterns
    """
    
    result = client.execute_read(query, {'frag_id': 'frag_001'})
    if result:
        data = result[0]
        print(f"Fragment: {dict(data['f']).get('content', '')[:50]}...")
        print(f"Event: {dict(data['e']).get('content', '')[:50]}...")
        print(f"Entity: {dict(data['p']).get('name', '')}")
        patterns = [dict(p) for p in data['patterns'] if p]
        print(f"Patterns: {len(patterns)}")
        for pat in patterns:
            print(f"  - {pat.get('content', '')[:50]}...")
    
    # 统计
    stats = client.get_stats()
    print(f"\n📊 统计: {stats['node_count']} 节点, {stats['relationship_count']} 关系")
    
    client.close()
    print("\n✅ 分层结构测试完成！")


if __name__ == "__main__":
    # 运行基本测试
    test_basic_operations()
    
    # 运行分层结构测试
    test_layer_structure()

