"""
测试各层 Neo4j Storage 实现

验证 Layer1、Layer2、Layer3 的存储功能
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

from core.infrastructure.neo4j_client import Neo4jClient
from core.layer1.neo4j_storage import Layer1Neo4jStorage
from core.layer2.neo4j_storage import Layer2Neo4jStorage
from core.layer3.neo4j_storage import Layer3Neo4jStorage


def test_layer1_storage():
    """测试 Layer1 Storage"""
    print("=" * 60)
    print("🧪 测试 Layer1 Neo4j Storage")
    print("=" * 60)
    
    # 创建客户端和存储
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_layer1"
    storage = Layer1Neo4jStorage(client, namespace)
    
    # 初始化存储
    storage.initialize_storage("test_data.json")
    print(f"✅ 初始化存储，命名空间: {storage.namespace}")
    
    # 测试保存实体
    print("\n📝 测试保存实体:")
    print("-" * 60)
    
    entity1 = {
        'id': 'entity_001',
        'name': 'Alice',
        'content': 'Alice is a software engineer working on AI systems',
        'layer': 1,
        'active': True
    }
    storage.save_entity(entity1, "test_data.json")
    print(f"✅ 保存实体: {entity1['name']} (id: {entity1['id']})")
    
    entity2 = {
        'id': 'entity_002',
        'name': 'Bob',
        'content': 'Bob is a data scientist',
        'layer': 1,
        'active': True
    }
    storage.save_entity(entity2, "test_data.json")
    print(f"✅ 保存实体: {entity2['name']} (id: {entity2['id']})")
    
    # 测试保存关系
    print("\n🔗 测试保存关系:")
    print("-" * 60)
    
    relationship = {
        'id': 'edge_001',
        'source': 'entity_001',
        'target': 'entity_002',
        'type': 'RELATED_TO',
        'content': 'Alice and Bob work together',
        'layer': 1,
        'active': True
    }
    storage.save_relationship(relationship, namespace)
    print(f"✅ 保存关系: {relationship['source']} -> {relationship['target']}")
    
    # 测试查询实体
    print("\n🔍 测试查询实体:")
    print("-" * 60)
    
    entities = storage.get_entities("test_data.json")
    print(f"✅ 查询到 {len(entities)} 个实体:")
    for entity in entities:
        print(f"   - {entity.get('name')} (id: {entity.get('id')})")
    
    # 测试查询关系
    print("\n🔍 测试查询关系:")
    print("-" * 60)
    
    relationships = storage.get_relationships("test_data.json")
    print(f"✅ 查询到 {len(relationships)} 个关系:")
    for rel in relationships:
        print(f"   - {rel.get('source')} -> {rel.get('target')} ({rel.get('type')})")
    
    # 测试更新节点
    print("\n✏️  测试更新节点:")
    print("-" * 60)
    
    storage.update_node('entity_001', content='Updated: Alice is a senior software engineer', namespace=namespace)
    # 重新查询验证更新
    updated_entity = storage.get_node('entity_001', labels=['Entity', 'Layer1'])
    if updated_entity:
        print(f"✅ 更新成功: {updated_entity.get('content')}")
    else:
        print("❌ 更新后查询失败")
    
    # 测试统计信息
    print("\n📊 测试统计信息:")
    print("-" * 60)
    
    stats = storage.get_storage_stats("test_data.json")
    print(f"✅ 统计信息:")
    print(f"   命名空间: {stats.get('namespace')}")
    print(f"   节点总数: {stats.get('node_count')}")
    print(f"   关系总数: {stats.get('relationship_count')}")
    print(f"   实体数量: {stats.get('total_entities')}")
    print(f"   关系数量: {stats.get('total_relationships')}")
    
    client.close()
    print("\n✅ Layer1 测试完成！\n")
    return True


def test_layer2_storage():
    """测试 Layer2 Storage"""
    print("=" * 60)
    print("🧪 测试 Layer2 Neo4j Storage")
    print("=" * 60)
    
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_layer2"
    storage = Layer2Neo4jStorage(client, namespace)
    
    # 测试保存事件节点
    print("\n📝 测试保存事件节点:")
    print("-" * 60)
    
    event = {
        'id': 'event_001',
        'content': 'Alice attended a conference on AI',
        'conversation_time': '2024-01-01T10:00:00Z',
        'relative_time': 'morning',
        'participants': ['Alice'],
        'location': 'San Francisco'
    }
    storage.save_timeline_node(event, namespace, 'event')
    print(f"✅ 保存事件: {event['id']}")
    
    # 测试保存状态节点
    state = {
        'id': 'state_001',
        'content': 'Alice is currently working on a project',
        'conversation_time': '2024-01-01T11:00:00Z',
        'participants': ['Alice'],
        'duration': 'ongoing'
    }
    storage.save_timeline_node(state, namespace, 'state')
    print(f"✅ 保存状态: {state['id']}")
    
    # 测试保存上下文节点
    context = {
        'id': 'context_001',
        'content': 'The team is preparing for a major release',
        'conversation_time': '2024-01-01T12:00:00Z',
        'affected_entities': ['Alice', 'Bob'],
        'impact': 'high'
    }
    storage.save_timeline_node(context, namespace, 'context')
    print(f"✅ 保存上下文: {context['id']}")
    
    # 测试创建 fragment 连接边
    print("\n🔗 测试创建 fragment 连接边:")
    print("-" * 60)
    
    storage.create_fragment_connection_edge('fragment_001', 'event_001', 'contains', namespace)
    print("✅ 创建 fragment -> event 关系")
    
    # 测试创建结构性边
    storage.create_structural_edge('event_001', 'entity_001', 'involves', namespace)
    print("✅ 创建 event -> entity 关系")
    
    # 测试查询节点
    print("\n🔍 测试查询节点:")
    print("-" * 60)
    
    events = storage.query_nodes(labels=['Event', 'Layer2'], filters={'type': 'event'})
    print(f"✅ 查询到 {len(events)} 个事件节点")
    for event in events:
        print(f"   - {event.get('content', '')[:50]}...")
    
    client.close()
    print("\n✅ Layer2 测试完成！\n")
    return True


def test_layer3_storage():
    """测试 Layer3 Storage"""
    print("=" * 60)
    print("🧪 测试 Layer3 Neo4j Storage")
    print("=" * 60)
    
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_layer3"
    storage = Layer3Neo4jStorage(client, namespace)
    
    # 测试保存事件聚类
    print("\n📝 测试保存事件聚类:")
    print("-" * 60)
    
    cluster = {
        'id': 'cluster_001',
        'content': 'Work-related events',
        'cluster_type': 'work',
        'participants': ['Alice', 'Bob'],
        'time_span': '2024-01-01 to 2024-01-31',
        'significance': 'high'
    }
    storage.save_event_cluster(cluster, namespace)
    print(f"✅ 保存事件聚类: {cluster['id']}")
    
    # 测试保存模式
    pattern = {
        'id': 'pattern_001',
        'person': 'Alice',
        'pattern_type': 'work_schedule',
        'content': 'Alice typically works from 9am to 6pm'
    }
    storage.save_pattern(pattern, namespace)
    print(f"✅ 保存模式: {pattern['id']}")
    
    # 测试保存偏好
    preference = {
        'id': 'preference_001',
        'person': 'Alice',
        'category': 'technology',
        'content': 'Alice prefers Python over Java'
    }
    storage.save_preference(preference, namespace)
    print(f"✅ 保存偏好: {preference['id']}")
    
    # 测试保存行为规则
    rule = {
        'id': 'rule_001',
        'person': 'Alice',
        'rule_type': 'communication',
        'content': 'Alice responds to emails within 24 hours'
    }
    storage.save_behavior_rule(rule, namespace)
    print(f"✅ 保存行为规则: {rule['id']}")
    
    # 测试创建关系
    print("\n🔗 测试创建关系:")
    print("-" * 60)
    
    storage.create_cluster_event_edge('cluster_001', 'event_001', namespace)
    print("✅ 创建 cluster -> event 关系")
    
    storage.create_pattern_person_edge('pattern_001', 'Alice', namespace)
    print("✅ 创建 pattern -> person 关系")
    
    # 测试查询节点
    print("\n🔍 测试查询节点:")
    print("-" * 60)
    
    patterns = storage.query_nodes(labels=['Pattern', 'Layer3'])
    print(f"✅ 查询到 {len(patterns)} 个模式节点")
    for pattern in patterns:
        print(f"   - {pattern.get('content', '')[:50]}...")
    
    client.close()
    print("\n✅ Layer3 测试完成！\n")
    return True


def test_integrated_flow():
    """测试完整的集成流程"""
    print("=" * 60)
    print("🧪 测试完整集成流程 (Layer1 -> Layer2 -> Layer3)")
    print("=" * 60)
    
    client = Neo4jClient()
    if not client.connect():
        print("❌ 无法连接到 Neo4j")
        return False
    
    namespace = "test_integrated"
    
    # Layer1: 创建实体
    layer1_storage = Layer1Neo4jStorage(client, namespace)
    entity = {
        'id': 'person_alice',
        'name': 'Alice',
        'content': 'Alice is a software engineer',
        'layer': 1,
        'active': True
    }
    layer1_storage.save_entity(entity, "test.json")
    print("✅ Layer1: 创建实体")
    
    # Layer2: 创建事件并连接到实体
    layer2_storage = Layer2Neo4jStorage(client, namespace)
    event = {
        'id': 'event_alice_work',
        'content': 'Alice worked on a project',
        'conversation_time': '2024-01-01T10:00:00Z',
        'participants': ['Alice']
    }
    layer2_storage.save_timeline_node(event, namespace, 'event')
    layer2_storage.create_structural_edge('event_alice_work', 'person_alice', 'involves', namespace)
    print("✅ Layer2: 创建事件并连接到实体")
    
    # Layer3: 创建模式并连接到实体
    layer3_storage = Layer3Neo4jStorage(client, namespace)
    pattern = {
        'id': 'pattern_alice_work',
        'person': 'Alice',
        'pattern_type': 'work',
        'content': 'Alice works on software projects'
    }
    layer3_storage.save_pattern(pattern, namespace)
    layer3_storage.create_pattern_person_edge('pattern_alice_work', 'person_alice', namespace)
    print("✅ Layer3: 创建模式并连接到实体")
    
    # 查询完整路径
    print("\n🔍 查询完整路径:")
    print("-" * 60)
    
    query = """
    MATCH (p:Entity:Layer1 {id: $person_id, namespace: $namespace})
    OPTIONAL MATCH (p)<-[:INVOLVES]-(e:Event:Layer2 {namespace: $namespace})
    OPTIONAL MATCH (p)<-[:DESCRIBES]-(pat:Pattern:Layer3 {namespace: $namespace})
    RETURN p, collect(DISTINCT e) as events, collect(DISTINCT pat) as patterns
    """
    
    result = client.execute_read(query, {'person_id': 'person_alice', 'namespace': namespace})
    if result and len(result) > 0:
        data = result[0]
        person = dict(data.get('p', {}))
        events = [dict(e) for e in data.get('events', []) if e]
        patterns = [dict(p) for p in data.get('patterns', []) if p]
        
        if person:
            print(f"✅ 实体: {person.get('name', 'person_alice')}")
            print(f"   关联事件数: {len(events)}")
            print(f"   关联模式数: {len(patterns)}")
            for event in events:
                print(f"   - 事件: {event.get('content', '')[:50]}...")
            for pattern in patterns:
                print(f"   - 模式: {pattern.get('content', '')[:50]}...")
        else:
            print("⚠️  未找到实体节点")
    else:
        print("⚠️  查询无结果")
    
    client.close()
    print("\n✅ 集成流程测试完成！\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 开始测试各层 Neo4j Storage 实现")
    print("=" * 60 + "\n")
    
    # 运行各层测试
    success = True
    success &= test_layer1_storage()
    success &= test_layer2_storage()
    success &= test_layer3_storage()
    success &= test_integrated_flow()
    
    if success:
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 部分测试失败")
        print("=" * 60)

