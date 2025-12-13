"""
测试LLM调用时长的脚本

分别测试：
1. Fragment分片的LLM调用时长
2. Layer1提取的LLM调用时长
3. Layer2提取的LLM调用时长
"""

import json
import time
import sys
from typing import Dict, List, Any
from config import Config
from core.infrastructure import LLMClient, EmbeddingManager, UnifiedCache
from core.fragment import FragmentProcessor, BufferManager
from core.layer1.extractor import Layer1Extractor
from core.layer2.extractor import Layer2Extractor
from test_locomo_dataset import load_locomo_dataset, convert_conversation_to_hamem_format


def test_fragment_splitting(conversation_data: Dict[str, Any], config: Config, llm_provider: str = "deepseek"):
    """测试Fragment分片的LLM调用时长"""
    print("\n" + "="*60)
    print(f"📄 测试Fragment分片 (使用 {llm_provider})")
    print("="*60)
    
    # 转换数据格式
    hamem_format = convert_conversation_to_hamem_format(conversation_data)
    messages = hamem_format['messages']
    
    # 初始化组件
    llm_client = LLMClient(config)
    fragment_processor = FragmentProcessor(llm_client, default_provider=llm_provider)
    buffer_manager = BufferManager(max_length=5000)
    
    # 统计信息
    total_llm_calls = 0
    total_llm_time = 0.0
    fragments_created = 0
    
    print(f"\n处理 {len(messages)} 条消息...")
    
    for i, msg in enumerate(messages, 1):
        turn = {
            'role': msg.get('role', 'user'),
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', ''),
            'metadata': msg.get('metadata', {})
        }
        
        # 检查是否需要LLM判断分片
        fragment, needs_llm = buffer_manager.add_turn(turn, timestamp=msg.get('timestamp'))
        
        if fragment:
            fragments_created += 1
            print(f"  ✅ Fragment {fragments_created} 创建 (时间戳变化)")
        
        if needs_llm:
            # 需要LLM判断分片
            print(f"  🤖 消息 {i}: 需要LLM判断分片 (buffer长度: {buffer_manager.get_buffer_length()})")
            
            turns = buffer_manager.get_turns_for_llm()
            if turns:
                start_time = time.time()
                result = fragment_processor.process_fragment(turns)
                llm_time = time.time() - start_time
                
                total_llm_calls += 1
                total_llm_time += llm_time
                
                should_split = result.get('should_split', False)
                split_point = result.get('split_point')
                
                print(f"    ⏱️  LLM调用耗时: {llm_time:.3f} 秒 ({llm_provider})")
                print(f"    📊 结果: should_split={should_split}, split_point={split_point}")
                
                if should_split and split_point:
                    fragment = buffer_manager.extract_fragment(split_point)
                    if fragment:
                        fragments_created += 1
                        print(f"    ✅ Fragment {fragments_created} 创建 (LLM判断分片)")
                    buffer_manager.keep_remaining(split_point)
    
    # 处理剩余的buffer
    if not buffer_manager.is_empty():
        fragment = buffer_manager._save_timestamp_fragment()
        if fragment:
            fragments_created += 1
            print(f"  ✅ Fragment {fragments_created} 创建 (最终buffer)")
    
    print(f"\n📊 Fragment分片统计:")
    print(f"  - 总LLM调用次数: {total_llm_calls}")
    print(f"  - 总LLM调用时长: {total_llm_time:.3f} 秒")
    print(f"  - 平均每次LLM调用时长: {total_llm_time/total_llm_calls:.3f} 秒" if total_llm_calls > 0 else "  - 平均每次LLM调用时长: N/A")
    print(f"  - 创建的Fragment数量: {fragments_created}")
    
    return {
        'total_llm_calls': total_llm_calls,
        'total_llm_time': total_llm_time,
        'avg_llm_time': total_llm_time / total_llm_calls if total_llm_calls > 0 else 0,
        'fragments_created': fragments_created
    }


def test_layer1_extraction(fragments: List[Dict[str, Any]], config: Config, return_entities: bool = False, llm_provider: str = "deepseek"):
    """
    测试Layer1提取的LLM调用时长（只测试提取，不做召回和冲突检测）
    
    Args:
        fragments: Fragment列表
        config: 配置对象
        return_entities: 是否返回提取的实体列表（用于Layer2测试）
    
    Returns:
        统计信息，如果return_entities=True，还会返回每个fragment对应的实体列表
    """
    print("\n" + "="*60)
    print("👥 测试Layer1提取（仅提取阶段）")
    print("="*60)
    
    # 初始化组件（只使用Extractor，不初始化Processor）
    llm_client = LLMClient(config)
    layer1_extractor = Layer1Extractor(llm_client, default_provider=llm_provider)
    
    # 统计信息
    total_llm_calls = 0
    total_llm_time = 0.0
    total_entities = 0
    total_relationships = 0
    
    # 存储每个fragment的实体（用于Layer2）
    fragment_entities_map = {}  # {fragment_id: [entities]}
    all_entities = []  # 累积所有实体
    
    print(f"\n处理 {len(fragments)} 个fragments...")
    
    for i, fragment in enumerate(fragments, 1):
        fragment_id = fragment.get('id', 'unknown')
        print(f"\n  📄 Fragment {i}: {fragment_id}")
        
        # 测量Layer1提取时间（只测试提取阶段）
        start_time = time.time()
        extraction_result = layer1_extractor.extract_from_fragment(fragment)
        llm_time = time.time() - start_time
        
        entities = extraction_result.get('entities', [])
        relationships = extraction_result.get('relationships', [])
        
        total_entities += len(entities)
        total_relationships += len(relationships)
        total_llm_calls += 1  # 每个fragment一次LLM调用
        total_llm_time += llm_time
        
        # 保存实体（用于Layer2）
        if return_entities:
            fragment_entities_map[fragment_id] = entities
            all_entities.extend(entities)
        
        print(f"    ⏱️  LLM调用耗时: {llm_time:.3f} 秒")
        print(f"    📊 提取: {len(entities)} 个实体, {len(relationships)} 个关系")
    
    print(f"\n📊 Layer1提取统计:")
    print(f"  - LLM调用次数: {total_llm_calls}")
    print(f"  - 总LLM调用时长: {total_llm_time:.3f} 秒")
    print(f"  - 平均每次LLM调用时长: {total_llm_time/total_llm_calls:.3f} 秒" if total_llm_calls > 0 else "  - 平均每次LLM调用时长: N/A")
    print(f"  - 总提取实体数: {total_entities}")
    print(f"  - 总提取关系数: {total_relationships}")
    
    result = {
        'total_llm_calls': total_llm_calls,
        'total_llm_time': total_llm_time,
        'avg_llm_time': total_llm_time / total_llm_calls if total_llm_calls > 0 else 0,
        'total_entities': total_entities,
        'total_relationships': total_relationships
    }
    
    if return_entities:
        result['fragment_entities_map'] = fragment_entities_map
        result['all_entities'] = all_entities
    
    return result


def test_layer2_extraction(fragments: List[Dict[str, Any]], config: Config, fragment_entities_map: Dict[str, List[Dict[str, Any]]] = None, all_entities: List[Dict[str, Any]] = None, llm_provider: str = "deepseek"):
    """
    测试Layer2提取的LLM调用时长（只测试提取，不做其他处理）
    
    Args:
        fragments: Fragment列表
        config: 配置对象
        fragment_entities_map: 每个fragment对应的实体映射 {fragment_id: [entities]}
        all_entities: 所有累积的实体列表（用于Layer2提取）
    """
    print("\n" + "="*60)
    print("📅 测试Layer2提取（仅提取阶段）")
    print("="*60)
    
    # 初始化组件（只使用Extractor，不初始化Processor）
    llm_client = LLMClient(config)
    layer2_extractor = Layer2Extractor(llm_client, default_provider=llm_provider)
    
    # 统计信息
    total_llm_calls = 0
    total_llm_time = 0.0
    total_events = 0
    total_states = 0
    total_contexts = 0
    
    # 累积所有实体（用于后续fragment的Layer2提取）
    current_all_entities = all_entities.copy() if all_entities else []
    
    print(f"\n处理 {len(fragments)} 个fragments...")
    
    for i, fragment in enumerate(fragments, 1):
        fragment_id = fragment.get('id', 'unknown')
        print(f"\n  📄 Fragment {i}: {fragment_id}")
        
        # 获取当前fragment对应的Layer1实体
        # 优先使用fragment_entities_map，否则使用累积的所有实体
        if fragment_entities_map and fragment_id in fragment_entities_map:
            layer1_entities = fragment_entities_map[fragment_id]
        else:
            # 如果没有提供映射，使用累积的所有实体
            layer1_entities = current_all_entities
        
        print(f"    📌 使用 {len(layer1_entities)} 个Layer1实体作为参考")
        
        # 测量Layer2提取时间（只测试提取阶段）
        start_time = time.time()
        extraction_result = layer2_extractor.extract_from_fragment(fragment, layer1_entities)
        llm_time = time.time() - start_time
        
        events = extraction_result.get('events', [])
        states = extraction_result.get('states', [])
        contexts = extraction_result.get('contexts', [])
        
        total_events += len(events)
        total_states += len(states)
        total_contexts += len(contexts)
        
        total_llm_calls += 1  # 每个fragment一次LLM调用
        total_llm_time += llm_time
        
        # 更新累积的实体列表（用于后续fragment）
        if fragment_entities_map and fragment_id in fragment_entities_map:
            current_all_entities.extend(fragment_entities_map[fragment_id])
        
        print(f"    ⏱️  LLM调用耗时: {llm_time:.3f} 秒")
        print(f"    📊 提取: {len(events)} events, {len(states)} states, {len(contexts)} contexts")
    
    print(f"\n📊 Layer2提取统计:")
    print(f"  - LLM调用次数: {total_llm_calls}")
    print(f"  - 总LLM调用时长: {total_llm_time:.3f} 秒")
    print(f"  - 平均每次LLM调用时长: {total_llm_time/total_llm_calls:.3f} 秒" if total_llm_calls > 0 else "  - 平均每次LLM调用时长: N/A")
    print(f"  - 总提取: {total_events} events, {total_states} states, {total_contexts} contexts")
    
    return {
        'total_llm_calls': total_llm_calls,
        'total_llm_time': total_llm_time,
        'avg_llm_time': total_llm_time / total_llm_calls if total_llm_calls > 0 else 0,
        'total_events': total_events,
        'total_states': total_states,
        'total_contexts': total_contexts
    }


def main():
    """主函数"""
    print("="*60)
    print("🚀 LLM调用时长测试")
    print("="*60)
    
    # 解析命令行参数
    if len(sys.argv) < 3:
        print("用法: python test_llm_timing.py <conversation_idx> <test_type> [llm_provider]")
        print("  conversation_idx: 0-9 (选择哪个conversation)")
        print("  test_type: fragment, layer1, layer2, all")
        print("  llm_provider: deepseek (默认) 或 openai")
        sys.exit(1)
    
    conversation_idx = int(sys.argv[1])
    test_type = sys.argv[2].lower()
    llm_provider = sys.argv[3].lower() if len(sys.argv) > 3 else "deepseek"
    
    if llm_provider not in ["deepseek", "openai"]:
        print(f"❌ 不支持的LLM提供商: {llm_provider}，使用默认值: deepseek")
        llm_provider = "deepseek"
    
    print(f"\n🤖 使用LLM提供商: {llm_provider}")
    if llm_provider == "openai":
        print(f"   模型: gpt-4o-mini")
    
    # 加载配置
    print("\n📝 加载配置...")
    config = Config.from_env()
    print("✅ 配置加载完成")
    
    # 加载数据集
    dataset_path = "/home/zhiyu_zheng/DCL/Others/locomo/data/locomo10.json"
    print(f"\n📂 加载数据集: {dataset_path}")
    data = load_locomo_dataset(dataset_path)
    print(f"✅ 加载完成，共 {len(data)} 个conversation")
    
    if conversation_idx >= len(data):
        print(f"❌ Conversation索引 {conversation_idx} 超出范围 (0-{len(data)-1})")
        sys.exit(1)
    
    conversation_data = data[conversation_idx]
    print(f"\n✅ 选择Conversation {conversation_idx}")
    
    # 转换数据格式
    hamem_format = convert_conversation_to_hamem_format(conversation_data)
    
    results = {}
    fragments = []
    
    # 为了测试Layer1和Layer2，需要先创建fragments
    if test_type in ['layer1', 'layer2', 'all']:
        print("\n📝 先创建fragments用于Layer1/Layer2测试...")
        # 创建fragments（不测试时间，只获取fragments列表）
        llm_client = LLMClient(config)
        fragment_processor = FragmentProcessor(llm_client)
        buffer_manager = BufferManager(max_length=5000)
        
        for msg in hamem_format['messages']:
            turn = {
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', ''),
                'metadata': msg.get('metadata', {})
            }
            
            fragment, needs_llm = buffer_manager.add_turn(turn, timestamp=msg.get('timestamp'))
            if fragment:
                fragments.append(fragment)
            
            if needs_llm:
                turns = buffer_manager.get_turns_for_llm()
                if turns:
                    result = fragment_processor.process_fragment(turns)
                    if result.get('should_split') and result.get('split_point'):
                        fragment = buffer_manager.extract_fragment(result['split_point'])
                        if fragment:
                            fragments.append(fragment)
                        buffer_manager.keep_remaining(result['split_point'])
        
        # 处理剩余的buffer
        if not buffer_manager.is_empty():
            fragment = buffer_manager._save_timestamp_fragment()
            if fragment:
                fragments.append(fragment)
        
        print(f"✅ 创建了 {len(fragments)} 个fragments")
    
    # 测试Fragment分片（带时间统计）
    if test_type in ['fragment', 'all']:
        fragment_result = test_fragment_splitting(conversation_data, config, llm_provider=llm_provider)
        results['fragment'] = fragment_result
    
    # 如果test_type是'all'，按照正确顺序处理：分片 -> Layer1 -> Layer2（按fragment顺序）
    if test_type == 'all' and fragments:
        print("\n" + "="*60)
        print("🔄 按顺序处理：分片 -> Layer1 -> Layer2（每个fragment依次处理）")
        print("="*60)
        
        # 初始化组件
        llm_client = LLMClient(config)
        layer1_extractor = Layer1Extractor(llm_client, default_provider=llm_provider)
        layer2_extractor = Layer2Extractor(llm_client, default_provider=llm_provider)
        
        # 统计信息
        layer1_stats = {
            'total_llm_calls': 0,
            'total_llm_time': 0.0,
            'total_entities': 0,
            'total_relationships': 0
        }
        layer2_stats = {
            'total_llm_calls': 0,
            'total_llm_time': 0.0,
            'total_events': 0,
            'total_states': 0,
            'total_contexts': 0
        }
        
        # 按fragment顺序处理
        for i, fragment in enumerate(fragments, 1):
            fragment_id = fragment.get('id', 'unknown')
            print(f"\n{'#'*60}")
            print(f"📄 Fragment {i}/{len(fragments)}: {fragment_id}")
            print(f"{'#'*60}")
            
            # Step 1: Layer1提取
            print(f"\n👥 Step 1: Layer1提取")
            start_time = time.time()
            layer1_result = layer1_extractor.extract_from_fragment(fragment)
            layer1_time = time.time() - start_time
            
            entities = layer1_result.get('entities', [])
            relationships = layer1_result.get('relationships', [])
            
            layer1_stats['total_llm_calls'] += 1
            layer1_stats['total_llm_time'] += layer1_time
            layer1_stats['total_entities'] += len(entities)
            layer1_stats['total_relationships'] += len(relationships)
            
            print(f"  ⏱️  LLM调用耗时: {layer1_time:.3f} 秒")
            print(f"  📊 提取: {len(entities)} 个实体, {len(relationships)} 个关系")
            
            # Step 2: Layer2提取（只使用当前fragment的Layer1实体）
            print(f"\n📅 Step 2: Layer2提取（使用当前fragment的 {len(entities)} 个实体）")
            start_time = time.time()
            layer2_result = layer2_extractor.extract_from_fragment(fragment, entities)
            layer2_time = time.time() - start_time
            
            events = layer2_result.get('events', [])
            states = layer2_result.get('states', [])
            contexts = layer2_result.get('contexts', [])
            
            layer2_stats['total_llm_calls'] += 1
            layer2_stats['total_llm_time'] += layer2_time
            layer2_stats['total_events'] += len(events)
            layer2_stats['total_states'] += len(states)
            layer2_stats['total_contexts'] += len(contexts)
            
            print(f"  ⏱️  LLM调用耗时: {layer2_time:.3f} 秒")
            print(f"  📊 提取: {len(events)} events, {len(states)} states, {len(contexts)} contexts")
        
        # 保存统计结果
        results['layer1'] = {
            'total_llm_calls': layer1_stats['total_llm_calls'],
            'total_llm_time': layer1_stats['total_llm_time'],
            'avg_llm_time': layer1_stats['total_llm_time'] / layer1_stats['total_llm_calls'] if layer1_stats['total_llm_calls'] > 0 else 0,
            'total_entities': layer1_stats['total_entities'],
            'total_relationships': layer1_stats['total_relationships']
        }
        results['layer2'] = {
            'total_llm_calls': layer2_stats['total_llm_calls'],
            'total_llm_time': layer2_stats['total_llm_time'],
            'avg_llm_time': layer2_stats['total_llm_time'] / layer2_stats['total_llm_calls'] if layer2_stats['total_llm_calls'] > 0 else 0,
            'total_events': layer2_stats['total_events'],
            'total_states': layer2_stats['total_states'],
            'total_contexts': layer2_stats['total_contexts']
        }
        
        print(f"\n📊 Layer1提取统计:")
        print(f"  - LLM调用次数: {layer1_stats['total_llm_calls']}")
        print(f"  - 总LLM调用时长: {layer1_stats['total_llm_time']:.3f} 秒")
        print(f"  - 平均每次LLM调用时长: {layer1_stats['total_llm_time']/layer1_stats['total_llm_calls']:.3f} 秒" if layer1_stats['total_llm_calls'] > 0 else "  - 平均每次LLM调用时长: N/A")
        print(f"  - 总提取实体数: {layer1_stats['total_entities']}")
        print(f"  - 总提取关系数: {layer1_stats['total_relationships']}")
        
        print(f"\n📊 Layer2提取统计:")
        print(f"  - LLM调用次数: {layer2_stats['total_llm_calls']}")
        print(f"  - 总LLM调用时长: {layer2_stats['total_llm_time']:.3f} 秒")
        print(f"  - 平均每次LLM调用时长: {layer2_stats['total_llm_time']/layer2_stats['total_llm_calls']:.3f} 秒" if layer2_stats['total_llm_calls'] > 0 else "  - 平均每次LLM调用时长: N/A")
        print(f"  - 总提取: {layer2_stats['total_events']} events, {layer2_stats['total_states']} states, {layer2_stats['total_contexts']} contexts")
    
    # 单独测试Layer1提取
    elif test_type == 'layer1':
        if fragments:
            layer1_result = test_layer1_extraction(fragments, config, return_entities=False, llm_provider=llm_provider)
            results['layer1'] = layer1_result
        else:
            print("\n⚠️  没有fragments，跳过Layer1测试")
    
    # 单独测试Layer2提取
    elif test_type == 'layer2':
        if fragments:
            # 单独测试Layer2时，使用空实体列表
            layer2_result = test_layer2_extraction(fragments, config, llm_provider=llm_provider)
            results['layer2'] = layer2_result
        else:
            print("\n⚠️  没有fragments，跳过Layer2测试")
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.3f}")
            else:
                print(f"  - {key}: {value}")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()

