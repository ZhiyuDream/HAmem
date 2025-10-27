"""
Layer3处理器

协调聚类、提取和存储的完整流程
"""

from typing import Dict, List, Any
from core.infrastructure import LLMClient, UnifiedCache
from .clustering import EventClusterer
from .extractor import Layer3Extractor
from .storage import Layer3Storage


class Layer3Processor:
    """Layer3处理器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        cache: UnifiedCache,
        storage_dir: str = "storage",
        layer2_threshold: int = 60,  # 累积60个Layer2节点触发分析
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 7  # 最小cluster大小
    ):
        self.clusterer = EventClusterer(
            cache, 
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        self.extractor = Layer3Extractor(llm_client)
        self.storage = Layer3Storage(storage_dir)
        self.cache = cache
        
        self.layer2_threshold = layer2_threshold
        self.last_analyzed_layer2_count = 0  # 记录上次分析时的Layer2节点总数
        
        # 计数器
        self.cluster_counter = 0
        self.pattern_counter = 0
        self.preference_counter = 0
        self.rule_counter = 0
    
    def should_trigger_analysis(self, namespace: str) -> bool:
        """
        判断是否应该触发Layer3分析
        基于Layer2节点总数（Event + State + Context）
        
        Args:
            namespace: 命名空间
        
        Returns:
            bool: 是否应该触发
        """
        # 统计当前的Layer2节点数量（Event + State + Context）
        all_nodes = self.cache.cache['nodes']
        current_layer2_count = sum(
            1 for node in all_nodes.values()
            if node.get('layer') == 2 and node.get('type') in ['event', 'state', 'context']
        )
        
        # 计算新增的Layer2节点数
        new_layer2_count = current_layer2_count - self.last_analyzed_layer2_count
        
        # 详细日志
        print(f"  📊 当前Layer2节点总数: {current_layer2_count}")
        print(f"  📊 已分析节点数: {self.last_analyzed_layer2_count}")
        print(f"  📊 新增节点数: {new_layer2_count}")
        print(f"  📊 触发阈值: {self.layer2_threshold}")
        
        if new_layer2_count >= self.layer2_threshold:
            print(f"  ✅ 达到阈值，触发Layer3分析")
            return True
        else:
            print(f"  ℹ️  未达到阈值，继续累积")
            return False
    
    def analyze_patterns(self, namespace: str) -> Dict[str, Any]:
        """
        分析模式（处理新增的Layer2节点）
        
        Args:
            namespace: 命名空间
        
        Returns:
            统计信息
        """
        print(f"\n{'='*60}")
        print(f"🧠 Layer3模式分析")
        print(f"{'='*60}")
        
        stats = {
            'clusters_created': 0,
            'patterns_created': 0,
            'preferences_created': 0,
            'rules_created': 0
        }
        
        # 1. 获取所有Layer2节点
        all_nodes = self.cache.cache['nodes']
        all_layer2_nodes = [
            node for node in all_nodes.values()
            if node.get('layer') == 2 and node.get('type') in ['event', 'state', 'context']
        ]
        
        current_layer2_count = len(all_layer2_nodes)
        
        # 只取新增的节点（增量分析）
        new_layer2_nodes = all_layer2_nodes[self.last_analyzed_layer2_count:]
        
        print(f"\n📊 Layer2节点统计:")
        print(f"  - 总节点数: {current_layer2_count}")
        print(f"  - 上次分析: {self.last_analyzed_layer2_count}")
        print(f"  - 新增节点: {len(new_layer2_nodes)}")
        
        # 按类型统计
        new_events = [n for n in new_layer2_nodes if n.get('type') == 'event']
        new_states = [n for n in new_layer2_nodes if n.get('type') == 'state']
        new_contexts = [n for n in new_layer2_nodes if n.get('type') == 'context']
        
        print(f"  - 新增Event: {len(new_events)}")
        print(f"  - 新增State: {len(new_states)}")
        print(f"  - 新增Context: {len(new_contexts)}")
        
        if not new_layer2_nodes:
            print("  ℹ️  无新增节点，跳过分析")
            return stats
        
        # 2. 对新增的Layer2节点进行统一聚类（Event + State + Context）
        print(f"\n🔍 Layer2节点聚类（混合Event/State/Context）...")
        clusters = self.clusterer.cluster_layer2_nodes(new_layer2_nodes)
        
        if not clusters:
            print("  ℹ️  未发现聚类")
            # 更新计数器
            self.last_analyzed_layer2_count = current_layer2_count
            return stats
        
        print(f"  ✅ 发现{len(clusters)}个clusters")
        
        # 3. 对每个cluster进行模式提取
        print(f"\n🤖 模式提取（处理{len(clusters)}个clusters）...")
        
        for i, cluster_nodes in enumerate(clusters, 1):
            # 按类型分组cluster内的节点
            cluster_events = [n for n in cluster_nodes if n.get('type') == 'event']
            cluster_states = [n for n in cluster_nodes if n.get('type') == 'state']
            cluster_contexts = [n for n in cluster_nodes if n.get('type') == 'context']
            
            print(f"\n  Cluster {i}/{len(clusters)}:")
            print(f"    - Events: {len(cluster_events)}")
            print(f"    - States: {len(cluster_states)}")
            print(f"    - Contexts: {len(cluster_contexts)}")
            
            # LLM提取模式（传入该cluster的所有节点）
            extraction_result = self.extractor.extract_patterns_from_cluster(
                cluster_events,
                cluster_states,   # 只传入该cluster的states
                cluster_contexts  # 只传入该cluster的contexts
            )
            
            # 创建节点
            cluster_stats = self._create_layer3_nodes(
                extraction_result,
                namespace,
                cluster_events  # 用于创建cluster → event的连接边
            )
            
            # 累加统计
            for key in stats:
                stats[key] += cluster_stats.get(key, 0)
        
        # 更新计数器
        self.last_analyzed_layer2_count = current_layer2_count
        
        print(f"\n✅ Layer3分析完成")
        print(f"  - 创建聚类: {stats['clusters_created']}")
        print(f"  - 创建模式: {stats['patterns_created']}")
        print(f"  - 创建偏好: {stats['preferences_created']}")
        print(f"  - 创建规则: {stats['rules_created']}")
        
        return stats
    
    def _create_layer3_nodes(
        self,
        extraction_result: Dict[str, Any],
        namespace: str,
        cluster_events: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        创建Layer3节点
        
        Args:
            extraction_result: LLM提取结果
            namespace: 命名空间
            cluster_events: 聚类的事件列表
        
        Returns:
            统计信息
        """
        stats = {
            'clusters_created': 0,
            'patterns_created': 0,
            'preferences_created': 0,
            'rules_created': 0
        }
        
        nodes_to_create = []
        
        # 1. 创建事件聚类节点
        event_cluster_data = extraction_result.get('event_cluster')
        if event_cluster_data:
            self.cluster_counter += 1
            cluster_id = f"cluster_{self.cluster_counter}"
            
            cluster_node = {
                "id": cluster_id,
                "content": event_cluster_data.get('description', ''),
                "cluster_type": event_cluster_data.get('cluster_type'),
                "participants": event_cluster_data.get('participants', []),
                "time_span": event_cluster_data.get('time_span'),
                "significance": event_cluster_data.get('significance'),
                "layer": 3
            }
            
            nodes_to_create.append(cluster_node)
            stats['clusters_created'] += 1
            
            # 保存到storage
            self.storage.save_event_cluster(cluster_node, namespace)
            
            # 创建cluster → event的连接边
            for event in cluster_events:
                self.storage.create_cluster_event_edge(cluster_id, event.get('id'), namespace)
        
        # 2. 创建模式节点
        for pattern_data in extraction_result.get('patterns', []):
            self.pattern_counter += 1
            pattern_id = f"pattern_{self.pattern_counter}"
            
            pattern_node = {
                "id": pattern_id,
                "person": pattern_data.get('person'),
                "pattern_type": pattern_data.get('pattern_type'),
                "content": pattern_data.get('description', ''),
                "layer": 3
            }
            
            nodes_to_create.append(pattern_node)
            stats['patterns_created'] += 1
            
            # 保存到storage
            self.storage.save_pattern(pattern_node, namespace)
            
            # 创建pattern → person的连接边
            person_name = pattern_data.get('person')
            if person_name:
                self.storage.create_pattern_person_edge(pattern_id, person_name, namespace)
        
        # 3. 创建偏好节点
        for pref_data in extraction_result.get('preferences', []):
            self.preference_counter += 1
            pref_id = f"preference_{self.preference_counter}"
            
            pref_node = {
                "id": pref_id,
                "person": pref_data.get('person'),
                "category": pref_data.get('category'),
                "content": pref_data.get('description', ''),
                "layer": 3
            }
            
            nodes_to_create.append(pref_node)
            stats['preferences_created'] += 1
            
            # 保存到storage
            self.storage.save_preference(pref_node, namespace)
            
            # 创建preference → person的连接边
            person_name = pref_data.get('person')
            if person_name:
                self.storage.create_pattern_person_edge(pref_id, person_name, namespace)
        
        # 4. 创建行为规则节点
        for rule_data in extraction_result.get('behavior_rules', []):
            self.rule_counter += 1
            rule_id = f"rule_{self.rule_counter}"
            
            rule_node = {
                "id": rule_id,
                "person": rule_data.get('person'),
                "rule_type": rule_data.get('rule_type'),
                "content": rule_data.get('description', ''),
                "layer": 3
            }
            
            nodes_to_create.append(rule_node)
            stats['rules_created'] += 1
            
            # 保存到storage
            self.storage.save_behavior_rule(rule_node, namespace)
            
            # 创建rule → person的连接边
            person_name = rule_data.get('person')
            if person_name:
                self.storage.create_pattern_person_edge(rule_id, person_name, namespace)
        
        # 5. 批量添加到cache
        if nodes_to_create:
            print(f"  📦 批量添加{len(nodes_to_create)}个Layer3节点到cache...")
            self.cache.batch_add_nodes(nodes_to_create)
        
        return stats

