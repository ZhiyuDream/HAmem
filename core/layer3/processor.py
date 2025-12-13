"""
Layer3处理器

协调聚类、提取和存储的完整流程
"""

from typing import Dict, List, Any, Optional
from core.infrastructure import LLMClient, UnifiedCache
from core.infrastructure.neo4j_client import Neo4jClient
from .clustering import EventClusterer
from .extractor import Layer3Extractor
from .neo4j_storage import Layer3Neo4jStorage


class Layer3Processor:
    """Layer3处理器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        cache: UnifiedCache,
        neo4j_client: Optional[Neo4jClient] = None,
        namespace: str = "default",
        layer2_threshold: int = 60,  # 累积60个Layer2节点触发分析
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 7,  # 最小cluster大小
        token_tracker=None
    ):
        self.clusterer = EventClusterer(
            cache, 
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        self.extractor = Layer3Extractor(llm_client, token_tracker=token_tracker)
        # 使用Neo4jStorage替代文件存储
        if neo4j_client:
            self.storage = Layer3Neo4jStorage(neo4j_client, namespace)
        else:
            self.storage = None  # 如果没有Neo4j客户端，storage为None（数据将通过统一写入）
        self.cache = cache
        self.neo4j_client = neo4j_client
        self.namespace = namespace
        
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
            'rules_created': 0,
            'created_nodes': [],  # 新增：返回创建的节点列表
            'created_edges': []   # 新增：返回创建的边列表
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
            
            # 创建节点（返回节点和边列表）
            cluster_stats = self._create_layer3_nodes(
                extraction_result,
                namespace,
                cluster_events  # 用于创建cluster → event的连接边
            )
            
            # 累加统计
            for key in ['clusters_created', 'patterns_created', 'preferences_created', 'rules_created']:
                stats[key] += cluster_stats.get(key, 0)
            
            # 收集创建的节点和边
            if 'created_nodes' in cluster_stats:
                stats['created_nodes'].extend(cluster_stats['created_nodes'])
            if 'created_edges' in cluster_stats:
                stats['created_edges'].extend(cluster_stats['created_edges'])
        
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
    ) -> Dict[str, Any]:
        """
        创建Layer3节点
        
        Args:
            extraction_result: LLM提取结果
            namespace: 命名空间
            cluster_events: 聚类的事件列表
        
        Returns:
            统计信息和创建的节点/边列表
        """
        stats = {
            'clusters_created': 0,
            'patterns_created': 0,
            'preferences_created': 0,
            'rules_created': 0,
            'created_nodes': [],  # 新增：返回创建的节点列表
            'created_edges': []   # 新增：返回创建的边列表
        }
        
        nodes_to_create = []
        edges_to_create = []
        
        # 1. 创建事件聚类节点
        current_cluster_id = None  # 初始化，供后续pattern/preference/behavior_rule连接使用
        event_cluster_data = extraction_result.get('event_cluster')
        if event_cluster_data:
            self.cluster_counter += 1
            cluster_id = f"cluster_{self.cluster_counter}"
            current_cluster_id = cluster_id  # 保存cluster_id，供后续pattern/preference/behavior_rule连接使用
            
            cluster_node = {
                "id": cluster_id,
                "type": "event_cluster",  # 新增：明确节点类型
                "content": event_cluster_data.get('description', ''),
                "cluster_type": event_cluster_data.get('cluster_type'),
                "participants": event_cluster_data.get('participants', []),
                "time_span": event_cluster_data.get('time_span'),
                "significance": event_cluster_data.get('significance'),
                "layer": 3,
                "active": True
            }
            
            nodes_to_create.append(cluster_node)
            stats['clusters_created'] += 1
            
            # 收集cluster → event的连接边（不再直接写入storage）
            for event in cluster_events:
                event_id = event.get('id')
                if event_id:
                    edges_to_create.append({
                        "id": f"cluster_event_{cluster_id}_{event_id}",
                        "source": cluster_id,
                        "target": event_id,
                        "type": "CONTAINS",
                        "layer": 3,
                        "active": True
                    })
        
        # 2. 创建模式节点
        for pattern_data in extraction_result.get('patterns', []):
            self.pattern_counter += 1
            pattern_id = f"pattern_{self.pattern_counter}"
            
            pattern_node = {
                "id": pattern_id,
                "type": "pattern",  # 新增：明确节点类型
                "person": pattern_data.get('person'),
                "pattern_type": pattern_data.get('pattern_type'),
                "content": pattern_data.get('description', ''),
                "layer": 3,
                "active": True
            }
            
            nodes_to_create.append(pattern_node)
            stats['patterns_created'] += 1
            
            # 收集pattern → person的连接边（不再直接写入storage）
            person_name = pattern_data.get('person')
            if person_name:
                # 注意：person可能是实体名称，需要找到对应的entity_id
                # 这里先创建边，entity_id会在memory.py中解析
                edges_to_create.append({
                    "id": f"pattern_person_{pattern_id}_{person_name}",
                    "source": pattern_id,
                    "target": person_name,  # 可能是名称，需要后续解析为entity_id
                    "type": "RELATED_TO",
                    "layer": 3,
                    "active": True
                })
            
            # 新增：event_cluster → pattern的连接边（通过cluster间接连接到event）
            if current_cluster_id:
                edges_to_create.append({
                    "id": f"cluster_pattern_{current_cluster_id}_{pattern_id}",
                    "source": current_cluster_id,
                    "target": pattern_id,
                    "type": "CONTAINS",
                    "layer": 3,
                    "active": True
                })
        
        # 3. 创建偏好节点
        for pref_data in extraction_result.get('preferences', []):
            self.preference_counter += 1
            pref_id = f"preference_{self.preference_counter}"
            
            pref_node = {
                "id": pref_id,
                "type": "preference",  # 新增：明确节点类型
                "person": pref_data.get('person'),
                "category": pref_data.get('category'),
                "content": pref_data.get('description', ''),
                "layer": 3,
                "active": True
            }
            
            nodes_to_create.append(pref_node)
            stats['preferences_created'] += 1
            
            # 收集preference → person的连接边（不再直接写入storage）
            person_name = pref_data.get('person')
            if person_name:
                edges_to_create.append({
                    "id": f"preference_person_{pref_id}_{person_name}",
                    "source": pref_id,
                    "target": person_name,  # 可能是名称，需要后续解析为entity_id
                    "type": "RELATED_TO",
                    "layer": 3,
                    "active": True
                })
            
            # 新增：event_cluster → preference的连接边（通过cluster间接连接到event）
            if current_cluster_id:
                edges_to_create.append({
                    "id": f"cluster_preference_{current_cluster_id}_{pref_id}",
                    "source": current_cluster_id,
                    "target": pref_id,
                    "type": "CONTAINS",
                    "layer": 3,
                    "active": True
                })
        
        # 4. 创建行为规则节点
        for rule_data in extraction_result.get('behavior_rules', []):
            self.rule_counter += 1
            rule_id = f"rule_{self.rule_counter}"
            
            rule_node = {
                "id": rule_id,
                "type": "behavior_rule",  # 新增：明确节点类型
                "person": rule_data.get('person'),
                "rule_type": rule_data.get('rule_type'),
                "content": rule_data.get('description', ''),
                "layer": 3,
                "active": True
            }
            
            nodes_to_create.append(rule_node)
            stats['rules_created'] += 1
            
            # 收集rule → person的连接边（不再直接写入storage）
            person_name = rule_data.get('person')
            if person_name:
                edges_to_create.append({
                    "id": f"rule_person_{rule_id}_{person_name}",
                    "source": rule_id,
                    "target": person_name,  # 可能是名称，需要后续解析为entity_id
                    "type": "RELATED_TO",
                    "layer": 3,
                    "active": True
                })
        
            # 新增：event_cluster → behavior_rule的连接边（通过cluster间接连接到event）
            if current_cluster_id:
                edges_to_create.append({
                    "id": f"cluster_rule_{current_cluster_id}_{rule_id}",
                    "source": current_cluster_id,
                    "target": rule_id,
                    "type": "CONTAINS",
                    "layer": 3,
                    "active": True
                })
        
        # 5. 不直接写入storage，而是返回节点和边列表（供memory.py统一处理）
        stats['created_nodes'] = nodes_to_create
        stats['created_edges'] = edges_to_create
        
        return stats

