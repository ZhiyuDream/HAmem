"""
Layer2处理器

协调提取、存储和缓存的完整流程
"""

from typing import Dict, List, Any
from core.infrastructure import LLMClient, UnifiedCache
from .extractor import Layer2Extractor
from .storage import Layer2Storage


class Layer2Processor:
    """Layer2处理器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        cache: UnifiedCache,
        storage_dir: str = "storage"
    ):
        self.extractor = Layer2Extractor(llm_client)
        self.storage = Layer2Storage(storage_dir)
        self.cache = cache
        
        # 计数器
        self.event_counter = 0
        self.state_counter = 0
        self.context_counter = 0
    
    def process_fragment(
        self,
        fragment: Dict[str, Any],
        namespace: str,
        layer1_entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        处理单个fragment，提取时间线信息
        
        Args:
            fragment: fragment数据
            namespace: 命名空间
            layer1_entities: Layer1提取的实体列表
        
        Returns:
            处理统计信息
        """
        fragment_id = fragment.get('id')
        
        print(f"\n{'='*60}")
        print(f"📅 Layer2处理Fragment: {fragment_id}")
        print(f"{'='*60}")
        
        # Phase 1: 提取时间线信息 (LLM调用)
        print("\n🤖 提取时间线信息...")
        extraction_result = self.extractor.extract_from_fragment(fragment, layer1_entities)
        
        events = extraction_result.get('events', [])
        states = extraction_result.get('states', [])
        contexts = extraction_result.get('contexts', [])
        
        print(f"✅ 提取了 {len(events)} 个事件, {len(states)} 个状态, {len(contexts)} 个上下文")
        
        if not events and not states and not contexts:
            print("ℹ️  无提取结果，跳过Layer2处理")
            return {
                'events_created': 0,
                'states_created': 0,
                'contexts_created': 0
            }
        
        # Phase 2: 批量创建节点（无冲突检测）
        nodes_to_create = []
        event_ids = []
        state_ids = []
        context_ids = []
        
        # 创建事件节点
        for event in events:
            self.event_counter += 1
            event_id = f"event_{self.event_counter}"
            event_ids.append(event_id)
            
            node = {
                "id": event_id,
                "type": "event",
                "content": event.get('content', ''),
                "participants": event.get('participants', []),
                "location": event.get('location'),
                "conversation_time": event.get('conversation_time'),
                "relative_time": event.get('relative_time'),
                "layer": 2
            }
            nodes_to_create.append(node)
        
        # 创建状态节点
        for state in states:
            self.state_counter += 1
            state_id = f"state_{self.state_counter}"
            state_ids.append(state_id)
            
            node = {
                "id": state_id,
                "type": "state",
                "content": state.get('content', ''),
                "participants": state.get('participants', []),
                "conversation_time": state.get('conversation_time'),
                "relative_time": state.get('relative_time'),
                "duration": state.get('duration'),
                "layer": 2
            }
            nodes_to_create.append(node)
        
        # 创建上下文节点
        for context in contexts:
            self.context_counter += 1
            context_id = f"context_{self.context_counter}"
            context_ids.append(context_id)
            
            node = {
                "id": context_id,
                "type": "context",
                "content": context.get('content', ''),
                "affected_entities": context.get('affected_entities', []),
                "conversation_time": context.get('conversation_time'),
                "relative_time": context.get('relative_time'),
                "impact": context.get('impact'),
                "layer": 2
            }
            nodes_to_create.append(node)
        
        # Phase 3: 批量添加到cache
        if nodes_to_create:
            print(f"\n📦 批量创建 {len(nodes_to_create)} 个时间线节点...")
            self.cache.batch_add_nodes(nodes_to_create)
        
        # Phase 4: 保存到storage并创建连接边
        print(f"\n💾 保存到storage...")
        
        # 保存事件
        for i, event_id in enumerate(event_ids):
            event_data = events[i]
            node = {
                "id": event_id,
                "content": event_data.get('content', ''),
                "participants": event_data.get('participants', []),
                "location": event_data.get('location'),
                "conversation_time": event_data.get('conversation_time'),
                "relative_time": event_data.get('relative_time')
            }
            self.storage.save_timeline_node(node, namespace, "event")
            self.storage.create_fragment_connection_edge(
                fragment_id, event_id, "contains", namespace
            )
            
            # 创建event -> entity的参与者边（结构性边，无content）
            participants = event_data.get('participants', [])
            for participant in participants:
                if participant:
                    # 查找匹配的entity
                    matched_entities = self._find_matching_entities(participant, layer1_entities)
                    for entity_id in matched_entities:
                        self.storage.create_structural_edge(
                            event_id, entity_id, "involves", namespace
                        )
            
            print(f"  ✅ 创建事件: {event_data.get('content', '')[:50]}...")
        
        # 保存状态
        for i, state_id in enumerate(state_ids):
            state_data = states[i]
            node = {
                "id": state_id,
                "content": state_data.get('content', ''),
                "participants": state_data.get('participants', []),
                "conversation_time": state_data.get('conversation_time'),
                "relative_time": state_data.get('relative_time'),
                "duration": state_data.get('duration')
            }
            self.storage.save_timeline_node(node, namespace, "state")
            self.storage.create_fragment_connection_edge(
                fragment_id, state_id, "contains", namespace
            )
            
            # 创建state -> entity的参与者边
            participants = state_data.get('participants', [])
            for participant in participants:
                if participant:
                    matched_entities = self._find_matching_entities(participant, layer1_entities)
                    for entity_id in matched_entities:
                        self.storage.create_structural_edge(
                            state_id, entity_id, "describes", namespace
                        )
            
            print(f"  ✅ 创建状态: {state_data.get('content', '')[:50]}...")
        
        # 保存上下文
        for i, context_id in enumerate(context_ids):
            context_data = contexts[i]
            node = {
                "id": context_id,
                "content": context_data.get('content', ''),
                "affected_entities": context_data.get('affected_entities', []),
                "conversation_time": context_data.get('conversation_time'),
                "relative_time": context_data.get('relative_time'),
                "impact": context_data.get('impact')
            }
            self.storage.save_timeline_node(node, namespace, "context")
            self.storage.create_fragment_connection_edge(
                fragment_id, context_id, "occurs_in", namespace
            )
            
            # 创建context -> entity的影响边
            affected_entities = context_data.get('affected_entities', [])
            for affected_entity in affected_entities:
                if affected_entity:
                    matched_entities = self._find_matching_entities(affected_entity, layer1_entities)
                    for entity_id in matched_entities:
                        self.storage.create_structural_edge(
                            context_id, entity_id, "affects", namespace
                        )
            
            print(f"  ✅ 创建上下文: {context_data.get('content', '')[:50]}...")
        
        print(f"\n✅ Layer2处理完成")
        
        return {
            'events_created': len(event_ids),
            'states_created': len(state_ids),
            'contexts_created': len(context_ids)
        }
    
    def _find_matching_entities(self, entity_name: str, layer1_entities: List[Dict]) -> List[str]:
        """
        根据名称查找匹配的entity ID
        
        Args:
            entity_name: 实体名称（如 "user", "shopping mall"）
            layer1_entities: Layer1创建的实体列表
        
        Returns:
            匹配的entity ID列表
        """
        matched_ids = []
        
        if not entity_name or not layer1_entities:
            return matched_ids
        
        entity_name_lower = entity_name.lower().strip()
        
        for entity in layer1_entities:
            entity_node_name = entity.get('name', '').lower().strip()
            entity_id = entity.get('id')
            
            # 简单的名称匹配
            if entity_id and (entity_name_lower == entity_node_name or 
                             entity_name_lower in entity_node_name or
                             entity_node_name in entity_name_lower):
                matched_ids.append(entity_id)
        
        return matched_ids

