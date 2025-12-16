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
        storage_dir: str = "storage",
        token_tracker=None
    ):
        self.extractor = Layer2Extractor(llm_client, token_tracker=token_tracker)
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
        layer1_entities: List[Dict[str, Any]],
        existing_layer2_nodes: List[Dict[str, Any]] = None
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
        
        # Phase 0: 使用已召回的已有Layer2节点（如果未提供，则自己召回）
        if existing_layer2_nodes is None:
            print("\n🔍 Phase 0: 召回已有Layer2节点...")
            fragment_text = fragment.get('content', '')
            existing_layer2_nodes = []
            if fragment_text:
                try:
                    # 使用fragment content生成embedding并召回已有Layer2节点
                    fragment_embedding, _, _ = self.cache.get_or_generate_embedding(fragment_text)
                    existing_candidates = self.cache.filter_and_search(
                        fragment_embedding,
                        filters={'layer': 2},  # 召回Layer2节点（event/state/context）
                        top_k=10  # 召回top-10个已有节点
                    )
                    
                    # 转换为节点格式
                    for candidate in existing_candidates:
                        node = candidate.get('node', {})
                        if node:
                            existing_layer2_nodes.append(node)
                    
                    print(f"✅ 召回了 {len(existing_layer2_nodes)} 个已有Layer2节点")
                    if existing_layer2_nodes:
                        node_types = {}
                        for node in existing_layer2_nodes[:5]:
                            node_type = node.get('type', 'unknown')
                            node_types[node_type] = node_types.get(node_type, 0) + 1
                        print(f"   类型分布: {dict(node_types)}")
                except Exception as e:
                    print(f"  ⚠️  召回已有Layer2节点失败: {e}")
        else:
            print(f"\n✅ 使用已召回的 {len(existing_layer2_nodes)} 个已有Layer2节点")
        
        # Phase 1: 提取时间线信息（包含关联判断，一次LLM调用）
        print("\n🤖 Phase 1: 提取时间线信息（包含关联判断）...")
        extraction_result = self.extractor.extract_from_fragment(
            fragment, 
            layer1_entities,
            existing_layer2_nodes=existing_layer2_nodes
        )
        
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
        
        # Phase 2: 批量创建节点（处理link_to_existing）
        nodes_to_create = []
        event_ids = []
        state_ids = []
        context_ids = []
        fragment_edges = []  # Fragment到Layer2节点的连接边 [(fragment_id, node_id, rel_type)]
        
        # 创建事件节点
        for event in events:
            action = event.get('action', 'create_new')  # 默认为create_new
            
            if action == 'create_new':
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
                
                # 创建fragment → 新事件的连接边
                fragment_edges.append((fragment_id, event_id, "contains"))
                
                # 处理link_to_existing（新事件与已有事件建立关联）
                link_to_existing = event.get('link_to_existing', [])
                if link_to_existing:
                    for existing_node_id in link_to_existing:
                        if isinstance(existing_node_id, str):
                            # 创建fragment → 已有事件的连接边（表示关联）
                            fragment_edges.append((fragment_id, existing_node_id, "related_to"))
                            print(f"  🔗 新事件 {event_id} 关联到已有事件 {existing_node_id}")
        
        # 创建状态节点
        for state in states:
            action = state.get('action', 'create_new')  # 默认为create_new
            
            if action == 'create_new':
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
                
                # 创建fragment → 新状态的连接边
                fragment_edges.append((fragment_id, state_id, "contains"))
                
                # 处理link_to_existing
                link_to_existing = state.get('link_to_existing', [])
                if link_to_existing:
                    for existing_node_id in link_to_existing:
                        if isinstance(existing_node_id, str):
                            fragment_edges.append((fragment_id, existing_node_id, "related_to"))
                            print(f"  🔗 新状态 {state_id} 关联到已有状态 {existing_node_id}")
        
        # 创建上下文节点
        for context in contexts:
            action = context.get('action', 'create_new')  # 默认为create_new
            
            if action == 'create_new':
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
                
                # 创建fragment → 新上下文的连接边
                fragment_edges.append((fragment_id, context_id, "occurs_in"))
                
                # 处理link_to_existing
                link_to_existing = context.get('link_to_existing', [])
                if link_to_existing:
                    for existing_node_id in link_to_existing:
                        if isinstance(existing_node_id, str):
                            fragment_edges.append((fragment_id, existing_node_id, "related_to"))
                            print(f"  🔗 新上下文 {context_id} 关联到已有上下文 {existing_node_id}")
        
        # Phase 3: 不立即添加到cache，而是返回节点（供后续统一批量生成embedding和写入Neo4j）
        if nodes_to_create:
            print(f"\n📦 准备创建 {len(nodes_to_create)} 个时间线节点...")
        
        # Phase 4: 准备结构性边信息（Layer2节点到Entity的边）
        structural_edges = []  # Layer2节点到Entity的结构性边 [(source_id, target_id, rel_type)]
        
        # 准备事件到Entity的边
        for i, event_id in enumerate(event_ids):
            event_data = events[i]
            
            # 创建event -> entity的参与者边（结构性边，无content）
            participants = event_data.get('participants', [])
            for participant in participants:
                if participant:
                    # 查找匹配的entity
                    matched_entities = self._find_matching_entities(participant, layer1_entities)
                    for entity_id in matched_entities:
                        structural_edges.append((event_id, entity_id, "involves"))
            
            print(f"  ✅ 准备事件: {event_data.get('content', '')[:50]}...")
        
        # 准备状态到Entity的边
        for i, state_id in enumerate(state_ids):
            state_data = states[i]
            
            # 创建state -> entity的参与者边
            participants = state_data.get('participants', [])
            for participant in participants:
                if participant:
                    matched_entities = self._find_matching_entities(participant, layer1_entities)
                    for entity_id in matched_entities:
                        structural_edges.append((state_id, entity_id, "describes"))
            
            print(f"  ✅ 准备状态: {state_data.get('content', '')[:50]}...")
        
        # 准备上下文到Entity的边
        for i, context_id in enumerate(context_ids):
            context_data = contexts[i]
            
            # 创建context -> entity的影响边
            affected_entities = context_data.get('affected_entities', [])
            for affected_entity in affected_entities:
                if affected_entity:
                    matched_entities = self._find_matching_entities(affected_entity, layer1_entities)
                    for entity_id in matched_entities:
                        structural_edges.append((context_id, entity_id, "affects"))
            
            print(f"  ✅ 准备上下文: {context_data.get('content', '')[:50]}...")
        
        print(f"\n✅ Layer2处理完成")
        
        return {
            'events_created': len(event_ids),
            'states_created': len(state_ids),
            'contexts_created': len(context_ids),
            'created_nodes': nodes_to_create,  # 返回节点列表
            'fragment_edges': fragment_edges,  # Fragment到Layer2节点的连接边
            'structural_edges': structural_edges  # Layer2节点到Entity的结构性边
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

