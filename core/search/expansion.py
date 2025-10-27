"""
Graph Expansion模块

基于edges进行图遍历和扩展
"""

from typing import List, Dict, Any, Set, Tuple


class GraphExpansion:
    """
    图扩展模块
    
    基于知识图谱的edges进行邻居节点扩展
    """
    
    def __init__(self, storage, cache):
        """
        Args:
            storage: Storage实例（读取edges）
            cache: UnifiedCache实例（读取nodes）
        """
        self.storage = storage
        self.cache = cache
    
    def expand_nodes(
        self,
        node_ids: List[str],
        max_neighbors: int = 20
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        扩展指定节点的邻居
        
        Args:
            node_ids: 要扩展的节点ID列表
            max_neighbors: 最多扩展的邻居数量
        
        Returns:
            (expanded_nodes, connecting_edges)
        """
        if not node_ids:
            return [], []
        
        print(f"\n🔗 图扩展: 扩展 {len(node_ids)} 个节点的邻居...")
        
        # 查找连接的edges
        connecting_edges = self._find_connecting_edges(node_ids)
        
        # 提取邻居节点IDs
        neighbor_ids = set()
        for edge in connecting_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # 添加邻居（排除已有节点）
            if source not in node_ids:
                neighbor_ids.add(source)
            if target not in node_ids:
                neighbor_ids.add(target)
        
        # 限制邻居数量
        neighbor_ids = list(neighbor_ids)[:max_neighbors]
        
        # 获取邻居节点
        expanded_nodes = self._get_nodes_by_ids(neighbor_ids)
        
        print(f"  ✅ 找到 {len(connecting_edges)} 条边")
        print(f"  ✅ 扩展 {len(expanded_nodes)} 个邻居节点")
        
        return expanded_nodes, connecting_edges
    
    def find_candidates(
        self,
        current_node_ids: List[str],
        max_candidates: int = 50
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        查找候选节点（用于LLM决策）
        
        策略：仅通过边连接查找候选
        
        Args:
            current_node_ids: 当前已有的节点IDs
            max_candidates: 最多候选数量
        
        Returns:
            (candidate_nodes, connecting_edges)
        """
        # 查找所有连接的edges
        connecting_edges = self._find_connecting_edges(current_node_ids)
        
        # 提取候选节点IDs
        candidate_ids = set()
        for edge in connecting_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in current_node_ids:
                candidate_ids.add(source)
            if target not in current_node_ids:
                candidate_ids.add(target)
        
        # 限制候选数量
        candidate_ids = list(candidate_ids)[:max_candidates]
        
        # 获取候选节点
        candidate_nodes = self._get_nodes_by_ids(candidate_ids)
        
        return candidate_nodes, connecting_edges
    
    def _find_connecting_edges(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        查找连接到指定节点的所有edges
        
        Args:
            node_ids: 节点ID列表
        
        Returns:
            边列表
        """
        node_id_set = set(node_ids)
        connecting_edges = []
        
        # 获取所有edges
        all_edges = self.cache.get_all_edges()
        
        for edge in all_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # 如果edge的source或target在node_ids中，则连接
            if source in node_id_set or target in node_id_set:
                connecting_edges.append(edge)
        
        return connecting_edges
    
    def _get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        根据IDs获取节点
        
        Args:
            node_ids: 节点ID列表
        
        Returns:
            节点列表
        """
        nodes = []
        
        for node_id in node_ids:
            if node_id in self.cache.cache['nodes']:
                nodes.append(self.cache.cache['nodes'][node_id])
        
        return nodes
    
    def get_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 3
    ) -> List[str]:
        """
        查找两个节点之间的最短路径（BFS）
        
        Args:
            start_node_id: 起始节点ID
            end_node_id: 目标节点ID
            max_depth: 最大搜索深度
        
        Returns:
            路径上的节点ID列表（不包括起始节点）
        """
        if start_node_id == end_node_id:
            return []
        
        # BFS
        visited = {start_node_id}
        queue = [(start_node_id, [])]  # (current_node, path)
        
        all_edges = self.cache.get_all_edges()
        
        # 构建邻接表
        adjacency = {}
        for edge in all_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []
            
            adjacency[source].append(target)
            adjacency[target].append(source)  # 无向图
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.pop(0)
            
            # 获取邻居
            neighbors = adjacency.get(current, [])
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                # 找到目标
                if neighbor == end_node_id:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return []  # 未找到路径

