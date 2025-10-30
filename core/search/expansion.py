"""
Graph Expansion module

Traverse and expand the graph based on edges
"""

from typing import List, Dict, Any, Set, Tuple


class GraphExpansion:
    """
    Expand neighbor nodes via knowledge graph edges
    """
    
    def __init__(self, storage, cache):
        """
        Args:
            storage: Storage (read edges)
            cache: UnifiedCache (read nodes)
        """
        self.storage = storage
        self.cache = cache
    
    def expand_nodes(
        self,
        node_ids: List[str],
        max_neighbors: int = 20
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand neighbors of given nodes
        """
        if not node_ids:
            return [], []
        
        print(f"\n🔗 Graph expansion: expanding neighbors of {len(node_ids)} nodes...")
        
        # Find connecting edges
        connecting_edges = self._find_connecting_edges(node_ids)
        
        # Collect neighbor IDs
        neighbor_ids = set()
        for edge in connecting_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # Add neighbor (exclude existing)
            if source not in node_ids:
                neighbor_ids.add(source)
            if target not in node_ids:
                neighbor_ids.add(target)
        
        # Limit neighbors
        neighbor_ids = list(neighbor_ids)[:max_neighbors]
        
        # Fetch neighbor nodes
        expanded_nodes = self._get_nodes_by_ids(neighbor_ids)
        
        print(f"  ✅ Found {len(connecting_edges)} connecting edges")
        print(f"  ✅ Expanded {len(expanded_nodes)} neighbor nodes")
        
        return expanded_nodes, connecting_edges
    
    def find_candidates(
        self,
        current_node_ids: List[str],
        max_candidates: int = 50
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Find candidate nodes (for LLM decisions)
        """
        # Find all connecting edges
        connecting_edges = self._find_connecting_edges(current_node_ids)
        
        # Extract candidate IDs
        candidate_ids = set()
        for edge in connecting_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source not in current_node_ids:
                candidate_ids.add(source)
            if target not in current_node_ids:
                candidate_ids.add(target)
        
        # Limit candidates
        candidate_ids = list(candidate_ids)[:max_candidates]
        
        # Fetch candidate nodes
        candidate_nodes = self._get_nodes_by_ids(candidate_ids)
        
        return candidate_nodes, connecting_edges
    
    def _find_connecting_edges(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find all edges connecting to given node IDs
        """
        node_id_set = set(node_ids)
        connecting_edges = []
        
        # Get all edges
        all_edges = self.cache.get_all_edges()
        
        for edge in all_edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # If either source or target is in node_ids, it's connecting
            if source in node_id_set or target in node_id_set:
                connecting_edges.append(edge)
        
        return connecting_edges
    
    def _get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get nodes by IDs
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
        Shortest path between two nodes (BFS)
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
            adjacency[target].append(source)  # undirected
        
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
        
        return []  # not found

