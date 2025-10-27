"""
Layer3节点聚类模块

基于embedding相似度的图聚类算法，支持Event/State/Context混合聚类
"""

import numpy as np
from typing import List, Dict, Any, Optional


class EventClusterer:
    """Layer2节点聚类器（支持Event/State/Context混合聚类）"""
    
    def __init__(self, cache, similarity_threshold: float = 0.6, min_cluster_size: int = 7):
        """
        初始化聚类器
        
        Args:
            cache: UnifiedCache实例
            similarity_threshold: 相似度阈值（0-1）
            min_cluster_size: 最小cluster大小
        """
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
    
    def cluster_layer2_nodes(self, nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        对Layer2节点进行聚类（Event + State + Context混合）
        
        Args:
            nodes: Layer2节点列表（可包含event, state, context）
        
        Returns:
            List[List[node]]: 聚类结果，每个cluster是一个节点列表
        """
        n = len(nodes)
        
        if n < self.min_cluster_size:
            print(f"  ℹ️  节点数量({n})不足，跳过聚类")
            return []
        
        print(f"  🔍 开始聚类 {n} 个Layer2节点...")
        
        # 统计节点类型
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"    节点分布: {node_types}")
        
        # 1. 获取所有节点的embedding
        embeddings = []
        valid_indices = []
        
        for i, node in enumerate(nodes):
            emb_idx = node.get('embedding_idx')
            if emb_idx is not None and emb_idx < len(self.cache.embeddings):
                embeddings.append(self.cache.embeddings[emb_idx])
                valid_indices.append(i)
            else:
                print(f"    ⚠️  节点{node.get('id')}无embedding，跳过")
        
        if len(valid_indices) < self.min_cluster_size:
            print(f"  ℹ️  有效节点数({len(valid_indices)})不足")
            return []
        
        # 2. 批量计算相似度矩阵（numpy加速）
        embeddings_array = np.array(embeddings)
        
        # 归一化
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-8)
        
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(normalized, normalized.T)
        
        print(f"  ✅ 计算相似度矩阵完成 ({len(valid_indices)}x{len(valid_indices)})")
        
        # 3. 构建邻接图（相似度>threshold则连边）
        graph = {i: [] for i in range(len(valid_indices))}
        edge_count = 0
        
        for i in range(len(valid_indices)):
            for j in range(i+1, len(valid_indices)):
                if similarity_matrix[i][j] > self.similarity_threshold:
                    graph[i].append(j)
                    graph[j].append(i)
                    edge_count += 1
        
        print(f"  ✅ 构建图完成：{edge_count} 条边（相似度>{self.similarity_threshold}）")
        
        # 4. DFS提取连通分量作为clusters
        visited = [False] * len(valid_indices)
        clusters = []
        
        def dfs(node, current_cluster):
            """深度优先搜索"""
            visited[node] = True
            current_cluster.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor, current_cluster)
        
        for i in range(len(valid_indices)):
            if not visited[i]:
                cluster_indices = []
                dfs(i, cluster_indices)
                
                # 只保留满足最小大小的cluster
                if len(cluster_indices) >= self.min_cluster_size:
                    cluster_nodes = [nodes[valid_indices[idx]] for idx in cluster_indices]
                    clusters.append(cluster_nodes)
        
        print(f"  ✅ 聚类完成：发现 {len(clusters)} 个clusters")
        
        # 输出cluster详情（按类型统计）
        for i, cluster in enumerate(clusters, 1):
            cluster_types = {}
            for node in cluster:
                node_type = node.get('type', 'unknown')
                cluster_types[node_type] = cluster_types.get(node_type, 0) + 1
            print(f"    Cluster {i}: {len(cluster)} 个节点 {cluster_types}")
        
        return clusters
    
    def cluster_events(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        对事件进行聚类（兼容旧接口）
        
        Args:
            events: 事件列表
        
        Returns:
            List[List[event]]: 聚类结果
        """
        # 直接调用通用聚类方法
        return self.cluster_layer2_nodes(events)
