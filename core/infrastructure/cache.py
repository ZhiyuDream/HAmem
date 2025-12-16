"""
统一缓存管理模块

功能：
1. Embedding去重 - 相同content只计算一次embedding
2. 高性能检索 - FAISS毫秒级查询
3. 持久化 - 跨程序运行保持数据
4. 命名空间隔离 - 不同测试互不干扰
"""

import os
import json
import hashlib
import numpy as np
import faiss
from typing import Dict, List, Optional, Any, Tuple
from .embedding import EmbeddingManager


class UnifiedCache:
    """统一的embedding缓存系统"""
    
    def __init__(self, cache_dir: str, namespace: str, embedding_manager: EmbeddingManager = None):
        """
        初始化缓存
        
        Args:
            cache_dir: 缓存根目录
            namespace: 命名空间（用于隔离不同测试）
            embedding_manager: Embedding管理器（可选）
        """
        self.cache_dir = os.path.join(cache_dir, namespace)
        self.namespace = namespace
        self.embedding_manager = embedding_manager
        
        # 文件路径
        self.cache_file = os.path.join(self.cache_dir, 'cache.json')
        self.embeddings_file = os.path.join(self.cache_dir, 'embeddings.npy')
        self.faiss_file = os.path.join(self.cache_dir, 'faiss_index.bin')
        self.hash_map_file = os.path.join(self.cache_dir, 'content_hash_map.json')
        
        # 确保目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 内存数据结构
        self.cache = {
            'metadata': {
                'version': '1.0',
                'embedding_model': 'text-embedding-3-small',
                'embedding_dim': 1536,
                'namespace': namespace
            },
            'nodes': {},
            'edges': {}
        }
        
        # Embedding管理（去重）
        self.content_hash_to_idx = {}  # content_hash → embedding_idx
        self.idx_to_node_id = {}       # embedding_idx → [node_id, ...]
        self.idx_to_edge_id = {}       # embedding_idx → [edge_id, ...]
        self.embeddings = []           # List[np.ndarray]
        self.faiss_index = None        # FAISS索引
        
        # 加载现有缓存
        self._load_cache()
    
    def _compute_content_hash(self, content: str) -> str:
        """计算content的hash值"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """从磁盘加载缓存到内存"""
        print(f"📂 正在加载cache: {self.namespace}...")
        
        # 1. 加载cache.json
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            print(f"✅ 加载了 {len(self.cache['nodes'])} 个节点, {len(self.cache['edges'])} 个边")
        else:
            print(f"ℹ️  Cache不存在，创建新cache")
        
        # 2. 加载embeddings.npy
        if os.path.exists(self.embeddings_file):
            embeddings_array = np.load(self.embeddings_file)
            self.embeddings = list(embeddings_array)
            print(f"✅ 加载了 {len(self.embeddings)} 个embedding向量")
        
        # 3. 加载FAISS索引
        if os.path.exists(self.faiss_file):
            self.faiss_index = faiss.read_index(self.faiss_file)
            print(f"✅ 加载了FAISS索引，包含 {self.faiss_index.ntotal} 个向量")
        
        # 4. 加载content_hash映射
        if os.path.exists(self.hash_map_file):
            with open(self.hash_map_file, 'r', encoding='utf-8') as f:
                self.content_hash_to_idx = json.load(f)
            print(f"✅ 加载了 {len(self.content_hash_to_idx)} 个hash映射")
        
        # 5. 加载storage中的所有边（包括结构性边）
        storage_edges_file = os.path.join(
            os.path.dirname(self.cache_dir),
            'storage',
            self.namespace,
            'edges.jsonl'
        )
        
        if os.path.exists(storage_edges_file):
            structural_edges_count = 0
            with open(storage_edges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        edge = json.loads(line)
                        edge_id = edge.get('id')
                        
                        # 如果cache中已经有这条边（有embedding的边），跳过
                        if edge_id in self.cache['edges']:
                            continue
                        
                        # 这是结构性边（没有content，不需要embedding）
                        self.cache['edges'][edge_id] = {
                            **edge,
                            'embedding_idx': -1,  # 标记为无embedding
                            'content_hash': ''
                        }
                        structural_edges_count += 1
            
            if structural_edges_count > 0:
                print(f"✅ 加载了 {structural_edges_count} 条结构性边（无embedding）")
        
        # 6. 重建反向索引
        for node_id, node in self.cache['nodes'].items():
            embedding_idx = node.get('embedding_idx')
            if embedding_idx is not None and embedding_idx != -1:
                if embedding_idx not in self.idx_to_node_id:
                    self.idx_to_node_id[embedding_idx] = []
                self.idx_to_node_id[embedding_idx].append(node_id)
        
        for edge_id, edge in self.cache['edges'].items():
            embedding_idx = edge.get('embedding_idx')
            if embedding_idx is not None and embedding_idx != -1:
                if embedding_idx not in self.idx_to_edge_id:
                    self.idx_to_edge_id[embedding_idx] = []
                self.idx_to_edge_id[embedding_idx].append(edge_id)
        
        print(f"🚀 Cache加载完成！")
    
    def get_or_generate_embedding(self, content: str) -> Tuple[np.ndarray, int, bool]:
        """
        获取或生成embedding（自动去重）
        
        Args:
            content: 文本内容
        
        Returns:
            (embedding, embedding_idx, is_new)
        """
        if not content or not content.strip():
            # 空内容返回零向量
            zero_embedding = np.zeros(1536, dtype='float32')
            return zero_embedding, -1, False
        
        # 计算content hash
        content_hash = self._compute_content_hash(content)
        
        # 检查是否已存在
        if content_hash in self.content_hash_to_idx:
            idx = self.content_hash_to_idx[content_hash]
            embedding = self.embeddings[idx]
            return embedding, idx, False  # 复用
        
        # 生成新embedding
        if self.embedding_manager is None:
            raise ValueError("EmbeddingManager未初始化，无法生成embedding")
        
        embedding = self.embedding_manager.get_embedding(content)
        
        # 验证embedding
        if not embedding or len(embedding) != 1536:
            raise ValueError(f"无效的embedding: 长度={len(embedding) if embedding else 0}, 期望1536")
        
        embedding_array = np.array(embedding, dtype='float32')
        
        # 验证数组形状
        if embedding_array.shape != (1536,):
            raise ValueError(f"Embedding数组形状错误: {embedding_array.shape}, 期望(1536,)")
        
        # 添加到索引
        idx = len(self.embeddings)
        self.embeddings.append(embedding_array)
        self.content_hash_to_idx[content_hash] = idx
        
        # 更新FAISS索引
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(1536)  # 内积索引
        self.faiss_index.add(embedding_array.reshape(1, -1))
        
        return embedding_array, idx, True  # 新生成
    
    def add_node(self, node: Dict[str, Any]) -> None:
        """
        添加节点到缓存（自动处理embedding去重）
        
        Args:
            node: 节点数据，必须包含id和content
        """
        node_id = node.get('id')
        if not node_id:
            raise ValueError("节点必须包含id字段")
        
        content = node.get('content', '')
        
        # 获取或生成embedding
        embedding, embedding_idx, is_new = self.get_or_generate_embedding(content)
        
        if is_new:
            print(f"🔄 生成新embedding [{embedding_idx}]: {content[:50]}...")
        else:
            print(f"✅ 复用embedding [{embedding_idx}]: {content[:50]}...")
        
        # 计算content hash
        content_hash = self._compute_content_hash(content)
        
        # 保存节点数据
        self.cache['nodes'][node_id] = {
            **node,
            'embedding_idx': embedding_idx,
            'content_hash': content_hash
        }
        
        # 维护反向索引
        if embedding_idx not in self.idx_to_node_id:
            self.idx_to_node_id[embedding_idx] = []
        if node_id not in self.idx_to_node_id[embedding_idx]:
            self.idx_to_node_id[embedding_idx].append(node_id)
    
    def add_edge(self, edge: Dict[str, Any]) -> None:
        """
        添加边到缓存
        
        Args:
            edge: 边数据，必须包含id和content
        """
        edge_id = edge.get('id')
        if not edge_id:
            raise ValueError("边必须包含id字段")
        
        content = edge.get('content', '')
        
        # 获取或生成embedding
        embedding, embedding_idx, is_new = self.get_or_generate_embedding(content)
        
        if is_new:
            print(f"🔄 生成新embedding [{embedding_idx}]: {content[:50]}...")
        else:
            print(f"✅ 复用embedding [{embedding_idx}]: {content[:50]}...")
        
        # 计算content hash
        content_hash = self._compute_content_hash(content)
        
        # 保存边数据
        self.cache['edges'][edge_id] = {
            **edge,
            'embedding_idx': embedding_idx,
            'content_hash': content_hash
        }
        
        # 维护反向索引
        if embedding_idx not in self.idx_to_edge_id:
            self.idx_to_edge_id[embedding_idx] = []
        if edge_id not in self.idx_to_edge_id[embedding_idx]:
            self.idx_to_edge_id[embedding_idx].append(edge_id)
    
    def batch_add_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """
        批量添加节点（优化：一次性生成所有embedding）
        
        Args:
            nodes: 节点列表，每个节点必须包含id和content
        """
        if not nodes:
            return
        
        # 1. 收集需要生成embedding的content
        contents_to_generate = []
        content_to_nodes = {}  # content -> [nodes]
        
        for node in nodes:
            node_id = node.get('id')
            if not node_id:
                raise ValueError("节点必须包含id字段")
            
            content = node.get('content', '')
            content_hash = self._compute_content_hash(content)
            
            # 检查是否已存在embedding
            if content_hash not in self.content_hash_to_idx:
                if content not in content_to_nodes:
                    contents_to_generate.append(content)
                    content_to_nodes[content] = []
                content_to_nodes[content].append((node, content_hash))
        
        # 2. 批量生成embedding（一次API调用）
        if contents_to_generate:
            # 过滤和记录空内容
            valid_contents = []
            content_index_map = {}  # 原始索引 → 有效内容索引
            
            for i, content in enumerate(contents_to_generate):
                if content and content.strip():
                    content_index_map[i] = len(valid_contents)
                    valid_contents.append(content)
                else:
                    print(f"⚠️  跳过空内容")
            
            if not valid_contents:
                print("⚠️  所有内容为空，跳过embedding生成")
                # 仍需添加节点，但没有embedding
                for node in nodes:
                    node_id = node.get('id')
                    self.cache['nodes'][node_id] = {**node, 'embedding_idx': -1, 'content_hash': ''}
                return
            
            print(f"🚀 批量生成 {len(valid_contents)} 个embedding...")
            embeddings = self.embedding_manager.batch_get_embeddings(valid_contents)
            
            # 更新embedding索引（只处理有效内容）
            for i, content in enumerate(contents_to_generate):
                if i in content_index_map:
                    # 有效内容，获取对应的embedding
                    valid_idx = content_index_map[i]
                    embedding = embeddings[valid_idx] if valid_idx < len(embeddings) else None
                else:
                    # 空内容，没有embedding
                    embedding = None
                
                if embedding:  # 确保embedding生成成功
                    embedding_array = np.array(embedding, dtype='float32')
                    idx = len(self.embeddings)
                    self.embeddings.append(embedding_array)
                    
                    content_hash = self._compute_content_hash(content)
                    self.content_hash_to_idx[content_hash] = idx
                    
                    # 添加到FAISS索引
                    if self.faiss_index is None:
                        self.faiss_index = faiss.IndexFlatIP(1536)
                    self.faiss_index.add(embedding_array.reshape(1, -1))
                    
                    print(f"  ✅ 新embedding [{idx}]: {content[:50]}...")
        
        # 3. 添加所有节点
        for node in nodes:
            node_id = node.get('id')
            content = node.get('content', '')
            content_hash = self._compute_content_hash(content)
            
            # 获取embedding索引
            embedding_idx = self.content_hash_to_idx.get(content_hash, -1)
            
            if embedding_idx == -1:
                print(f"  ℹ️  复用embedding: {content[:50]}...")
                # 查找已存在的embedding
                for existing_hash, idx in self.content_hash_to_idx.items():
                    if existing_hash == content_hash:
                        embedding_idx = idx
                        break
            
            # 保存节点数据
            self.cache['nodes'][node_id] = {
                **node,
                'embedding_idx': embedding_idx,
                'content_hash': content_hash
            }
            
            # 维护反向索引
            if embedding_idx != -1:
                if embedding_idx not in self.idx_to_node_id:
                    self.idx_to_node_id[embedding_idx] = []
                if node_id not in self.idx_to_node_id[embedding_idx]:
                    self.idx_to_node_id[embedding_idx].append(node_id)
    
    def batch_add_edges(self, edges: List[Dict[str, Any]]) -> None:
        """
        批量添加边（优化：一次性生成所有embedding）
        
        Args:
            edges: 边列表，每个边必须包含id和content
        """
        if not edges:
            return
        
        # 1. 收集需要生成embedding的content
        contents_to_generate = []
        content_to_edges = {}
        
        for edge in edges:
            edge_id = edge.get('id')
            if not edge_id:
                raise ValueError("边必须包含id字段")
            
            content = edge.get('content', '')
            content_hash = self._compute_content_hash(content)
            
            # 检查是否已存在embedding
            if content_hash not in self.content_hash_to_idx:
                if content not in content_to_edges:
                    contents_to_generate.append(content)
                    content_to_edges[content] = []
                content_to_edges[content].append((edge, content_hash))
        
        # 2. 批量生成embedding
        if contents_to_generate:
            # 过滤空内容
            valid_contents = []
            content_index_map = {}
            
            for i, content in enumerate(contents_to_generate):
                if content and content.strip():
                    content_index_map[i] = len(valid_contents)
                    valid_contents.append(content)
                else:
                    print(f"⚠️  跳过空边内容")
            
            if not valid_contents:
                print("⚠️  所有边内容为空，跳过embedding生成")
            else:
                print(f"🚀 批量生成 {len(valid_contents)} 个边embedding...")
                embeddings = self.embedding_manager.batch_get_embeddings(valid_contents)
                
                # 更新embedding索引
                for i, content in enumerate(contents_to_generate):
                    if i not in content_index_map:
                        continue  # 跳过空内容
                    
                    valid_idx = content_index_map[i]
                    embedding = embeddings[valid_idx] if valid_idx < len(embeddings) else None
                    
                    if not embedding:
                        continue
                    embedding_array = np.array(embedding, dtype='float32')
                    idx = len(self.embeddings)
                    self.embeddings.append(embedding_array)
                    
                    content_hash = self._compute_content_hash(content)
                    self.content_hash_to_idx[content_hash] = idx
                    
                    # 添加到FAISS索引
                    if self.faiss_index is None:
                        self.faiss_index = faiss.IndexFlatIP(1536)
                    self.faiss_index.add(embedding_array.reshape(1, -1))
                    
                    print(f"  ✅ 新边embedding [{idx}]: {content[:50]}...")
        
        # 3. 添加所有边
        for edge in edges:
            edge_id = edge.get('id')
            content = edge.get('content', '')
            content_hash = self._compute_content_hash(content)
            
            # 获取embedding索引
            embedding_idx = self.content_hash_to_idx.get(content_hash, -1)
            
            # 保存边数据
            self.cache['edges'][edge_id] = {
                **edge,
                'embedding_idx': embedding_idx,
                'content_hash': content_hash
            }
            
            # 维护反向索引
            if embedding_idx != -1:
                if embedding_idx not in self.idx_to_edge_id:
                    self.idx_to_edge_id[embedding_idx] = []
                if edge_id not in self.idx_to_edge_id[embedding_idx]:
                    self.idx_to_edge_id[embedding_idx].append(edge_id)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取单个节点"""
        return self.cache['nodes'].get(node_id)
    
    def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取节点"""
        return [self.cache['nodes'][nid] for nid in node_ids if nid in self.cache['nodes']]
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """获取单个边"""
        return self.cache['edges'].get(edge_id)
    
    def get_edges_by_node(self, node_id: str) -> List[Dict[str, Any]]:
        """获取节点相关的所有边"""
        edges = []
        for edge_id, edge in self.cache['edges'].items():
            if edge.get('source') == node_id or edge.get('target') == node_id:
                edges.append(edge)
        return edges
    
    def get_all_edges(self, layer: int = None) -> List[Dict[str, Any]]:
        """
        获取所有边
        
        Args:
            layer: 如果指定，只返回该layer的边
        
        Returns:
            边列表
        """
        if layer is None:
            return list(self.cache['edges'].values())
        else:
            return [
                edge for edge in self.cache['edges'].values()
                if edge.get('layer') == layer
            ]
    
    def get_nodes_by_filter(self, **filters) -> List[Dict[str, Any]]:
        """
        根据条件过滤节点
        
        Args:
            **filters: 过滤条件（如 type='event', layer=2）
        
        Returns:
            符合条件的节点列表
        """
        result = []
        for node in self.cache['nodes'].values():
            match = True
            for key, value in filters.items():
                if node.get(key) != value:
                    match = False
                    break
            if match:
                result.append(node)
        return result
    
    def filter_nodes(self, **filters) -> List[str]:
        """
        按条件过滤节点
        
        Args:
            **filters: 过滤条件，如 type='event', layer=2
        
        Returns:
            符合条件的node_id列表
        """
        results = []
        for node_id, node in self.cache['nodes'].items():
            match = True
            for key, value in filters.items():
                if key == 'participants':
                    # 特殊处理participants（列表包含）
                    if not any(p in node.get('participants', []) for p in value):
                        match = False
                        break
                else:
                    if node.get(key) != value:
                        match = False
                        break
            if match:
                results.append(node_id)
        return results
    
    def search_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        快速检索相似节点
        
        Args:
            query_embedding: 查询向量
            top_k: 返回top-k结果
        
        Returns:
            相似节点列表，包含node和similarity
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("⚠️  FAISS索引为空")
            return []
        
        # 确保query_embedding是正确的形状
        query_array = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        # FAISS检索
        D, I = self.faiss_index.search(query_array, k=min(top_k, self.faiss_index.ntotal))
        
        # 获取完整节点信息
        results = []
        for idx, similarity in zip(I[0], D[0]):
            if idx == -1:  # FAISS返回-1表示无效
                continue
            
            node_ids = self.idx_to_node_id.get(int(idx), [])
            for node_id in node_ids:
                node = self.cache['nodes'].get(node_id)
                if node:
                    results.append({
                        'node': node,
                        'similarity': float(similarity)
                    })
        
        return results[:top_k]
    
    def filter_and_search(self, query_embedding: np.ndarray, 
                         filters: Dict[str, Any] = None, 
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """
        过滤 + 检索
        
        Args:
            query_embedding: 查询向量
            filters: 过滤条件
            top_k: 返回top-k结果
        
        Returns:
            相似节点列表
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("⚠️  FAISS索引为空")
            return []
        
        # 确保query_embedding是正确的形状
        query_array = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        # FAISS检索（召回更多候选）
        # 如果有过滤条件（特别是layer过滤），需要搜索更多候选以确保能找到足够的匹配节点
        if filters and 'layer' in filters:
            # 对于layer过滤，搜索更多候选（因为可能大部分节点都不是目标layer）
            k_candidates = min(top_k * 50, self.faiss_index.ntotal)
        else:
            k_candidates = min(top_k * 5, self.faiss_index.ntotal)
        
        D, I = self.faiss_index.search(query_array, k=k_candidates)
        
        # 应用过滤条件
        results = []
        for idx, similarity in zip(I[0], D[0]):
            if idx == -1:
                continue
            
            node_ids = self.idx_to_node_id.get(int(idx), [])
            for node_id in node_ids:
                node = self.cache['nodes'].get(node_id)
                if not node:
                    continue
                
                # 过滤条件检查
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key == 'participants':
                            if not any(p in node.get('participants', []) for p in value):
                                match = False
                                break
                        else:
                            if node.get(key) != value:
                                match = False
                                break
                    if not match:
                        continue
                
                results.append({
                    'node': node,
                    'similarity': float(similarity)
                })
                
                if len(results) >= top_k:
                    break
            
            if len(results) >= top_k:
                break
        
        return results
    
    def update_node(self, node_id: str, new_content: str = None, **kwargs) -> None:
        """
        更新节点
        
        Args:
            node_id: 节点ID
            new_content: 新的content（如果提供，会重新生成embedding）
            **kwargs: 其他要更新的字段
        """
        if node_id not in self.cache['nodes']:
            raise ValueError(f"节点不存在: {node_id}")
        
        node = self.cache['nodes'][node_id]
        
        # 如果更新content，需要重新生成embedding
        if new_content is not None and new_content != node.get('content'):
            embedding, embedding_idx, is_new = self.get_or_generate_embedding(new_content)
            content_hash = self._compute_content_hash(new_content)
            
            node['content'] = new_content
            node['embedding_idx'] = embedding_idx
            node['content_hash'] = content_hash
            
            # 更新反向索引
            if embedding_idx not in self.idx_to_node_id:
                self.idx_to_node_id[embedding_idx] = []
            if node_id not in self.idx_to_node_id[embedding_idx]:
                self.idx_to_node_id[embedding_idx].append(node_id)
        
        # 更新其他字段
        for key, value in kwargs.items():
            node[key] = value
    
    def save(self) -> None:
        """保存缓存到磁盘"""
        print(f"💾 保存cache到磁盘: {self.namespace}...")
        
        # 更新metadata
        self.cache['metadata']['total_nodes'] = len(self.cache['nodes'])
        self.cache['metadata']['total_edges'] = len(self.cache['edges'])
        self.cache['metadata']['total_embeddings'] = len(self.embeddings)
        
        # 1. 保存cache.json
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        print(f"✅ 保存cache.json: {len(self.cache['nodes'])} 个节点, {len(self.cache['edges'])} 个边")
        
        # 2. 保存embeddings.npy
        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype='float32')
            np.save(self.embeddings_file, embeddings_array)
            print(f"✅ 保存embeddings.npy: {len(self.embeddings)} 个向量")
        
        # 3. 保存FAISS索引
        if self.faiss_index and self.faiss_index.ntotal > 0:
            faiss.write_index(self.faiss_index, self.faiss_file)
            print(f"✅ 保存FAISS索引: {self.faiss_index.ntotal} 个向量")
        
        # 4. 保存content_hash映射
        with open(self.hash_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.content_hash_to_idx, f, indent=2)
        print(f"✅ 保存content_hash映射: {len(self.content_hash_to_idx)} 个hash")
        
        print(f"💾 Cache保存完成！")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_nodes = len(self.cache['nodes'])
        total_edges = len(self.cache['edges'])
        total_items = total_nodes + total_edges
        total_embeddings = len(self.embeddings)
        dedup_rate = (total_items - total_embeddings) / total_items if total_items > 0 else 0
        
        # 按类型统计
        nodes_by_type = {}
        for node in self.cache['nodes'].values():
            node_type = node.get('type', 'unknown')
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
        
        # 按layer统计
        nodes_by_layer = {}
        for node in self.cache['nodes'].values():
            layer = node.get('layer', -1)
            nodes_by_layer[layer] = nodes_by_layer.get(layer, 0) + 1
        
        return {
            'namespace': self.namespace,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_items': total_items,
            'total_embeddings': total_embeddings,
            'dedup_rate': round(dedup_rate, 3),
            'api_calls_saved': total_items - total_embeddings,
            'nodes_by_type': nodes_by_type,
            'nodes_by_layer': nodes_by_layer,
            'cache_dir': self.cache_dir
        }

