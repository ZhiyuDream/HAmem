"""
递归高层抽象处理器

Layer N（N>=3）新增节点数达到阈值后，聚类并抽象出 Layer N+1 节点，
然后继续检查 Layer N+1，如此递归向上，直到某层新增不足阈值自然停止。
阈值天然限制层数：每轮抽象节点数按 min_cluster_size 倍缩水，
越高层越难攒够阈值，无需硬性层数上限。
"""

from typing import Dict, List, Any, Optional
from core.infrastructure import LLMClient, UnifiedCache
from core.infrastructure.neo4j_client import Neo4jClient
from .clustering import EventClusterer
from .higher_extractor import HigherLayerExtractor


class HigherLayerProcessor:
    """递归高层抽象处理器（N>=3 -> N+1）"""

    def __init__(
        self,
        llm_client: LLMClient,
        cache: UnifiedCache,
        neo4j_client: Optional[Neo4jClient] = None,
        namespace: str = "default",
        abstraction_threshold: int = 60,  # 与 layer2->3 相同的触发标准
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 7,
        default_provider: str = "openai",
        token_tracker=None
    ):
        self.cache = cache
        self.namespace = namespace
        self.abstraction_threshold = abstraction_threshold
        self.min_cluster_size = min_cluster_size

        self.clusterer = EventClusterer(
            cache,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        self.extractor = HigherLayerExtractor(
            llm_client,
            token_tracker=token_tracker,
            default_provider=default_provider
        )

        # 按层记录上次处理时的节点总数（增量触发）
        self.last_analyzed_count: Dict[int, int] = {}
        # 按层独立计数器，用于生成节点ID
        self.counters: Dict[int, int] = {}

    def _nodes_at_layer(self, layer: int) -> List[Dict[str, Any]]:
        """获取cache中指定层的所有节点"""
        return [
            node for node in self.cache.cache['nodes'].values()
            if node.get('layer') == layer
        ]

    def _init_counter(self, layer: int) -> None:
        """根据cache中已有的同层抽象节点初始化计数器，避免ID冲突"""
        if layer in self.counters:
            return
        prefix = f"layer{layer}_abs_"
        existing = [
            n for n in self._nodes_at_layer(layer)
            if str(n.get('id', '')).startswith(prefix)
        ]
        max_idx = 0
        for node in existing:
            try:
                max_idx = max(max_idx, int(str(node.get('id')).rsplit('_', 1)[1]))
            except (ValueError, IndexError):
                continue
        self.counters[layer] = max_idx

    def process(
        self,
        namespace: str,
        pending_nodes: Optional[List[Dict[str, Any]]] = None,
        pending_edges: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        从 Layer3 开始向上递归抽象

        Args:
            namespace: 命名空间
            pending_nodes: 本批次刚创建、尚未写入cache的节点（如本批的layer3节点）
            pending_edges: 本批次刚创建、尚未写入cache的边

        Returns:
            {'created_nodes': [...], 'created_edges': [...],
             'abstractions_by_layer': {layer: count}}
        """
        stats = {
            'created_nodes': [],
            'created_edges': [],
            'abstractions_by_layer': {}
        }

        # 先把本批次pending的节点/边写入cache（生成embedding），使它们参与统计与聚类
        if pending_nodes:
            self.cache.batch_add_nodes(pending_nodes)
        if pending_edges:
            self.cache.batch_add_edges(pending_edges)

        layer = 3
        while True:
            nodes_n = self._nodes_at_layer(layer)
            total = len(nodes_n)
            last = self.last_analyzed_count.get(layer, 0)
            new_count = total - last

            if new_count < self.abstraction_threshold:
                if last > 0 or total > 0:
                    print(f"  📊 Layer{layer}: 共{total}个，新增{new_count}个 "
                          f"(阈值{self.abstraction_threshold})，停止向上抽象")
                break

            print(f"\n{'='*60}")
            print(f"🧠 Layer{layer} 新增 {new_count} 个节点 (>= 阈值{self.abstraction_threshold})，"
                  f"触发 Layer{layer + 1} 抽象")
            print(f"{'='*60}")

            # 只处理新增节点（增量聚类）
            new_nodes = nodes_n[last:]

            clusters = self.clusterer.cluster_layer2_nodes(new_nodes)
            if not clusters:
                print(f"  ℹ️  Layer{layer} 新增节点未形成聚类，跳过")
                self.last_analyzed_count[layer] = total
                layer += 1
                continue

            self._init_counter(layer + 1)

            extracted_count = 0
            for cluster_nodes in clusters:
                extraction = self.extractor.abstract_cluster(
                    cluster_nodes, source_layer=layer, target_layer=layer + 1
                )
                abstraction = extraction.get('abstraction')
                if not abstraction:
                    print(f"  ⚠️  Layer{layer + 1}抽象提取失败（cluster {extracted_count + 1}/{len(clusters)}），跳过")
                    continue

                extracted_count += 1

                self.counters[layer + 1] += 1
                abs_id = f"layer{layer + 1}_abs_{self.counters[layer + 1]}"

                abs_node = {
                    "id": abs_id,
                    "type": "abstraction",
                    "content": abstraction.get('summary', ''),
                    "themes": abstraction.get('themes', []),
                    "significance": abstraction.get('significance'),
                    "layer": layer + 1,
                    "active": True
                }
                stats['created_nodes'].append(abs_node)

                for child in cluster_nodes:
                    child_id = child.get('id')
                    if child_id:
                        stats['created_edges'].append({
                            "id": f"{abs_id}_contains_{child_id}",
                            "source": abs_id,
                            "target": child_id,
                            "type": "CONTAINS",
                            "layer": layer + 1,
                            "active": True
                        })

            created_this_round = len(stats['created_nodes'])
            print(f"  ✅ 创建 {extracted_count} 个 Layer{layer + 1} 抽象节点"
                  f"（累计 {created_this_round} 个）")

            # 新节点/边立即写入cache（有embedding），供下一层聚类使用
            newly = [
                n for n in stats['created_nodes']
                if n.get('id') not in self.cache.cache['nodes']
            ]
            new_edges = [
                e for e in stats['created_edges']
                if e.get('id') not in self.cache.cache['edges']
            ]
            if newly:
                self.cache.batch_add_nodes(newly)
            if new_edges:
                self.cache.batch_add_edges(new_edges)

            stats['abstractions_by_layer'][layer + 1] = \
                stats['abstractions_by_layer'].get(layer + 1, 0) + len(newly)

            self.last_analyzed_count[layer] = total
            layer += 1

        return stats
