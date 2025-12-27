"""
Retrieval Engine and QA System initialization

Provides RetrievalEngine and QASystem classes for core.main
"""

from typing import Dict, Any, Optional
from config import Config
from core.infrastructure import UnifiedCache, EmbeddingManager, LLMClient
from core.infrastructure.neo4j_client import Neo4jClient
from core.search.qa_system import QASystem
from core.search.recall import SearchRecall


class RetrievalEngine:
    """
    Retrieval Engine wrapper
    
    Provides a simple interface for memory search
    """
    
    def __init__(self, config: Config):
        """
        Initialize retrieval engine
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize embedding manager and cache
        self.embedding_manager = EmbeddingManager(config)
        self.cache = UnifiedCache(
            cache_dir=config.cache_dir,
            namespace="default",
            embedding_manager=self.embedding_manager
        )
        
        # Initialize storage (will be set by StorageManager)
        self.storage = None
    
    def set_storage(self, storage):
        """Set storage instance"""
        self.storage = storage
    
    def search(self, query: str, top_k: int = 10, namespace: str = "default") -> list:
        """
        Search memory
        
        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Namespace for Neo4j search
            
        Returns:
            List of search results
        """
        if not self.storage:
            raise ValueError("Storage not set. Call set_storage() first.")
        
        # 如果启用了混合检索，使用Neo4j向量搜索
        if self.config.use_hybrid_search and self.config.use_neo4j:
            try:
                from core.infrastructure.neo4j_client import Neo4jClient
                from core.search.neo4j_hybrid_recall import Neo4jHybridRecall
                
                neo4j_client = Neo4jClient(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_username,
                    password=self.config.neo4j_password,
                    database=self.config.neo4j_database
                )
                
                if neo4j_client.connect():
                    recall = Neo4jHybridRecall(self.cache, neo4j_client, namespace)
                    results = recall.multi_layer_recall(query, layer1_top_k=top_k, layer2_top_k=top_k, layer3_top_k=top_k)
                    
                    # Flatten results
                    all_results = []
                    for layer in ['layer1', 'layer2', 'layer3']:
                        if layer in results:
                            all_results.extend(results[layer][:top_k])
                    
                    neo4j_client.close()
                    return all_results[:top_k]
            except Exception as e:
                print(f"⚠️  Neo4j搜索失败，降级到FAISS: {e}")
        
        # 降级到FAISS搜索
        recall = SearchRecall(self.cache, self.storage)
        results = recall.multi_layer_recall(query, top_k, top_k, top_k)
        
        # Flatten results
        all_results = []
        for layer in ['layer1', 'layer2', 'layer3']:
            if layer in results:
                all_results.extend(results[layer][:top_k])
        
        return all_results[:top_k]


def create_qa_system(
    config: Config,
    cache: UnifiedCache,
    storage,
    namespace: str = "default",
    default_provider: str = "deepseek"
) -> QASystem:
    """
    Create QA System with optional hybrid search support
    
    Args:
        config: Configuration object
        cache: UnifiedCache instance
        storage: Storage instance
        namespace: Namespace
        default_provider: 默认LLM提供商 ("openai" 或 "deepseek")
        
    Returns:
        QASystem instance
    """
    # Initialize LLM client
    llm_client = LLMClient(config)
    
    # Check if hybrid search is enabled
    use_hybrid_search = config.use_hybrid_search and config.use_neo4j
    
    neo4j_client = None
    if use_hybrid_search:
        try:
            neo4j_client = Neo4jClient(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            )
            if not neo4j_client.connect():
                print("⚠️  Neo4j connection failed, falling back to standard search")
                use_hybrid_search = False
                neo4j_client = None
            else:
                print("✅ Neo4j connected, using hybrid search mode")
        except Exception as e:
            print(f"⚠️  Failed to initialize Neo4j: {e}, falling back to standard search")
            use_hybrid_search = False
            neo4j_client = None
    
    # Create QA System
    qa_system = QASystem(
        cache=cache,
        storage=storage,
        llm_client=llm_client,
        namespace=namespace,
        max_hops=2,
        use_hybrid_search=use_hybrid_search,
        neo4j_client=neo4j_client,
        default_provider=default_provider
    )
    
    return qa_system

