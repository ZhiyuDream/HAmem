"""
HAmem entry point

Provides a simple public API
"""

import os
import sys
from typing import Dict, Any, List

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from core.memory import MemoryBuilder, ConversationData, StorageManager
from core.retrieval import RetrievalEngine, create_qa_system
from core.utils.input_processor import process_input_file


class HAmem:
    
    
    def __init__(self, config: Config = None):
        
        if config is None:
            config = Config.from_env()
        
        self.config = config
        self.config.validate()
        
        # Initialize core components
        self.memory_builder = MemoryBuilder(config)
        self.storage = StorageManager(config)
        self.retrieval_engine = RetrievalEngine(config)
        self.retrieval_engine.set_storage(self.storage)
        
        # Initialize cache for QA system
        from core.infrastructure import UnifiedCache, EmbeddingManager
        embedding_manager = EmbeddingManager(config)
        cache = UnifiedCache(
            cache_dir=config.cache_dir,
            namespace="default",
            embedding_manager=embedding_manager
        )
        
        # Create QA system with optional hybrid search
        self.qa_system = create_qa_system(
            config=config,
            cache=cache,
            storage=self.storage,
            namespace="default"
        )
        
        print("🚀 HAmem initialized successfully!")
    
    def build_memory(self, conversation_data: Dict[str, Any], namespace: str = "default", llm_provider: str = None) -> Dict[str, Any]:
        """
        Build memory from conversation data
        
        Args:
            conversation_data: Conversation data in HAmem format
            namespace: Namespace for storage isolation (will be used as Neo4j database name)
            llm_provider: LLM provider (if None, uses config default)
        """
        print(f"🧠 Building memory (namespace: {namespace})...")
        
        # Convert input data format
        conversation = ConversationData.from_dict(conversation_data)
        
        # Use config provider if not specified
        if llm_provider is None:
            llm_provider = self.config.llm_config.provider
        
        # Build memory（纯Neo4j架构：每个fragment处理完后已直接写入Neo4j）
        result = self.memory_builder.build_memory(conversation, namespace=namespace, llm_provider=llm_provider)
        
        print(f"✅ Memory built: {result.total_fragments} fragments processed")
        return result.to_dict()
    
    def build_memory_from_file(self, file_path: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Build memory from conversation file
        
        Args:
            file_path: Path to conversation file (JSON format)
            namespace: Namespace for storage isolation (will be used as Neo4j database name)
        
        Returns:
            Memory build result dictionary
        """
        print(f"📂 Loading conversation from file: {file_path}")
        
        # Process input file (auto-detect format and convert to standard format)
        conversation_data = process_input_file(file_path)
        
        # Build memory using standard format
        return self.build_memory(conversation_data, namespace=namespace)
    
    def search_memory(self, query: str, top_k: int = 10, namespace: str = "default", similarity_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search memory using the same logic as QA system (embedding search + similarity)
        
        This uses the same recall logic as QA system but returns raw search results
        instead of generating answers. Same as experiment/test_qa.py logic.
        
        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Namespace for search
            similarity_threshold: Minimum similarity score (0.0-1.0), results below this will be filtered out
        """
        print(f"🔍 Searching memory for: {query} (similarity_threshold: {similarity_threshold})")
        
        # Use QA system's recall logic (same as experiment/test_qa.py)
        # Load cache for the namespace
        from core.infrastructure import UnifiedCache, EmbeddingManager
        embedding_manager = EmbeddingManager(self.config)
        cache = UnifiedCache(
            cache_dir=self.config.cache_dir,
            namespace=namespace,
            embedding_manager=embedding_manager
        )
        
        # Use the same recall logic as QA system
        if self.config.use_hybrid_search and self.config.use_neo4j:
            # Use hybrid search (same as QA system)
            from core.infrastructure.neo4j_client import Neo4jClient
            from core.search.neo4j_hybrid_recall import Neo4jHybridRecall
            
            neo4j_client = Neo4jClient(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            if neo4j_client.connect():
                recall = Neo4jHybridRecall(cache, neo4j_client, namespace)
                recalled = recall.multi_layer_recall_with_expansion(
                    query,
                    layer0_top_k=2,  # Fragment召回top2
                    layer1_top_k=top_k,
                    layer2_top_k=top_k * 2,
                    layer3_top_k=top_k // 2,
                    max_hops=1,  # 只做初始召回，不扩展
                    expand_limit=30
                )
                
                # Flatten results (same format as QA system) and filter by similarity
                all_results = []
                for layer_key in ['layer0', 'layer1', 'layer2', 'layer3']:
                    if layer_key in recalled:
                        layer_data = recalled[layer_key]
                        layer_nodes = layer_data.get('all_nodes', [])
                        # 过滤相似度低于阈值的结果
                        for node in layer_nodes:
                            similarity = node.get('similarity_score', node.get('similarity', 0.0))
                            if similarity >= similarity_threshold:
                                all_results.append((node, similarity))
                
                # 按相似度排序
                all_results.sort(key=lambda x: x[1], reverse=True)
                # 提取节点（已按相似度排序）
                results = [node for node, _ in all_results[:top_k]]
                
                neo4j_client.close()
            else:
                # Fallback to standard search
                from core.search.recall import SearchRecall
                recall = SearchRecall(cache, self.storage)
                recalled = recall.multi_layer_recall(
                    query,
                    layer0_top_k=2,
                    layer1_top_k=top_k,
                    layer2_top_k=top_k * 2,
                    layer3_top_k=top_k // 2
                )
                all_results = (
                    recalled.get('layer0', []) + 
                    recalled.get('layer1', []) + 
                    recalled.get('layer2', []) + 
                    recalled.get('layer3', [])
                )
                results = all_results[:top_k]
        else:
            # Use standard search (same as QA system)
            from core.search.recall import SearchRecall
            recall = SearchRecall(cache, self.storage)
            recalled = recall.multi_layer_recall(
                query,
                layer0_top_k=2,
                layer1_top_k=top_k,
                layer2_top_k=top_k * 2,
                layer3_top_k=top_k // 2
            )
            all_results = (
                recalled.get('layer0', []) + 
                recalled.get('layer1', []) + 
                recalled.get('layer2', []) + 
                recalled.get('layer3', [])
            )
            results = all_results[:top_k]
        
        print(f"✅ Found {len(results)} results")
        # results可能是dict或对象，需要兼容处理
        return [r if isinstance(r, dict) else r.to_dict() for r in results]
    
    def ask_question(self, question: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Answer a question
        
        Args:
            question: Question to answer
            namespace: Namespace for the question (should match the namespace used in build_memory)
        """
        print(f"❓ Answering question: {question} (namespace: {namespace})")
        
        # 如果namespace不同，需要重新创建QA系统（使用正确的namespace对应的cache）
        # 或者如果当前cache的FAISS索引为空，也尝试切换namespace
        should_recreate = False
        if hasattr(self.qa_system, 'namespace') and self.qa_system.namespace != namespace:
            should_recreate = True
        elif hasattr(self.qa_system, 'cache') and (
            self.qa_system.cache.faiss_index is None or 
            self.qa_system.cache.faiss_index.ntotal == 0
        ):
            # 当前cache为空，尝试使用指定的namespace
            if namespace != self.qa_system.namespace:
                print(f"  ⚠️  当前cache为空，尝试切换到namespace: {namespace}")
                should_recreate = True
        
        if should_recreate:
            print(f"  🔄 切换namespace: {self.qa_system.namespace} -> {namespace}")
            # 重新创建cache（使用正确的namespace）
            from core.infrastructure import UnifiedCache, EmbeddingManager
            embedding_manager = EmbeddingManager(self.config)
            cache = UnifiedCache(
                cache_dir=self.config.cache_dir,
                namespace=namespace,
                embedding_manager=embedding_manager
            )
            
            # 重新创建QA系统（使用正确的namespace）
            from core.retrieval import create_qa_system
            self.qa_system = create_qa_system(
                config=self.config,
                cache=cache,
                storage=self.storage,
                namespace=namespace
            )
            print(f"  ✅ QA系统已更新为namespace: {namespace}")
        
        answer = self.qa_system.answer_question(question)
        
        # answer可能是dict或对象，需要兼容处理
        if isinstance(answer, dict):
            print(f"✅ Answer generated")
            return answer
        else:
            print(f"✅ Answer generated with confidence: {answer.confidence}")
        return answer.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system stats"""
        stats = {
            'config': self.config.to_dict(),
        }
        
        # 获取cache统计信息
        if hasattr(self.memory_builder, 'cache') and self.memory_builder.cache:
            try:
                stats['cache_stats'] = self.memory_builder.cache.get_cache_stats()
            except Exception as e:
                stats['cache_stats'] = {'error': str(e)}
        else:
            stats['cache_stats'] = {}
        
        # 获取storage统计信息
        if hasattr(self.storage, 'get_stats'):
            try:
                stats['storage_stats'] = self.storage.get_stats()
            except Exception as e:
                stats['storage_stats'] = {'error': str(e)}
        else:
            stats['storage_stats'] = {}
        
        return stats


def main():
    """Main function - example usage"""
    print("🚀 HAmem - Hierarchical Memory System")
    print("=" * 50)
    
    # Check configuration
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Please configure API keys:")
        print("   Option 1: Set environment variables")
        print("     - OPENAI_API_KEY / DEEPSEEK_API_KEY")
        print("   Option 2: Use new flexible config:")
        print("     from InputConfig import LlmConfig, EmbeddingConfig")
        print("     config = Config(")
        print("         llm_config=LlmConfig.create_deepseek(api_key='...'),")
        print("         embedding_config=EmbeddingConfig.create_openai(api_key='...')")
        print("     )")
        return
    
    # Show configuration
    print("✅ Configuration loaded")
    if config.llm_config:
        print(f"   - LLM Provider: {config.llm_config.provider}")
        print(f"   - LLM Model: {config.llm_config.get_model()}")
    if config.embedding_config:
        print(f"   - Embedding Provider: {config.embedding_config.provider}")
        print(f"   - Embedding Model: {config.embedding_config.get_model()}")
    
    # Initialize HAmem
    try:
        hamem = HAmem()
    except Exception as e:
        print(f"❌ Failed to initialize HAmem: {e}")
        return
    
    # Example usage
    print("\n📝 Example usage:")
    print("hamem = HAmem()")
    print("result = hamem.build_memory(conversation_data)")
    print("answer = hamem.ask_question('What did we discuss?')")
    
    # Show full configuration
    print(f"\n⚙️  Full Configuration:")
    stats = hamem.get_stats()
    config_dict = stats['config']
    print(f"  - LLM: {config_dict.get('llm_provider', 'N/A')} / {config_dict.get('llm_model', 'N/A')}")
    print(f"  - Embedding: {config_dict.get('embedding_provider', 'N/A')} / {config_dict.get('embedding_model', 'N/A')}")
    print(f"  - Cache Directory: {config_dict.get('cache_dir', 'N/A')}")
    print(f"  - Max Workers: {config_dict.get('max_workers', 'N/A')}")


if __name__ == "__main__":
    main()
