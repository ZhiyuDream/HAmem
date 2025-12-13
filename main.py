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
from memory import MemoryBuilder, ConversationData, StorageManager
from retrieval import RetrievalEngine, create_qa_system


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
    
    def build_memory(self, conversation_data: Dict[str, Any], namespace: str = "default") -> Dict[str, Any]:
        """
        Build memory from conversation data
        
        Args:
            conversation_data: Conversation data in HAmem format
            namespace: Namespace for storage isolation (will be used as Neo4j database name)
        """
        print(f"🧠 Building memory (namespace: {namespace})...")
        
        # Convert input data format
        conversation = ConversationData.from_dict(conversation_data)
        
        # Build memory（纯Neo4j架构：每个fragment处理完后已直接写入Neo4j）
        result = self.memory_builder.build_memory(conversation, namespace=namespace)
        
        print(f"✅ Memory built: {result.total_fragments} fragments processed")
        return result.to_dict()
    
    def search_memory(self, query: str, top_k: int = 10, namespace: str = "default") -> List[Dict[str, Any]]:
        """Search memory"""
        print(f"🔍 Searching memory for: {query}")
        
        results = self.retrieval_engine.search(query, top_k=top_k, namespace=namespace)
        
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
        if hasattr(self.qa_system, 'namespace') and self.qa_system.namespace != namespace:
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
            from retrieval import create_qa_system
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
        return {
            'config': self.config.to_dict(),
            'cache_stats': self.memory_builder.embedding_manager.cache.get_cache_stats(),
            'storage_stats': self.storage.get_stats()
        }


def main():
    """Main function - example usage"""
    print("🚀 HAmem - Hierarchical Memory System")
    print("=" * 50)
    
    # Check API keys
    config = Config()
    if not config.openai_api_key or not config.deepseek_api_key:
        print("❌ Please configure API keys in .env file or environment variables")
        print("   - OPENAI_API_KEY: For embeddings")
        print("   - DEEPSEEK_API_KEY: For LLM calls")
        return
    
    print("✅ API configuration loaded")
    print("   - OpenAI API: For embeddings")
    print("   - DeepSeek API: For LLM calls")
    
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
    
    # Show configuration
    print(f"\n⚙️  Configuration:")
    stats = hamem.get_stats()
    print(f"  - LLM Model: {stats['config']['llm_model']} (DeepSeek)")
    print(f"  - Embedding Model: {stats['config']['embedding_model']} (OpenAI)")
    print(f"  - Cache Directory: {stats['config']['cache_dir']}")
    print(f"  - Max Workers: {stats['config']['max_workers']}")


if __name__ == "__main__":
    main()
