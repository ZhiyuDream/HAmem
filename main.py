"""
HAmem 主入口

提供简单的API接口
"""

import os
import sys
from typing import Dict, Any, List

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from memory import MemoryBuilder, ConversationData, StorageManager
from retrieval import RetrievalEngine, QASystem


class HAmem:
    
    
    def __init__(self, config: Config = None):
        
        if config is None:
            config = Config.from_env()
        
        self.config = config
        self.config.validate()
        
        # 初始化核心组件
        self.memory_builder = MemoryBuilder(config)
        self.storage = StorageManager(config)
        self.retrieval_engine = RetrievalEngine(config)
        self.qa_system = QASystem(config)
        
        print("🚀 HAmem initialized successfully!")
    
    def build_memory(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        
        print("🧠 Building memory...")
        
        # 转换数据格式
        conversation = ConversationData.from_dict(conversation_data)
        
        # 构建记忆
        result = self.memory_builder.build_memory(conversation)
        
        print(f"✅ Memory built: {result.total_fragments} fragments processed")
        return result.to_dict()
    
    def search_memory(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索记忆"""
        print(f"🔍 Searching memory for: {query}")
        
        results = self.retrieval_engine.search(query, top_k)
        
        print(f"✅ Found {len(results)} results")
        return [result.to_dict() for result in results]
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """回答问题"""
        print(f"❓ Answering question: {question}")
        
        answer = self.qa_system.answer_question(question)
        
        print(f"✅ Answer generated with confidence: {answer.confidence}")
        return answer.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'config': self.config.to_dict(),
            'cache_stats': self.memory_builder.embedding_manager.cache.get_cache_stats(),
            'storage_stats': self.storage.get_stats()
        }


def main():
    """主函数 - 示例用法"""
    print("🚀 HAmem - Hierarchical Memory System")
    print("=" * 50)
    
    # 检查API密钥
    config = Config()
    if not config.openai_api_key or not config.deepseek_api_key:
        print("❌ Please configure API keys in .env file or environment variables")
        print("   - OPENAI_API_KEY: For embeddings")
        print("   - DEEPSEEK_API_KEY: For LLM calls")
        return
    
    print("✅ API configuration loaded")
    print("   - OpenAI API: For embeddings")
    print("   - DeepSeek API: For LLM calls")
    
    # 初始化HAmem
    try:
        hamem = HAmem()
    except Exception as e:
        print(f"❌ Failed to initialize HAmem: {e}")
        return
    
    # 示例用法
    print("\n📝 Example usage:")
    print("hamem = HAmem()")
    print("result = hamem.build_memory(conversation_data)")
    print("answer = hamem.ask_question('What did we discuss?')")
    
    # 显示配置
    print(f"\n⚙️  Configuration:")
    stats = hamem.get_stats()
    print(f"  - LLM Model: {stats['config']['llm_model']} (DeepSeek)")
    print(f"  - Embedding Model: {stats['config']['embedding_model']} (OpenAI)")
    print(f"  - Cache Directory: {stats['config']['cache_dir']}")
    print(f"  - Max Workers: {stats['config']['max_workers']}")


if __name__ == "__main__":
    main()
