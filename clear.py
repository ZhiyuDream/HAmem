"""
清除Neo4j数据库中的所有数据

使用方法：
    python clear_neo4j.py [namespace]
    
如果不提供namespace，将清除所有数据
如果提供namespace，只清除该namespace的数据
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(Path(__file__).parent, '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from core.infrastructure.neo4j_client import Neo4jClient
from config import Config
import shutil


def clear_cache(namespace: str = None):
    """清除缓存数据"""
    config = Config()
    cache_root = config.cache_dir
    
    if not os.path.exists(cache_root):
        print(f"ℹ️  缓存目录不存在: {cache_root}")
        return
    
    if namespace:
        # 清除指定namespace的缓存
        cache_dir = os.path.join(cache_root, namespace)
        if os.path.exists(cache_dir):
            print(f"\n🗑️  清除namespace '{namespace}' 的缓存...")
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ 已删除缓存目录: {cache_dir}")
            except Exception as e:
                print(f"❌ 删除缓存目录失败: {e}")
        else:
            print(f"ℹ️  缓存目录不存在: {cache_dir}")
    else:
        # 清除所有缓存
        print(f"\n🗑️  清除所有缓存...")
        try:
            if os.path.exists(cache_root):
                shutil.rmtree(cache_root)
                os.makedirs(cache_root, exist_ok=True)  # 重新创建空目录
                print(f"✅ 已删除所有缓存目录: {cache_root}")
            else:
                print(f"ℹ️  缓存根目录不存在: {cache_root}")
        except Exception as e:
            print(f"❌ 删除缓存目录失败: {e}")


def clear_neo4j(namespace: str = None):
    """清除Neo4j数据"""
    print("=" * 60)
    print("🗑️  清除Neo4j数据")
    print("=" * 60)
    
    # 创建客户端
    client = Neo4jClient()
    
    # 连接
    if not client.connect():
        print("❌ 无法连接到Neo4j，请检查服务是否启动")
        return False
    
    print("✅ Neo4j连接成功")
    
    try:
        if namespace:
            # 只清除指定namespace的数据
            print(f"\n🗑️  清除namespace '{namespace}' 的数据...")
            
            # 先统计要删除的数据
            count_query = """
            MATCH (n)
            WHERE n.namespace = $namespace
            RETURN count(n) as node_count
            """
            result = client.execute_read(count_query, {'namespace': namespace})
            node_count = result[0]['node_count'] if result else 0
            
            # 删除所有关系和节点
            delete_query = """
            MATCH (n)
            WHERE n.namespace = $namespace
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            result = client.execute_write(delete_query, {'namespace': namespace})
            deleted_count = result[0]['deleted'] if result else 0
            
            print(f"✅ 已删除 {deleted_count} 个节点（及其关系）")
        else:
            # 清除所有数据
            print("\n🗑️  清除所有数据...")
            
            # 先统计
            count_query = "MATCH (n) RETURN count(n) as node_count"
            result = client.execute_read(count_query, {})
            node_count = result[0]['node_count'] if result else 0
            
            # 删除所有
            result = client.clear_database()
            
            print(f"✅ 已清除所有数据（共 {node_count} 个节点）")
        
        print("\n✅ Neo4j清除完成！")
        
        # 同时清除对应的缓存
        clear_cache(namespace)
        
        print("\n✅ 全部清除完成！")
        return True
        
    except Exception as e:
        print(f"❌ 清除失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        client.close()


if __name__ == "__main__":
    namespace = sys.argv[1] if len(sys.argv) > 1 else None
    
    if namespace:
        print(f"⚠️  将清除namespace '{namespace}' 的所有数据（包括Neo4j和缓存）")
    else:
        print("⚠️  将清除Neo4j中的所有数据（包括所有缓存）")
    
    confirm = input("确认继续？(yes/no): ")
    if confirm.lower() in ['yes', 'y']:
        clear_neo4j(namespace)
    else:
        print("❌ 已取消")

