"""
JSON修复工具

使用json_repair库修复LLM返回的格式错误的JSON
"""

import json
from json_repair import repair_json
from typing import Any, Dict, Optional


class JSONFixer:
    """JSON修复器，用于处理LLM返回的格式错误的JSON"""
    
    @staticmethod
    def parse_json(json_string: str, default_value: Any = None) -> Any:
        """
        尝试解析JSON字符串，如果失败则尝试修复后再解析
        
        Args:
            json_string: 待解析的JSON字符串
            default_value: 解析失败时返回的默认值
        
        Returns:
            解析后的Python对象，或default_value
        """
        if not json_string or not json_string.strip():
            print("⚠️  JSON字符串为空")
            return default_value
        
        # 第一层：直接解析
        try:
            obj = json.loads(json_string)
            return obj
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON解析失败: {e}")
            print(f"   尝试修复JSON...")
            
            # 第二层：修复后解析
            try:
                fixed_json = repair_json(json_string)
                obj = json.loads(fixed_json)
                print(f"✅ JSON修复成功")
                return obj
            except Exception as repair_error:
                print(f"❌ JSON修复失败: {repair_error}")
                print(f"   原始JSON: {json_string[:200]}...")
                return default_value
    
    @staticmethod
    def clean_llm_response(response: str) -> str:
        """
        清理LLM响应，移除markdown代码块标记
        
        Args:
            response: LLM原始响应
        
        Returns:
            清理后的JSON字符串
        """
        cleaned = response.strip()
        
        # 移除markdown代码块标记
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        return cleaned.strip()
    
    @staticmethod
    def parse_llm_response(
        response: str, 
        expected_keys: Optional[list] = None,
        default_value: Any = None
    ) -> Any:
        """
        解析LLM响应为JSON对象
        
        Args:
            response: LLM原始响应
            expected_keys: 期望的顶层键列表（用于验证）
            default_value: 解析失败时返回的默认值
        
        Returns:
            解析后的Python对象
        """
        # 1. 清理响应
        cleaned_response = JSONFixer.clean_llm_response(response)
        
        # 2. 解析JSON
        obj = JSONFixer.parse_json(cleaned_response, default_value)
        
        # 3. 验证期望的键
        if obj and expected_keys:
            missing_keys = [key for key in expected_keys if key not in obj]
            if missing_keys:
                print(f"⚠️  缺少期望的键: {missing_keys}")
                # 添加缺失的键，使用空值
                for key in missing_keys:
                    if isinstance(obj, dict):
                        obj[key] = [] if key.endswith('s') else None
        
        return obj
    
    @staticmethod
    def safe_get_field(obj: Dict, field: str, default: Any = None) -> Any:
        """
        安全获取字段值
        
        Args:
            obj: 字典对象
            field: 字段名
            default: 默认值
        
        Returns:
            字段值或默认值
        """
        if not isinstance(obj, dict):
            return default
        return obj.get(field, default)


# 便捷函数
def parse_llm_json(response: str, expected_keys: list = None, default: Any = None) -> Any:
    """
    便捷函数：解析LLM返回的JSON
    
    Args:
        response: LLM响应
        expected_keys: 期望的键列表
        default: 默认值
    
    Returns:
        解析后的对象
    """
    return JSONFixer.parse_llm_response(response, expected_keys, default)

