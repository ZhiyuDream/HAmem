"""
Layer1 Prompt模块
"""

from .extraction_prompt import build_layer1_extraction_prompt
from .conflict_resolution_prompt import build_conflict_resolution_prompt

__all__ = [
    'build_layer1_extraction_prompt',
    'build_conflict_resolution_prompt'
]

