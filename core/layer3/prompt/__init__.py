"""
Layer3 Prompt模块
"""

from .extraction_prompt import build_pattern_analysis_prompt
from .abstraction_prompt import build_abstraction_prompt

__all__ = ['build_pattern_analysis_prompt', 'build_abstraction_prompt']

