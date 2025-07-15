# -*- coding: utf-8 -*-
"""
Data Analysis Agent Package

基于开源 Qwen 的智能数据分析代理，专为 Jupyter Notebook 环境设计。
"""

import os
from typing import Optional, List, Dict, Any
from .utils.llm_helper_qwen import LLMHelperQwen
from .utils.code_executor import CodeExecutor
from .data_analysis_agent import DataAnalysisAgent

__version__ = "1.0.0"
__author__ = "Data Analysis Agent Team"

# 公开接口
__all__ = [
    "DataAnalysisAgent",
    "LLMHelperQwen",
    "CodeExecutor",
]


def create_agent(
    llm_helper: LLMHelperQwen = None,
    output_dir: str           = "outputs",
    max_rounds: int           = 30,
    absolute_path: bool       = False
) -> DataAnalysisAgent:
    """
    创建一个数据分析智能体实例

    Args:
        llm_helper: LLMHelperQwen 实例，如果为 None 则自动创建（基于环境变量配置）
        output_dir: 输出目录
        max_rounds: 最大分析轮数
        absolute_path: 是否使用绝对路径

    Returns:
        DataAnalysisAgent: 智能体实例
    """
    if llm_helper is None:
        llm_helper = LLMHelperQwen(
            model_name    = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen1.5-7B-Chat"),
            max_new_tokens= int(os.getenv("MAX_NEW_TOKENS", "1024")),
            temperature   = float(os.getenv("QWEN_TEMPERATURE", "0.7")),
            top_p         = float(os.getenv("QWEN_TOP_P", "0.9")),
        )
    return DataAnalysisAgent(
        llm_helper   = llm_helper,
        output_dir   = output_dir,
        max_rounds   = max_rounds,
        absolute_path= absolute_path,
    )


def quick_analysis(
    query: str,
    files: Optional[List[str]] = None,
    llm_helper: LLMHelperQwen  = None,
    output_dir: str            = "outputs",
    max_rounds: int            = 10,
    absolute_path: bool        = False
) -> Dict[str, Any]:
    """
    快速数据分析函数

    Args:
        query: 分析需求（自然语言）
        files: 数据文件路径列表
        llm_helper: LLMHelperQwen 实例，如果为 None 则自动创建
        output_dir: 输出目录
        max_rounds: 最大分析轮数
        absolute_path: 是否使用绝对路径

    Returns:
        dict: 分析结果
    """
    agent = create_agent(
        llm_helper   = llm_helper,
        output_dir   = output_dir,
        max_rounds   = max_rounds,
        absolute_path= absolute_path,
    )
    return agent.analyze(query, files)
