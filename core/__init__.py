"""
Optimus-Megatron Dynamic Parallelism Framework
This module implements dynamic parallelism strategies for efficient training of large language models.
"""

from .hardware_profiler import HardwareProfiler
from .model_profiler import ModelProfiler
from .dataset_profiler import DatasetProfiler
from .strategy_selector import DynamicStrategySelector
from .parallelism_manager import ParallelismManager

__all__ = [
    'HardwareProfiler',
    'ModelProfiler',
    'DatasetProfiler',
    'DynamicStrategySelector',
    'ParallelismManager',
]
