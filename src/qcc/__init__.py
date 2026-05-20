"""
QCC Framework: Evidence Elicitation in Interactive Diagnosis Benchmark

This package provides evaluation tools for assessing LLM diagnostic capabilities
through multi-turn patient-doctor interactions.
"""

from qcc.config import ModelConfig, get_model_config, load_config
from qcc.benchmark import Benchmark

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "ModelConfig",
    "get_model_config",
    "load_config",
]
