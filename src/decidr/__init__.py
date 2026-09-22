"""decidr -- typed decisions from local LLMs, in one forward pass."""

from .backend import Backend, LiteLLMBackend, OllamaBackend
from .calibrate import CalibrationResult, evaluate_out_of_fold, fit_temperature
from .core import (
    Client,
    Decision,
    DecisionError,
    softmax,
    validate_row,
)
from .prefix import build_prefix_messages

__all__ = [
    "Backend",
    "CalibrationResult",
    "Client",
    "Decision",
    "DecisionError",
    "LiteLLMBackend",
    "OllamaBackend",
    "build_prefix_messages",
    "evaluate_out_of_fold",
    "fit_temperature",
    "softmax",
    "validate_row",
]
__version__ = "0.7.1"
