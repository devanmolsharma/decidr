"""decidr -- typed decisions from an LLM, in one forward pass."""

from .backend import Backend, DecisionError, OpenAIBackend
from .calibrate import CalibrationResult, evaluate_out_of_fold, fit_temperature
from .core import (
    MAX_BRANCHES_PER_LEVEL,
    MAX_ID_LENGTH,
    MIN_ID_LENGTH,
    Client,
    Decision,
    confidence,
    is_reliable,
    softmax,
    validate_row,
)
from .prefix import build_prefix_messages
from .token_cache import TokenCache

__all__ = [
    "MAX_BRANCHES_PER_LEVEL",
    "MAX_ID_LENGTH",
    "MIN_ID_LENGTH",
    "Backend",
    "CalibrationResult",
    "Client",
    "Decision",
    "DecisionError",
    "OpenAIBackend",
    "TokenCache",
    "build_prefix_messages",
    "confidence",
    "evaluate_out_of_fold",
    "fit_temperature",
    "is_reliable",
    "softmax",
    "validate_row",
]
__version__ = "0.8.0"
