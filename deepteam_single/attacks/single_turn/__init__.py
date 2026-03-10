from .base_single_turn_attack import BaseSingleTurnAttack
from .base64 import Base64
from .roleplay import Roleplay
from .rot13 import ROT13

# Generic enhancement attacks (moved from agentic)

__all__ = [
    "BaseSingleTurnAttack",
    "Base64",
    "MathProblem",
    "Multilingual",
    "Roleplay",
    "ROT13",
]


def __getattr__(name: str):
    if name == "MathProblem":
        from .math_problem import MathProblem  # lazy import to avoid runpy warning
        return MathProblem
    if name == "Multilingual":
        from .multilingual import Multilingual  # lazy import to avoid runpy warning
        return Multilingual
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
