__all__ = ["MathProblem"]

def __getattr__(name: str):
    if name == "MathProblem":
        from .math_problem import MathProblem
        return MathProblem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")