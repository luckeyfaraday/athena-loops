from .mock import MockAgent

__all__ = ["MockAgent"]

# ClaudeAgent is imported lazily so the package works without the anthropic SDK.
def __getattr__(name):
    if name == "ClaudeAgent":
        from .claude import ClaudeAgent
        return ClaudeAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
