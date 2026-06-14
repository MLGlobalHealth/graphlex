"""graphlex — deterministic, LLM-interpretable language for NetworkX graphs."""
from .core.facts import facts
from .verbalize.render import verbalize

__all__ = ["facts", "verbalize"]
__version__ = "0.0.1"
