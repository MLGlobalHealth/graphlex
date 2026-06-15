"""graphlex — deterministic, LLM-interpretable language for NetworkX graphs."""
from .core.facts import facts
from .verbalize.render import verbalize, verbalize_node

__all__ = ["facts", "verbalize", "verbalize_node"]
__version__ = "0.0.1"
