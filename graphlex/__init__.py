"""graphlex — deterministic, LLM-interpretable language for NetworkX graphs."""
from .core.facts import (
    facts, feature_vector, feature_names,
    GROUPS, SCALAR_GROUPS, ALL_GROUPS, FEATURE_VERSION,
)
from .verbalize.render import verbalize, verbalize_node

__all__ = [
    "facts", "verbalize", "verbalize_node",
    "feature_vector", "feature_names",
    "GROUPS", "SCALAR_GROUPS", "ALL_GROUPS", "FEATURE_VERSION",
]
__version__ = "0.0.1"
