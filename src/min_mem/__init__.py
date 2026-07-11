"""Min-mem: semantic-preserving memory minification via a minimal synonym dictionary."""

from min_mem.agent import MemoryMinifier, SessionStats
from min_mem.converter import MinMemConverter, MinifyResult

__all__ = [
    "MinMemConverter",
    "MinifyResult",
    "MemoryMinifier",
    "SessionStats",
]
__version__ = "0.2.0"
