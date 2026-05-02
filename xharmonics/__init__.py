"""Top-level package exports for xharmonics."""

from . import accessor as _accessor  # noqa: F401  (registers xarray accessors on import)
from .core import evaluate, fit, infer_sampling_frequency

__all__ = [
    "fit",
    "evaluate",
    "infer_sampling_frequency",
]
