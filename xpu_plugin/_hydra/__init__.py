"""
Reinplementation of linear-related apis via hydra.
"""

from ._linear import LinearWithGradAccumulationAndAsyncCommunication
from ._linear_with_frozen_weight import LinearWithFrozenWeight

__all__ = ["LinearWithGradAccumulationAndAsyncCommunication", "LinearWithFrozenWeight"]
