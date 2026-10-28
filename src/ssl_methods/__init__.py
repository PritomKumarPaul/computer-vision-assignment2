"""Self-supervised objectives trained only on the supplied images."""

from .dino import DINO
from .moco import MoCo

__all__ = ["DINO", "MoCo"]

