from importlib.metadata import version
from .frontend import multiscale_deformable_attention, MultiscaleDeformableAttention


__version__ = version("msda_triton")

__all__ = [
    "multiscale_deformable_attention",
    "MultiscaleDeformableAttention",
]
