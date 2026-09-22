"""Original MeanFlow: official JAX training, strict conversion, PyTorch inference.

Imports are lazy: JAX training does not import torch or initialize a second GPU
allocator, and PyTorch inference does not import JAX.
"""

__all__ = ["MeanFlowGenerator"]


def __getattr__(name):
    if name == "MeanFlowGenerator":
        from .torch_model import MeanFlowGenerator
        return MeanFlowGenerator
    raise AttributeError(name)
