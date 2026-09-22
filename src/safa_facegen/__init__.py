"""Unconditional face generators and traceable pretraining."""

__version__ = "0.1.0"


def load_generator(model_id, checkpoint, **kwargs):
    from .generator import load_generator as load
    return load(model_id, checkpoint, **kwargs)
