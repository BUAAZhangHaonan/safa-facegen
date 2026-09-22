"""Stable PyTorch delivery interface for all six generator identities."""
from pathlib import Path
from .common import MODEL_IDS


def load_generator(model_id, checkpoint, *, device="cuda", codec=None, **kwargs):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported generator: {model_id}")
    if model_id.startswith("MeanFlow-"):
        from .meanflow.generator import MeanFlowGenerator
        return MeanFlowGenerator.from_pretrained(Path(checkpoint), codec=codec, device=device,
                                                expected_model_id=model_id, **kwargs)
    from .torch_models import load_generator as load
    return load(model_id, checkpoint, device=device, codec_checkpoint=codec, **kwargs)
