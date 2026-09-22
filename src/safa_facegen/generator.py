"""Stable PyTorch delivery interface for all six generator identities."""
import json
from pathlib import Path
from .common import MODEL_IDS, project_root


def registered_codec_path(model_id):
    root = project_root()
    local = root / "configs/local.json"
    settings = json.loads(local.read_text(encoding="utf-8")) if local.exists() else {}
    paths = settings.get("models", {}).get(model_id, {}).get("paths", {})
    default = "models/codecs/SD-VAE-EMA" if model_id.startswith("MeanFlow-") else "models/codecs/LDM-VQ4.pt"
    value = paths["codec"] if "codec" in paths else default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A registered codec path is required for {model_id}")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    return path.resolve(strict=True)


def load_generator(model_id, checkpoint, *, device="cuda", codec=None, **kwargs):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported generator: {model_id}")
    if codec is None and model_id != "RectifiedFlow-NCSNpp":
        codec = registered_codec_path(model_id)
    if model_id.startswith("MeanFlow-"):
        from .meanflow.generator import MeanFlowGenerator
        return MeanFlowGenerator.from_pretrained(Path(checkpoint), codec=codec, device=device,
                                                expected_model_id=model_id, **kwargs)
    from .torch_models import load_generator as load
    return load(model_id, checkpoint, device=device, codec_checkpoint=codec, **kwargs)
