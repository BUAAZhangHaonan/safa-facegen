"""Portable runtime using the project's existing generator interfaces."""
from __future__ import annotations
import json
from pathlib import Path
import sys


def read_release(root: Path) -> dict:
    root = root.resolve()
    value = json.loads((root / 'release.json').read_text(encoding='utf-8'))
    if value.get('additional_training_updates') != 0 or len(value.get('models', [])) != 6:
        raise ValueError('Expected the six-model pretrained release manifest')
    return value


def resolve_local(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Runtime artifacts use relative paths')
    value = (root / p).resolve(strict=True)
    if not value.is_relative_to(root.resolve()):
        raise ValueError('Runtime artifact leaves the release directory')
    return value


def load_selected(root: Path, model_id: str | None = None, device: str = 'cuda:0'):
    root = root.resolve()
    manifest = read_release(root)
    model_id = model_id or manifest['default_model']
    rows = [r for r in manifest['models'] if r['model_id'] == model_id]
    if len(rows) != 1:
        raise ValueError(f'Unknown selected model: {model_id}')
    sys.path.insert(0, str(root / 'src'))
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('The selected CUDA runtime is unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from safa_facegen import load_generator
    import safa_facegen
    if not Path(safa_facegen.__file__).resolve().is_relative_to(root / 'src'):
        raise RuntimeError('Start a fresh process using the released source tree')
    r = rows[0]
    checkpoint = resolve_local(root, r['checkpoint'])
    codec = str(resolve_local(root, r['codec'])) if r.get('codec') else None
    generator = load_generator(model_id, str(checkpoint), device=device, codec=codec, **r['sampling'])
    identity = getattr(generator, 'ema_identity', None)
    identity = identity if isinstance(identity, dict) else {}
    # Legacy exports can leave selected provenance fields in their adjacent manifest.
    actual_id = identity.get('checkpoint_id')
    if actual_id and actual_id != r['checkpoint_id']:
        raise ValueError('Loaded checkpoint differs from the selected release identity')
    if getattr(generator, 'model_id', None) != model_id:
        raise ValueError('Loaded generator has a different model identity')
    return generator, r
