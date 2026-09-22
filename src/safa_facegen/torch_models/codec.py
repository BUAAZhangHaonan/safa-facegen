"""Inference-only official FFHQ VQ-f4 architecture, without Lightning/loss modules.

The VQ operation keeps upstream VectorQuantizer2's straight-through estimator.
Input gradients use that declared surrogate for the discrete argmin operation.
"""
from pathlib import Path
import json
import torch
from torch import nn
from .vendor import ldm_imports, ROOT

ldm_imports()
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2

DDCONFIG = dict(double_z=False, z_channels=3, resolution=256, in_channels=3,
                out_ch=3, ch=128, ch_mult=[1, 2, 4], num_res_blocks=2,
                attn_resolutions=[], dropout=0.0)


class VQModelInterface(nn.Module):
    representation = "prequant"
    gradient_estimator = "official_VectorQuantizer2_straight_through"
    scaling_factor = 1.0

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(**DDCONFIG)
        self.decoder = Decoder(**DDCONFIG)
        self.quantize = VectorQuantizer2(8192, 3, beta=0.25, remap=None, sane_index_shape=False)
        self.quant_conv = nn.Conv2d(3, 3, 1)
        self.post_quant_conv = nn.Conv2d(3, 3, 1)

    def encode(self, image):
        return self.quant_conv(self.encoder(image))

    def decode(self, latent, force_not_quantize=False):
        quant = latent if force_not_quantize else self.quantize(latent)[0]
        return self.decoder(self.post_quant_conv(quant))


def read_checkpoint(path):
    """Only use trusted official/project checkpoint files; pickle is not a sandbox."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def registered_codec(checkpoint):
    """Reuse the extraction/transfer registration without scanning codec tensors."""
    path=Path(checkpoint)
    metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if int(metadata['bytes'])!=path.stat().st_size:
        raise ValueError('Codec size differs from its registered identity')
    if not metadata.get('sha256') or not metadata.get('family'):
        raise ValueError('Codec registration requires family and the existing transfer sha256')
    return metadata


def load_codec(checkpoint, device="cpu", *, payload=None):
    if payload is None:payload = read_checkpoint(checkpoint)
    state = payload.get("state_dict", payload)
    prefix = "first_stage_model."
    codec_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not codec_state:
        codec_state = state
    codec = VQModelInterface()
    # Do not ignore missing codec tensors or mix codecs from a different LDM.
    codec.load_state_dict(codec_state, strict=True)
    return codec.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
