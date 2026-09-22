"""PyTorch inference port of the pinned official JAX DiT, with input gradients.

The architecture follows vendor/meanflow (MIT); training stays in official JAX.
State names deliberately expose each operation for reversible conversion.
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .spec import LATENT_SCALE, NULL_LABEL, UPSTREAM_COMMIT, ModelSpec, get_spec


def position_embedding(hidden_size, grid_size):
    # Exact official NumPy construction: x-coordinate followed by y-coordinate.
    grid = np.meshgrid(np.arange(grid_size, dtype=np.float32),
                       np.arange(grid_size, dtype=np.float32))
    omega = 1.0 / 10000 ** (np.arange(hidden_size // 4, dtype=np.float64) / (hidden_size / 4))
    components = []
    for coord in grid:
        phase = np.einsum("m,d->md", coord.reshape(-1), omega)
        components.extend([np.sin(phase), np.cos(phase)])
    return torch.from_numpy(np.concatenate(components, axis=1).astype(np.float32))[None]


class TimeEmbedder(nn.Module):
    def __init__(self, width, frequency_size=256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(nn.Linear(frequency_size, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        half = self.frequency_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * frequencies[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.frequency_size % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.scale = (width // heads) ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Explicit attention supports both reverse input gradients and JVP.
        attn = ((q * self.scale) @ k.transpose(-1, -2)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, d))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = spec.hidden_size
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, spec.num_heads)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * spec.mlp_ratio)),
                                 nn.GELU(approximate="tanh"), nn.Linear(int(d * spec.mlp_ratio), d))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x, condition):
        a, b, c, d, e, f = self.adaLN_modulation(condition).chunk(6, -1)
        x = x + c[:, None] * self.attn(modulate(self.norm1(x), a, b))
        return x + f[:, None] * self.mlp(modulate(self.norm2(x), d, e))


class FinalLayer(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = spec.hidden_size
        self.norm_final = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d, spec.patch_size ** 2 * spec.in_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, -1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class TorchMeanFlow(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        d, p = spec.hidden_size, spec.patch_size
        self.x_embedder = nn.Conv2d(spec.in_channels, d, p, stride=p)
        self.t_embedder = TimeEmbedder(d, spec.time_embedding_size)
        self.h_embedder = TimeEmbedder(d, spec.time_embedding_size)
        self.y_embedder = nn.Embedding(spec.num_classes + 1, d)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(spec.depth))
        self.final_layer = FinalLayer(spec)
        self.register_buffer("pos_embed", position_embedding(d, spec.input_size // p), persistent=False)

    def forward(self, x, t, h):
        spec = self.spec
        hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed.to(hidden.dtype)
        c = self.t_embedder(t) + self.h_embedder(h) + self.y_embedder.weight[spec.num_classes][None]
        for block in self.blocks:
            hidden = block(hidden, c)
        p = spec.patch_size
        grid = spec.input_size // p
        patches = self.final_layer(hidden, c).reshape(x.shape[0], grid, grid, p, p, spec.in_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], spec.in_channels, spec.input_size, spec.input_size)


class MeanFlowGenerator(nn.Module):
    """Frozen EMA generator whose noise input remains differentiable for SAFA."""
    noise_shape = (4, 32, 32)
    output_range = (-1.0, 1.0)
    state_role = "ema"
    step_noise_count = 0

    def __init__(self, network: TorchMeanFlow, codec: nn.Module, *, model_id, latent_scale=LATENT_SCALE):
        super().__init__()
        self.model_id = model_id
        self.network = network
        self.codec = codec
        self.latent_scale = float(latent_scale)
        self.ema_sha256 = None
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, export_dir, *, codec, device="cpu", dtype=torch.float32,
                        expected_model_id=None):
        root = Path(export_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "safa-meanflow-torch" or manifest.get("format_version") != 1:
            raise ValueError("Unsupported MeanFlow export format")
        if expected_model_id is not None and manifest.get("model_id") != expected_model_id:
            raise ValueError("MeanFlow export differs from the requested model_id")
        if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("MeanFlow export references a different official implementation")
        if manifest.get("null_label") != NULL_LABEL or manifest.get("latent_scale") != LATENT_SCALE:
            raise ValueError("MeanFlow export NULL index or latent scale differs from the implementation")
        if manifest.get("ema_weights") != "ema.safetensors":
            raise ValueError("MeanFlow delivery requires the dedicated EMA safetensors")
        spec = get_spec(manifest["model_id"])
        if manifest.get("architecture") != spec.to_dict():
            raise ValueError("Export architecture does not match model_id")
        weights = root / "ema.safetensors"
        expected = manifest.get("sha256", {}).get(weights.name)
        if (not isinstance(expected,str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                or not weights.is_file()):
            raise ValueError("EMA artifact identity or weights missing")
        if codec is None or not Path(codec).is_dir():
            raise ValueError("A local registered MeanFlow VAE directory is required")
        from ..cache import codec_identity
        codec_info = codec_identity(codec)
        trained_codec = manifest.get("metadata",{}).get("identity",{}).get("codec",{}).get("sha256")
        if trained_codec and trained_codec != codec_info["sha256"]:
            raise ValueError("VAE identity differs from the MeanFlow training cache")
        from safetensors.torch import load_file
        from diffusers import AutoencoderKL
        net = TorchMeanFlow(spec)
        net.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        codec = AutoencoderKL.from_pretrained(str(codec), local_files_only=True)
        if (codec.config.latent_channels != 4 or codec.config.in_channels != 3
                or codec.config.out_channels != 3 or len(codec.config.block_out_channels) != 4
                or float(codec.config.scaling_factor) != LATENT_SCALE):
            raise ValueError("MeanFlow requires the registered RGB SD-VAE with four channels and downsample factor eight")
        generator = cls(net, codec, model_id=manifest["model_id"],
                        latent_scale=manifest["latent_scale"]).to(device=device, dtype=dtype)
        generator.ema_sha256 = expected
        generator.codec_identity = codec_info
        return generator

    def sample(self, noise, step_noises=None, grad_enabled=False):
        if noise.ndim != 4 or tuple(noise.shape[1:]) != self.noise_shape:
            raise ValueError(f"Expected noise [N,4,32,32], got {tuple(noise.shape)}")
        if step_noises is not None and len(step_noises):
            raise ValueError("Original one-step MeanFlow has no step noise inputs")
        if noise.device != self.network.x_embedder.weight.device:
            raise ValueError("Noise and generator must be on the same device")
        # enable_grad overrides an outer no_grad context when SAFA asks for it.
        with torch.enable_grad() if grad_enabled else torch.no_grad():
            noise = noise.to(self.network.x_embedder.weight.dtype)
            t = torch.ones(noise.shape[0], device=noise.device, dtype=torch.float32)
            latent = noise - self.network(noise, t, t)
            result = self.codec.decode(latent / self.latent_scale)
            rgb = result.sample if hasattr(result, "sample") else result[0]
            return rgb.clamp(-1.0, 1.0)

    def forward(self, noise):
        return self.sample(noise, grad_enabled=torch.is_grad_enabled())
