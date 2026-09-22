# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
from functools import partial

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

from .timm_models import Attention, Mlp, PatchEmbed
from .torch_models import TorchEmbedding, TorchLayerNorm, TorchLinear

DiTLinear = partial(TorchLinear, weight_init='xavier_uniform', bias_init='zeros')
DiTAttention = partial(Attention, linear_layer=DiTLinear, norm_layer=TorchLayerNorm)
DiTMlp = partial(Mlp, linear_layer=DiTLinear)

def unsqueeze(t, dim):
    return jnp.expand_dims(t, axis=dim)

def modulate(x, shift, scale):
    """do an affine transformation on x"""
    return x * (1 + unsqueeze(scale, 1)) + unsqueeze(shift, 1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256
    
    def setup(self):
        frequency_embedding_size = self.frequency_embedding_size; hidden_size = self.hidden_size
        self.mlp = nn.Sequential([
            TorchLinear(frequency_embedding_size, hidden_size, bias=True, weight_init='0.02', bias_init='zeros'),
            nn.silu,
            TorchLinear(hidden_size, hidden_size, bias=True, weight_init='0.02', bias_init='zeros'),
        ])

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    num_classes: int
    hidden_size: int
    dropout_prob: float

    def setup(self):
        num_classes = self.num_classes; hidden_size = self.hidden_size; dropout_prob = self.dropout_prob
        use_cfg_embedding = True
        self.embedding_table = TorchEmbedding(num_classes + use_cfg_embedding, hidden_size)

    def token_drop(self, labels, force_drop_ids=None, rng=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = random.uniform(shape=(labels.shape[0],), key=rng) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels) # mysterious
        return labels

    def __call__(self, labels, train, force_drop_ids=None, rng=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids, rng=rng)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    
    def setup(self):
        hidden_size = self.hidden_size; num_heads = self.num_heads; mlp_ratio = self.mlp_ratio
        self.norm1 = TorchLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = TorchLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: partial(nn.gelu, approximate=True)
        self.mlp = DiTMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            TorchLinear(hidden_size, 6 * hidden_size, bias=True, weight_init='zeros', bias_init='zeros')
        ])

    def __call__(self, x, c): # c: condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_modulation(c), 6, axis=1)
        x = x + unsqueeze(gate_msa, 1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + unsqueeze(gate_mlp, 1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        hidden_size = self.hidden_size; patch_size = self.patch_size; out_channels = self.out_channels
        self.norm_final = TorchLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = TorchLinear(hidden_size, patch_size * patch_size * out_channels, bias=True, weight_init='zeros', bias_init='zeros')
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            TorchLinear(hidden_size, 2 * hidden_size, bias=True, weight_init='zeros', bias_init='zeros')
        ])

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = False
    
    def setup(self):
        input_size = self.input_size; patch_size = self.patch_size; in_channels = self.in_channels; hidden_size = self.hidden_size; depth = self.depth; num_heads = self.num_heads; mlp_ratio = self.mlp_ratio; class_dropout_prob = self.class_dropout_prob; num_classes = self.num_classes; learn_sigma = self.learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(get_2d_sincos_pos_embed(hidden_size, int(num_patches ** 0.5))).astype(jnp.float32)

        self.blocks = nn.Sequential([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        ---
        return: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def __call__(self, x, t, h, y, train=False, key=None):
        """
        __call__ pass of DiT.
        x: (B, H, W, C) input noisy data
        t: (B,) diffusion timesteps
        y: (B,) class labels
        ---
        return: (B, H, W, C) 
        """
        x = self.x_embedder(x) + self.pos_embed_func()  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                          # (N, D)
        h = self.h_embedder(h)                          # (N, D)
        y = self.y_embedder(y, train=train, rng=key)    # (N, D)
        c = t + h + y                                   # (N, D)
        for block in self.blocks.layers:
            x = block(x, c)                             # (N, T, D)
        x = self.final_layer(x, c)                      # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                          # (N, H, W, out_channels)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

DiT_XL_2 = partial(DiT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
DiT_XL_4 = partial(DiT, depth=28, hidden_size=1152, patch_size=4, num_heads=16)
DiT_XL_8 = partial(DiT, depth=28, hidden_size=1152, patch_size=8, num_heads=16)
DiT_L_2 = partial(DiT, depth=24, hidden_size=1024, patch_size=2, num_heads=16)
DiT_L_4 = partial(DiT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
DiT_L_8 = partial(DiT, depth=24, hidden_size=1024, patch_size=8, num_heads=16)
DiT_M_2 = partial(DiT, depth=16, hidden_size=1024, patch_size=2, num_heads=16)
DiT_M_4 = partial(DiT, depth=16, hidden_size=1024, patch_size=4, num_heads=16)
DiT_M_8 = partial(DiT, depth=16, hidden_size=1024, patch_size=8, num_heads=16)
DiT_B_2 = partial(DiT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
DiT_B_4 = partial(DiT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
DiT_B_8 = partial(DiT, depth=12, hidden_size=768, patch_size=8, num_heads=12)
DiT_S_2 = partial(DiT, depth=12, hidden_size=384, patch_size=2, num_heads=6)
DiT_S_4 = partial(DiT, depth=12, hidden_size=384, patch_size=4, num_heads=6)
DiT_S_8 = partial(DiT, depth=12, hidden_size=384, patch_size=8, num_heads=6)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-M/2':  DiT_M_2,   'DiT-M/4':  DiT_M_4,   'DiT-M/8':  DiT_M_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

if __name__ == '__main__':
    model = DiT_S_4(num_classes=1000)
    variables = model.init(random.PRNGKey(0), jnp.ones((2, 32, 32, 4)), jnp.ones((2,)), jnp.ones((2,), dtype=jnp.int32))
    x = np.random.randn(2, 32, 32, 4).astype(np.float32)
    t = np.random.randint(0, 1000, (2,)).astype(np.float32)
    y = np.random.randint(0, 1000, (2,)).astype(np.int32)

    out = model.apply(variables, x, t, t-t, y=y, train=True, key=random.PRNGKey(0))
    print(out.min(), out.max())
    print(out.shape)
