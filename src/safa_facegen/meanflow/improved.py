"""iMF boundary-velocity objective on the EXISTING MeanFlow backbone.
Mathematical adaptation of arXiv:2512.02012 Algorithm 1, not the full iMF model.
apply_u(params, z, t, r) MUST internally call the old net with h=t-r.
Input clean is scaled posterior-sampled latent [B,H,W,C], not RGB/cache mean_std.
"""
from __future__ import annotations
from collections.abc import Callable
import jax
import jax.numpy as jnp


def configure_precision() -> None:
    """Call in trainer startup BEFORE model initialization and any JIT/pmap trace."""
    jax.config.update("jax_default_matmul_precision", "highest")


def improved_loss(params, apply_u: Callable, clean, key, *, equal_fraction: float = 0.5,
                  time_mean: float = -0.4, time_std: float = 1.0,
                  norm_p: float = 1.0, norm_eps: float = 0.01):
    if clean.ndim != 4 or clean.shape[0] < 2:
        raise ValueError("Expected [B,H,W,C], B>=2")
    if not 0 <= equal_fraction <= 1 or time_std <= 0 or norm_eps <= 0 or norm_p < 0:
        raise ValueError("Invalid objective configuration")
    b = clean.shape[0]
    key_t, key_r, key_e = jax.random.split(key, 3)
    shape = (b, 1, 1, 1)
    t = jax.nn.sigmoid(time_mean + time_std * jax.random.normal(key_t, shape))
    r = jax.nn.sigmoid(time_mean + time_std * jax.random.normal(key_r, shape))
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    boundary = jnp.arange(b) < int(b * equal_fraction)
    r = jnp.where(boundary.reshape(shape), t, r)
    epsilon = jax.random.normal(key_e, clean.shape, dtype=jnp.float32)
    clean = clean.astype(jnp.float32)
    z = (1 - t) * clean + t * epsilon
    conditional_velocity = epsilon - clean
    # Only the tangent direction is changed vs the original MF objective.
    # Stop its parameter gradient: the whole JVP outcome is stop-gradient below.
    tangent = jax.lax.stop_gradient(apply_u(params, z, t, t))
    fn = lambda zz, tt, rr: apply_u(params, zz, tt, rr)
    u, derivative = jax.jvp(fn, (z, t, r),
                            (tangent, jnp.ones_like(t), jnp.zeros_like(r)))
    compound_velocity = u + (t - r) * jax.lax.stop_gradient(derivative)
    error = compound_velocity - conditional_velocity
    # Preserve the original SUM-over-latent-coordinates adaptive normalization.
    squared_sum = jnp.sum(jnp.square(error), axis=(1, 2, 3))
    weighted = squared_sum / jax.lax.stop_gradient((squared_sum + norm_eps) ** norm_p)
    per_coordinate_mse = squared_sum / (clean.shape[1] * clean.shape[2] * clean.shape[3])
    def masked_mean(mask):
        return jnp.sum(jnp.where(mask, per_coordinate_mse, 0)) / jnp.maximum(jnp.sum(mask), 1)
    metrics = {
        "loss": jnp.mean(weighted),
        "raw_residual_mse": jnp.mean(per_coordinate_mse),
        "boundary_residual_mse": masked_mean(boundary),
        "interval_residual_mse": masked_mean(~boundary),
        "boundary_count": jnp.sum(boundary),
        "interval_count": jnp.sum(~boundary),
        "t_mean": jnp.mean(t),
        "interval_mean": jnp.mean(t-r),
    }
    return jnp.mean(weighted), metrics
