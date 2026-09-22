"""Dependency-free format and architecture definitions."""
from dataclasses import asdict, dataclass

UPSTREAM_COMMIT = "d70cb55d298ee03c53bf6da67bec281082e4e2d9"
FORMAT_VERSION = 1
NULL_LABEL = 1000
LATENT_SCALE = 0.18215

FIXED_RECIPE = {
    "precision": "fp32",
    "adam_betas": [0.9, 0.95],
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "ema_decay": 0.9999,
    "time_sampling": {"distribution": "logit_normal", "mean": -0.4, "std": 1.0},
    "data_proportion": 0.75,
    "norm_p": 1.0,
    "norm_eps": 0.01,
    "null_label": NULL_LABEL,
    "guidance": "none",
    "class_dropout_prob": 0.0,
    "latent_scale": LATENT_SCALE,
}


def validate_recipe(config):
    for key, expected in FIXED_RECIPE.items():
        if config.get(key) != expected:
            raise ValueError(f"Original MeanFlow implementation requires {key}={expected!r}; "
                             f"configuration provided {config.get(key)!r}")


@dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    depth: int
    num_heads: int
    patch_size: int
    input_size: int = 32
    in_channels: int = 4
    mlp_ratio: float = 4.0
    time_embedding_size: int = 256
    num_classes: int = 1000

    def to_dict(self):
        return asdict(self)


SPECS = {
    "MeanFlow-B-4": ModelSpec(768, 12, 12, 4),
    "MeanFlow-B-2": ModelSpec(768, 12, 12, 2),
    "MeanFlow-L-2": ModelSpec(1024, 24, 16, 2),
}
OFFICIAL_NAMES = {
    "MeanFlow-B-4": "DiT_B_4",
    "MeanFlow-B-2": "DiT_B_2",
    "MeanFlow-L-2": "DiT_L_2",
}


def get_spec(model_id):
    try:
        return SPECS[model_id]
    except KeyError:
        raise ValueError(f"Unknown MeanFlow model {model_id!r}; expected {list(SPECS)}") from None
