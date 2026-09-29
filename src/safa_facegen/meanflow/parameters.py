"""Strict MeanFlow parameter-layout conversion and EMA export."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts import MEANFLOW_EXPORT_FORMAT
from .io import atomic_json
from .spec import FORMAT_VERSION, LATENT_SCALE, UPSTREAM_COMMIT, get_spec


def canonical_shapes(spec):
    d, p, f = spec.hidden_size, spec.patch_size, spec.time_embedding_size
    result = {"x_embedder.weight": (d, spec.in_channels, p, p), "x_embedder.bias": (d,),
              "y_embedder.weight": (spec.num_classes + 1, d)}
    for name in ("t_embedder", "h_embedder"):
        result.update({f"{name}.mlp.0.weight": (d, f), f"{name}.mlp.0.bias": (d,),
                       f"{name}.mlp.2.weight": (d, d), f"{name}.mlp.2.bias": (d,)})
    for i in range(spec.depth):
        for name, size in (("attn.qkv", (3*d, d)), ("attn.proj", (d, d)),
                           ("mlp.0", (int(d*spec.mlp_ratio), d)),
                           ("mlp.2", (d, int(d*spec.mlp_ratio))),
                           ("adaLN_modulation.1", (6*d, d))):
            result[f"blocks.{i}.{name}.weight"] = size
            result[f"blocks.{i}.{name}.bias"] = (size[0],)
    for name, size in (("linear", (p*p*spec.in_channels, d)), ("adaLN_modulation.1", (2*d, d))):
        result[f"final_layer.{name}.weight"] = size
        result[f"final_layer.{name}.bias"] = (size[0],)
    return result


def strict_check(state, shapes):
    missing, extra = sorted(set(shapes) - set(state)), sorted(set(state) - set(shapes))
    if missing or extra:
        raise ValueError(f"State keys mismatch: missing={missing}, unexpected={extra}")
    for key, shape in shapes.items():
        value = state[key]
        if tuple(value.shape) != shape:
            raise ValueError(f"Shape mismatch {key}: {value.shape} != {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite weights in {key}")


def torch_key_to_flax(key):
    """Pinned official Linen parameter paths; checked against actual init tree."""
    if key == "y_embedder.weight":
        return ("net", "y_embedder", "embedding_table", "_flax_embedding", "embedding")
    if key.startswith("x_embedder."):
        return ("net", "x_embedder", "proj", "kernel" if key.endswith("weight") else "bias")
    pieces = key.split(".")
    if pieces[0] in ("t_embedder", "h_embedder"):
        path = [pieces[0], "mlp", "layers_"+pieces[2]]
    elif pieces[0] == "blocks":
        path = ["blocks", "layers_"+pieces[1]]
        if pieces[2] == "mlp":
            path += ["mlp", "fc1" if pieces[3] == "0" else "fc2"]
        elif pieces[2] == "adaLN_modulation":
            path += ["adaLN_modulation", "layers_1"]
        else:
            path += pieces[2:-1]
    elif pieces[0] == "final_layer":
        path = ["final_layer", pieces[1]]
        if pieces[1] == "adaLN_modulation":
            path += ["layers_1"]
    else:
        raise ValueError(f"Unmapped key {key}")
    return tuple(["net", *path, "_flax_linear", "kernel" if pieces[-1] == "weight" else "bias"])


def canonical_to_flax(state, spec, template=None):
    from flax.traverse_util import flatten_dict, unflatten_dict
    strict_check(state, canonical_shapes(spec))
    flat = {}
    for key, array in state.items():
        if key == "x_embedder.weight":
            array = array.transpose(2, 3, 1, 0)
        elif key.endswith(".weight") and key != "y_embedder.weight":
            array = array.T
        flat[torch_key_to_flax(key)] = np.ascontiguousarray(array)
    if template is not None:
        expected = flatten_dict(template)
        strict_check(flat, {k: tuple(v.shape) for k, v in expected.items()})
    return unflatten_dict(flat)


def flax_to_canonical(params, spec):
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    expected = canonical_shapes(spec)
    wanted = {torch_key_to_flax(k) for k in expected}
    if set(flat) != wanted:
        raise ValueError(f"Flax paths mismatch: missing={wanted-set(flat)}, extra={set(flat)-wanted}")
    output = {}
    for key in expected:
        array = np.asarray(flat[torch_key_to_flax(key)])
        if key == "x_embedder.weight":
            array = array.transpose(3, 2, 0, 1)
        elif key.endswith(".weight") and key != "y_embedder.weight":
            array = array.T
        output[key] = np.ascontiguousarray(array, dtype=np.float32)
    strict_check(output, expected)
    return output


def write_export(output, model_id, ema, *, raw=None, metadata=None, register_identity=True, integrity_mode="metadata"):
    from safetensors.numpy import save_file
    if integrity_mode != "metadata":
        raise ValueError("New exports require metadata integrity")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    strict_check(ema, canonical_shapes(get_spec(model_id)))
    save_file(ema, str(output / "ema.safetensors"))
    names = ["ema.safetensors"]
    if raw is not None:
        strict_check(raw, canonical_shapes(get_spec(model_id)))
        save_file(raw, str(output / "raw.safetensors"))
        names.append("raw.safetensors")
    manifest = {"format": MEANFLOW_EXPORT_FORMAT, "format_version": FORMAT_VERSION,
                "model_id": model_id, "architecture": get_spec(model_id).to_dict(),
                "upstream_commit": UPSTREAM_COMMIT, "latent_scale": LATENT_SCALE,
                "null_label": 1000, "ema_weights": "ema.safetensors",
                "integrity_mode": integrity_mode,
                "files": [{"path": name, "bytes": (output/name).stat().st_size} for name in names],
                "sha256": {},
                "temporary_calibration": not register_identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}}
    source = metadata or {}
    manifest.update(state_role="ema", checkpoint_id=source.get("checkpoint", source.get("checkpoint_id")),
                    stage_id=source.get("stage_id"), objective_id=source.get("objective_id"),
                    step=source.get("hq_step", source.get("step")),
                    parent_checkpoint_id=source.get("parent_checkpoint_id"),
                    codec_registration=source.get("codec_registration", source.get("identity", {}).get("codec")))
    atomic_json(output / "manifest.json", manifest)
    return manifest
