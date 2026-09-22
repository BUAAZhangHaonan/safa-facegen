"""Strict legacy SAFA -> canonical official-coordinate MeanFlow conversion.

Legacy checkpoints are trusted local artifacts. No pickle is accepted by the
training or inference runtime: conversion is the one explicit torch.load site.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .io import atomic_json, sha256_file
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


def convert_legacy_state(state, spec):
    """Return canonical torch-layout NumPy tensors; no JAX dependency.

    P swaps hidden halves. Residual activations become P*x while time/NULL
    conditioning coordinates stay unchanged. QKV head coordinates are unchanged.
    """
    source = {}
    for name, value in state.items():
        name = name.removeprefix("module.").removeprefix("_orig_mod.")
        name = name.removeprefix("vector_field.")
        if name in source:
            raise ValueError(f"Duplicate normalized checkpoint key {name}")
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().numpy()
        source[name] = np.asarray(value, dtype=np.float32)
    expected = canonical_shapes(spec)
    old_shapes = {k.replace("h_embedder.", "r_embedder."): v for k, v in expected.items()
                  if k != "y_embedder.weight"}
    if "z_embedder.0.weight" not in source:
        raise ValueError("Expected SAFA legacy z_embedder and learned NULL checkpoint")
    embedding_size = source["z_embedder.0.weight"].shape[1]
    d = spec.hidden_size
    old_shapes.update({"z_embedder.0.weight": (d, embedding_size), "z_embedder.0.bias": (d,),
                       "z_embedder.2.weight": (d, d), "z_embedder.2.bias": (d,),
                       "null_condition.embedding": (embedding_size,)})
    # Historical LearnedNullCondition uses [D]; a serialized [1,D] is equivalent.
    if source.get("null_condition.embedding", np.empty(0)).shape == (1, embedding_size):
        source["null_condition.embedding"] = source["null_condition.embedding"][0]
    strict_check(source, old_shapes)
    condition = source["z_embedder.0.weight"] @ source["null_condition.embedding"] + source["z_embedder.0.bias"]
    sigmoid = np.exp(-np.logaddexp(0.0, -condition))
    condition = source["z_embedder.2.weight"] @ (condition * sigmoid) + source["z_embedder.2.bias"]
    result = {k.replace("r_embedder.", "h_embedder."): v.copy() for k, v in source.items()
              if not (k.startswith("z_embedder.") or k.startswith("null_condition."))}
    table = np.zeros((spec.num_classes + 1, d), dtype=np.float32)
    table[spec.num_classes] = condition
    result["y_embedder.weight"] = table
    permutation = np.r_[np.arange(d//2, d), np.arange(d//2)]
    result["x_embedder.weight"] = result["x_embedder.weight"][permutation]
    result["x_embedder.bias"] = result["x_embedder.bias"][permutation]
    for i in range(spec.depth):
        prefix = f"blocks.{i}."
        for name in ("attn.qkv.weight", "mlp.0.weight"):
            result[prefix+name] = result[prefix+name][:, permutation]
        for name in ("attn.proj", "mlp.2"):
            result[prefix+name+".weight"] = result[prefix+name+".weight"][permutation]
            result[prefix+name+".bias"] = result[prefix+name+".bias"][permutation]
        rows = np.concatenate([permutation + group*d for group in range(6)])
        for suffix in ("weight", "bias"):
            key = prefix + "adaLN_modulation.1." + suffix
            result[key] = result[key][rows]
    rows = np.r_[permutation, permutation+d]
    for suffix in ("weight", "bias"):
        key = "final_layer.adaLN_modulation.1." + suffix
        result[key] = result[key][rows]
    result["final_layer.linear.weight"] = result["final_layer.linear.weight"][:, permutation]
    result = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in result.items()}
    strict_check(result, expected)
    return result


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


def write_export(output, model_id, ema, *, raw=None, metadata=None, register_identity=True):
    from safetensors.numpy import save_file
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    strict_check(ema, canonical_shapes(get_spec(model_id)))
    save_file(ema, str(output / "ema.safetensors"))
    names = ["ema.safetensors"]
    if raw is not None:
        strict_check(raw, canonical_shapes(get_spec(model_id)))
        save_file(raw, str(output / "raw.safetensors"))
        names.append("raw.safetensors")
    manifest = {"format": "safa-meanflow-torch", "format_version": FORMAT_VERSION,
                "model_id": model_id, "architecture": get_spec(model_id).to_dict(),
                "upstream_commit": UPSTREAM_COMMIT, "latent_scale": LATENT_SCALE,
                "null_label": 1000, "ema_weights": "ema.safetensors",
                "sha256": {name: sha256_file(output/name) for name in names} if register_identity else {},
                "temporary_calibration": not register_identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def source_progress(payload):
    epoch = payload.get("epoch")
    epoch_field = "epoch" if epoch is not None else None
    if isinstance(payload.get("metrics"), dict):
        for name in ("epoch_1based", "stage_epoch_1based"):
            if payload["metrics"].get(name) is not None:
                epoch = payload["metrics"][name]
                epoch_field = "metrics."+name
                break
    step = payload.get("global_step", payload.get("step"))
    step_field = "global_step/step" if step is not None else None
    if payload.get("checkpoint_format") == "safa_meanflow_b2_null_prior_pilot_v2":
        epoch = payload["progress"]["epoch"]
        epoch_field = "progress.epoch (completed)"
        step = payload["progress"]["global_step"]
        step_field = "progress.global_step"
    if step is None and isinstance(payload.get("metrics"),dict):
        step = payload["metrics"].get("optimizer_steps")
        step_field = "metrics.optimizer_steps" if step is not None else None
    if step is None:
        moments = payload.get("optimizer_state_dict", {}).get("state", {})
        counts = {int(value["step"]) for value in moments.values() if "step" in value}
        if len(counts) == 1:
            step = counts.pop()
            step_field = "optimizer_state_dict.state[*].step (all equal)"
    return {"source_epoch": int(epoch) if epoch is not None else None,
            "source_epoch_field": epoch_field, "source_step": int(step) if step is not None else None,
            "source_step_field": step_field}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-key", default="model_state_dict")
    parser.add_argument("--ema-key", default="ema_model_state_dict")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    spec = get_spec(args.model_id)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if args.raw_key not in payload or args.ema_key not in payload:
        raise ValueError("Both explicitly selected raw and EMA dictionaries are required")
    # Fold the EMA's own NULL and own nonlinear zMLP, never reuse folded raw.
    raw = convert_legacy_state(payload[args.raw_key], spec)
    ema = convert_legacy_state(payload[args.ema_key], spec)
    metadata = {"source_path": str(Path(args.checkpoint).resolve()),
                "source_sha256": sha256_file(args.checkpoint),
                **source_progress(payload),
                "phase": "HQ", "hq_epoch": 0, "hq_step": 0,
                "optimizer_reset": True, "position_conversion": "residual_half_permutation",
                "raw_key": args.raw_key, "ema_key": args.ema_key}
    manifest = write_export(args.output, args.model_id, ema, raw=raw, metadata=metadata)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
