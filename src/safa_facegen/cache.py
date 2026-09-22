"""Build codec-specific float32 mmap caches; original and flip are encoded separately."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .data import HQDataset, atomic_json, sha256_file


def file_stat(path: str | Path) -> dict:
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def codec_identity(checkpoint: str | Path, registered: dict | None = None) -> dict:
    """Use the identity registered at acquisition; do not rescan immutable weights."""
    path = Path(checkpoint).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if registered is None:
        registry = path.with_suffix(".json") if path.is_file() else path.with_name(path.name + ".json")
        if not registry.is_file():
            raise FileNotFoundError(f"Codec acquisition identity is required: {registry}")
        registered = json.loads(registry.read_text(encoding="utf-8"))
    records = registered.get("files") or [{"path": path.name, "bytes": registered["bytes"]}]
    stats = {}
    for record in records:
        target = path if path.is_file() else path / record["path"]
        if not target.resolve().is_relative_to(path if path.is_dir() else path.parent):
            raise ValueError("Codec registration escapes its directory")
        stats[record["path"]] = file_stat(target)
        if stats[record["path"]]["bytes"] != record["bytes"]:
            raise ValueError(f"Codec size differs from its acquisition identity: {target}")
        previous = registered.get("verified_stats", {}).get(record["path"])
        if previous and stats[record["path"]] != previous:
            raise ValueError(f"Codec changed after acquisition: {target}")
    return {"sha256": registered["sha256"], "files": records, "verified_stats": stats}


def load_codec(family: str, checkpoint: str | Path, device: str, factory: str | None):
    if factory:
        module, separator, name = factory.partition(":")
        if not separator:
            raise ValueError("codec factory must be module:function")
        codec = getattr(importlib.import_module(module), name)(str(checkpoint), device)
    elif family == "meanflow":
        from diffusers import AutoencoderKL
        codec = AutoencoderKL.from_pretrained(str(checkpoint), local_files_only=True).to(device)
    else:
        raise ValueError("LDM cache requires --codec-factory for the official VQModelInterface")
    if hasattr(codec, "eval"):
        codec.eval()
    if hasattr(codec, "requires_grad_"):
        codec.requires_grad_(False)
    return codec


def encode_view(codec, family: str, images: torch.Tensor) -> torch.Tensor:
    if family == "meanflow":
        encoded = codec.encode(images)
        posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded
        if not hasattr(posterior, "mean") or not hasattr(posterior, "std"):
            raise TypeError("MeanFlow codec must return a diagonal Gaussian posterior with mean and std")
        value = torch.cat((posterior.mean, posterior.std), dim=1)
        expected = (8, 32, 32)
        if torch.any(posterior.std < 0):
            raise ValueError("Posterior standard deviation must be non-negative")
    else:
        # VQModelInterface.encode returns pre-quantization continuous latents.
        value = codec.encode(images)
        expected = (3, 64, 64)
    if not torch.is_tensor(value) or tuple(value.shape[1:]) != expected:
        raise ValueError(f"Codec returned {getattr(value, 'shape', type(value))}; expected [N,{expected}]")
    if not torch.isfinite(value).all():
        raise FloatingPointError("Codec produced non-finite values")
    return value.float()


def build_cache(*, manifest: str | Path, output_root: str | Path, family: str,
                checkpoint: str | Path, device: str = "cuda:0", batch_size: int = 32,
                factory: str | None = None, expected_codec_sha256: str | None = None,
                image_root: str | Path | None = None,
                resume_manifest: str | Path | None = None) -> dict:
    if family not in ("meanflow", "ldm") or batch_size < 1:
        raise ValueError("family must be meanflow/ldm and batch size must be positive")
    dataset = HQDataset(manifest, random_flip=False, image_root=image_root)
    dataset_stat = file_stat(manifest)
    previous = json.loads(Path(resume_manifest).read_text()) if resume_manifest else None
    if previous and previous.get("status") not in ("building", "failed"):
        raise ValueError("Only a building/failed partial cache can be resumed")
    # This small metadata hash binds ordering on first build/resume, never image bytes.
    dataset_hash = sha256_file(manifest)
    if previous and previous["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("Resume dataset manifest differs from the original cache")
    identity = codec_identity(checkpoint, previous["codec"] if previous else None)
    if expected_codec_sha256 and identity["sha256"] != expected_codec_sha256:
        raise ValueError("Codec checkpoint differs from the requested SHA256")
    output = Path(resume_manifest).resolve().parent if previous else Path(output_root).resolve() / f"{family}-{identity['sha256'][:16]}-{dataset_hash[:16]}"
    if not previous:
        output.mkdir(parents=True, exist_ok=False)
    representation = "mean_std" if family == "meanflow" else "prequant"
    shape = [len(dataset), 2, 8, 32, 32] if family == "meanflow" else [len(dataset), 2, 3, 64, 64]
    required_bytes = int(np.prod(shape)) * 4 + 4096
    if not previous and shutil.disk_usage(output).free < required_bytes:
        raise OSError(f"Insufficient free disk space for {required_bytes} byte cache")
    cache_manifest = previous or {
        "schema_version": 1, "status": "building", "dataset_manifest_sha256": dataset_hash,
        "codec": {"family": family, **identity}, "array": "latents.npy",
        "shape": shape, "dtype": "float32", "layout": "N,F,C,H,W",
        "representation": representation, "flip_axis": ["original", "horizontal_flip_before_encoding"],
        "scaled": False, "rows_completed": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if previous and (previous["codec"]["family"] != family or previous["shape"] != shape
                     or previous["dtype"] != "float32" or previous["representation"] != representation):
        raise ValueError("Resume cache layout/family differs from this invocation")
    start_row = int(cache_manifest["rows_completed"])
    if not 0 <= start_row <= len(dataset):
        raise ValueError("Invalid committed cache row boundary")
    cache_manifest.update(status="building", dataset_manifest_stat=dataset_stat, codec={"family": family, **identity})
    cache_manifest.pop("error", None)
    status_path = output / "manifest.json"
    atomic_json(status_path, cache_manifest)
    partial_path = output / "latents.partial.npy"
    array = None
    try:
        codec = load_codec(family, checkpoint, device, factory)
        array = np.lib.format.open_memmap(partial_path, mode="r+" if previous else "w+", dtype=np.float32, shape=tuple(shape))
        if tuple(array.shape) != tuple(shape) or array.dtype != np.float32:
            raise ValueError("Partial cache header differs from its manifest")
        if partial_path.stat().st_size != array.offset + array.nbytes:
            raise ValueError("Partial cache is truncated")
        with torch.inference_mode():
            for batch_number, start in enumerate(range(start_row, len(dataset), batch_size), 1):
                stop = min(start + batch_size, len(dataset))
                for index in range(start, stop):
                    record = dataset.records[index]
                    actual = file_stat(dataset.image_path(index))
                    if actual["bytes"] != record["bytes"] or actual["mtime_ns"] != record["mtime_ns"]:
                        raise ValueError(f"Image changed after manifest validation: {dataset.records[index]['id']}")
                batch = torch.stack([dataset[index] for index in range(start, stop)]).to(device)
                for flip in (0, 1):
                    view = batch if flip == 0 else batch.flip(-1)
                    array[start:stop, flip] = encode_view(codec, family, view).cpu().numpy()
                if batch_number % 100 == 0 or stop == len(dataset):
                    array.flush()
                    with partial_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    cache_manifest["rows_completed"] = stop
                    atomic_json(status_path, cache_manifest)
                    print(json.dumps({"cache": str(output), "rows_completed": stop, "total": len(dataset)}), flush=True)
        array.flush()
        del array
        array = None
        if file_stat(manifest) != dataset_stat or codec_identity(checkpoint, identity)["verified_stats"] != identity["verified_stats"]:
            raise RuntimeError("Dataset manifest or codec checkpoint changed during cache construction")
        os.replace(partial_path, output / "latents.npy")
        cache_manifest.update(status="complete", rows_completed=len(dataset),
                              array_bytes=(output / "latents.npy").stat().st_size)
        atomic_json(status_path, cache_manifest)
        return {"manifest": str(status_path), **cache_manifest}
    except BaseException as exc:
        if array is not None:
            array.flush()
        cache_manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(status_path, cache_manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--family", choices=["meanflow", "ldm"], required=True)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--codec-factory")
    parser.add_argument("--expected-codec-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-root")
    parser.add_argument("--resume-manifest", help="Resume a flushed partial cache without changing its identity or path")
    args = parser.parse_args()
    result = build_cache(manifest=args.manifest, output_root=args.output_root, family=args.family,
                         checkpoint=args.codec_checkpoint, factory=args.codec_factory,
                         expected_codec_sha256=args.expected_codec_sha256, device=args.device,
                         batch_size=args.batch_size, image_root=args.image_root, resume_manifest=args.resume_manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
