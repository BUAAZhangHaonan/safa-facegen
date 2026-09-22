"""Fixed 1024-sample EMA evaluation without image rejection or metric fallbacks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import HQDataset, atomic_json, sha256_file
from .cache import file_stat

SAMPLE_COUNT = 1024
GRID_COUNT = 256


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def make_fixed_noises(generator, indices: list[int], seed: int, device: str,
                      max_step_noise_bytes: int = 512 * 1024 * 1024):
    """Each sample has its own CPU RNG stream, independent of evaluation batch size."""
    shape = tuple(generator.noise_shape)
    count = int(getattr(generator, "step_noise_count", 0))
    step_shape = tuple(getattr(generator, "step_noise_shape", shape))
    if count < 0 or not shape or any(int(x) <= 0 for x in shape + step_shape):
        raise ValueError("Invalid generator noise contract")
    budget = len(indices) * count * int(np.prod(step_shape)) * 4
    if budget > max_step_noise_bytes:
        raise MemoryError(f"Explicit step noise needs {budget} bytes; choose a smaller evaluation batch explicitly")
    initial, per_step = [], [[] for _ in range(count)]
    for index in indices:
        rng = torch.Generator(device="cpu").manual_seed(seed + index)
        initial.append(torch.randn(shape, generator=rng, dtype=torch.float32))
        for step in range(count):
            per_step[step].append(torch.randn(step_shape, generator=rng, dtype=torch.float32))
    noise = torch.stack(initial).to(device)
    steps = [torch.stack(values).to(device) for values in per_step] if count else None
    return noise, steps


def checkpoint_stat(path: str | Path) -> dict:
    path = Path(path)
    if path.is_file():
        return {path.name: file_stat(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("EMA export must contain regular files and no symlinks")
    return {p.relative_to(path).as_posix(): file_stat(p) for p in files}


def evaluation_asset(path: Path) -> dict:
    registry = json.loads((path.parent / "assets.json").read_text(encoding="utf-8"))
    record = next((row for row in registry["files"] if row["path"] == path.name), None)
    if record is None or file_stat(path) != {key: record[key] for key in ("bytes", "mtime_ns")}:
        raise ValueError(f"Evaluation asset differs from its acquisition registration: {path}")
    return record


def generator_options(model_id: str, ema_sha256: str | None) -> dict:
    return {"ema_sha256": ema_sha256} if ema_sha256 and not model_id.startswith("MeanFlow-") else {}


def codec_provenance(checkpoint, codec):
    if codec is None:
        return None
    from .cache import codec_identity
    result = {"path": str(Path(codec).resolve()), **codec_identity(codec)}
    manifest = Path(checkpoint) / "manifest.json"
    if Path(checkpoint).is_dir() and manifest.is_file():
        expected = json.loads(manifest.read_text()).get("metadata", {}).get("identity", {}).get("codec", {}).get("sha256")
        if expected and expected != result["sha256"]:
            raise ValueError("Evaluation codec differs from the MeanFlow training cache codec")
        result["training_codec_identity_present"] = bool(expected)
    return result


def _save_contact_sheet(paths: list[Path], invalid: set[int], output: Path,
                        count: int = GRID_COUNT, columns: int = 16) -> None:
    if len(paths) < count:
        raise ValueError("Cannot make the contact sheet from incomplete sampling")
    canvas = Image.new("RGB", (columns * 256, math.ceil(count / columns) * 256), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths[:count]):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), ((index % columns) * 256, (index // columns) * 256))
        if index in invalid:
            x, y = (index % columns) * 256, (index // columns) * 256
            draw.rectangle((x, y, x + 255, y + 255), outline="red", width=5)
            draw.text((x + 8, y + 8), f"NONFINITE {index:04d}", fill="red")
    canvas.save(output)


def _extract_features(paths: list[Path], extractor, output: Path, device: str, batch_size: int):
    features = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(paths), 2048))
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            arrays = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    arrays.append(torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8, copy=True)).permute(2, 0, 1))
            batch = torch.stack(arrays).to(device)
            values = extractor(batch)[0].detach().cpu().float()
            if tuple(values.shape) != (len(arrays), 2048) or not torch.isfinite(values).all():
                raise FloatingPointError("Official Inception extractor returned invalid features")
            features[start:start + len(arrays)] = values.numpy()
    features.flush()
    return features


def distribution_metrics(generated: list[Path], real: list[Path], *, weights: Path,
                         output: Path, device: str, batch_size: int, seed: int) -> dict:
    if len(generated) != SAMPLE_COUNT or len(real) != SAMPLE_COUNT:
        raise ValueError("FID1024/KID requires exactly 1024 generated and real images")
    if not weights.is_file():
        raise FileNotFoundError(f"Official Inception weights must already exist: {weights}")
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], feature_extractor_weights_path=str(weights),
    ).to(device).eval()
    generated_features = _extract_features(generated, extractor, output / "generated-inception.npy", device, batch_size)
    real_features = _extract_features(real, extractor, output / "real-inception.npy", device, batch_size)
    del extractor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    generated_tensor = torch.from_numpy(generated_features)
    real_tensor = torch.from_numpy(real_features)
    generated_stats = fid_features_to_statistics(generated_tensor)
    real_stats = fid_features_to_statistics(real_tensor)
    np.savez(output / "generated-inception-stats.npz", **generated_stats)
    np.savez(output / "real-inception-stats.npz", **real_stats)
    fid = fid_statistics_to_metric(generated_stats, real_stats, verbose=False)
    kid = kid_features_to_metric(generated_tensor, real_tensor, kid_subsets=100, kid_subset_size=1000,
                                 rng_seed=seed, verbose=False)
    metrics = {**fid, **kid}
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite distribution metrics: {metrics}")
    return {"status": "complete", "num_generated": SAMPLE_COUNT, "num_real": SAMPLE_COUNT,
            "feature_extractor": "torch-fidelity inception-v3-compat", "feature_dimension": 2048,
            "torch_fidelity_version": _version("torch-fidelity"), "weights_sha256": evaluation_asset(weights)["sha256"],
            "kid_subsets": 100, "kid_subset_size": 1000, "rng_seed": seed, **metrics}


def face_metrics(paths: list[Path], invalid: set[int], *, detector_path: Path,
                 output: Path, detector_size: int = 640, detector_threshold: float = 0.5,
                 cpu_threads: int = 8) -> dict:
    if not detector_path.is_file():
        raise FileNotFoundError(f"Face detection ONNX must already exist: {detector_path}")
    import insightface
    import onnxruntime
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    # Direct ONNX path avoids FaceAnalysis model downloads and recognition-model loading.
    detector = insightface.model_zoo.get_model(str(detector_path), providers=["CPUExecutionProvider"], sess_options=options)
    if detector is None or not hasattr(detector, "detect"):
        raise TypeError("The supplied ONNX is not an InsightFace face detector")
    detector.prepare(ctx_id=-1, input_size=(detector_size, detector_size), det_thresh=detector_threshold)
    counts = {"zero": 0, "single": 0, "multiple": 0, "nonfinite": len(invalid)}
    with (output / "face-records.jsonl").open("x", encoding="utf-8") as handle:
        for index, path in enumerate(paths):
            if index in invalid:
                record = {"sample_index": index, "status": "nonfinite", "face_count": None}
            else:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                boxes, _ = detector.detect(np.ascontiguousarray(rgb[:, :, ::-1]), max_num=0)
                count = len(boxes)
                counts["zero" if count == 0 else "single" if count == 1 else "multiple"] += 1
                record = {"sample_index": index, "status": "complete", "face_count": count,
                          "scores": [float(value) for value in boxes[:, 4]]}
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    return {"status": "complete", "samples": len(paths), **counts,
            "single_face_rate": counts["single"] / SAMPLE_COUNT,
            "denominator_includes_blank_and_nonfinite": True,
            "detector": detector_path.name, "detector_sha256": evaluation_asset(detector_path)["sha256"],
            "detector_size": [detector_size, detector_size], "threshold": detector_threshold,
            "cpu_threads": cpu_threads,
            "providers": ["CPUExecutionProvider"], "insightface_version": _version("insightface")}


def evaluate(*, model_id: str, checkpoint: str | Path, dataset_manifest: str | Path,
             output: str | Path, inception_weights: str | Path, face_detector: str | Path,
             codec: str | None = None, image_root: str | Path | None = None,
             device: str = "cuda:0", batch_size: int = 16, seed: int = 42,
             detector_size: int = 640, detector_threshold: float = 0.5,
             cpu_threads: int = 8, ema_sha256: str | None = None,
             dataset_manifest_sha256: str | None = None) -> dict:
    if batch_size < 1 or cpu_threads < 1 or detector_size < 1 or not 0 < detector_threshold < 1:
        raise ValueError("Batch size, CPU threads and detector size must be positive; threshold must be in (0,1)")
    torch.set_num_threads(cpu_threads)
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "status": "running", "model_id": model_id,
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "samples_required": SAMPLE_COUNT, "preview_count": GRID_COUNT,
               "seed": seed, "batch_size": batch_size, "device": device,
               "checkpoint": str(checkpoint), "errors": []}
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    try:
        from .generator import load_generator
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for dependency in [Path(inception_weights), Path(face_detector)]:
            if not dependency.is_file():
                raise FileNotFoundError(f"Required evaluation weights missing: {dependency}")
        summary["checkpoint_stat"] = checkpoint_stat(checkpoint)
        summary["codec"] = codec_provenance(checkpoint, codec)
        summary["dataset_manifest_sha256"] = dataset_manifest_sha256 or sha256_file(dataset_manifest)
        summary["dataset_manifest_stat"] = file_stat(dataset_manifest)
        dataset = HQDataset(dataset_manifest, random_flip=False, image_root=image_root)
        if len(dataset) < SAMPLE_COUNT:
            raise ValueError("Real dataset has fewer than 1024 records")
        real_indices = np.random.RandomState(seed).choice(len(dataset), SAMPLE_COUNT, replace=False).tolist()
        real_records = [dataset.records[index] for index in real_indices]
        atomic_json(output / "real-records.json", {"dataset_manifest_sha256": summary["dataset_manifest_sha256"],
                    "seed": seed, "indices": real_indices, "records": real_records})
        real_paths = [dataset.image_path(index) for index in real_indices]
        for path, record in zip(real_paths, real_records):
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"Real evaluation image changed since audit: {record['id']}")
        generator = load_generator(model_id, str(checkpoint), device=device, codec=codec,
                                   **generator_options(model_id, ema_sha256))
        if getattr(generator, "model_id", model_id) != model_id:
            raise ValueError("Loaded generator identity differs from the requested model")
        if getattr(generator, "state_role", None) != "ema":
            raise ValueError("Generator must confirm state_role='ema'; raw checkpoints are not evaluated silently")
        ema_hash = getattr(generator, "ema_sha256", None)
        if not isinstance(ema_hash, str) or len(ema_hash) != 64 or any(c not in "0123456789abcdef" for c in ema_hash):
            raise ValueError("Generator must expose the selected EMA's sha256")
        summary["ema_sha256"] = ema_hash
        if ema_sha256 and ema_hash != ema_sha256:
            raise ValueError("Loaded EMA identity differs from the verified replica")
        summary["state_role"] = "ema"
        summary["noise_protocol"] = {"kind": "per_sample_cpu_float32_torch_randn", "seed_rule": "seed + sample_index",
            "noise_shape": list(generator.noise_shape), "step_noise_count": int(getattr(generator, "step_noise_count", 0)),
            "step_noise_shape": list(getattr(generator, "step_noise_shape", generator.noise_shape)),
            "initial_then_step_noises_in_same_rng_stream": True}
        atomic_json(summary_path, summary)
        images_dir = output / "images"
        images_dir.mkdir()
        paths, invalid, blank_count, out_of_range_count = [], set(), 0, 0
        with (output / "generation-records.jsonl").open("x", encoding="utf-8") as ledger:
            for start in range(0, SAMPLE_COUNT, batch_size):
                indices = list(range(start, min(start + batch_size, SAMPLE_COUNT)))
                noise, steps = make_fixed_noises(generator, indices, seed, device)
                with torch.inference_mode():
                    values = generator.sample(noise, step_noises=steps, grad_enabled=False)
                if not torch.is_tensor(values) or tuple(values.shape) != (len(indices), 3, 256, 256):
                    raise ValueError(f"Generator must return RGB [N,3,256,256], got {getattr(values, 'shape', type(values))}")
                values = values.detach().cpu().float()
                for offset, index in enumerate(indices):
                    value = values[offset]
                    finite = bool(torch.isfinite(value).all())
                    path = images_dir / f"sample-{index:06d}.png"
                    record = {"sample_index": index, "seed": seed + index, "finite": finite}
                    if not finite:
                        invalid.add(index)
                        np.save(output / f"nonfinite-{index:06d}.npy", value.numpy(), allow_pickle=False)
                        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
                        record["placeholder_png"] = True
                    else:
                        outside = bool((value < -1).any() or (value > 1).any())
                        out_of_range_count += int(outside)
                        pixels = value.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
                        spatial_std = float(pixels.astype(np.float32).std(axis=(0, 1)).max())
                        blank = spatial_std < 1.0
                        blank_count += int(blank)
                        record.update(minimum=float(value.min()), maximum=float(value.max()),
                                      out_of_range_before_png=outside, blank_spatial_std_lt1=blank, maximum_channel_spatial_std=spatial_std)
                    Image.fromarray(pixels).save(path)
                    ledger.write(json.dumps(record, allow_nan=False) + "\n")
                    paths.append(path)
                print(json.dumps({"generated": len(paths), "nonfinite": len(invalid), "required": SAMPLE_COUNT}), flush=True)
                del noise, steps, values
        del generator
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        summary["generation"] = {"status": "failed" if invalid else "complete", "samples": len(paths),
                                 "nonfinite": len(invalid), "blank_spatial_std_lt1": blank_count,
                                 "out_of_range_before_png": out_of_range_count, "filtered_samples": 0}
        _save_contact_sheet(paths, invalid, output / "first-0256-contact-sheet.png")
        try:
            summary["face_detection"] = face_metrics(paths, invalid, detector_path=Path(face_detector), output=output,
                                                     detector_size=detector_size, detector_threshold=detector_threshold,
                                                     cpu_threads=cpu_threads)
        except Exception as exc:
            summary["face_detection"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            summary["errors"].append(summary["face_detection"]["error"])
        if invalid:
            summary["distribution"] = {"status": "failed", "error": "Nonfinite outputs retained; placeholder pixels must not receive FID/KID"}
            summary["errors"].append(summary["distribution"]["error"])
        else:
            try:
                summary["distribution"] = distribution_metrics(paths, real_paths, weights=Path(inception_weights), output=output,
                                                                device=device, batch_size=batch_size, seed=seed)
            except Exception as exc:
                summary["distribution"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                summary["errors"].append(summary["distribution"]["error"])
        if checkpoint_stat(checkpoint) != summary["checkpoint_stat"] or file_stat(dataset_manifest) != summary["dataset_manifest_stat"]:
            summary["errors"].append("Checkpoint or dataset manifest changed during evaluation")
        if codec and codec_provenance(checkpoint, codec)["verified_stats"] != summary["codec"]["verified_stats"]:
            summary["errors"].append("Codec changed during evaluation")
        summary["status"] = "failed" if summary["errors"] else "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(summary_path, summary)
    if summary["status"] != "complete":
        raise RuntimeError(f"Evaluation failed; partial evidence retained at {summary_path}")
    return summary


def preview(*, model_id: str, checkpoint: str | Path, output: str | Path,
            codec: str | None = None, device: str = "cuda:0", batch_size: int = 8,
            seed: int = 42, cpu_threads: int = 8, ema_sha256: str | None = None) -> dict:
    """Load the real EMA and generate exactly 64 unfiltered images; no FID claim."""
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch_size and cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "mode": "preview64", "status": "running",
               "model_id": model_id, "seed": seed, "samples_required": 64,
               "batch_size": batch_size, "device": device, "filtered_samples": 0,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": "not_requested"}
    atomic_json(output / "summary.json", summary)
    generator = None
    try:
        from .generator import load_generator
        summary["checkpoint_stat"] = checkpoint_stat(checkpoint)
        summary["codec"] = codec_provenance(checkpoint, codec)
        generator = load_generator(model_id, str(checkpoint), device=device, codec=codec,
                                   **generator_options(model_id, ema_sha256))
        if getattr(generator, "model_id", model_id) != model_id:
            raise ValueError("Loaded generator identity differs from the requested model")
        ema_hash = getattr(generator, "ema_sha256", None)
        if getattr(generator, "state_role", None) != "ema" or not isinstance(ema_hash, str) or len(ema_hash) != 64 or any(c not in "0123456789abcdef" for c in ema_hash):
            raise ValueError("Preview requires a confirmed EMA with its SHA256")
        if ema_sha256 and ema_hash != ema_sha256:
            raise ValueError("Loaded EMA identity differs from the verified replica")
        summary.update(state_role="ema", ema_sha256=ema_hash,
                       noise_protocol={"seed_rule": "seed + sample_index", "noise_shape": list(generator.noise_shape),
                                       "step_noise_count": int(getattr(generator, "step_noise_count", 0)),
                                       "step_noise_shape": list(getattr(generator, "step_noise_shape", generator.noise_shape))})
        images = output / "images"; images.mkdir()
        paths, invalid = [], set()
        with (output / "generation-records.jsonl").open("x", encoding="utf-8") as ledger:
            for start in range(0, 64, batch_size):
                indices = list(range(start, min(start + batch_size, 64)))
                noise, steps = make_fixed_noises(generator, indices, seed, device)
                with torch.inference_mode():
                    values = generator.sample(noise, step_noises=steps, grad_enabled=False)
                if not torch.is_tensor(values) or tuple(values.shape) != (len(indices), 3, 256, 256):
                    raise ValueError("Generator preview must return RGB [N,3,256,256]")
                for index, value in zip(indices, values.detach().cpu().float()):
                    finite = bool(torch.isfinite(value).all())
                    if not finite:
                        invalid.add(index)
                        np.save(output / f"nonfinite-{index:06d}.npy", value.numpy(), allow_pickle=False)
                        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
                    else:
                        pixels = value.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
                    path = images / f"sample-{index:06d}.png"
                    Image.fromarray(pixels).save(path); paths.append(path)
                    ledger.write(json.dumps({"sample_index": index, "seed": seed + index,
                                 "finite": finite, "placeholder_png": not finite}) + "\n")
                del noise, steps, values
        _save_contact_sheet(paths, invalid, output / "first-0064-contact-sheet.png", count=64, columns=8)
        summary.update(samples=len(paths), nonfinite=len(invalid), status="complete" if not invalid else "failed")
        if checkpoint_stat(checkpoint) != summary["checkpoint_stat"]:
            raise ValueError("EMA checkpoint changed during preview")
        if codec and codec_provenance(checkpoint, codec)["verified_stats"] != summary["codec"]["verified_stats"]:
            raise ValueError("Codec changed during preview")
        if invalid:
            raise FloatingPointError("Nonfinite preview samples retained in place")
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        del generator
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Generate 64 unfiltered images without distribution metrics")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema-sha256", help="EMA identity already verified by checkpoint transfer")
    parser.add_argument("--dataset-manifest-sha256", help="Dataset identity already verified by data preparation")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inception-weights")
    parser.add_argument("--face-detector")
    parser.add_argument("--codec")
    parser.add_argument("--image-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    kwargs = vars(args)
    is_preview = kwargs.pop("preview")
    if is_preview:
        for key in ("dataset_manifest", "dataset_manifest_sha256", "inception_weights", "face_detector", "image_root", "detector_size", "detector_threshold"):
            kwargs.pop(key)
        result = preview(**kwargs)
    else:
        if not all(kwargs[key] for key in ("dataset_manifest", "inception_weights", "face_detector")):
            parser.error("Full evaluation requires --dataset-manifest, --inception-weights and --face-detector")
        result = evaluate(**kwargs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
