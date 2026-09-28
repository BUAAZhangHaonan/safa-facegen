"""Fixed 1024-sample EMA evaluation without image rejection or metric fallbacks."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import HQDataset, atomic_json
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


def generator_identity(generator, checkpoint, expected_hash=None):
    if getattr(generator, "state_role", None) != "ema":
        raise ValueError("Evaluation requires a confirmed EMA")
    digest = getattr(generator, "ema_sha256", None)
    identity = getattr(generator, "ema_identity", None)
    if getattr(generator, "integrity_mode", None) == "metadata":
        if not isinstance(identity, dict) or identity.get("state_role") != "ema" or not identity.get("bytes"):
            raise ValueError("Generator must expose metadata for its selected EMA")
        if identity.get("model_id") != generator.model_id:
            raise ValueError("Generator metadata model mismatch")
    elif not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Generator requires registered metadata or a historical EMA identity")
    if expected_hash and digest != expected_hash:
        raise ValueError("Loaded EMA identity differs from the replica registration")
    return {"integrity_mode": getattr(generator, "integrity_mode", "sha256"),
            "ema_sha256": digest, "ema_identity": identity}


def generator_protocol(generator):
    """Record the loaded sampler and runtime precision, without generating images."""
    network = getattr(generator, "network", None)
    if network is None:
        network = getattr(generator, "net", None)
    if network is None:
        raise ValueError("Generator has no inspectable sampling network")
    parameters = list(network.parameters())
    if not parameters:
        raise ValueError("Sampling network has no parameters")
    device_type = parameters[0].device.type
    try:
        ambient_autocast = torch.is_autocast_enabled(device_type)
    except TypeError:
        ambient_autocast = (torch.is_autocast_enabled() if device_type == "cuda"
                            else torch.is_autocast_cpu_enabled())
    meanflow = generator.model_id.startswith("MeanFlow-")
    if meanflow:
        sampling = {"method": "meanflow_one_step", "steps": 1, "guidance": False,
                    "null_label": 1000, "latent_scale": float(generator.latent_scale),
                    "time": 1.0, "interval": 1.0}
        network_autocast = ambient_autocast
    else:
        sampling = dict(generator.sampling_config)
        sampling["guidance"] = False
        kind = generator.kind
        if kind == "diffusion":
            from .torch_models.models import make_ddim_timesteps
            sampling.update(method="ddim", timesteps=make_ddim_timesteps(
                "uniform", generator.steps, 1000, verbose=False)[::-1].tolist())
        elif kind == "latent_consistency":
            grid = (np.arange(1, 51) * 20 - 1)[::-1]
            indices = np.floor(np.linspace(0, len(grid), num=generator.steps, endpoint=False)).astype(np.int64)
            sampling.update(method="latent_consistency", timesteps=grid[indices].tolist(), teacher_grid_points=50)
        elif kind == "rectified_flow":
            sampling.update(method="rk45", atol=generator.ode_tol, rtol=generator.ode_tol,
                            time_label="fp32(t)*999")
        else:
            raise ValueError("Unknown generator sampling family")
        network_autocast = bool(generator.bf16 and device_type == "cuda")
    codec = getattr(generator, "codec", None)
    def dtypes(module):
        return sorted({str(p.dtype).removeprefix("torch.") for p in module.parameters()}) if module is not None else []
    autocast_dtype = None
    if network_autocast:
        if not meanflow:
            autocast_dtype = "bfloat16"
        else:
            try:
                autocast_dtype = str(torch.get_autocast_dtype(device_type)).removeprefix("torch.")
            except AttributeError:
                autocast_dtype = str(torch.get_autocast_gpu_dtype() if device_type == "cuda"
                                     else torch.get_autocast_cpu_dtype()).removeprefix("torch.")
    precision = {"network_parameter_dtypes": dtypes(network), "codec_parameter_dtypes": dtypes(codec),
                 "network_autocast_enabled": bool(network_autocast), "network_autocast_dtype": autocast_dtype,
                 "ambient_autocast_enabled": bool(ambient_autocast), "device_type": device_type,
                 "noise_dtype": "float32", "ode_state_dtype": "float64" if sampling["method"] == "rk45" else None,
                 "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                 "torch_float32_matmul_precision": torch.get_float32_matmul_precision()}
    return {"sampling": sampling, "generation_precision": precision}


@contextmanager
def fp32_tf32_disabled(device: str):
    """Scope the new Diffusion protocol without changing other queued models."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def quality_v1_plan(generator, settings, *, seed, batch_size, real_indices, real_records, weights, codec_info):
    """Require an explicit protocol only for the new metadata Min-SNR objective."""
    identity = getattr(generator, "ema_identity", None) or {}
    if (getattr(generator, "integrity_mode", None) != "metadata"
            or identity.get("objective_id") != "min_snr_epsilon_v1"):
        return None
    if generator.model_id != "Diffusion-LDM-UNet":
        raise ValueError("Min-SNR evaluation requires the Diffusion model identity")
    if not isinstance(settings, dict):
        raise ValueError("min_snr_epsilon_v1 requires explicit quality_v1_evaluation settings")
    required = {"objective_id": "min_snr_epsilon_v1", "seed": 42,
                "generation_batch_size": 8, "feature_batch_size": 32, "precision": "fp32"}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("quality_v1_evaluation differs from the registered Diffusion protocol")
    if (settings.get("cuda_matmul_allow_tf32") is not False
            or settings.get("cudnn_allow_tf32") is not False
            or not isinstance(settings.get("protocol_id"), str) or not settings["protocol_id"].strip()):
        raise ValueError("Explicit FP32/TF32-disabled protocol identity is required")
    if seed != settings["seed"] or batch_size != settings["generation_batch_size"]:
        raise ValueError("Evaluation seed or generation batch differs from the registered protocol")
    if generator.steps != 200 or generator.eta != 1.0 or generator.bf16:
        raise ValueError("Min-SNR comparison requires DDIM200 eta1 with network BF16 disabled")
    if not settings.get("reference_review") or not settings.get("baseline_checkpoint_id"):
        raise ValueError("Explicit baseline checkpoint and reference review are required")
    review = Path(settings["reference_review"]).resolve(strict=True)
    source_paths = [review / name for name in ("summary.json", "real-records.json", "real-inception.npy")]
    before = {str(path): file_stat(path) for path in source_paths}
    baseline = json.loads(source_paths[0].read_text(encoding="utf-8"))
    reference = json.loads(source_paths[1].read_text(encoding="utf-8"))
    if baseline.get("status") != "complete" or baseline.get("checkpoint_id") != settings["baseline_checkpoint_id"]:
        raise ValueError("Configured Diffusion baseline identity is not a complete review")
    baseline_precision = baseline.get("protocol", {}).get("generation_precision", {})
    if baseline_precision != {"dtype": "fp32", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}:
        raise ValueError("Baseline does not register the expected TF32-disabled generation protocol")
    sampling = baseline.get("sampling", {})
    if (baseline.get("model_id") != generator.model_id or baseline.get("seed") != seed
            or sampling.get("steps") != 200 or sampling.get("eta") != 1.0 or sampling.get("bf16") is not False):
        raise ValueError("Baseline sampling differs from the new Diffusion comparison")
    if not codec_info or not codec_info.get("sha256") or baseline.get("codec", {}).get("sha256") != codec_info["sha256"]:
        raise ValueError("Diffusion baseline and candidate require the same registered codec")
    if (len(real_indices) != SAMPLE_COUNT or len(real_records) != SAMPLE_COUNT
            or reference.get("seed") != seed or reference.get("indices") != real_indices
            or reference.get("records") != real_records):
        raise ValueError("Fixed real reference identities or order differ from the baseline")
    expected_stat = settings.get("reference_feature_stat")
    if not isinstance(expected_stat, dict) or before[str(source_paths[2])] != expected_stat:
        raise ValueError("Fixed real feature metadata differs from explicit registration")
    distribution = baseline.get("distribution", {})
    if (distribution.get("weights_sha256") != evaluation_asset(Path(weights))["sha256"]
            or distribution.get("torch_fidelity_version") != _version("torch-fidelity")
            or distribution.get("num_real") != SAMPLE_COUNT or distribution.get("num_generated") != SAMPLE_COUNT
            or distribution.get("kid_subsets") != 100 or distribution.get("kid_subset_size") != 1000
            or distribution.get("rng_seed") != seed):
        raise ValueError("Baseline Inception/KID registration differs from the active evaluator")
    features = np.load(source_paths[2], mmap_mode="r", allow_pickle=False)
    if features.shape != (SAMPLE_COUNT, 2048) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Fixed real reference features must be finite FP32 [1024,2048]")
    return {"features": features, "source_stats": before, "feature_batch_size": settings["feature_batch_size"],
            "record": {"protocol_id": settings["protocol_id"], "objective_id": identity["objective_id"],
                "baseline_checkpoint_id": settings["baseline_checkpoint_id"], "reference_review": str(review),
                "reference_features": str(source_paths[2]), "reference_records": str(source_paths[1]),
                "reference_feature_stat": expected_stat, "reference_count": SAMPLE_COUNT,
                "reference_records_and_order_verified": True, "reference_features_recomputed": False,
                "reference_runtime_precision": "historical runtime flags unrecorded; identical fixed matrix reused",
                "reference_origin": baseline.get("provenance", {}).get("reference_source"),
                "generation_batch_size": batch_size, "generated_feature_batch_size": settings["feature_batch_size"],
                "seed": seed, "precision": "fp32", "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False, "integrity_mode": "metadata"}}


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
                         output: Path, device: str, batch_size: int, seed: int,
                         fixed_reference: dict | None = None) -> dict:
    if len(generated) != SAMPLE_COUNT or len(real) != SAMPLE_COUNT:
        raise ValueError("FID1024/KID requires exactly 1024 generated and real images")
    if fixed_reference is not None and (batch_size != fixed_reference["feature_batch_size"]
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.is_autocast_enabled(torch.device(device).type)):
        raise ValueError("Fixed-reference Diffusion features require their explicit FP32/TF32-disabled scope")
    if not weights.is_file():
        raise FileNotFoundError(f"Official Inception weights must already exist: {weights}")
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048"], feature_extractor_weights_path=str(weights),
    ).to(device).eval()
    if fixed_reference is not None:
        extractor = extractor.float()
    feature_dtype = str(next(extractor.parameters()).dtype).removeprefix("torch.")
    generated_features = _extract_features(generated, extractor, output / "generated-inception.npy", device, batch_size)
    if fixed_reference is None:
        real_features = _extract_features(real, extractor, output / "real-inception.npy", device, batch_size)
    else:
        real_features = fixed_reference["features"]
        np.save(output / "real-inception.npy", real_features, allow_pickle=False)
    del extractor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    generated_tensor = torch.from_numpy(generated_features)
    real_tensor = torch.from_numpy(np.array(real_features, copy=True) if fixed_reference is not None else real_features)
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
    result = {"status": "complete", "num_generated": SAMPLE_COUNT, "num_real": SAMPLE_COUNT,
            "feature_extractor": "torch-fidelity inception-v3-compat", "feature_dimension": 2048,
            "torch_fidelity_version": _version("torch-fidelity"), "weights_sha256": evaluation_asset(weights)["sha256"],
            "kid_subsets": 100, "kid_subset_size": 1000, "rng_seed": seed, **metrics}
    if fixed_reference is not None:
        result.update(fixed_reference=fixed_reference["record"], generated_feature_precision={
            "parameter_dtype": feature_dtype, "autocast_enabled": torch.is_autocast_enabled(torch.device(device).type),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(), "batch_size": batch_size})
    return result


def face_metrics(paths: list[Path], invalid: set[int], *, detector_path: Path,
                 output: Path, detector_size: int = 256, detector_threshold: float = 0.5,
                 cpu_threads: int = 8) -> dict:
    if not detector_path.is_file():
        raise FileNotFoundError(f"Face detection ONNX must already exist: {detector_path}")
    import insightface
    import onnxruntime
    from insightface.model_zoo.retinaface import RetinaFace
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    # Direct ONNX path avoids FaceAnalysis model downloads and recognition-model loading.
    # InsightFace's model router does not forward sess_options. Bind the session
    # explicitly so the recorded CPU thread limits apply to the real detector.
    session = onnxruntime.InferenceSession(str(detector_path), sess_options=options,
                                          providers=["CPUExecutionProvider"])
    detector = RetinaFace(model_file=str(detector_path), session=session)
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
            "cpu_threads": session.get_session_options().intra_op_num_threads,
            "cpu_inter_threads": session.get_session_options().inter_op_num_threads,
            "providers": session.get_providers(), "insightface_version": _version("insightface")}


def evaluate(*, model_id: str, checkpoint: str | Path, dataset_manifest: str | Path,
             output: str | Path, inception_weights: str | Path, face_detector: str | Path,
             codec: str | None = None, image_root: str | Path | None = None,
             device: str = "cuda:0", batch_size: int = 16, seed: int = 42,
             detector_size: int = 256, detector_threshold: float = 0.5,
             cpu_threads: int = 8, ema_sha256: str | None = None,
             dataset_manifest_sha256: str | None = None,
             quality_v1_evaluation: dict | None = None) -> dict:
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
    precision_scope = ExitStack()
    quality_plan = None
    try:
        from .generator import load_generator
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for dependency in [Path(inception_weights), Path(face_detector)]:
            if not dependency.is_file():
                raise FileNotFoundError(f"Required evaluation weights missing: {dependency}")
        summary["checkpoint_stat"] = checkpoint_stat(checkpoint)
        summary["codec"] = codec_provenance(checkpoint, codec)
        summary["dataset_manifest_sha256"] = dataset_manifest_sha256
        summary["dataset_integrity_mode"] = "metadata"
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
        summary.update(generator_identity(generator, checkpoint, ema_sha256))
        quality_plan = quality_v1_plan(generator, quality_v1_evaluation, seed=seed, batch_size=batch_size,
                                      real_indices=real_indices, real_records=real_records, weights=inception_weights,
                                      codec_info=summary["codec"])
        if quality_plan is not None:
            precision_scope.enter_context(fp32_tf32_disabled(device))
            summary["quality_v1_evaluation"] = quality_plan["record"]
        summary.update(generator_protocol(generator))
        if quality_plan is not None:
            precision = summary["generation_precision"]
            if (precision["network_parameter_dtypes"] != ["float32"]
                    or precision["codec_parameter_dtypes"] != ["float32"]
                    or precision["network_autocast_enabled"] or precision["ambient_autocast_enabled"]):
                raise ValueError("The new Diffusion generator did not load in the required FP32 protocol")
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
                    device=device, batch_size=quality_plan["feature_batch_size"] if quality_plan else batch_size,
                    seed=seed, fixed_reference=quality_plan)
            except Exception as exc:
                summary["distribution"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                summary["errors"].append(summary["distribution"]["error"])
        if checkpoint_stat(checkpoint) != summary["checkpoint_stat"] or file_stat(dataset_manifest) != summary["dataset_manifest_stat"]:
            summary["errors"].append("Checkpoint or dataset manifest changed during evaluation")
        if codec and codec_provenance(checkpoint, codec)["verified_stats"] != summary["codec"]["verified_stats"]:
            summary["errors"].append("Codec changed during evaluation")
        if quality_plan is not None and any(file_stat(path) != info for path, info in quality_plan["source_stats"].items()):
            summary["errors"].append("Fixed reference metadata changed during evaluation")
        summary["status"] = "failed" if summary["errors"] else "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        precision_scope.close()
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
        summary.update(generator_identity(generator, checkpoint, ema_sha256))
        summary.update(generator_protocol(generator))
        summary.update(state_role="ema",
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
    parser.add_argument("--detector-size", type=int, default=256)
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
