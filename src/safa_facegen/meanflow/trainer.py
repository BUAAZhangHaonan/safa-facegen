"""Single-process/four-device original MeanFlow trainer.

Launch only after a strict legacy conversion. No torch, VAE or Inception GPU
context is created here. Completed EMA exports queue K100 preview/review work.
"""
from ..contracts import MEANFLOW_ORIGINAL, IMF_BOUNDARY, MEANFLOW_CACHE_SCHEMA, MEANFLOW_CHECKPOINT_SCHEMA, MEANFLOW_EXPORT_FORMAT, QUALITY_STAGE_SCHEMA
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from .io import append_event, atomic_json
from .spec import FIXED_RECIPE, LATENT_SCALE, NULL_LABEL, UPSTREAM_COMMIT, get_spec, validate_recipe

EXIT_RECOVERY_EXHAUSTED = 78


class RecoveryExhausted(RuntimeError):
    """Controller must stop this model instead of starting another recovery cycle."""


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("model_id", "learning_rate", "microbatch", "paths", "seed",
                "save_interval_seconds", "preview_interval_seconds", "review_interval_seconds", "limits")
    for key in required:
        if key not in config:
            raise ValueError(f"Missing configuration key {key}")
    get_spec(config["model_id"])
    validate_recipe(config)
    for key in ("dataset_manifest", "latent_cache", "initial_checkpoint", "codec", "output"):
        if key not in config["paths"]:
            raise ValueError(f"Missing paths.{key}")
    if config["learning_rate"] <= 0 or config["microbatch"] < 1:
        raise ValueError("learning_rate and microbatch must be positive")
    return config


class LatentCache:
    def __init__(self, manifest_path, dataset_path):
        manifest_path, dataset_path = Path(manifest_path), Path(dataset_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"schema_version": MEANFLOW_CACHE_SCHEMA, "layout": "N,F,C,H,W", "representation": "mean_std",
                    "dtype": "float32", "status": "complete"}
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f"Latent cache {key} must be {value!r}")
        dataset_stat = dataset_path.stat()
        self.dataset_stat = {"bytes": dataset_stat.st_size, "mtime_ns": dataset_stat.st_mtime_ns}
        if self.manifest.get("dataset_manifest_stat") != self.dataset_stat:
            raise ValueError("Latent cache dataset registration differs from the current manifest")
        self.dataset_sha256 = self.manifest["dataset_manifest_sha256"]
        manifest_stat = manifest_path.stat()
        self.manifest_stat = {"bytes": manifest_stat.st_size, "mtime_ns": manifest_stat.st_mtime_ns}
        self.codec = self.manifest["codec"]
        array_path = (manifest_path.parent / self.manifest["array"]).resolve()
        # Cache construction and transfer own the one-time large-array checksum.
        # Training checks metadata and each accessed batch without rereading 6 GiB.
        self.data = np.load(array_path, mmap_mode="r", allow_pickle=False)
        # The completed builder records N after reading the dataset once. Its
        # unchanged registration above avoids rereading the 100k-record JSON.
        count = int(self.manifest["shape"][0])
        if count <= 0:
            raise ValueError("Latent cache must contain at least one record")
        if tuple(self.data.shape) != (count, 2, 8, 32, 32):
            raise ValueError(f"Unexpected cache shape {self.data.shape}, expected {(count,2,8,32,32)}")
        if list(self.data.shape) != self.manifest["shape"] or self.data.dtype != np.float32:
            raise ValueError("Cache shape/dtype differs from its manifest")
        self.count = count

    def batch(self, indices, flips):
        value = np.asarray(self.data[indices, flips], dtype=np.float32)
        if not np.isfinite(value).all() or np.any(value[:, 4:] < 0):
            raise ValueError("Latent cache contains nonfinite values or negative posterior std")
        return value.transpose(0, 2, 3, 1).copy()


def memory_status(limits):
    used = None
    for filename in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            used = int(Path(filename).read_text())
            break
        except (OSError, ValueError):
            pass
    if used is None:
        import psutil
        used = psutil.Process().memory_info().rss
    gib = used / 2**30
    level = "hard" if gib >= limits["memory_hard_gib"] else "soft" if gib >= limits["memory_soft_gib"] else "ok"
    return level, gib


def gpu_memory_gib():
    """Allocator peak per JAX device; CPU devices return no allocator stats."""
    import jax
    result = {}
    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats:
            result[str(device)] = float(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))) / 2**30
    return result


def check_runtime():
    """Fail before loading large weights if a backend silently miscomputes GEMM."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(391)
    a = rng.normal(size=(2,256)).astype(np.float32)
    b = rng.normal(size=(256,32)).astype(np.float32)
    reference = a @ b
    for device in jax.local_devices():
        multiply = jax.jit(lambda x,y: jnp.matmul(x,y,precision=jax.lax.Precision.HIGHEST))
        actual = np.asarray(multiply(jax.device_put(a,device),jax.device_put(b,device)))
        np.testing.assert_allclose(actual,reference,rtol=2e-4,atol=1e-4,
                                   err_msg=f"JAX backend matmul failed on {device}")
        x = rng.normal(size=(2,32,32,4)).astype(np.float32)
        kernel = rng.normal(size=(4,4,4,32)).astype(np.float32)
        expected = np.einsum("nhpwqc,pqco->nhwo",x.reshape(2,8,4,8,4,4),kernel)
        conv = jax.jit(lambda xx,kk: jax.lax.conv_general_dilated(
            xx,kk,(4,4),"VALID",dimension_numbers=("NHWC","HWIO","NHWC"),
            precision=jax.lax.Precision.HIGHEST))
        actual = np.asarray(conv(jax.device_put(x,device),jax.device_put(kernel,device)))
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-4,
                                   err_msg=f"JAX backend convolution failed on {device}")


def create_state(model, spec, initial, learning_rate, seed, initialization_mode=None):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from safetensors.numpy import load_file
    from .parameters import canonical_to_flax

    class State(train_state.TrainState):
        ema_params: object
        rng: object
        learning_rate: object

    root = Path(initial)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != MEANFLOW_EXPORT_FORMAT or manifest.get("architecture") != spec.to_dict():
        raise ValueError("Initial checkpoint architecture/format mismatch")
    if manifest.get('integrity_mode') == 'metadata':
        from ..integrity import validate_files
        validate_files(root, [row for row in manifest['files'] if row['path'] == 'ema.safetensors'])
    # Migration/transfer validates these files once. Safetensors and strict tensor
    # shape/key checks below protect loading without repeating multi-GiB hashes.
    key = jax.random.PRNGKey(seed)
    template = jax.eval_shape(model.init, {"params": key},
                              jax.ShapeDtypeStruct((1, spec.input_size, spec.input_size, spec.in_channels), jnp.float32),
                              jax.ShapeDtypeStruct((1,), jnp.float32),
                              jax.ShapeDtypeStruct((1,), jnp.int32))["params"]
    ema = canonical_to_flax(load_file(str(root/"ema.safetensors")), spec, template)
    raw = (jax.tree_util.tree_map(lambda x: jnp.array(x), ema)
           if initialization_mode == 'ema_warm_start_new_optimizer' else
           canonical_to_flax(load_file(str(root/"raw.safetensors")), spec, template))
    # Unit Adam normalization; scalar LR is a saved leaf so bounded recovery can
    # lower it without resetting moments or silently changing the optimizer.
    tx = optax.scale_by_adam(b1=FIXED_RECIPE["adam_betas"][0],
                             b2=FIXED_RECIPE["adam_betas"][1], eps=FIXED_RECIPE["adam_eps"])
    state = State.create(apply_fn=model.apply, params=raw, tx=tx, ema_params=ema,
                         rng=key, learning_rate=jnp.asarray(learning_rate, jnp.float32))
    return state, manifest


def make_train_step(model, ema_decay=FIXED_RECIPE["ema_decay"], objective_id=MEANFLOW_ORIGINAL):
    import jax
    import jax.numpy as jnp
    import optax

    def train_step(state, cached):
        next_rng, step_rng = jax.random.split(state.rng)
        local_rng = jax.random.fold_in(step_rng, jax.lax.axis_index("devices"))
        posterior_rng, objective_rng = jax.random.split(local_rng)
        mean, std = jnp.split(cached, 2, axis=-1)
        images = (mean + std * jax.random.normal(posterior_rng, mean.shape)) * LATENT_SCALE
        labels = jnp.full((images.shape[0],), NULL_LABEL, dtype=jnp.int32)

        def objective(params):
            if objective_id == IMF_BOUNDARY:
                from .boundary import boundary_loss
                def apply_u(p, z, t, r):
                    return model.apply({'params': p}, z, t, t-r, labels,
                                       method=model.u_fn, rngs={'gen': objective_rng})
                return boundary_loss(params, apply_u, images, objective_rng)
            if objective_id != MEANFLOW_ORIGINAL:
                raise ValueError('Unknown MeanFlow objective')
            return model.apply({"params": params}, imgs=images, labels=labels,
                               method=model.forward, rngs={"gen": objective_rng})

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "devices")
        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        finite = jax.lax.pmin(finite.astype(jnp.int32), "devices").astype(bool)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        updates = jax.tree_util.tree_map(lambda x: -state.learning_rate * x, updates)
        params = optax.apply_updates(state.params, updates)
        finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(params)]))
        ema = jax.tree_util.tree_map(lambda e, p: ema_decay*e + (1-ema_decay)*p, state.ema_params, params)
        proposed = state.replace(step=state.step+1, params=params, opt_state=opt_state,
                                 ema_params=ema, rng=next_rng)
        # No donation: a failed/OOM step must leave the last valid state usable.
        result = jax.lax.cond(finite, lambda _: proposed, lambda _: state, operand=None)
        extra = {}
        if objective_id == IMF_BOUNDARY:
            extra = dict(metrics)
            for part in ('boundary', 'interval'):
                count = jax.lax.psum(metrics[part+'_count'], 'devices')
                numerator = jax.lax.psum(metrics[part+'_residual_mse']*metrics[part+'_count'], 'devices')
                extra[part+'_residual_mse'] = numerator/jnp.maximum(count, 1)
                extra[part+'_count'] = count
        else:
            extra['v_loss'] = jnp.mean(metrics['v_loss'])
        metrics = {**extra, "loss": jnp.mean(loss),
                   "grad_norm": grad_norm, "finite": finite.astype(jnp.float32),
                   "learning_rate": state.learning_rate}
        return result, jax.lax.pmean(metrics, "devices")

    return jax.pmap(train_step, axis_name="devices")


def save_checkpoint(state, progress, config, identity, *, export=False):
    level, memory = memory_status(config["limits"])
    if level != "ok":
        # Do not allocate a second host state after the shared RAM threshold.
        # This also guards final, signal-driven, calibration and recovery saves.
        runtime = sys.modules.get("jax")
        if runtime is not None:
            runtime.clear_caches()
        gc.collect()
        append_event(Path(config["paths"]["output"])/"events.jsonl", "checkpoint_save_skipped",
                     reason=f"RAM {level} limit", memory_gib=memory, latest_complete_preserved=True)
        raise MemoryError(f"RAM {level} limit prevents checkpoint save ({memory:.2f} GiB); "
                          "preserving the latest complete checkpoint")
    import jax
    from flax import jax_utils, serialization
    import orbax.checkpoint as ocp
    from .parameters import flax_to_canonical, write_export

    root = Path(config["paths"]["output"])
    temporary_calibration = config.get("calibration",False) and not config.get("resume_validation",False)
    epoch = progress["valid_samples"] // identity["dataset_count"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{config['model_id']}-{epoch:04d}ep-{timestamp}"
    final = root / "checkpoints" / name
    staging = final.with_name("." + name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    # Unreplicate on device before copying: never transfer four full states.
    host_state = jax.device_get(jax_utils.unreplicate(state))
    host_progress = copy.deepcopy(progress)
    stage = config.get('bounded_stage', {})
    metadata = {"schema_version": MEANFLOW_CHECKPOINT_SCHEMA, "model_id": config["model_id"], "phase": "HQ",
                "integrity_mode": "metadata", "checkpoint_id": name, "state_role": "training_state",
                "stage_id": stage.get('stage_id'), "objective_id": config.get('objective_id', MEANFLOW_ORIGINAL),
                "parent_checkpoint_id": stage.get('parent_checkpoint_id'), "codec_registration": identity['codec'],
                "temporary_calibration": temporary_calibration,
                "upstream_commit": UPSTREAM_COMMIT, "identity": identity,
                "progress": host_progress, "config": config,
                "step": int(host_state.step), "learning_rate": float(host_state.learning_rate)}
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        checkpointer.save(str((staging / "state").resolve()), serialization.to_state_dict(host_state))
    finally:
        checkpointer.close()
    metadata["state_files"] = []
    for path in sorted((staging/"state").rglob("*")):
        if path.is_file():
            entry = {"path": path.relative_to(staging/"state").as_posix(), "size": path.stat().st_size}
            metadata["state_files"].append(entry)
    if not metadata["state_files"]:
        raise RuntimeError("Orbax checkpoint did not produce any state files")
    export_path = None
    if export:
        write_export(staging / "export", config["model_id"],
                     flax_to_canonical(host_state.ema_params, get_spec(config["model_id"])),
                     register_identity=not temporary_calibration,
                     metadata={"checkpoint": name, "hq_epoch": epoch, "hq_step": int(host_state.step),
                               "valid_samples": progress["valid_samples"], "identity": identity,
                               "stage_id": stage.get('stage_id'), "objective_id": config.get('objective_id'),
                               "parent_checkpoint_id": stage.get('parent_checkpoint_id'),
                               "codec_registration": identity['codec']})
        export_path = str(final / "export")
    # COMPLETE only becomes visible after all state chunks and metadata exist.
    atomic_json(staging / "manifest.json", metadata)
    from ..integrity import completion_record, validate_files
    validate_files(staging/'state', metadata['state_files'], exact=True)
    atomic_json(staging / "COMPLETE.json", completion_record(staging/'manifest.json', name))
    os.replace(staging, final)
    pointer = {"checkpoint": str(final), "step": int(host_state.step), "checkpoint_id": name,
               "name": name, "export": export_path, "complete": True, "model_id": config['model_id'],
               "stage_id": stage.get('stage_id'), "objective_id": config.get('objective_id'),
               "valid_samples": progress['valid_samples'], "integrity_mode": "metadata"}
    atomic_json(root / "latest.json", pointer)
    if stage.get('schema_version') == QUALITY_STAGE_SCHEMA:
        atomic_json(root / ('latest-' + stage['stage_id'] + '.json'), pointer)
    append_event(root/"events.jsonl", "checkpoint_complete", checkpoint=str(final),
                 step=int(host_state.step), valid_samples=progress["valid_samples"], export=export_path)
    if not config.get("validation_scope",False) and not config.get("calibration",False):
        append_event(root/"requests.jsonl", "save", model_id=config["model_id"],
                     checkpoint=str(final.resolve()), export=export_path,
                     step=int(host_state.step), hq_epoch=epoch,
                     valid_samples=progress["valid_samples"], integrity_mode='metadata',
                     checkpoint_id=name, stage_id=stage.get('stage_id'))
    return final, export_path


def verify_checkpoint(path):
    from ..integrity import verify_meanflow_checkpoint
    return verify_meanflow_checkpoint(path)


def restore_checkpoint(path, template, identity):
    from flax import serialization
    import orbax.checkpoint as ocp
    path = Path(path)
    metadata = verify_checkpoint(path)
    if metadata["identity"] != identity:
        raise ValueError("Resume model/data/cache/initial-checkpoint identity differs")
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        data = checkpointer.restore(str((path/"state").resolve()))
    finally:
        checkpointer.close()
    return serialization.from_state_dict(template, data), metadata["progress"]


def apply_recovery_overrides(state, progress, config, *, checkpoint=None):
    """Apply an explicit controller recovery after restoring the saved state."""
    requested = config.get("recovery_overrides") or {}
    if not requested:
        return state, progress
    unsupported = set(requested)-{"microbatch", "learning_rate", "precision"}
    if unsupported:
        raise ValueError(f"Unsupported MeanFlow recovery overrides: {sorted(unsupported)}; "
                         "this trainer reads memmap batches directly and has no DataLoader workers or prefetch queue")
    before = {"microbatch":int(progress["microbatch"]),
              "learning_rate":float(np.asarray(state.learning_rate)), "precision":"fp32"}
    after = dict(before)
    if "microbatch" in requested:
        value = requested["microbatch"]
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError("Recovery microbatch must be a positive integer")
        after["microbatch"] = value
    if "learning_rate" in requested:
        value = float(requested["learning_rate"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Recovery learning_rate must be finite and positive")
        after["learning_rate"] = float(np.asarray(value,dtype=np.float32))
    if requested.get("precision","fp32") != "fp32":
        raise ValueError("Original MeanFlow recovery must retain fp32 precision")
    restored_progress = copy.deepcopy(progress)
    restored_progress["microbatch"] = after["microbatch"]
    if "learning_rate" in requested:
        state = state.replace(learning_rate=np.asarray(after["learning_rate"],dtype=np.float32))
    append_event(Path(config["paths"]["output"])/"events.jsonl", "recovery_overrides_applied",
                 source="controller.recovery_overrides", checkpoint=str(checkpoint) if checkpoint else None,
                 requested=requested, before=before, after=after)
    return state, restored_progress


def run(config, *, resume=None, max_steps=None):
    validate_recipe(config)
    from ..bounded_stage import guard_trainer_launch
    from ..quality_stage import is_quality_stage, scheduled_learning_rate
    stage = config.get('bounded_stage', {})
    quality = is_quality_stage(stage)
    project = Path(config.get('project_root', Path(__file__).resolve().parents[3]))
    if quality:
        guard_trainer_launch(config, project)
        from ..quality_stage import guard_initialization
        guard_initialization(config, project, resume)
    # Must precede the first JAX import/device initialization.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    import jax
    # Must be effective before device discovery, model init and all tracing.
    jax.config.update('jax_default_matmul_precision', 'highest')
    import jax.numpy as jnp
    from flax import jax_utils
    from .official import create_model

    cache_root = Path(config.get("project_root",Path(__file__).resolve().parents[3])) / ".cache" / "jax"
    jax.config.update("jax_compilation_cache_dir",str(cache_root.resolve()))

    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    limits = config["limits"]
    level, memory = memory_status(limits)
    if level != "ok":
        raise MemoryError(f"Admission requires RAM below soft limit; observed {memory:.2f} GiB")
    devices = jax.local_device_count()
    if devices != config.get("device_count", 4):
        raise ValueError(f"Expected {config.get('device_count',4)} local devices, found {devices}")
    if jax.process_count() != 1:
        raise ValueError("This runner uses one process across all local GPUs")
    if any(d.platform != "gpu" for d in jax.local_devices()):
        raise RuntimeError("Training requires the configured NVIDIA GPUs; CPU fallback is disabled")
    check_runtime()
    cache = LatentCache(config["paths"]["latent_cache"], config["paths"]["dataset_manifest"])
    if cache.count % devices:
        raise ValueError("Dataset count must divide device count for exact, unpadded epoch accounting")
    model = create_model(config['model_id'], config)
    state, initial = create_state(model, get_spec(config["model_id"]),
                                  config["paths"]["initial_checkpoint"], config["learning_rate"], config["seed"],
                                  config.get('initialization_mode'))
    if quality and (initial.get('checkpoint_id') or initial.get('metadata', {}).get('checkpoint')) != stage['parent_checkpoint_id']:
        raise ValueError('Warm-start export differs from the registered parent EMA')
    identity = {"model_id": config["model_id"], "dataset_count": cache.count,
                "dataset_manifest_sha256": cache.dataset_sha256,
                "dataset_manifest_stat": cache.dataset_stat,
                "cache_manifest_stat": cache.manifest_stat,
                "codec": cache.codec, "initial_weights": initial.get("sha256") or initial.get('files'),
                "seed": config["seed"], "upstream_commit": UPSTREAM_COMMIT}
    if quality:
        identity.update(stage_contract=stage, parent_checkpoint_id=stage['parent_checkpoint_id'],
                        objective_id=stage['objective_id'], matmul_precision=str(jax.config.jax_default_matmul_precision))
    now = time.time()
    rng = np.random.default_rng(config["seed"])
    progress = {"sampler_epoch": 0, "position": 0, "samples_seen": 0, "valid_samples": 0,
                "attempted_samples": 0, "rejected_steps": 0, "oom_events": 0,
                "numpy_rng": rng.bit_generator.state, "microbatch": config["microbatch"],
                "last_save": now, "last_preview": now, "last_review": now}
    if quality:
        progress['recovery_scale'] = float(config.get('recovery_scale', 1))
    if resume == "latest":
        pointer = ('latest-' + stage['stage_id'] + '.json') if quality else 'latest.json'
        resume = json.loads((output/pointer).read_text())["checkpoint"]
    if resume:
        state, progress = restore_checkpoint(resume, state, identity)
        rng.bit_generator.state = progress["numpy_rng"]
    elif (output/('latest-' + stage['stage_id'] + '.json' if quality else 'latest.json')).exists():
        raise FileExistsError("Output already has a checkpoint; pass --resume latest")
    if quality:
        progress['recovery_scale'] = min(progress['recovery_scale'], config.get('recovery_scale', 1))
        config['recovery_scale'] = progress['recovery_scale']
        config['learning_rate'] = stage['recipe']['learning_rate'] * progress['recovery_scale']
        overrides = copy.deepcopy(config.get('recovery_overrides', {}))
        overrides.pop('learning_rate', None)
        recovery_config = {**config, 'recovery_overrides': overrides}
        state, progress = apply_recovery_overrides(state, progress, recovery_config, checkpoint=resume)
    else:
        state, progress = apply_recovery_overrides(state,progress,config,checkpoint=resume)
    actual_learning_rate = float(np.asarray(state.learning_rate))
    state = jax_utils.replicate(state)
    step_fn = make_train_step(model, config['ema_decay'], config.get('objective_id', MEANFLOW_ORIGINAL))
    append_event(output/"events.jsonl", "training_start", identity=identity, resume=resume,
                 microbatch=progress["microbatch"], learning_rate=actual_learning_rate, device_count=devices,
                 jax=jax.__version__, devices=[str(d) for d in jax.local_devices()],
                 matmul_precision=str(jax.config.jax_default_matmul_precision), bounded_stage=stage)
    if quality and not resume:
        save_checkpoint(state, progress, config, identity, export=True)
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    order_epoch = None
    failures = 0
    failure_kind = None
    completed = 0
    started = time.monotonic()
    while not stop["requested"] and (max_steps is None or completed < max_steps):
        committed_step = int(np.asarray(jax.device_get(state.step))[0])
        if quality:
            if not stage['source_step'] <= committed_step <= stage['stop_step']:
                raise ValueError('State step outside immutable stage budget')
            if committed_step == stage['stop_step']:
                break
            rate = scheduled_learning_rate(config, committed_step, progress['recovery_scale'])
            state = state.replace(learning_rate=jnp.full((devices,), rate, dtype=jnp.float32))
        if config.get("max_epochs") is not None and progress["sampler_epoch"] >= config["max_epochs"]:
            break
        level, ram = memory_status(limits)
        if level != "ok":
            append_event(output/"events.jsonl", "memory_stop", level=level, memory_gib=ram)
            jax.clear_caches()
            gc.collect()
            raise MemoryError(f"RAM {level} limit reached; stopped without a new CPU state copy; "
                              "preserving the latest complete checkpoint")
        if order_epoch != progress["sampler_epoch"]:
            order_epoch = progress["sampler_epoch"]
            order = np.random.default_rng(np.random.SeedSequence([config["seed"], order_epoch])).permutation(cache.count)
        batch_size = min(progress["microbatch"]*devices, cache.count-progress["position"])
        indices = order[progress["position"]:progress["position"]+batch_size]
        before_rng = copy.deepcopy(rng.bit_generator.state)
        # Shared augmentation contract across all algorithms and worker counts.
        flips = np.fromiter((hashlib.blake2b(f"{config['seed']}:{order_epoch}:{int(i)}".encode(),
                             digest_size=1).digest()[0] & 1 for i in indices), dtype=np.int64, count=batch_size)
        batch = cache.batch(indices, flips).reshape(devices, batch_size//devices, 32, 32, 8)
        progress["attempted_samples"] += batch_size
        try:
            candidate, metrics = step_fn(state, batch)
            metrics = jax.device_get(metrics)
            values = {key: float(np.asarray(value).mean()) for key, value in metrics.items()}
            if values["finite"] != 1.0:
                raise FloatingPointError("Nonfinite loss/gradient/updated parameters; state not committed")
            # Ensure device work has completed before accounting or saving it.
            jax.block_until_ready(candidate)
        except Exception as exc:
            is_oom = any(token in str(exc).lower() for token in ("out of memory", "resource_exhausted", "failed to allocate"))
            if not is_oom and not isinstance(exc, FloatingPointError):
                raise
            rng.bit_generator.state = before_rng
            kind = "oom" if is_oom else "nonfinite"
            failures = failures + 1 if failure_kind == kind else 1
            failure_kind = kind
            progress["rejected_steps"] += 1
            progress["oom_events"] += int(is_oom)
            if config.get("calibration",False) or config.get("max_recovery_failures",2) == 0:
                progress["numpy_rng"] = rng.bit_generator.state
                append_event(output/"events.jsonl","calibration_failed",reason=type(exc).__name__,
                             detail=str(exc),microbatch=progress["microbatch"])
                save_checkpoint(state,progress,config,identity,export=False)
                raise RuntimeError("Calibration candidate failed; no automatic batch/LR change") from exc
            old_batch = progress["microbatch"]
            old_lr = float(np.asarray(jax.device_get(state.learning_rate))[0])
            if is_oom:
                progress["microbatch"] = max(1, old_batch//2)
            else:
                state = state.replace(learning_rate=jnp.full((devices,), old_lr*0.5, dtype=jnp.float32))
                if quality:
                    progress['recovery_scale'] *= .5
                    config['recovery_scale'] = progress['recovery_scale']
                    config['learning_rate'] = stage['recipe']['learning_rate'] * progress['recovery_scale']
            append_event(output/"events.jsonl", "automatic_recovery", reason=type(exc).__name__, detail=str(exc),
                         old_microbatch=old_batch, new_microbatch=progress["microbatch"],
                         old_learning_rate=old_lr, new_learning_rate=old_lr if is_oom else old_lr*0.5,
                         attempt=failures, previous_state_preserved=True)
            del batch
            gc.collect()
            jax.clear_caches()
            if failures >= config.get("max_recovery_failures", 2) or (is_oom and old_batch <= (2 if quality else 1)):
                progress["numpy_rng"] = rng.bit_generator.state
                save_checkpoint(state, progress, config, identity, export=False)
                raise RecoveryExhausted("Bounded automatic recovery exhausted; last valid state saved") from exc
            continue
        del batch
        failures = 0
        failure_kind = None
        state = candidate
        completed += 1
        progress["position"] += batch_size
        progress["samples_seen"] += batch_size
        progress["valid_samples"] += batch_size
        if progress["position"] == cache.count:
            progress["sampler_epoch"] += 1
            progress["position"] = 0
        progress["numpy_rng"] = rng.bit_generator.state
        gpu_peak = gpu_memory_gib()
        append_event(output/"metrics.jsonl", "train_step", step=int(np.asarray(jax.device_get(state.step))[0]),
                     hq_epoch=progress["valid_samples"]/cache.count, valid_samples=progress["valid_samples"],
                     batch_size=batch_size, memory_gib=ram, gpu_peak_gib=gpu_peak,
                     seconds=time.monotonic()-started, **values)
        if any(value > limits["gpu_gib"] for value in gpu_peak.values()):
            if config.get("calibration",False):
                save_checkpoint(state,progress,config,identity,export=False)
                raise MemoryError("Calibration exceeded GPU memory budget")
            # Historical allocator peak cannot be reset reliably in-process.
            # Save the valid state and stop; the supervisor restarts a smaller batch.
            append_event(output/"events.jsonl", "gpu_memory_stop", gpu_peak_gib=gpu_peak,
                         microbatch=progress["microbatch"], recovery_owner="controller")
            print("gpu_memory_stop: GPU memory budget exceeded; saving the current valid state; "
                  "controller will choose the recovery microbatch", file=sys.stderr, flush=True)
            break
        now = time.time()
        preview = now-progress["last_preview"] >= config["preview_interval_seconds"]
        review = now-progress["last_review"] >= config["review_interval_seconds"]
        if now-progress["last_save"] >= config["save_interval_seconds"] or preview or review:
            progress["last_save"] = now
            if preview:
                progress["last_preview"] = now
            if review:
                progress["last_review"] = now
            checkpoint, export_path = save_checkpoint(state, progress, config, identity, export=preview or review)
            for kind, due in (("preview", preview), ("review", review)):
                if due and not config.get("validation_scope",False) and not config.get("calibration",False):
                    append_event(output/"requests.jsonl", kind, model_id=config["model_id"],
                                 checkpoint=str(checkpoint), export=export_path, codec=config["paths"]["codec"],
                                 seed=config["seed"], valid_samples=progress["valid_samples"],
                                 integrity_mode='metadata', checkpoint_id=checkpoint.name,
                                 stage_id=stage.get('stage_id'), step=int(np.asarray(jax.device_get(state.step))[0]))
    progress["numpy_rng"] = rng.bit_generator.state
    checkpoint, export_path = save_checkpoint(state, progress, config, identity, export=True)
    append_event(output/"events.jsonl", "training_stopped", checkpoint=str(checkpoint),
                 export=export_path, signal=stop["requested"], completed_steps=completed)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Complete checkpoint directory or latest")
    parser.add_argument("--max-steps", type=int, help="Absolute update limit for this invocation; registered stages enforce their fixed cap")
    parser.add_argument("--microbatch", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.microbatch is not None:
        if args.microbatch < 1:
            raise ValueError("--microbatch must be positive")
        config["microbatch"] = args.microbatch
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    run(config, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    try:
        main()
    except RecoveryExhausted as exc:
        print(f"RECOVERY_EXHAUSTED: {exc}",file=sys.stderr,flush=True)
        raise SystemExit(EXIT_RECOVERY_EXHAUSTED)
