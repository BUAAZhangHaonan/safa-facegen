"""Sequential four-GPU training with explicit human approvals and bounded recovery."""
from __future__ import annotations

import argparse
import copy
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import atomic_json, append_event, check_limits, git_commit, MODEL_IDS, project_root, sha256_file, utc_now, validation_identity, same_validated_implementation, EXECUTION_PATHS
from .retention import retire_states


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_config(root, campaign, model_id):
    config = read_json(root / campaign["model_configs"][model_id])
    local = root / "configs/local.json"
    overrides = read_json(local).get("models", {}).get(model_id, {}) if local.exists() else {}
    for key, value in overrides.items():
        if key == "paths":
            config.setdefault("paths", {}).update(value)
        else:
            config[key] = value
    config["project_root"] = str(root)
    config["model_id"] = model_id
    config["paths"]["output"] = str(root / "runs" / model_id)
    for key, value in list(config["paths"].items()):
        if value and not Path(value).is_absolute():
            config["paths"][key] = str(root / value)
    for key in ("save_interval_seconds", "preview_interval_seconds", "review_interval_seconds", "limits"):
        config[key] = copy.deepcopy(campaign[key])
    config["max_recovery_failures"] = campaign["max_consecutive_recovery_failures"]
    config["required_world_size"] = 4
    config.pop("max_steps", None)
    config.pop("max_epochs", None)
    config.pop("max_hq_epochs", None)
    return config


def training_command(root, config_file, config, resume=False):
    if config["model_id"].startswith("MeanFlow-"):
        command = [str(root / ".venv-jax/bin/python"), "-m", "safa_facegen.meanflow.trainer", "--config", str(config_file)]
        if resume:
            command += ["--resume", "latest"]
        return command
    return [str(root / ".venv-torch/bin/python"), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            "-m", "safa_facegen.train_torch", "--config", str(config_file)]


def process_environment(root):
    env = os.environ.copy()
    local = root / 'configs/local.json'
    if local.exists():
        settings = read_json(local).get('environment', {})
        for key in ('LD_LIBRARY_PATH', 'PATH'):
            if key in settings:
                env[key] = settings[key]
    env.update(PYTHONPATH=str(root / "src"), SAFA_FACEGEN_ROOT=str(root), CUDA_VISIBLE_DEVICES="0,1,2,3",
               XLA_PYTHON_CLIENT_PREALLOCATE="true", XLA_PYTHON_CLIENT_MEM_FRACTION="0.85",
               OMP_NUM_THREADS="8", TMPDIR=str(root / "tmp"), XDG_CACHE_HOME=str(root / ".cache"),
               HF_HOME=str(root / ".cache/huggingface"), TORCH_HOME=str(root / ".cache/torch"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", JAX_COMPILATION_CACHE_DIR=str(root / ".cache/jax"),
               PYTHONUNBUFFERED="1")
    for key, relative in {
        'CUDA_CACHE_PATH': '.cache/cuda', 'TRITON_CACHE_DIR': '.cache/triton',
        'TORCHINDUCTOR_CACHE_DIR': '.cache/torchinductor', 'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions',
        'PYTHONPYCACHEPREFIX': '.cache/python', 'XDG_CONFIG_HOME': '.cache/config',
        'XDG_DATA_HOME': '.cache/share', 'MPLCONFIGDIR': '.cache/matplotlib',
    }.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=10, check=True)
    return [{"index": int(row[0]), "used_bytes": int(row[1]) * 2**20, "total_bytes": int(row[2]) * 2**20}
            for line in result.stdout.splitlines() if (row := [item.strip() for item in line.split(",")])]


def verify_launch(root, config):
    model_id = config["model_id"]
    identity = validation_identity(root, config)
    for receipt in ("training", "batch"):
        path = root / "reports/validation" / model_id / (receipt + ".json")
        value = read_json(path)
        if value.get("status") != "passed":
            raise ValueError(f"Required validation has not passed: {path}")
        if value.get("model_id") != model_id:
            raise ValueError(f"Validation identity mismatch: {path}")
        if not same_validated_implementation(root, value.get('validation_identity'), identity):
            raise ValueError(f'Validation does not cover the current sources, recipe, and data: {path}')
        if receipt == "batch":
            config["microbatch"] = int(value["microbatch"])
            if value["peak_gpu_gib"] > config["limits"]["gpu_gib"] or value["peak_memory_gib"] >= config["limits"]["memory_soft_gib"]:
                raise ValueError("Profile exceeds configured resource limit")
    for key in ("dataset_manifest", "initial_checkpoint", "codec"):
        value = config["paths"].get(key)
        if value and not Path(value).exists():
            raise FileNotFoundError(value)
    if model_id != "RectifiedFlow-NCSNpp" and not Path(config["paths"]["latent_cache"]).is_file():
        raise FileNotFoundError(config["paths"]["latent_cache"])
    devices = gpu_snapshot()
    if [x["index"] for x in devices] != [0, 1, 2, 3]:
        raise RuntimeError("Expected exactly the four authorized H100 GPUs")
    return devices


def latest_state(root, model_id):
    output = root / "runs" / model_id
    path = output / ("latest.json" if model_id.startswith("MeanFlow-") else "last.json")
    return read_json(path) if path.exists() else None


def read_approval(root, model_id):
    path = root / "runs/approvals" / (model_id + ".json")
    if not path.exists():
        return None
    approval = read_json(path)
    if approval.get("model_id") != model_id or approval.get("decision") != "approved" or approval.get("source") != "explicit_user_confirmation":
        raise ValueError(f"Invalid approval: {path}")
    weight = Path(approval["ema_path"])
    if not weight.is_absolute():
        weight = root / weight
    if root.resolve() not in weight.resolve().parents:
        raise ValueError("Approval weight is outside the project")
    info = weight.stat()
    if approval_weight_digest(str(weight), info.st_size, info.st_mtime_ns) != approval["ema_sha256"]:
        raise ValueError("Approved EMA content has changed")
    return approval


@lru_cache(maxsize=24)
def approval_weight_digest(path, byte_count, modified_ns):
    # One content check when accepting a new explicit user decision. Repeated
    # graceful-stop polling reuses it while the immutable file stat is unchanged.
    return sha256_file(path)


def stop_group(process, *, hard=False):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)


def log_tail(path, size=32000):
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - size))
        return handle.read().decode("utf-8", errors="replace")


def recovery_config(config, state, reason):
    result = copy.deepcopy(config)
    changes = {}
    if reason in ("gpu_memory", "out_of_memory"):
        changes["microbatch"] = max(1, result["microbatch"] // 2)
        if changes["microbatch"] == result["microbatch"]:
            raise RuntimeError("No smaller microbatch is available")
    elif reason == "nonfinite":
        changes["learning_rate"] = result["learning_rate"] * 0.5
        changes["precision"] = "fp32"
    elif reason == "host_memory":
        changes["num_workers"] = max(0, result.get("num_workers", 4) // 2)
        changes["prefetch_factor"] = 1
    else:
        raise RuntimeError("Failure requires code diagnosis before recovery")
    result.update(changes)
    result["recovery_overrides"] = changes
    if state and not result["model_id"].startswith("MeanFlow-"):
        result["paths"]["resume"] = state["state_path"]
    return result, changes


def run_campaign(root, campaign_path):
    root = Path(root).resolve()
    campaign = read_json(campaign_path)
    state_dir = root / "runs/controller"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("tmp", "logs/training", "runs/approvals", "models/formal"):
        (root / name).mkdir(parents=True, exist_ok=True)
    events = state_dir / "events.jsonl"
    revision = git_commit(root)
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', '--quiet', 'HEAD', '--',
                    *EXECUTION_PATHS], check=True)
    append_event(events, "campaign_start", code_commit=revision, pid=os.getpid())
    stop_requested = [False]
    def request_stop(*_):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    process = None
    try:
        for model_id in campaign["model_order"]:
            if read_approval(root, model_id):
                continue
            config = load_model_config(root, campaign, model_id)
            if model_id == "LatentConsistency-LDM-UNet":
                teacher = read_approval(root, "Diffusion-LDM-UNet")
                if not teacher:
                    raise RuntimeError("LCM requires the user-approved project Diffusion teacher")
                config["paths"]["teacher_checkpoint"] = teacher["ema_path"]
                config["paths"]["initial_checkpoint"] = None
            verify_launch(root, config)
            attempts = 0
            previous_failure = None
            while not stop_requested[0]:
                state = latest_state(root, model_id)
                if state and not model_id.startswith("MeanFlow-"):
                    config["paths"]["resume"] = state["state_path"]
                runtime = state_dir / (model_id + ".json")
                atomic_json(runtime, config)
                log_path = root / "logs/training" / (model_id + ".log")
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.Popen(training_command(root, runtime, config, resume=bool(state)),
                                               cwd=root, env=process_environment(root), stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                append_event(events, "model_started", model_id=model_id, pid=process.pid,
                             microbatch=config["microbatch"], learning_rate=config["learning_rate"], resume=state)
                reason = None
                approval = None
                stop_sent = None
                peak_gpu = 0
                peak_memory = 0
                recovered_step = int((state or {}).get('step', 0))
                retention_checked = 0.0
                while process.poll() is None:
                    level, info = check_limits(config["limits"])
                    devices = gpu_snapshot()
                    peak_gpu = max(peak_gpu, max(x["used_bytes"] for x in devices))
                    peak_memory = max(peak_memory, info["memory_bytes"])
                    approval = read_approval(root, model_id)
                    completed = latest_state(root, model_id)
                    if time.monotonic() - retention_checked > 60:
                        retire_states(root, model_id)
                        retention_checked = time.monotonic()
                    if attempts and completed and int(completed.get('step', 0)) >= recovered_step + 100:
                        append_event(events, 'recovery_stable', model_id=model_id,
                                     previous_failure=previous_failure, step=completed['step'])
                        attempts = 0
                        previous_failure = None
                    if stop_requested[0] or approval:
                        if stop_sent is None:
                            stop_group(process)
                            stop_sent = time.monotonic()
                    elif level == "hard":
                        reason = "host_memory"
                        stop_group(process, hard=True)
                    elif level == "soft" or max(x["used_bytes"] for x in devices) > config["limits"]["gpu_gib"] * 2**30:
                        reason = "host_memory" if level == "soft" else "gpu_memory"
                        if stop_sent is None:
                            stop_group(process)
                            stop_sent = time.monotonic()
                    if stop_sent is not None and time.monotonic() - stop_sent > 300:
                        stop_group(process, hard=True)
                    atomic_json(state_dir / "status.json", {"status": "stopping" if stop_sent else "training",
                                "model_id": model_id, "pid": process.pid, "code_commit": revision,
                                "updated_at": utc_now(), "gpu": devices, **info,
                                "peak_gpu_gib": peak_gpu / 2**30, "peak_memory_gib": peak_memory / 2**30,
                                "approval_received": bool(approval), "recovery_attempt": attempts})
                    time.sleep(5)
                exit_code = process.wait()
                process = None
                if stop_requested[0]:
                    atomic_json(state_dir / "status.json", {"status": "stopped", "model_id": model_id, "time": utc_now()})
                    return
                if approval:
                    atomic_json(root / "models/formal" / (model_id + ".json"), approval)
                    append_event(events, "model_approved", model_id=model_id, approval=approval, exit_code=exit_code)
                    break
                tail = log_tail(log_path).lower()
                if reason is None:
                    if "automatic recovery exhausted" in tail:
                        raise RuntimeError("Trainer exhausted its bounded recovery; inspect events")
                    if "out of memory" in tail or "resource_exhausted" in tail:
                        reason = "out_of_memory"
                    elif "non-finite" in tail or "nonfinite" in tail:
                        reason = "nonfinite"
                    elif "resource_limit" in tail or "memory_stop" in tail:
                        reason = "host_memory"
                    else:
                        reason = "runtime_error"
                attempts = attempts + 1 if reason == previous_failure else 1
                previous_failure = reason
                append_event(events, "model_failure", model_id=model_id, reason=reason, exit_code=exit_code,
                             consecutive_failures=attempts, log_path=str(log_path))
                if attempts > campaign["max_consecutive_recovery_failures"]:
                    raise RuntimeError("Two recovery attempts failed; current model stopped")
                config, changes = recovery_config(config, latest_state(root, model_id), reason)
                append_event(events, "recovery_scheduled", model_id=model_id, reason=reason, changes=changes,
                             attempt=attempts)
            if stop_requested[0]:
                return
        atomic_json(state_dir / "status.json", {"status": "all_models_approved", "time": utc_now()})
    except BaseException as exc:
        if process is not None:
            stop_group(process)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                stop_group(process, hard=True)
                process.wait(timeout=30)
        atomic_json(state_dir / "status.json", {"status": "attention_required", "time": utc_now(),
                                               "error": f"{type(exc).__name__}: {exc}"})
        append_event(events, "attention_required", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--campaign", default="configs/campaign.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    campaign = Path(args.campaign)
    run_campaign(root, campaign if campaign.is_absolute() else root / campaign)


if __name__ == "__main__":
    main()
