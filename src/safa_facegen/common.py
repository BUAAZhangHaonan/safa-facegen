"""Small shared utilities for immutable model identities and resource limits."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

MODEL_IDS = (
    "Diffusion-LDM-UNet", "LatentConsistency-LDM-UNet", "RectifiedFlow-NCSNpp",
    "MeanFlow-B-4", "MeanFlow-B-2", "MeanFlow-L-2",
)
EXECUTION_PATHS = ('src', 'vendor', 'configs', 'requirements', 'requirements.txt',
                   'requirements-jax.txt', 'pyproject.toml')


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def fsync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def append_event(path, event, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    return record


def checkpoint_name(model_id, samples_seen, dataset_size, timestamp=None):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model: {model_id}")
    if dataset_size <= 0 or samples_seen < 0:
        raise ValueError("Invalid sample accounting")
    return f"{model_id}-{int(samples_seen // dataset_size):04d}ep-{timestamp or utc_now()}"


def git_commit(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def validation_identity(root, config):
    """Reuse Git revision and registered metadata without rescanning asset contents."""
    root = Path(root).resolve()
    paths = config.get('paths', {})
    artifacts = {}
    artifact_keys = ('dataset_manifest', 'latent_cache', 'codec') if config.get('model_id') == 'LatentConsistency-LDM-UNet' else ('dataset_manifest', 'latent_cache', 'codec', 'initial_checkpoint')
    for key in artifact_keys:
        if not paths.get(key):
            continue
        path = Path(paths[key])
        if not path.is_absolute():
            path = root / path
        info = path.stat()
        artifacts[key] = {'path': str(path.relative_to(root)) if root in path.parents else str(path),
                          'bytes': info.st_size if path.is_file() else None,
                          'mtime_ns': info.st_mtime_ns}
    keys = ('model_id', 'learning_rate', 'gradient_accumulation_steps', 'precision', 'ema_decay',
            'adam_betas', 'weight_decay', 'gradient_clip', 'seed', 'sampling', 'required_world_size')
    return {'code_commit': git_commit(root), 'artifacts': artifacts,
            'teacher_model_id': 'Diffusion-LDM-UNet' if config.get('model_id') == 'LatentConsistency-LDM-UNet' else None,
            'recipe': {key: config.get(key) for key in keys}}


def same_validated_implementation(root, previous, current):
    if not isinstance(previous, dict):
        return False
    if {k: v for k, v in previous.items() if k != 'code_commit'} != {
            k: v for k, v in current.items() if k != 'code_commit'}:
        return False
    if previous.get('code_commit') == current['code_commit']:
        return True
    # Documentation and training-record commits do not invalidate completed GPU
    # runs. The exact validated and current revisions remain in their records.
    revision = previous.get('code_commit', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        return False
    return subprocess.run(['git', '-C', str(root), 'diff', '--quiet', revision,
                           current['code_commit'], '--', *EXECUTION_PATHS],
                          capture_output=True).returncode == 0


def memory_bytes():
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    # K100 may expose only a host cgroup. Report process RSS separately; never
    # invent a container memory limit from the host's installed RAM.
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def check_limits(limits):
    used = memory_bytes()
    info = {"memory_bytes": used, "time": time.time()}
    hard = float(limits.get("memory_hard_gib", 224)) * 2**30
    soft = float(limits.get("memory_soft_gib", 192)) * 2**30
    return ("hard" if used >= hard else "soft" if used >= soft else "ok"), info


def project_root():
    return Path(os.environ.get("SAFA_FACEGEN_ROOT", Path(__file__).resolve().parents[2])).resolve()


def require_inside(path, root=None):
    root = Path(root or project_root()).resolve()
    path = Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be a child of project root: {path}")
    return path
