"""Small shared utilities for immutable model identities and resource limits."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import stat
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


def release_file_cache(paths, root):
    """Release clean pages of named project files without changing their contents."""
    before = memory_bytes()
    count = advised = 0
    errors = []
    for filename in dict.fromkeys(map(str, paths)):
        original = Path(filename)
        if original.is_symlink():
            raise ValueError(f"Symlink in checkpoint cache path: {original}")
        path = require_inside(original, root)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"Cache release requires a regular file: {path}")
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
                advised += info.st_size
            finally:
                os.close(fd)
        except FileNotFoundError:
            # Retention can retire a completed state while its metadata remains.
            continue
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"files_advised": count, "file_bytes_advised": advised,
            "memory_before_bytes": before, "memory_after_bytes": memory_bytes(),
            "errors": errors}


def release_checkpoint_cache(root, model_id):
    """Only completed checkpoints of this model are eligible for cache release."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model: {model_id}")
    root = Path(root).resolve()
    output = require_inside(root / "runs" / model_id, root)
    paths = []
    if model_id.startswith("MeanFlow-"):
        for checkpoint in (output / "checkpoints").glob(model_id + "-*"):
            if checkpoint.is_symlink():
                raise ValueError(f"Symlink in checkpoint directory: {checkpoint}")
            if (checkpoint / "COMPLETE.json").is_file():
                paths.extend(p for p in checkpoint.rglob("*") if p.is_file())
    else:
        for metadata in output.glob(model_id + "-*.json"):
            if metadata.name.endswith(".config.json"):
                continue
            record = json.loads(metadata.read_text())
            if not record.get("complete"):
                continue
            for key in ("state_path", "ema_path"):
                if record.get(key):
                    path = root / record[key]
                    if path.parent.resolve() != output:
                        raise ValueError(f"Checkpoint path is outside its model directory: {path}")
                    paths.append(path)
    return release_file_cache(paths, root)


def project_root():
    return Path(os.environ.get("SAFA_FACEGEN_ROOT", Path(__file__).resolve().parents[2])).resolve()


def require_inside(path, root=None):
    root = Path(root or project_root()).resolve()
    path = Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be a child of project root: {path}")
    return path
