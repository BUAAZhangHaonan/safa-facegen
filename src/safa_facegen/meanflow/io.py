"""Atomic metadata and content identity; independent of either ML runtime."""
import hashlib
import json
import os
from pathlib import Path
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_event(path, event, **values):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "event": event, **values},
                                ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
