"""Metadata integrity without reading whole checkpoint files for digests.

Historical digests remain provenance only. Byte inventory, atomic completion,
strict runtime deserialization and source stability are independent checks.
"""
from pathlib import Path
import json


def file_record(path, *, relative=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing, empty or nonregular artifact: {path}")
    return {"path": relative or path.name, "bytes": path.stat().st_size}


def validate_files(root, records, *, exact=False):
    root = Path(root)
    seen = set()
    for row in records:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError("Invalid or duplicate artifact relative path")
        seen.add(relative.as_posix())
        path = root / relative
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Artifact symlinks are forbidden")
        size = row.get("bytes", row.get("size"))
        if not isinstance(size, int) or size <= 0 or file_record(path)["bytes"] != size:
            raise ValueError(f"Artifact byte count mismatch: {relative}")
    if not seen:
        raise ValueError("Empty artifact inventory")
    if exact and seen != {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}:
        raise ValueError("Artifact inventory mismatch")


def completion_record(manifest_path, checkpoint_id):
    return {"integrity_mode": "metadata", "checkpoint_id": checkpoint_id,
            "manifest_bytes": file_record(manifest_path)["bytes"]}


def validate_completion(manifest, complete, *, manifest_bytes, checkpoint_id):
    if manifest.get("integrity_mode", "sha256") == "metadata":
        if (complete.get("integrity_mode") != "metadata"
                or complete.get("checkpoint_id") != checkpoint_id
                or manifest.get("checkpoint_id") != checkpoint_id
                or complete.get("manifest_bytes") != manifest_bytes):
            raise ValueError("Metadata completion marker mismatch")
    elif manifest.get("integrity_mode", "sha256") == "sha256":
        if not complete.get("manifest_sha256"):
            raise ValueError("Legacy completion identity missing")
    else:
        raise ValueError("Unknown integrity mode")


def verify_meanflow_checkpoint(path):
    path = Path(path)
    if (path / "STATE_RETIRED.json").exists():
        raise ValueError("Retired state cannot be restored")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    validate_completion(manifest, complete, manifest_bytes=manifest_path.stat().st_size,
                        checkpoint_id=path.name)
    if manifest.get("temporary_calibration"):
        raise ValueError("Temporary calibration cannot be restored")
    validate_files(path / "state", manifest.get("state_files", []), exact=True)
    return manifest
