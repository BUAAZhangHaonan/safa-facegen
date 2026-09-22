"""Validated HQ image manifests and read-only memory mapped training datasets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {path}")
    if "records" in value and value.get("count") != len(value["records"]):
        raise ValueError(f"Manifest record count mismatch: {path}")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _inspect_image(task: tuple[str, str, str]) -> dict[str, Any]:
    source, relative, root = task
    path = Path(root) / relative
    before = path.stat()
    digest = sha256_file(path)
    with Image.open(path) as image:
        image.load()  # Full decode; reading headers alone is not validation.
        original_mode, image_format = image.mode, image.format
        if image.size != (256, 256):
            raise ValueError(f"Expected 256x256, got {image.size}: {path}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during validation: {path}")
    return {
        "id": f"{source}:{path.stem}", "path": relative, "source": source,
        "sha256": digest, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
        "width": 256, "height": 256, "mode": original_mode, "format": image_format,
        "constant_rgb": bool(np.all(rgb == rgb[0, 0])),
    }


def validate_hq_dataset(
    project_root: str | Path, image_root: str | Path,
    output: str | Path, report_dir: str | Path, *, workers: int = 8,
) -> dict[str, Any]:
    """Decode/hash all 100k source records without removing duplicates or moving images."""
    project_root = Path(project_root).resolve()
    image_root = Path(image_root).resolve()
    output, report_dir = Path(output).resolve(), Path(report_dir).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    for target, permitted in [(output, project_root / "data/hq256"),
                              (report_dir, project_root / "reports/data")]:
        try:
            target.relative_to(permitted)
        except ValueError:
            raise ValueError(f"Output must remain inside {permitted}: {target}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing manifest: {output}")
    report_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for source, expected in [("ffhq", 70000), ("celeba_hq", 30000)]:
        files = sorted(p for p in (image_root / source).rglob("*") if p.is_file())
        if len(files) != expected:
            raise ValueError(f"{source}: expected {expected} files, found {len(files)}")
        tasks.extend((source, p.relative_to(image_root).as_posix(), str(image_root)) for p in files)
    raw_seen: dict[str, str] = {}
    rgb_seen: dict[str, str] = {}
    errors, records = [], []
    duplicate_path = report_dir / "duplicates.jsonl"
    error_path = report_dir / "invalid-images.jsonl"
    if duplicate_path.exists() or error_path.exists():
        raise FileExistsError("Audit output already exists; use a new report subdirectory")
    with duplicate_path.open("x", encoding="utf-8") as duplicates, error_path.open("x", encoding="utf-8") as invalid:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bounded submission avoids holding 100k Future objects and decoded images.
            for start in range(0, len(tasks), workers * 4):
                submitted = [(task, pool.submit(_inspect_image, task)) for task in tasks[start:start + workers * 4]]
                for task, future in submitted:
                    try:
                        record = future.result()
                    except Exception as exc:
                        error = {"source": task[0], "path": task[1], "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(error)
                        invalid.write(json.dumps(error, ensure_ascii=False) + "\n")
                        continue
                    for field, seen in [("sha256", raw_seen), ("rgb_sha256", rgb_seen)]:
                        previous = seen.setdefault(record[field], record["id"])
                        if previous != record["id"]:
                            record[f"{field}_duplicate_of"] = previous
                            duplicates.write(json.dumps({"id": record["id"], "duplicate_of": previous, "hash_kind": field, "hash": record[field]}) + "\n")
                    records.append(record)
                if start // (workers * 4) % 100 == 0:
                    print(json.dumps({"validated": len(records), "errors": len(errors), "total": len(tasks)}), flush=True)
    summary = {
        "schema_version": 1, "status": "failed" if errors else "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_count": 100000, "valid_count": len(records), "error_count": len(errors),
        "source_counts": {source: sum(r["source"] == source for r in records) for source in ("ffhq", "celeba_hq")},
        "unique_encoded_sha256": len(raw_seen), "unique_rgb_sha256": len(rgb_seen),
        "encoded_duplicate_records": len(records) - len(raw_seen),
        "rgb_duplicate_records": len(records) - len(rgb_seen),
        "constant_rgb_records": sum(r["constant_rgb"] for r in records),
        "duplicates_removed": False, "full_decode": True, "workers": workers,
    }
    if not errors:
        manifest = {
            "schema_version": 1,
            "project_root_relative": os.path.relpath(project_root, output.parent).replace(os.sep, "/"),
            "image_root": os.path.relpath(image_root, project_root).replace(os.sep, "/"),
            "count": len(records), "pixel_size": [256, 256], "channels": 3,
            "preprocessing": "Existing RGB JPEG95; original bundle used LANCZOS resize, quality95, subsampling0",
            "duplicate_policy": "retain_all_100000_source_records_and_register_duplicates",
            "records": records,
        }
        atomic_json(output, manifest)
        summary["dataset_manifest_sha256"] = sha256_file(output)
        summary["manifest"] = str(output)
    atomic_json(report_dir / "summary.json", summary)
    if errors:
        raise RuntimeError(f"Dataset validation failed for {len(errors)} images; see {error_path}")
    return summary


class HQDataset:
    """Images normalized to [-1,1]; no cropping, filtering, or hidden fallback."""
    def __init__(self, manifest: str | Path, random_flip: bool = True, seed: int = 42, *, image_root: str | Path | None = None):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = read_manifest(self.manifest_path)
        self.records = self.manifest["records"]
        base = (self.manifest_path.parent / self.manifest.get("project_root_relative", "../..")).resolve()
        self.image_root = Path(image_root).resolve() if image_root is not None else (base / self.manifest["image_root"]).resolve()
        self.random_flip = random_flip
        self.seed = int(seed)
        self._epoch = multiprocessing.get_context("spawn").Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def flip_for_index(self, index: int) -> bool:
        key = f"{self.seed}:{self._epoch.value}:{index}".encode("ascii")
        return self.random_flip and bool(hashlib.blake2b(key, digest_size=1).digest()[0] & 1)

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, index: int) -> Path:
        relative = Path(self.records[index]["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest image path: {relative}")
        return self.image_root / relative

    def __getitem__(self, index: int) -> torch.Tensor:
        import torch
        with Image.open(self.image_path(index)) as image:
            if image.size != (256, 256):
                raise ValueError(f"Image shape changed after validation: {self.image_path(index)}")
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        value = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.flip_for_index(index):
            value = value.flip(-1)
        return value


class CachedLatentDataset:
    """Each read selects the original or *separately encoded* horizontal-flip view."""
    def __init__(self, cache_manifest: str | Path, random_flip: bool = True, seed: int = 42):
        self.manifest_path = Path(cache_manifest).resolve()
        self.manifest = read_manifest(self.manifest_path)
        if self.manifest.get("status") != "complete":
            raise ValueError("Latent cache is incomplete")
        expected_channels = {"mean_std": 8, "prequant": 3}.get(self.manifest.get("representation"))
        shape = self.manifest.get("shape", [])
        expected_hw = 32 if expected_channels == 8 else 64
        if len(shape) != 5 or shape[1:] != [2, expected_channels, expected_hw, expected_hw]:
            raise ValueError(f"Unexpected cache shape: {shape}")
        if self.manifest.get("dtype") != "float32" or self.manifest.get("layout") != "N,F,C,H,W":
            raise ValueError("Cache must use float32 N,F,C,H,W")
        relative_array = Path(self.manifest["array"])
        if relative_array.is_absolute() or ".." in relative_array.parts:
            raise ValueError("Cache array must have a safe relative path")
        self.array_path = (self.manifest_path.parent / relative_array).resolve()
        mapped = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
        if list(mapped.shape) != shape or mapped.dtype != np.float32:
            raise ValueError("Cache array header differs from its manifest")
        actual_bytes = self.array_path.stat().st_size
        if actual_bytes != mapped.offset + mapped.nbytes or actual_bytes != self.manifest.get("array_bytes", actual_bytes):
            raise ValueError("Cache array file size differs from its completed layout")
        self.random_flip = random_flip
        self.seed = int(seed)
        self._epoch = multiprocessing.get_context("spawn").Value("q", 0)
        self._array = mapped

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def flip_for_index(self, index: int) -> bool:
        key = f"{self.seed}:{self._epoch.value}:{index}".encode("ascii")
        return self.random_flip and bool(hashlib.blake2b(key, digest_size=1).digest()[0] & 1)

    def __len__(self) -> int:
        return self.manifest["shape"][0]

    def __getitem__(self, index: int) -> torch.Tensor:
        import torch
        if self._array is None:
            self._array = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
            if list(self._array.shape) != self.manifest["shape"] or self._array.dtype != np.float32:
                raise ValueError("Cache array differs from its manifest")
        flip = int(self.flip_for_index(index))
        # Copy one item only: torch must not mutate a read-only memory map.
        return torch.from_numpy(np.array(self._array[index, flip], copy=True))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(validate_hq_dataset(args.project_root, args.image_root, args.output, args.report_dir, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
