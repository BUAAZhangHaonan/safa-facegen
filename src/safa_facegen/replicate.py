"""K100 pull-only replication and a serial GPU evaluation queue.

SSH passwords and pinned host keys are accepted on stdin, never in config or files.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sqlite3
import stat
import sys
import threading
import time
import uuid
import errno

from .common import MODEL_IDS, fsync_directory
from .data import atomic_json, sha256_file

H100_ROOT = PurePosixPath("/home/apulis-dev/code/meanflow_e15_h100_bundle")
K100_ROOT = Path("/home/k100/projects/safa-facegen")
ALLOWED_REMOTE = (H100_ROOT / "runs", H100_ROOT / "models", H100_ROOT / "data/hq256")
RECEIPT_ROOT = H100_ROOT / "reports/replication/receipts"
TRANSFER_PRIORITY = {"review": 0, "preview": 1, "save": 2}


class TransferPaused(Exception):
    """A durable partial transfer yielded its connection to a higher priority EMA."""


def remote_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError("Parent traversal in remote path")
    if not path.is_absolute():
        path = H100_ROOT / path
    if not any(path == root or root in path.parents for root in ALLOWED_REMOTE):
        raise ValueError(f"Remote path is outside the read whitelist: {path}")
    if any(part.endswith((".partial", ".tmp")) or ".partial." in part for part in path.parts):
        raise ValueError(f"Partial remote artifact rejected: {path}")
    return path


def valid_hash(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A complete SHA256 is required")
    return value


class RemoteReader:
    """Only SFTP reads and sha256sum are exposed to the replication code."""
    def __init__(self, credentials: dict):
        import paramiko

        class PinnedKey(paramiko.MissingHostKeyPolicy):
            def __init__(self, expected):
                self.expected = base64.b64decode(expected, validate=True)

            def missing_host_key(self, client, hostname, key):
                if not hmac.compare_digest(self.expected, key.asbytes()):
                    raise paramiko.SSHException("Pinned SSH host key mismatch")

        def configure(client, entry):
            blob = base64.b64decode(entry["key"], validate=True)
            kind = paramiko.Message(blob).get_text()
            key = paramiko.PKey.from_type_string(kind, blob)
            host = entry["hostname"] if int(entry.get("port", 22)) == 22 else f"[{entry['hostname']}]:{entry['port']}"
            # Also advertise the pinned key algorithm first during negotiation.
            client.get_host_keys().add(host, kind, key)
            client.set_missing_host_key_policy(PinnedKey(entry["key"]))

        self.jump = self.target = self.sftp = None
        self.active = None
        self.retrying = []
        self.last_heartbeat = 0.0
        self.transfer_check = None
        # hashlib state is deliberately memory-only. Controlled scheduling pauses
        # reuse it; a process restart must read an unfinished prefix once.
        self.partial_digests = {}
        try:
            jump, target = credentials["jump"], credentials["h100"]
            self.jump = paramiko.SSHClient()
            configure(self.jump, jump)
            self.jump.connect(hostname=jump["hostname"], port=int(jump.get("port", 22)),
                              username=jump["user"], password=jump["password"],
                              allow_agent=False, look_for_keys=False, timeout=30, auth_timeout=30, banner_timeout=30)
            channel = self.jump.get_transport().open_channel(
                "direct-tcpip", (target["hostname"], int(target.get("port", 22))), ("127.0.0.1", 0), timeout=30)
            self.target = paramiko.SSHClient()
            configure(self.target, target)
            self.target.connect(hostname=target["hostname"], port=int(target.get("port", 22)),
                                username=target["user"], password=target["password"], sock=channel,
                                allow_agent=False, look_for_keys=False, timeout=30, auth_timeout=30, banner_timeout=30)
            self.target.get_transport().set_keepalive(30)
            self.sftp = self.target.open_sftp()
            self.sftp.get_channel().settimeout(120)
        except BaseException:
            self.close()
            raise

    def close(self):
        for obj in (self.sftp, self.target, self.jump):
            if obj is not None:
                with contextlib.suppress(Exception):
                    obj.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def checked(self, value: str | PurePosixPath) -> PurePosixPath:
        path = remote_path(str(value))
        # Do not allow a symlink to escape the root or change between inventory and read.
        for item in reversed((path, *path.parents)):
            if item == PurePosixPath("/"):
                continue
            if stat.S_ISLNK(self.sftp.lstat(str(item)).st_mode):
                raise ValueError(f"Remote symlink rejected: {item}")
        return path

    def read(self, value, limit=64 * 1024 * 1024) -> bytes:
        path = self.checked(value)
        attributes = self.sftp.stat(str(path))
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_size > limit:
            raise ValueError(f"Metadata is not a regular bounded file: {path}")
        with self.sftp.open(str(path), "rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Metadata grew beyond the read limit")
        return data

    def json(self, value):
        return json.loads(self.read(value))

    def inventory(self, value, expected_hashes: dict | None = None) -> list[dict]:
        path = self.checked(value)
        output = []

        def visit(current, relative):
            current = self.checked(current)
            before = self.sftp.stat(str(current))
            if stat.S_ISDIR(before.st_mode):
                for entry in sorted(self.sftp.listdir_attr(str(current)), key=lambda x: x.filename):
                    if entry.filename in (".", "..") or "/" in entry.filename:
                        raise ValueError("Invalid SFTP directory entry")
                    visit(current / entry.filename, relative / entry.filename)
            elif stat.S_ISREG(before.st_mode):
                declared = (expected_hashes or {}).get(str(relative))
                if declared:
                    digest = valid_hash(declared)
                else:
                    _, stdout, stderr = self.target.exec_command("sha256sum -- " + shlex.quote(str(current)), timeout=120)
                    response = stdout.read().decode("ascii").strip()
                    error = stderr.read().decode("utf-8", "replace")
                    if stdout.channel.recv_exit_status() != 0:
                        raise OSError(f"Remote SHA256 failed: {error[:300]}")
                    digest = valid_hash(response.split()[0])
                after = self.sftp.stat(str(current))
                if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
                    raise ValueError("Remote source changed during inventory")
                output.append({"source": str(current), "path": str(relative), "bytes": after.st_size,
                               "mtime": after.st_mtime, "sha256": digest})
            else:
                raise ValueError(f"Remote artifact is not a regular file/directory: {current}")

        if stat.S_ISDIR(self.sftp.stat(str(path)).st_mode):
            visit(path, PurePosixPath("."))
        else:
            visit(path, PurePosixPath(path.name))
        return output

    def download(self, record: dict, destination: Path):
        source = self.checked(record["source"])
        before = self.sftp.stat(str(source))
        if (before.st_size, before.st_mtime) != (record["bytes"], record["mtime"]):
            raise ValueError("Remote source changed before transfer")
        destination.parent.mkdir(parents=True, exist_ok=True)
        progress_key = (str(destination), str(source), record["sha256"], record["bytes"], record["mtime"])
        digest, total = hashlib.sha256(), 0
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or destination.stat().st_size > record["bytes"]:
                raise ValueError("Invalid resumable local partial file")
            info = destination.stat()
            previous = self.partial_digests.get(progress_key)
            if previous and previous["stat"] == (info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                digest, total = previous["digest"].copy(), info.st_size
            else:
                with destination.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b""):
                        digest.update(block); total += len(block)
            if total == record["bytes"]:
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("Completed staging file has an incorrect hash")
                self.partial_digests.pop(progress_key, None)
                return
        try:
            with self.sftp.open(str(source), "rb") as remote, destination.open("ab" if destination.exists() else "xb") as local:
                remote.seek(total)
                try:
                    while total < record["bytes"]:
                        # Drain each bounded window before yielding: Paramiko does not
                        # cancel a whole-file prefetch thread when its handle closes.
                        window_end = min(total + 1024 * 1024, record["bytes"])
                        remote.prefetch(file_size=window_end, max_concurrent_requests=32)
                        block = remote.read(window_end - total)
                        if not block:
                            break
                        local.write(block); digest.update(block); total += len(block)
                        self.heartbeat()
                        if self.transfer_check is not None:
                            self.transfer_check()
                finally:
                    local.flush(); os.fsync(local.fileno())
        finally:
            if destination.is_file():
                info = destination.stat()
                if info.st_size == total:
                    self.partial_digests[progress_key] = {"stat": (info.st_size, info.st_mtime_ns, info.st_ctime_ns), "digest": digest.copy()}
        after = self.sftp.stat(str(source))
        if total != record["bytes"] or digest.hexdigest() != record["sha256"] or (after.st_size, after.st_mtime) != (before.st_size, before.st_mtime):
            raise ValueError("Transferred file SHA256/size/source-stability check failed")
        self.partial_digests.pop(progress_key, None)
        fsync_directory(destination.parent)

    def _prepare_replication_directory(self):
        directory = RECEIPT_ROOT.parent
        for item in reversed((directory, *directory.parents)):
            if item == PurePosixPath("/"):
                continue
            try:
                info = self.sftp.lstat(str(item))
            except FileNotFoundError:
                if item != directory:
                    raise
                self.sftp.mkdir(str(item)); info = self.sftp.lstat(str(item))
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("Replication write parent is not a real directory")

    def heartbeat(self, *, force=False):
        if not force and (self.active is None or time.monotonic() - self.last_heartbeat < 30):
            return
        self._prepare_replication_directory()
        destination = RECEIPT_ROOT.parent / "active-transfer.json"
        try:
            if not stat.S_ISREG(self.sftp.lstat(str(destination)).st_mode):
                raise ValueError("Active transfer record is not a regular file")
        except FileNotFoundError:
            pass
        payload = {"schema_version": 1, "status": "active" if self.active else "idle",
                   "updated_at_unix": time.time(), "retrying": self.retrying, **(self.active or {})}
        temporary = destination.with_name(".active-transfer.partial." + uuid.uuid4().hex)
        with self.sftp.open(str(temporary), "wx") as handle:
            handle.write(json.dumps(payload, allow_nan=False).encode() + b"\n"); handle.flush()
        self.sftp.posix_rename(str(temporary), str(destination))
        self.last_heartbeat = time.monotonic()

    def begin_transfer(self, request, retrying):
        self.active = {"request_id": request["request_id"], "identity": request["identity"],
                       "model_id": request["model_id"], "source_checkpoint": request["checkpoint"], "event": request["event"]}
        self.retrying = retrying
        self.heartbeat(force=True)

    def end_transfer(self, retrying):
        self.active = None
        self.retrying = retrying
        self.heartbeat(force=True)

    def write_receipt(self, request_id: str, receipt: dict):
        """The sole H100 write surface: an atomic, immutable transfer acknowledgement."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,220}", request_id) or receipt.get("request_id") != request_id:
            raise ValueError("Unsafe or mismatched source receipt identity")
        for directory in reversed((RECEIPT_ROOT, *RECEIPT_ROOT.parents)):
            if directory == PurePosixPath("/"):
                continue
            try:
                info = self.sftp.lstat(str(directory))
            except FileNotFoundError:
                if directory not in (RECEIPT_ROOT.parent, RECEIPT_ROOT):
                    raise
                self.sftp.mkdir(str(directory))
                info = self.sftp.lstat(str(directory))
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("Source receipt parent must be a real directory")
        destination = RECEIPT_ROOT / (request_id + ".json")
        body = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Source receipt is unexpectedly large")
        try:
            info = self.sftp.lstat(str(destination))
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                raise ValueError("Existing receipt is not a bounded regular file")
            with self.sftp.open(str(destination), "rb") as handle:
                previous = json.load(handle)
            for field in ("request_id", "model_id", "identity", "event", "source_checkpoint", "source_ema", "files", "artifact_role", "received_complete"):
                if previous.get(field) != receipt.get(field):
                    raise ValueError("An existing source acknowledgement conflicts with this transfer")
            return
        temporary = RECEIPT_ROOT / ("." + request_id + ".partial." + uuid.uuid4().hex)
        with self.sftp.open(str(temporary), "wx") as handle:
            handle.write(body)
            handle.flush()
        self.sftp.posix_rename(str(temporary), str(destination))
        with self.sftp.open(str(destination), "rb") as handle:
            if hashlib.sha256(handle.read()).digest() != hashlib.sha256(body).digest():
                raise ValueError("Source acknowledgement readback mismatch")


def normalize_request(raw: dict, expected_model: str) -> dict:
    model = raw.get("model_id", expected_model)
    event = raw.get("type", raw.get("event", "save"))
    if model != expected_model or model not in MODEL_IDS or event not in ("save", "preview", "review"):
        raise ValueError("Invalid request model/event")
    checkpoint = remote_path(raw.get("checkpoint", raw.get("state_path", "")))
    if H100_ROOT / "runs" not in checkpoint.parents:
        raise ValueError("Checkpoint must be below runs")
    identity = raw.get("checkpoint_id", raw.get("identity"))
    if not isinstance(identity, str):
        identity = checkpoint.name.removesuffix(".state.pt")
    if not re.fullmatch(re.escape(model) + r"-\d+ep-\d{8}T\d{6}(?:\d{6})?Z", identity):
        raise ValueError(f"Noncanonical model-epoch-UTC identity: {identity}")
    if H100_ROOT / "runs" / model not in checkpoint.parents:
        raise ValueError("Checkpoint is outside its model's run directory")
    result = {"request_id": str(raw.get("request_id", f"{identity}-{event}")), "model_id": model,
              "identity": identity, "event": event, "checkpoint": str(checkpoint),
              "seed": int(raw.get("seed", 42)), "source": raw}
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,220}", result["request_id"]):
        raise ValueError("Unsafe request ID")
    if model.startswith("MeanFlow-"):
        result["kind"] = "meanflow"
        result["ema"] = str(remote_path(raw["export"])) if raw.get("export") else None
        if result["ema"] and PurePosixPath(result["ema"]) != checkpoint / "export":
            raise ValueError("MeanFlow EMA export must belong to the named checkpoint")
        result["hashes"] = {"manifest": raw.get("manifest_sha256")}
    else:
        if raw.get("complete") is not True:
            raise ValueError("Torch source has not declared its complete atomic save")
        result["kind"] = "torch"
        result["ema"] = str(remote_path(raw["ema_path"]))
        result["config"] = str(remote_path(raw.get("config_path", raw.get("config", ""))))
        if checkpoint.name != identity + ".state.pt" or PurePosixPath(result["ema"]) != checkpoint.parent / (identity + ".ema.pt") or PurePosixPath(result["config"]) != checkpoint.parent / (identity + ".config.json"):
            raise ValueError("Torch state, EMA and configuration must share the immutable identity")
        hashes = raw.get("hashes", {})
        result["hashes"] = {"state": valid_hash(hashes.get("state", raw.get("state_sha256"))),
                            "ema": valid_hash(hashes.get("ema", raw.get("ema_sha256", raw.get("sha256")))),
                            "config": valid_hash(hashes.get("config"))}
    if event != "save" and not result["ema"]:
        raise ValueError("Evaluation request has no immutable EMA export")
    return result


def copy_bundle(reader: RemoteReader, files: list[dict], destination: Path, *, role: str, request: dict) -> dict:
    """Verify the committed source identities while streaming an atomic local bundle."""
    if not files:
        raise ValueError("Refusing an empty artifact bundle")
    receipt_name = "REPLICA.json"
    if destination.exists():
        receipt = json.loads((destination / receipt_name).read_text())
        expected = [(x["path"], x["sha256"]) for x in files]
        if receipt["identity"] != request["identity"] or expected != [(x["path"], x["sha256"]) for x in receipt["files"]]:
            raise ValueError("Immutable local artifact conflicts with source identity")
        for record in receipt["files"]:
            local = (destination / record["path"]).stat()
            if receipt.get("local_stat", {}).get(record["path"]) != {"bytes": local.st_size, "mtime_ns": local.st_mtime_ns}:
                raise ValueError("Existing verified local replica changed")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name("." + destination.name + ".partial")
    receipt = {"schema_version": 1, "managed_by": "safa_facegen.replicate", "role": role,
               "identity": request["identity"], "model_id": request["model_id"], "status": "transferring", "files": files}
    if staging.exists():
        if staging.is_symlink():
            raise ValueError("Partial staging directory must not be a symlink")
        previous = json.loads((staging / receipt_name).read_text())
        if previous.get("identity") != request["identity"] or [(x["path"],x["sha256"],x["bytes"]) for x in previous["files"]] != [(x["path"],x["sha256"],x["bytes"]) for x in files]:
            raise ValueError("Partial staging content belongs to a different source identity")
        receipt["verified_files"] = previous.get("verified_files", {})
    else:
        staging.mkdir()
    atomic_json(staging / receipt_name, receipt)
    remaining = 0
    for record in files:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["path"] == receipt_name:
            raise ValueError("Unsafe bundle relative path")
        local = staging / relative
        if local.is_symlink():
            raise ValueError("Partial bundle file must not be a symlink")
        remaining += max(0, record["bytes"] - (local.stat().st_size if local.is_file() else 0))
    if shutil.disk_usage(destination.parent).free < remaining + 16 * 1024 * 1024:
        raise OSError("Insufficient disk for the remaining staging transfer")
    try:
        for record in files:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts or record["path"] == receipt_name:
                raise ValueError("Unsafe bundle relative path")
            if reader.transfer_check is not None:
                reader.transfer_check()
            local = staging / relative
            verified = receipt.get("verified_files", {}).get(record["path"])
            info = local.stat() if local.exists() else None
            if verified and info and not local.is_symlink() and verified == {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}:
                continue
            reader.download(record, local)
            info = local.stat()
            receipt.setdefault("verified_files", {})[record["path"]] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
            atomic_json(staging / receipt_name, receipt)
        receipt.update(status="transport_verified", copied_at=time.time(), restore_exercised=False)
        receipt["local_stat"] = {record["path"]: {"bytes": (staging / record["path"]).stat().st_size,
                                "mtime_ns": (staging / record["path"]).stat().st_mtime_ns} for record in files}
        atomic_json(staging / receipt_name, receipt)
        fsync_directory(staging)
        os.replace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException as exc:
        receipt.update(status="paused" if isinstance(exc, TransferPaused) else "failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(staging / receipt_name, receipt)
        raise
    return receipt


def replicate_artifact(reader: RemoteReader, request: dict, root: Path, role: str) -> Path:
    destination = root / "models/replicas" / request["model_id"] / role / request["identity"]
    if request["kind"] == "meanflow":
        checkpoint = PurePosixPath(request["checkpoint"])
        complete = reader.json(checkpoint / "COMPLETE.json")
        manifest_bytes = reader.read(checkpoint / "manifest.json")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if manifest["model_id"] != request["model_id"]:
            raise ValueError("MeanFlow checkpoint metadata model mismatch")
        if manifest_hash != complete["manifest_sha256"] or request["hashes"].get("manifest") not in (None, manifest_hash):
            raise ValueError("MeanFlow checkpoint complete marker/hash mismatch")
        path = checkpoint if role == "restore" else PurePosixPath(request["ema"])
        expected = {}
        if role == "ema":
            export = reader.json(path / "manifest.json")
            expected = export["sha256"]
        else:
            if not manifest.get("state_files"):
                raise ValueError("MeanFlow restore requires checkpoint-registered state file identities")
            expected = {"state/" + row["path"]: valid_hash(row["sha256"]) for row in manifest["state_files"]}
            expected["manifest.json"] = manifest_hash
            if request.get("ema"):
                export = reader.json(PurePosixPath(request["ema"]) / "manifest.json")
                expected.update({"export/" + name: value for name, value in export["sha256"].items()})
        files = reader.inventory(path, expected_hashes=expected)
        if role == "restore":
            actual = {record["path"]: record for record in files}
            if "STATE_RETIRED.json" in actual:
                raise ValueError("MeanFlow source restore state has been retired")
            declared_state = {"state/" + row["path"]: row for row in manifest["state_files"]}
            actual_state = {name for name in actual if name.startswith("state/")}
            if actual_state != set(declared_state):
                raise ValueError("MeanFlow source restore state is incomplete or contains unregistered files")
            if any(actual[name]["bytes"] != row["size"] for name, row in declared_state.items()):
                raise ValueError("MeanFlow source restore state size differs from its committed metadata")
        if role == "ema":
            actual = {r["path"]: r["sha256"] for r in files}
            if actual.get("ema.safetensors") != valid_hash(export["sha256"]["ema.safetensors"]):
                raise ValueError("MeanFlow exported EMA hash mismatch")
        copy_bundle(reader, files, destination, role=role, request=request)
        if reader.json(checkpoint / "COMPLETE.json") != complete:
            raise ValueError("MeanFlow completion marker changed during transfer")
        return destination
    roles = ("state", "ema", "config") if role == "restore" else ("ema", "config")
    sources = {"state": request["checkpoint"], "ema": request["ema"], "config": request["config"]}
    files = []
    for item in roles:
        records = reader.inventory(sources[item], expected_hashes={PurePosixPath(sources[item]).name: request["hashes"][item]})
        if len(records) != 1 or records[0]["sha256"] != request["hashes"][item]:
            raise ValueError(f"Torch {item} source hash mismatch")
        files.extend(records)
    copy_bundle(reader, files, destination, role=role, request=request)
    return destination / PurePosixPath(request["checkpoint"] if role == "restore" else request["ema"]).name


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, model TEXT, identity TEXT, event TEXT, payload TEXT, status TEXT, error TEXT, updated REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS pointers (model TEXT, role TEXT, payload TEXT, PRIMARY KEY(model,role))")

    def recover_interrupted(self):
        with self.connection() as db:
            db.execute("UPDATE requests SET status='retry_transfer',error='Transfer interrupted; resume verified partial files' WHERE status='copying'")
            db.execute("UPDATE requests SET status='failed',error='Evaluation interrupted; explicit retry required' WHERE status='evaluating'")

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, request):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM requests WHERE id=?", (request["request_id"],)).fetchone()
            if row:
                old = json.loads(row["payload"])
                for field in ("model_id", "identity", "event", "checkpoint", "ema", "hashes", "seed"):
                    if old.get(field) != request.get(field):
                        raise ValueError("A request ID was reused for different immutable content")
                return
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,'received',NULL,?)", (request["request_id"], request["model_id"], request["identity"], request["event"], json.dumps(request), time.time()))

    def reject(self, model, line: bytes, error: str):
        digest = hashlib.sha256(line).hexdigest()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,'failed',?,?)",
                       (f"rejected-{model}-{digest}", model, "rejected", "rejected",
                        json.dumps({"line_sha256": digest}), error, time.time()))

    def rows(self, status):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT rowid,* FROM requests WHERE status=? ORDER BY rowid", (status,))]

    def update(self, request_id, status, *, request=None, error=None):
        with self.connection() as db:
            if request is None:
                db.execute("UPDATE requests SET status=?,error=?,updated=? WHERE id=?", (status, error, time.time(), request_id))
            else:
                db.execute("UPDATE requests SET status=?,payload=?,error=?,updated=? WHERE id=?", (status, json.dumps(request), error, time.time(), request_id))

    def pointer(self, model, role, payload):
        with self.connection() as db:
            db.execute("INSERT INTO pointers VALUES (?,?,?) ON CONFLICT(model,role) DO UPDATE SET payload=excluded.payload", (model, role, json.dumps(payload)))

    def get_pointer(self, model, role):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM pointers WHERE model=? AND role=?", (model, role)).fetchone()
            return json.loads(row["payload"]) if row else None

    def change_if(self, request_id, expected, status, error=None):
        with self.connection() as db:
            return db.execute("UPDATE requests SET status=?,error=?,updated=? WHERE id=? AND status=?", (status, error, time.time(), request_id, expected)).rowcount == 1


def retire_previous_restore(previous: dict | None, current: Path, root: Path, model: str):
    if previous is None or Path(previous["path"]) == current:
        return
    path = Path(previous["path"]).resolve()
    directory = path if path.is_dir() else path.parent
    permitted = (root / "models/replicas" / model / "restore").resolve()
    if directory.parent != permitted or directory.name != previous["identity"] or directory.is_symlink():
        raise ValueError("Previous restore directory failed the managed-storage boundary check")
    receipt = json.loads((directory / "REPLICA.json").read_text())
    if receipt.get("managed_by") != "safa_facegen.replicate" or receipt.get("role") != "restore" or receipt.get("status") != "transport_verified":
        raise ValueError("Refusing to retire a non-managed restore artifact")
    # EMA candidates and published review outputs have separate directories and are retained.
    shutil.rmtree(directory)


def poll_requests(reader: RemoteReader, journal: Journal):
    for model in MODEL_IDS:
        run = H100_ROOT / "runs" / model
        try:
            lines = reader.read(run / "requests.jsonl").splitlines(keepends=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            journal.reject(model, str(run / "requests.jsonl").encode(), f"{type(exc).__name__}: {exc}")
            continue
        for line in lines:
            if line.endswith(b"\n"):
                try:
                    raw = json.loads(line)
                    if raw.get("type", raw.get("event")) in ("save", "preview", "review"):
                        journal.add(normalize_request(raw, model))
                except Exception as exc:
                    journal.reject(model, line, f"{type(exc).__name__}: {exc}")


def transfer_pending(reader: RemoteReader, journal: Journal, root: Path):
    import paramiko

    def waiting():
        rows = journal.rows("received") + journal.rows("retry_transfer") + journal.rows("paused_transfer")
        return sorted(rows, key=lambda item: (TRANSFER_PRIORITY[item["event"]],
                      item["status"] == "received", item["rowid"]))

    def retry_protection():
        return [{"request_id": row["id"], "identity": row["identity"], "model_id": row["model"],
                 "source_checkpoint": json.loads(row["payload"])["checkpoint"], "status": row["status"]}
                for row in journal.rows("retry_transfer") + journal.rows("paused_transfer")]

    while True:
        # Rebuild the eligible queue after every completed or paused artifact.
        poll_requests(reader, journal)
        pending = waiting()
        for candidate in pending:
            if candidate["status"] == "received" and candidate["event"] in ("save", "preview"):
                if any(item["model"] == candidate["model"] and item["event"] == candidate["event"]
                       and item["rowid"] > candidate["rowid"] for item in pending):
                    journal.update(candidate["id"], "superseded", error="Newer unstarted request of the same kind exists")
        eligible = [item for item in waiting() if json.loads(item["payload"]).get("retry_after_unix", 0) <= time.time()]
        if not eligible:
            break
        row = eligible[0]
        request = json.loads(row["payload"])
        if not journal.change_if(row["id"], row["status"], "copying"):
            continue
        last_poll = time.monotonic()

        def check_priority():
            nonlocal last_poll
            if row["event"] == "review" or time.monotonic() - last_poll < 30:
                return
            poll_requests(reader, journal)
            last_poll = time.monotonic()
            if any(TRANSFER_PRIORITY[item["event"]] < TRANSFER_PRIORITY[row["event"]]
                   and json.loads(item["payload"]).get("retry_after_unix", 0) <= time.time()
                   for item in waiting()):
                raise TransferPaused("Higher priority EMA request arrived; preserve and resume this partial transfer")

        reader.transfer_check = check_priority
        retry = False
        try:
            reader.begin_transfer(request, retry_protection())
            role = "restore" if row["event"] == "save" else "ema"
            path = replicate_artifact(reader, request, root, role)
            request["local_path"] = str(path)
            bundle_dir = path if path.is_dir() else path.parent
            bundle = json.loads((bundle_dir / "REPLICA.json").read_text())
            acknowledgement = {"schema_version": 1, "request_id": row["id"], "model_id": row["model"],
                               "identity": row["identity"], "event": row["event"], "artifact_role": role,
                               "source_checkpoint": request["checkpoint"], "source_ema": request["ema"],
                               "source_declared_hashes": request["hashes"], "files": bundle["files"],
                               "local_path": str(path), "received_complete": True,
                               "checkpoint_received": role == "restore", "restore_exercised": False,
                               "received_at_unix": time.time()}
            reader.write_receipt(row["id"], acknowledgement)
            request["source_receipt"] = str(RECEIPT_ROOT / (row["id"] + ".json"))
            if role == "restore":
                previous = journal.get_pointer(row["model"], "latest_restore")
                journal.pointer(row["model"], "latest_restore", {"identity": row["identity"], "path": str(path), "restore_exercised": False})
                journal.update(row["id"], "transport_verified", request=request)
                try:
                    retire_previous_restore(previous, path, root, row["model"])
                except Exception as exc:
                    request["retention_warning"] = f"{type(exc).__name__}: {exc}"
                    journal.update(row["id"], "transport_verified", request=request)
            else:
                # Supersede only queued previews. Published reviews and their EMA files remain.
                if row["event"] == "preview":
                    for prior in journal.rows("queued"):
                        if prior["model"] == row["model"] and prior["event"] == "preview":
                            journal.change_if(prior["id"], "queued", "superseded", "Newer preview transferred before evaluation started")
                journal.update(row["id"], "queued", request=request)
        except TransferPaused as exc:
            # Persist this state before clearing active: source retention always
            # sees either active or retrying protection for the unfinished state.
            request.pop("retry_after_unix", None)
            journal.update(row["id"], "paused_transfer", request=request, error=str(exc))
        except Exception as exc:
            retry = isinstance(exc, (EOFError, ConnectionError, TimeoutError, paramiko.SSHException)) or (
                isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET, errno.ETIMEDOUT, errno.ENETUNREACH, errno.EHOSTUNREACH))
            if retry:
                request["transfer_attempts"] = int(request.get("transfer_attempts", 0)) + 1
                request["retry_after_unix"] = time.time() + min(900, 30 * 2 ** min(request["transfer_attempts"] - 1, 5))
            journal.update(row["id"], "retry_transfer" if retry else "failed", request=request, error=f"{type(exc).__name__}: {exc}")
        finally:
            reader.transfer_check = None
            with contextlib.suppress(Exception):
                reader.end_transfer(retry_protection())
        if retry:
            # Reconnect on the next poll; do not poison unrelated requests on a dead SSH channel.
            break


def local_dependency(root: Path, value: str) -> str:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root not in path.parents or not path.exists():
        raise ValueError(f"Evaluation dependency must exist within K100 project: {path}")
    return str(path)


def evaluate_worker(journal: Journal, root: Path, config: dict, stop: threading.Event):
    while not stop.is_set():
        rows = journal.rows("queued")
        if not rows:
            stop.wait(5)
            continue
        # A periodic review is not starved by newer lightweight previews.
        row = min(rows, key=lambda value: (value["event"] != "review", value["rowid"]))
        request = json.loads(row["payload"])
        if not journal.change_if(row["id"], "queued", "evaluating"):
            continue
        try:
            from .evaluate import evaluate, preview
            settings = config["models"][row["model"]]
            codec = local_dependency(root, settings["codec"]) if settings.get("codec") else None
            output = root / "reports/evaluation" / row["identity"] / row["event"]
            kwargs = {"model_id": row["model"], "checkpoint": request["local_path"], "output": output,
                      "codec": codec, "device": config.get("device", "cuda:0"),
                      "batch_size": int(settings.get("batch_size", config.get("batch_size", 8))),
                      "seed": request["seed"], "cpu_threads": int(config.get("cpu_threads", 8)),
                      "ema_sha256": request.get("hashes", {}).get("ema")}
            if row["event"] == "preview":
                result = preview(**kwargs)
            else:
                result = evaluate(**kwargs,
                    dataset_manifest=local_dependency(root, config["dataset_manifest"]),
                    dataset_manifest_sha256=config.get("dataset_manifest_sha256"),
                    image_root=local_dependency(root, config["image_root"]),
                    inception_weights=local_dependency(root, config["inception_weights"]),
                    face_detector=local_dependency(root, config["face_detector"]))
                # Publish only after the real generator and every required metric succeeded.
                journal.pointer(row["model"], "current_review", {"identity": row["identity"], "path": request["local_path"],
                                "ema_path": str(Path(request["local_path"]) / "ema.safetensors") if request["kind"] == "meanflow" else request["local_path"],
                                "source_ema_path": str(PurePosixPath(request["ema"]) / "ema.safetensors") if request["kind"] == "meanflow" else request["ema"],
                                "source_checkpoint": request["checkpoint"], "dataset_manifest_sha256": result["dataset_manifest_sha256"],
                                "report": str(output / "summary.json"), "ema_sha256": result["ema_sha256"], "awaiting_user_decision": True})
            request["report"] = str(output / "summary.json")
            journal.update(row["id"], "complete", request=request)
        except Exception as exc:
            journal.update(row["id"], "failed", error=f"{type(exc).__name__}: {exc}")


def _run(credentials: dict, config: dict, *, once=False):
    root = K100_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if once and config.get("evaluate", True):
        raise ValueError("--once requires evaluate:false; real evaluation needs the persistent worker")
    journal = Journal(root / "reports/replication/journal.sqlite3")
    journal.recover_interrupted()
    stop = threading.Event()
    thread = None
    if config.get("evaluate", True):
        thread = threading.Thread(target=evaluate_worker, args=(journal, root, config, stop), name="safa-evaluation", daemon=True)
        thread.start()
    try:
        while True:
            try:
                with RemoteReader(credentials) as reader:
                    poll_requests(reader, journal)
                    transfer_pending(reader, journal, root)
                if thread is not None and not thread.is_alive():
                    raise RuntimeError("Evaluation worker exited unexpectedly")
                atomic_json(root / "reports/replication/status.json", {"status": "one_shot_complete" if once else "running", "last_poll": time.time(),
                            "queue_depth": len(journal.rows("queued")), "failed_requests": len(journal.rows("failed")),
                            "retrying_transfers": len(journal.rows("retry_transfer")),
                            "paused_transfers": len(journal.rows("paused_transfer"))})
            except Exception as exc:
                # Authentication values never form part of errors or structured logs.
                atomic_json(root / "reports/replication/status.json", {"status": "failed", "time": time.time(), "error": f"{type(exc).__name__}: {exc}"})
                if once:
                    raise
            if once:
                break
            stop.wait(float(config.get("poll_seconds", 30)))
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=60)


def run(credentials: dict, config: dict, *, once=False):
    # The worker runs on Linux/K100. A second process must not claim the same GPU queue.
    import fcntl
    directory = K100_ROOT / "reports/replication"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        _run(credentials, config, once=once)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Non-secret local worker configuration")
    parser.add_argument("--once", action="store_true", help="One transfer poll; use evaluate:false for transfer-only validation")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if any(key in config for key in ("password", "credentials", "jump", "h100")):
        raise ValueError("SSH credentials belong only on stdin")
    line = sys.stdin.buffer.readline(64 * 1024 + 1)
    if len(line) > 64 * 1024:
        raise ValueError("Credential input too large")
    credentials = json.loads(line)
    del line
    run(credentials, config, once=args.once)


if __name__ == "__main__":
    main()
