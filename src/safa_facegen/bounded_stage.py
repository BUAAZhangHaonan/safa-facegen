"""Optional, single-model, bounded continuation. No quality approval is written.

The production controller remains responsible for locking, four GPUs, limits,
recovery, and checkpoint publication. This module only adds a fixed stop budget
and an intentional LR reduction AFTER the existing resume/recovery resolution.
"""
from __future__ import annotations
import copy
import json
import math
import re
import os
from pathlib import Path

MODEL = "Diffusion-LDM-UNet"
SCHEMA = 1
FIELDS = {"schema_version", "stage_id", "model_id", "source_checkpoint_id",
          "source_step", "additional_steps", "learning_rate"}

# Each journal has one project writer. Remember the validated prefix and inspect
# only new bytes; an unchanged file does not need another full history scan.
_validated_journals: dict[str, tuple[int, int, int, int]] = {}


def _validate_open_journal(handle, path: Path) -> None:
    info = os.fstat(handle.fileno())
    identity = (info.st_dev, info.st_ino)
    cached = _validated_journals.get(str(path))
    start = 0
    if cached is not None and cached[:2] == identity:
        if info.st_size < cached[2]:
            raise ValueError(f"Journal was truncated: {path}")
        if info.st_size == cached[2] and info.st_mtime_ns == cached[3]:
            return
        if info.st_size > cached[2]:
            start = cached[2]
    handle.seek(start)
    for line in handle:
        if not line.endswith(b"\n"):
            raise ValueError(f"Incomplete journal at {path}; inspect before appending")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid journal at {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Journal record must be a JSON object: {path}")
    _validated_journals[str(path)] = (*identity, info.st_size, info.st_mtime_ns)


def validate_journal(path: Path) -> None:
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError(f"Journal must not be a symlink: {path}")
    try:
        with path.open("rb") as handle:
            _validate_open_journal(handle, path)
    except FileNotFoundError:
        _validated_journals.pop(str(path), None)


def append_event(path: Path, event: str, **payload) -> dict:
    """Reject incomplete/invalid existing bytes, including error-report writes."""
    from .common import utc_now
    path = Path(path).absolute()
    if path.is_symlink():
        raise ValueError(f"Journal must not be a symlink: {path}")
    record = {"time": utc_now(), "event": event, **payload}
    encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        _validate_open_journal(handle, path)
        handle.write(encoded)
        handle.flush()
        info = os.fstat(handle.fileno())
        _validated_journals[str(path)] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return record


def validate_stage(campaign: dict) -> dict | None:
    stage = campaign.get("bounded_stage")
    if stage is None:
        return None
    if not isinstance(stage, dict) or set(stage) != FIELDS:
        raise ValueError("bounded_stage has missing or unknown fields")
    if type(stage["schema_version"]) is not int or stage["schema_version"] != SCHEMA or stage["model_id"] != MODEL:
        raise ValueError("Only bounded Diffusion continuation is supported")
    if campaign.get("model_order") != [MODEL]:
        raise ValueError("A bounded campaign must contain Diffusion only; no automatic next model")
    for field in ("stage_id", "source_checkpoint_id"):
        value = stage[field]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"Invalid {field}")
    for field, lo, hi in (("source_step", 0, 10**12), ("additional_steps", 1, 30000)):
        value = stage[field]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"Invalid {field}")
    lr = stage["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 3e-6:
        raise ValueError("Bounded learning_rate must be positive and at most 3e-6")
    return copy.deepcopy(stage)


def goal(stage: dict) -> int:
    return stage["source_step"] + stage["additional_steps"]


def resolve_config(campaign: dict, config: dict, state: dict | None,
                   saved_config: dict | None, recovery: dict) -> dict:
    """Pure function: keep optimizer/RNG/cursor recovery intact; compare to SAVED LR.

    In a failure recovery the controller may already have reduced the runtime LR.
    Do not increase it. Explicit overrides must still describe differences from
    the source checkpoint, not differences from the already-resolved runtime.
    """
    stage = validate_stage(campaign)
    if stage is None:
        if (saved_config or {}).get("bounded_stage") is not None:
            raise ValueError("A bounded checkpoint cannot resume under an unbounded campaign")
        return config
    if config.get("model_id") != MODEL:
        raise ValueError("Bounded stage used for a different model")
    if not state or state.get("complete") is not True or state.get("model_id") != MODEL:
        raise ValueError("Bounded continuation requires a complete Diffusion checkpoint")
    if state.get("recovery_state_available") is False or not saved_config:
        raise ValueError("The complete recovery state/config must remain available")
    step = state.get("step")
    if type(step) is not int or not stage["source_step"] <= step <= goal(stage):
        raise ValueError("Checkpoint lies outside the immutable stage budget")
    at_source = state.get("checkpoint_id") == stage["source_checkpoint_id"]
    if at_source:
        if step != stage["source_step"]:
            raise ValueError("Source checkpoint step differs from the fixed anchor")
        if saved_config.get("bounded_stage") not in (None, stage):
            raise ValueError("Cannot chain another bounded stage without a separate decision")
    elif saved_config.get("bounded_stage") != stage:
        raise ValueError("Resume checkpoint is not a descendant of this bounded stage")
    if recovery.get("status") == "exhausted":
        raise ValueError("Exhausted recovery must not be reset by a bounded stage")
    result = copy.deepcopy(config)
    runtime_lr, saved_lr = float(result["learning_rate"]), float(saved_config["learning_rate"])
    if not all(math.isfinite(x) and x > 0 for x in (runtime_lr, saved_lr)):
        raise ValueError("Invalid saved/runtime learning rate")
    desired = min(runtime_lr, saved_lr, float(stage["learning_rate"]))
    changes = dict(result.get("recovery_overrides", {}))
    if desired != saved_lr:
        changes["learning_rate"] = desired
    else:
        changes.pop("learning_rate", None)
    result["learning_rate"] = desired
    result["max_steps"] = goal(stage)  # Absolute optimizer step; never latest+budget.
    result.pop("max_epochs", None)
    result.pop("max_hq_epochs", None)
    result["bounded_stage"] = stage
    if changes:
        result["recovery_overrides"] = changes
    else:
        result.pop("recovery_overrides", None)
    if at_source and recovery.get("status") == "idle":
        result["runtime_change_reason"] = "bounded_stage_started"
    return result


def registry_path(root: Path) -> Path:
    return Path(root) / "runs/controller" / (MODEL + ".bounded-stage.json")


def _read_json_file(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"Symlink is not allowed for stage metadata: {path}")
    data = path.read_bytes()
    if len(data) > 1024 * 1024:
        raise ValueError("Stage/config JSON exceeds 1 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def _read_saved_config(root: Path, state: dict) -> dict:
    from .common import require_inside
    value = Path(state["config_path"])
    original = value if value.is_absolute() else root / value
    if original.is_symlink():
        raise ValueError("Symlink in checkpoint configuration path")
    path = require_inside(original, root)
    return _read_json_file(path)


def read_registration(root: Path) -> dict | None:
    path = registry_path(root)
    if not path.exists() and not path.is_symlink():
        return None
    record = _read_json_file(path)
    stage = validate_stage({"model_order": [MODEL], "bounded_stage": record.get("stage")})
    if (record.get("schema_version") != 1 or stage is None or
            record.get("model_id") != MODEL or record.get("stop_step") != goal(stage)):
        raise ValueError("Invalid immutable stage registration")
    return record


def guard_campaign(root: Path, campaign: dict) -> None:
    """Block an old periodic launcher before it can resume or skip this model.

    An explicit campaign for another model is unaffected. The original six-model
    campaign includes Diffusion and must not silently bypass its bounded stage.
    """
    stage = validate_stage(campaign)
    if MODEL not in campaign.get("model_order", []):
        return
    registered = read_registration(root)
    if registered is not None and stage != registered["stage"]:
        raise ValueError("Diffusion has an immutable bounded stage; reuse its campaign")
    pointer = Path(root) / "runs" / MODEL / "last.json"
    if pointer.exists():
        saved = _read_saved_config(root, _read_json_file(pointer))
        previous = saved.get("bounded_stage")
        if previous is not None and previous != stage:
            raise ValueError("Checkpoint stage cannot be removed, replaced, or extended")


def bind_stage(root: Path, stage: dict) -> None:
    """Called under the existing controller lock, after resume validation.

    Publish a complete temporary JSON via an exclusive hard link. Existing
    registrations are never overwritten, including when two launchers race.
    """
    validate_stage({"model_order": [MODEL], "bounded_stage": stage})
    existing = read_registration(root)
    if existing is not None:
        if existing["stage"] != stage:
            raise ValueError("Cannot replace the registered stage or reset its budget")
        return
    from .common import utc_now, fsync_directory
    import tempfile
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "model_id": MODEL, "stage": stage,
              "stop_step": goal(stage), "registered_at_utc": utc_now()}
    fd, temp = tempfile.mkstemp(prefix=".bounded-stage-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, allow_nan=False, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temp, path)  # No replacement of an existing registration.
            fsync_directory(path.parent)
        except FileExistsError:
            existing = read_registration(root)
            if existing is None or existing["stage"] != stage:
                raise ValueError("Concurrent, incompatible stage registration")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def prepare_launch(root: Path, campaign: dict, config: dict, state: dict | None,
                   recovery: dict) -> dict:
    guard_campaign(root, campaign)
    stage = validate_stage(campaign)
    if stage is None:
        return config
    if not state:
        raise ValueError("No complete source checkpoint")
    saved = _read_saved_config(root, state)
    result = resolve_config(campaign, config, state, saved, recovery)
    bind_stage(root, stage)
    return result


def guard_trainer_launch(config: dict, root: Path) -> None:
    """Check registered stage even when a stale torchrun config omits resume."""
    if config.get("model_id") != MODEL:
        return
    requested = config.get("bounded_stage")
    record = read_registration(root)
    if record is not None and requested != record["stage"]:
        raise ValueError("Trainer cannot bypass the registered continuation budget")
    if requested is not None:
        stage = validate_stage({"model_order": [MODEL], "bounded_stage": requested})
        if record is None:
            raise ValueError("Bounded training must be registered by the controller")
        if not config.get("paths", {}).get("resume"):
            raise ValueError("Bounded training requires complete state resume, not reinitialization")
        if type(config.get("max_steps")) is not int or config["max_steps"] != goal(stage):
            raise ValueError("Trainer must preserve the exact absolute stop_step")


def validate_training_resume(config: dict, payload: dict, root: Path | None = None) -> None:
    """Enforce the same bound inside train_torch, not only in its controller.

    This is called before loading model/EMA/Adam state. Existing recipe equality
    and RNG/data-cursor restore checks remain in place in the original trainer.
    """
    previous = payload.get("config", {}).get("bounded_stage")
    requested = config.get("bounded_stage")
    if root is not None and config.get("model_id") == MODEL:
        record = read_registration(root)
        if record is not None and requested != record["stage"]:
            raise ValueError("Trainer cannot bypass the registered continuation budget")
        if requested is not None and record is None:
            raise ValueError("Bounded training must be registered by the controller")
    if previous is None and requested is None:
        return
    if requested is None or (previous is not None and previous != requested):
        raise ValueError("Trainer resume cannot erase or modify a bounded stage")
    stage = validate_stage({"model_order": [MODEL], "bounded_stage": requested})
    if config.get("model_id") != MODEL or payload.get("model_id") != MODEL:
        raise ValueError("Bounded trainer resume requires Diffusion states")
    step = payload.get("step")
    if (type(step) is not int or payload.get("progress", {}).get("step") != step or
            not stage["source_step"] <= step <= goal(stage)):
        raise ValueError("Resume progress is outside the immutable bound")
    if previous is None:
        source_file = Path(config.get("paths", {}).get("resume", "")).name
        if step != stage["source_step"] or source_file != stage["source_checkpoint_id"] + ".state.pt":
            raise ValueError("First bounded resume must use its exact complete source")
    if type(config.get("max_steps")) is not int or config["max_steps"] != goal(stage):
        raise ValueError("Trainer must preserve the exact absolute stop_step")
    if config.get("max_epochs") is not None or config.get("max_hq_epochs") is not None:
        raise ValueError("Conflicting epoch budget in bounded training")
    rate = config.get("learning_rate")
    saved_rate = payload.get("config", {}).get("learning_rate")
    if (type(rate) not in (int, float) or not math.isfinite(rate) or
            not 0 < rate <= stage["learning_rate"]):
        raise ValueError("Invalid bounded trainer learning rate")
    if previous is not None and saved_rate is not None and rate > float(saved_rate):
        raise ValueError("Resume must not raise a previously reduced learning rate")


def protected_identities(campaign: dict) -> tuple[str, ...]:
    stage = validate_stage(campaign)
    return (stage["source_checkpoint_id"],) if stage else ()


def _jsonl_has(path: Path, key: str, identity: str, expected: dict) -> bool:
    """Never append behind a truncated journal record; retain evidence and stop."""
    found = False
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete journal at {path}:{number}; inspect before appending")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid journal at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError("Journal record must be a JSON object")
            if row.get(key) == identity:
                if any(row.get(k) != v for k, v in expected.items()):
                    raise ValueError("Existing journal identity conflicts with final checkpoint")
                found = True
    return found


def _durable_append(path: Path, event: str, payload: dict) -> None:
    append_event(path, event, **payload)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def finish_if_budget_met(root: Path, campaign: dict, state: dict | None,
                         config: dict, events: Path) -> bool:
    stage = validate_stage(campaign)
    if stage is None or state is None or state.get("step", -1) < goal(stage):
        return False
    if (state.get("step") != goal(stage) or state.get("complete") is not True or
            state.get("model_id") != MODEL or config.get("bounded_stage") != stage):
        raise ValueError("Budget completion lacks an exact, complete stage checkpoint")
    from .common import atomic_json, utc_now, require_inside
    saved = _read_saved_config(root, state)
    if saved.get("bounded_stage") != stage:
        raise ValueError("Final checkpoint has no matching stage lineage")
    for key in ("state_path", "ema_path", "config_path"):
        value = Path(state[key])
        original = value if value.is_absolute() else root / value
        if original.is_symlink():
            raise ValueError("Final checkpoint must not be a symlink")
        path = require_inside(original, root)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Final checkpoint lacks {key}")
    registered = read_registration(root)
    if registered is None or registered["stage"] != stage:
        raise ValueError("Final checkpoint lacks its immutable stage registration")
    request_path = root / "runs" / MODEL / "requests.jsonl"
    request_id = state["checkpoint_id"] + "-review"
    request_identity = {"event": "review", "model_id": MODEL,
                        "checkpoint_id": state["checkpoint_id"]}
    queued = _jsonl_has(request_path, "request_id", request_id, request_identity)
    event_id = stage["stage_id"] + ":complete"
    event_identity = {"event": "bounded_stage_complete", "model_id": MODEL,
                      "checkpoint_id": state["checkpoint_id"]}
    recorded = _jsonl_has(events, "bounded_event_id", event_id, event_identity)
    if not queued:
        _durable_append(request_path, "review", {**state, "request_id": request_id, "type": "review"})
    status = {"status": "bounded_stage_complete_pending_review", "model_id": MODEL,
              "time": utc_now(), "stage": stage, "checkpoint_id": state["checkpoint_id"],
              "step": state["step"], "quality_approved": False, "next_model_started": False}
    if not recorded:
        _durable_append(events, "bounded_stage_complete",
                        {**{k: v for k, v in status.items() if k != "time"}, "bounded_event_id": event_id})
    atomic_json(root / "runs/controller/status.json", status)
    return True
