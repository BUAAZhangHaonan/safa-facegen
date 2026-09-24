"""Start an authorized next campaign after a bounded controller has exited."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from .bounded_stage import append_event, read_registration, validate_journal
from .common import atomic_json, check_limits, git_commit, require_inside, utc_now
from .controller import ensure_no_existing_trainer, process_environment


def completion_matches(plan, status, latest):
    if status.get("status") != "bounded_stage_complete_pending_review":
        return False
    if (status.get("stage", {}).get("stage_id") != plan["source_stage_id"]
            or status.get("model_id") != plan["source_model"]
            or status.get("step") != plan["stop_step"]
            or latest.get("step") != plan["stop_step"]
            or latest.get("complete") is not True
            or status.get("checkpoint_id") != latest.get("checkpoint_id")):
        raise ValueError("Completion does not match the authorized source stage")
    return True


def run(root, plan_path):
    root = Path(root).resolve()
    plan_path = require_inside(plan_path, root)
    plan = json.loads(plan_path.read_text())
    if (plan["source_model"] != "Diffusion-LDM-UNet"
            or plan["target_model"] != "RectifiedFlow-NCSNpp"
            or plan.get("authorized") is not True):
        raise ValueError("This handoff requires explicit Diffusion-to-RF authorization")
    state_path = plan_path.with_suffix(".status.json")
    lock = plan_path.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") in ("dispatching", "started", "attention_required"):
            return  # A restart cannot duplicate a dispatch or erase a failure.
    registration = read_registration(root)
    if (registration["stage"]["stage_id"] != plan["source_stage_id"]
            or registration["stop_step"] != plan["stop_step"]):
        raise ValueError("Source budget registration changed")
    campaign_path = require_inside(root / plan["target_campaign"], root)
    campaign = json.loads(campaign_path.read_text())
    if campaign["model_order"] != [plan["target_model"]] or "bounded_stage" in campaign:
        raise ValueError("Next campaign must contain only the authorized RF model")
    events = root / "runs/controller/events.jsonl"
    atomic_json(state_path, {"status": "waiting", "plan": plan, "time": utc_now()})
    while True:
        status = json.loads((root / "runs/controller/status.json").read_text())
        latest = json.loads((root / "runs" / plan["source_model"] / "last.json").read_text())
        if completion_matches(plan, status, latest):
            # Completion is published just before the original controller exits.
            with (root / "runs/controller/controller.lock").open("a") as controller_lock:
                try:
                    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(5)
                    continue
                ensure_no_existing_trainer(root)
                for key in ("state_path", "ema_path", "config_path"):
                    path = require_inside(root / latest[key], root)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError("Final complete checkpoint is missing an artifact")
                requests = root / "runs" / plan["source_model"] / "requests.jsonl"
                validate_journal(requests)
                identity = latest["checkpoint_id"] + "-review"
                if not any(json.loads(line).get("request_id") == identity
                           for line in requests.read_text().splitlines()):
                    raise ValueError("Final Diffusion review has not been requested")
                used = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=10)
                values = [int(v) for v in used.splitlines()]
                level, _ = check_limits(campaign["limits"])
                if len(values) != 4 or max(values) > 1024 or level != "ok":
                    time.sleep(5)
                    continue
                record = {"status": "dispatching", "time": utc_now(), "plan": plan,
                          "source_checkpoint": latest["checkpoint_id"],
                          "quality_approval_written": False, "code_commit": git_commit(root)}
                atomic_json(state_path, record)
            # The next original controller acquires the same controller lock.
            with (root / "logs/training/RectifiedFlow-NCSNpp-controller.log").open("ab") as log:
                process = subprocess.Popen([
                    str(root / ".venv-torch/bin/python"), "-m", "safa_facegen.controller",
                    "--root", str(root), "--campaign", str(campaign_path)],
                    cwd=root, env=process_environment(root), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="started", controller_pid=process.pid, time=utc_now())
            atomic_json(state_path, record)
            append_event(events, "authorized_model_handoff", **{k: v for k, v in record.items() if k != "time"})
            if process.wait() != 0:
                record.update(status="attention_required", time=utc_now())
                atomic_json(state_path, record)
            return
        if status.get("status") in ("attention_required", "stopped"):
            atomic_json(state_path, {"status": "attention_required", "time": utc_now(),
                                    "reason": "Source requires diagnosis before handoff", "plan": plan})
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    run(Path(args.root), Path(args.plan))


if __name__ == "__main__":
    main()
