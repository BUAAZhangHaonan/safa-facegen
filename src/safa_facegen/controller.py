"""Sequential four-GPU training with explicit human approvals and bounded recovery."""
from __future__ import annotations

import argparse
import copy
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import atomic_json, append_event, check_limits, git_commit, MODEL_IDS, project_root, sha256_file, utc_now, EXECUTION_PATHS
from .retention import retire_states, timestamp as checkpoint_timestamp


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
    config["configured_microbatch"] = config["microbatch"]
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


def gpu_snapshot(*, events=None):
    for attempt in range(1, 4):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                    text=True, capture_output=True, timeout=10, check=True)
            break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            if attempt == 3:
                raise RuntimeError('GPU memory query failed after three fresh attempts') from exc
            payload = {'attempt': attempt, 'next_attempt': attempt + 1, 'delay_seconds': 1,
                       'error': f'{type(exc).__name__}: {exc}'}
            if events is not None:
                append_event(events, 'gpu_query_retry', **payload)
            else:
                print(json.dumps({'event': 'gpu_query_retry', **payload}), flush=True)
            time.sleep(1)
    return [{"index": int(row[0]), "used_bytes": int(row[1]) * 2**20, "total_bytes": int(row[2]) * 2**20}
            for line in result.stdout.splitlines() if (row := [item.strip() for item in line.split(",")])]


def verify_launch(root, config, *, events=None):
    model_id = config["model_id"]
    if model_id not in MODEL_IDS or config.get("required_world_size") != 4:
        raise ValueError("Launch requires a registered model and four GPUs")
    if type(config.get("microbatch")) is not int or config["microbatch"] < 1:
        raise ValueError("Configured microbatch must be a positive integer")
    if config.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("The approved training recipe uses one gradient accumulation step")
    rate = config.get("learning_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    for key in ("dataset_manifest", "initial_checkpoint", "codec"):
        value = config["paths"].get(key)
        if value and not Path(value).exists():
            raise FileNotFoundError(value)
    if model_id != "RectifiedFlow-NCSNpp" and not Path(config["paths"]["latent_cache"]).is_file():
        raise FileNotFoundError(config["paths"]["latent_cache"])
    devices = gpu_snapshot(events=events)
    if [x["index"] for x in devices] != [0, 1, 2, 3]:
        raise RuntimeError("Expected exactly the four authorized H100 GPUs")
    return devices


def latest_state(root, model_id):
    output = root / "runs" / model_id
    path = output / ("latest.json" if model_id.startswith("MeanFlow-") else "last.json")
    return read_json(path) if path.exists() else None


def restore_torch_runtime(root, config, state):
    """Restore saved recovery settings after validating the original launch recipe."""
    if state is None or config['model_id'].startswith('MeanFlow-'):
        return config
    root = Path(root).resolve()
    model = config['model_id']
    if not isinstance(state, dict) or state.get('complete') is not True or state.get('model_id') != model:
        raise ValueError('Torch resume requires a complete checkpoint of the same model')
    identity = state.get('checkpoint_id')
    if not isinstance(identity, str) or state.get('identity') != identity:
        raise ValueError('Torch checkpoint identity is inconsistent')
    checkpoint_timestamp(identity, model)
    if state.get('root') != str(root) or state.get('recovery_state_available') is False:
        raise ValueError('Torch checkpoint project root or recovery availability is invalid')

    def committed_file(value, suffix):
        if not isinstance(value, str) or not value:
            raise ValueError('Torch checkpoint paths must be nonempty strings')
        path = Path(value)
        if '..' in path.parts:
            raise ValueError('Parent traversal in Torch checkpoint path')
        if not path.is_absolute():
            path = root / path
        expected = root / 'runs' / model / (identity + suffix)
        if path != expected:
            raise ValueError('Torch checkpoint file is outside its named model identity')
        for item in (path, *path.parents):
            if item.is_symlink():
                raise ValueError('Symlink in Torch checkpoint path')
            if item == root:
                break
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError('Torch checkpoint file is missing or empty')
        return path

    state_path = committed_file(state.get('state_path'), '.state.pt')
    if committed_file(state.get('checkpoint'), '.state.pt') != state_path:
        raise ValueError('Torch checkpoint and state_path disagree')
    config_path = committed_file(state.get('config_path'), '.config.json')
    with config_path.open('rb') as handle:
        encoded = handle.read(1024 * 1024 + 1)
    if len(encoded) > 1024 * 1024:
        raise ValueError('Saved Torch configuration exceeds the small-JSON limit')
    hashes = state.get('hashes')
    if not isinstance(hashes, dict):
        raise ValueError('Torch checkpoint has no committed configuration identity')
    expected_hash = hashes.get('config')
    if not isinstance(expected_hash, str) or hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError('Saved Torch configuration differs from its committed identity')
    saved = json.loads(encoded)
    if not isinstance(saved, dict) or saved.get('model_id') != model or saved.get('project_root') != str(root):
        raise ValueError('Saved Torch configuration model or project root differs')

    def integer(value, minimum, name):
        if type(value) is not int or value < minimum:
            raise ValueError(f'Invalid saved Torch {name}')
        return value

    batch = integer(saved.get('microbatch'), 1, 'microbatch')
    requested_batch = integer(config.get('microbatch'), 1, 'configured microbatch')
    previous_requested = integer(saved.get('configured_microbatch', batch), 1, 'previous configured microbatch')
    batch_changed = requested_batch != previous_requested
    if not batch_changed and batch > requested_batch:
        raise ValueError('Saved Torch microbatch exceeds the configured target')
    rate, original_rate = saved.get('learning_rate'), config.get('learning_rate')
    for value in (rate, original_rate):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid saved or original Torch learning_rate')
    if rate > original_rate:
        raise ValueError('Saved Torch learning_rate exceeds the original configuration')
    precision = saved.get('precision')
    if not isinstance(precision, str) or precision not in ('fp32', 'bf16'):
        raise ValueError('Invalid saved Torch precision')
    workers = integer(saved.get('num_workers'), 0, 'num_workers')
    prefetch = integer(saved.get('prefetch_factor'), 1, 'prefetch_factor')
    result = copy.deepcopy(config)
    result.update(microbatch=batch, learning_rate=rate, precision=precision,
                  num_workers=workers, prefetch_factor=prefetch)
    result.pop('recovery_overrides', None)
    result.pop('runtime_change_reason', None)
    if batch_changed and batch != requested_batch:
        result['microbatch'] = requested_batch
        result['recovery_overrides'] = {'microbatch': requested_batch}
        result['runtime_change_reason'] = 'configured_microbatch_changed'
    result['paths']['resume'] = str(state_path)
    return result


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


def stop_request_file(root, model_id):
    root = Path(root).resolve()
    if model_id not in MODEL_IDS:
        raise ValueError('Unknown model for stop request')
    path = root / 'runs/controller' / (model_id + '.stop.json')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Symlink in controller stop-request path')
        if item == root:
            break
    return path


def prepare_stop_request(root, config):
    if not config['model_id'].startswith('MeanFlow-'):
        path = stop_request_file(root, config['model_id'])
        path.unlink(missing_ok=True)
        config['stop_request_path'] = str(path)


def stop_training(process, config, reason, *, hard=False):
    if process.poll() is not None:
        return
    if hard:
        os.killpg(process.pid, signal.SIGKILL)
    elif config['model_id'].startswith('MeanFlow-'):
        os.kill(process.pid, signal.SIGTERM)
    else:
        path = stop_request_file(config['project_root'], config['model_id'])
        if config.get('stop_request_path') != str(path):
            raise ValueError('Torch stop_request_path differs from its project-local control file')
        if path.exists():
            if read_json(path).get('model_id') != config['model_id']:
                raise ValueError('Stop request belongs to a different model')
            return
        atomic_json(path, {'model_id': config['model_id'], 'reason': reason, 'created_at': utc_now()})


def log_tail(path, size=32000, start=0):
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(start, handle.tell() - size))
        return handle.read().decode("utf-8", errors="replace")


def recovery_config(config, state, reason):
    result = copy.deepcopy(config)
    result.pop('runtime_change_reason', None)
    changes = {}
    is_meanflow = result['model_id'].startswith('MeanFlow-')
    previous_batch = int(result['microbatch'])
    previous_learning_rate = float(result['learning_rate'])
    if is_meanflow and state:
        saved = read_json(Path(state['checkpoint']) / 'manifest.json')
        previous_batch = min(previous_batch, int(saved['progress']['microbatch']))
        previous_learning_rate = min(previous_learning_rate, float(saved['learning_rate']))
    if reason in ("gpu_memory", "out_of_memory"):
        changes["microbatch"] = max(1, previous_batch // 2)
        if changes["microbatch"] == previous_batch:
            raise RuntimeError("No smaller microbatch is available")
    elif reason == "nonfinite":
        changes["learning_rate"] = previous_learning_rate * 0.5
        changes["precision"] = "fp32"
    elif reason == "host_memory":
        if is_meanflow:
            raise RuntimeError('MeanFlow host-memory limit reached; direct memory-map input has no '
                               'prefetch workers to reduce. Inspect resident memory before restarting; '
                               'the latest complete checkpoint is retained.')
        changes["num_workers"] = max(0, result.get("num_workers", 4) // 2)
        changes["prefetch_factor"] = 1
    else:
        raise RuntimeError("Failure requires code diagnosis before recovery")
    result.update(changes)
    if state:
        result["recovery_overrides"] = changes
    else:
        # Before the first completed save, restart from the registered
        # initialization using the adjusted recipe and fresh optimizer.
        result.pop('recovery_overrides', None)
    if state and not result["model_id"].startswith("MeanFlow-"):
        result["paths"]["resume"] = state["state_path"]
    return result, changes


RECOVERABLE_FAILURES = {'gpu_memory', 'out_of_memory', 'nonfinite', 'host_memory'}
TORCH_RUNTIME_FIELDS = ('microbatch', 'learning_rate', 'precision', 'num_workers', 'prefetch_factor')


def runtime_parameters(config):
    fields = TORCH_RUNTIME_FIELDS[:3] if config['model_id'].startswith('MeanFlow-') else TORCH_RUNTIME_FIELDS
    return {key: config.get(key, 'fp32' if key == 'precision' else None) for key in fields}


def validate_runtime_parameters(parameters, bounds):
    if not isinstance(parameters, dict) or set(parameters) != set(runtime_parameters(bounds)):
        raise ValueError('Recovery parameters do not match the model runtime fields')
    for key, minimum in (('microbatch', 1), ('num_workers', 0), ('prefetch_factor', 1)):
        if key in parameters and (type(parameters[key]) is not int or parameters[key] < minimum):
            raise ValueError(f'Invalid persisted recovery {key}')
    if parameters['microbatch'] > bounds['microbatch']:
        raise ValueError('Recovery microbatch exceeds the configured target')
    rate = parameters['learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid persisted recovery learning rate')
    if rate > bounds['learning_rate'] and not (bounds['model_id'].startswith('MeanFlow-') and
                                              math.isclose(rate, bounds['learning_rate'], rel_tol=1e-6)):
        raise ValueError('Recovery learning rate exceeds the original configuration')
    allowed = ('fp32',) if bounds['model_id'].startswith('MeanFlow-') else ('fp32', 'bf16')
    if parameters['precision'] not in allowed:
        raise ValueError('Invalid persisted recovery precision')


def recovery_checkpoint_identity(state, model):
    if state is None:
        return None
    identity = Path(state['checkpoint']).name if model.startswith('MeanFlow-') else state.get('checkpoint_id')
    checkpoint_timestamp(identity, model)
    return identity


def checkpoint_runtime(root, bounds, state):
    if not bounds['model_id'].startswith('MeanFlow-'):
        return restore_torch_runtime(root, bounds, state)
    if state is None:
        return copy.deepcopy(bounds)
    model = bounds['model_id']
    identity = recovery_checkpoint_identity(state, model)
    path = Path(state['checkpoint'])
    if not path.is_absolute():
        path = root / path
    expected = root / 'runs' / model / 'checkpoints' / identity
    if path != expected or state.get('name', identity) != identity:
        raise ValueError('MeanFlow recovery checkpoint is outside its model identity')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Symlink in MeanFlow recovery checkpoint path')
        if item == root:
            break
    if not (path / 'COMPLETE.json').is_file() or (path / 'STATE_RETIRED.json').exists():
        raise ValueError('MeanFlow recovery checkpoint is unavailable or incomplete')
    saved = read_json(path / 'manifest.json')
    if saved.get('model_id') != model or saved.get('step') != state.get('step'):
        raise ValueError('MeanFlow recovery checkpoint metadata differs')
    result = copy.deepcopy(bounds)
    result.update(microbatch=saved['progress']['microbatch'], learning_rate=saved['learning_rate'], precision='fp32')
    validate_runtime_parameters(runtime_parameters(result), bounds)
    return result


def recovery_file(root, model):
    return stop_request_file(root, model).with_name(model + '.recovery.json')


def idle_recovery(model):
    return {'schema_version': 1, 'model_id': model, 'status': 'idle', 'failure_kind': None,
            'recoveries_scheduled': 0, 'recovery_origin_step': None}


def save_recovery(root, record):
    record['updated_at'] = utc_now()
    path = recovery_file(root, record['model_id'])
    if path.is_symlink():
        raise ValueError('Symlink in persisted recovery path')
    atomic_json(path, record)


def load_recovery(root, bounds, events, state, limit):
    model = bounds['model_id']
    path = recovery_file(root, model)
    if path.is_symlink():
        raise ValueError('Symlink in persisted recovery path')
    if path.exists():
        record = read_json(path)
    else:
        record = idle_recovery(model)
        # Older controllers journaled their recovery plan before launching it.
        # Only an explicit stable event clears that model's recovery history.
        scheduled = None
        failure = None
        if events.exists():
            with events.open(encoding='utf-8') as handle:
                for line in handle:
                    event = json.loads(line)
                    if event.get('model_id') != model:
                        continue
                    if event.get('event') == 'recovery_stable':
                        scheduled = failure = None
                    elif event.get('event') == 'recovery_scheduled':
                        scheduled, failure = event, None
                    elif event.get('event') == 'model_failure':
                        failure = event
        if scheduled is not None:
            parameters = runtime_parameters(checkpoint_runtime(root, bounds, state))
            changes = scheduled['changes']
            if not isinstance(changes, dict) or not set(changes) <= set(parameters):
                raise ValueError('Legacy recovery changes are invalid')
            parameters.update(changes)
            source = scheduled.get('recovery_state')
            record.update(status='pending', failure_kind=scheduled['reason'],
                          recoveries_scheduled=scheduled['attempt'],
                          recovery_origin_step=int((source or {}).get('step', 0)),
                          parameters=parameters, changes=changes,
                          source_identity=recovery_checkpoint_identity(source, model),
                          migrated_from_event_time=scheduled.get('time'))
            if failure is not None and failure.get('reason') == record['failure_kind']:
                # A recorded failure after the last plan has no durable next plan.
                # Reconstructing a new adjustment would hide an interrupted decision.
                record['status'] = 'exhausted'
                record['attention_reason'] = 'Legacy model failure has no subsequent recovery plan'
        elif failure is not None:
            raise ValueError('Legacy model failure has no durable recovery plan; inspect its event')
    if (not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('model_id') != model or
            record.get('status') not in ('idle', 'pending', 'running', 'exhausted')):
        raise ValueError('Invalid persisted recovery identity or status')
    count = record.get('recoveries_scheduled')
    if type(count) is not int or not 0 <= count <= limit:
        raise ValueError('Invalid persisted recovery budget')
    if record['status'] == 'idle':
        if count != 0 or record.get('failure_kind') is not None:
            raise ValueError('Idle recovery record contains an active budget')
    else:
        terminal = record['status'] == 'exhausted' and isinstance(record.get('attention_reason'), str) and bool(record['attention_reason'])
        valid_failure = record.get('failure_kind') in RECOVERABLE_FAILURES or (
            terminal and record.get('failure_kind') == 'trainer_recovery_exhausted')
        if not valid_failure or (count == 0 and not terminal):
            raise ValueError('Invalid persisted recovery failure kind')
        origin = record.get('recovery_origin_step')
        if type(origin) is not int or origin < 0:
            raise ValueError('Invalid persisted recovery origin')
        validate_runtime_parameters(record.get('parameters'), bounds)
        if record.get('source_identity') is not None:
            checkpoint_timestamp(record['source_identity'], model)
        changes = record.get('changes')
        if not isinstance(changes, dict) or (not changes and not terminal) or not set(changes) <= set(record['parameters']):
            raise ValueError('Invalid persisted recovery changes')
        if any(record['parameters'][key] != value for key, value in changes.items()):
            raise ValueError('Persisted recovery changes and parameters disagree')
    if not path.exists() and record['status'] != 'idle':
        save_recovery(root, record)
    return record


def apply_recovery_plan(root, bounds, state, record):
    if record['status'] == 'exhausted':
        raise RuntimeError(record.get('attention_reason', 'Two recovery attempts failed; current model stopped'))
    if record['status'] == 'idle':
        return checkpoint_runtime(root, bounds, state)
    config = checkpoint_runtime(root, bounds, state)
    identity = recovery_checkpoint_identity(state, bounds['model_id'])
    source = record['source_identity']
    if source is not None and (identity is None or checkpoint_timestamp(identity, bounds['model_id']) <
                              checkpoint_timestamp(source, bounds['model_id'])):
        raise ValueError('Recovery source checkpoint identity regressed or is missing')
    desired = record['parameters']
    actual = runtime_parameters(config)
    changes = {key: value for key, value in desired.items() if not
               (math.isclose(value, actual[key], rel_tol=1e-6) if key == 'learning_rate' else value == actual[key])}
    if state and identity != source and changes:
        if (bounds['model_id'].startswith('MeanFlow-') and actual['microbatch'] <= desired['microbatch'] and
                actual['learning_rate'] <= desired['learning_rate']):
            # MeanFlow can commit its own bounded OOM/nonfinite recovery between
            # controller polls. Its complete checkpoint records the resulting recipe.
            config.pop('recovery_overrides', None)
            return config
        raise ValueError('A newer checkpoint has an unexpected persisted recovery recipe')
    config.update(desired)
    config.pop('recovery_overrides', None)
    config.pop('runtime_change_reason', None)
    if state and changes:
        config['recovery_overrides'] = changes
    return config


def settle_recovery(root, bounds, record, completed, events):
    if record['status'] not in ('pending', 'running') or completed is None:
        return record
    if int(completed.get('step', 0)) < record['recovery_origin_step'] + 100:
        return record
    apply_recovery_plan(root, bounds, completed, record)
    previous = record
    record = idle_recovery(record['model_id'])
    save_recovery(root, record)
    append_event(events, 'recovery_stable', model_id=record['model_id'],
                 previous_failure=previous['failure_kind'], step=completed['step'])
    return record


def schedule_recovery(root, bounds, config, state, record, reason, limit):
    if reason not in RECOVERABLE_FAILURES:
        raise RuntimeError('Failure requires code diagnosis before recovery')
    used = record['recoveries_scheduled'] if record['failure_kind'] == reason else 0
    if used >= limit:
        record = dict(record, status='exhausted')
        save_recovery(root, record)
        raise RuntimeError('Two recovery attempts failed; current model stopped')
    try:
        changed, changes = recovery_config(config, state, reason)
    except RuntimeError as exc:
        stopped = dict(record, status='exhausted', failure_kind=reason, recoveries_scheduled=used,
                       recovery_origin_step=int((state or {}).get('step', 0)), parameters=runtime_parameters(config),
                       changes={}, source_identity=recovery_checkpoint_identity(state, config['model_id']),
                       attention_reason=str(exc))
        save_recovery(root, stopped)
        raise
    parameters = runtime_parameters(changed)
    validate_runtime_parameters(parameters, bounds)
    record = {'schema_version': 1, 'model_id': config['model_id'], 'status': 'pending',
              'failure_kind': reason, 'recoveries_scheduled': used + 1,
              'recovery_origin_step': int((state or {}).get('step', 0)),
              'parameters': parameters, 'changes': changes,
              'source_identity': recovery_checkpoint_identity(state, config['model_id'])}
    save_recovery(root, record)
    return record


def stop_exhausted_trainer(root, bounds, state, record, exit_code, tail):
    if not bounds['model_id'].startswith('MeanFlow-') or (exit_code != 78 and 'automatic recovery exhausted' not in tail):
        return
    actual = checkpoint_runtime(root, bounds, state)
    exhausted = dict(record, status='exhausted', failure_kind='trainer_recovery_exhausted',
                     recovery_origin_step=int((state or {}).get('step', 0)), parameters=runtime_parameters(actual),
                     changes={}, source_identity=recovery_checkpoint_identity(state, bounds['model_id']),
                     trainer_exit_code=exit_code, attention_reason='Trainer exhausted its bounded recovery; inspect events')
    save_recovery(root, exhausted)
    raise RuntimeError(exhausted['attention_reason'])


def ensure_no_existing_trainer(root):
    # The controller lock cannot cover a trainer orphaned by a controller SIGKILL.
    # Match the exact project runtime argument even before the trainer allocates a GPU.
    runtimes = {str(root / 'runs/controller' / (model + '.json')) for model in MODEL_IDS}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            arguments = (item / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if (runtimes.intersection(arguments) and any(module in arguments for module in
                    ('safa_facegen.train_torch', 'safa_facegen.meanflow.trainer')) and
                    (item / 'cwd').resolve() == root):
                raise RuntimeError(f'Existing project trainer PID {item.name} prevents a duplicate launch')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


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
            verify_launch(root, config, events=events)
            bounds = copy.deepcopy(config)
            recovery = load_recovery(root, bounds, events, latest_state(root, model_id),
                                     campaign['max_consecutive_recovery_failures'])
            while not stop_requested[0]:
                state = latest_state(root, model_id)
                recovery = settle_recovery(root, bounds, recovery, state, events)
                config = apply_recovery_plan(root, bounds, state, recovery)
                runtime = state_dir / (model_id + ".json")
                ensure_no_existing_trainer(root)
                prepare_stop_request(root, config)
                atomic_json(runtime, config)
                log_path = root / "logs/training" / (model_id + ".log")
                with log_path.open("ab", buffering=0) as log:
                    log_offset = log.tell()
                    process = subprocess.Popen(training_command(root, runtime, config, resume=bool(state)),
                                               cwd=root, env=process_environment(root), stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                if recovery['status'] != 'idle':
                    recovery['status'] = 'running'
                    recovery['trainer_pid'] = process.pid
                    save_recovery(root, recovery)
                append_event(events, "model_started", model_id=model_id, pid=process.pid,
                             microbatch=config["microbatch"], learning_rate=config["learning_rate"], resume=state)
                reason = None
                approval = None
                stop_sent = None
                peak_gpu = 0
                peak_memory = 0
                retention_checked = 0.0
                while process.poll() is None:
                    level, info = check_limits(config["limits"])
                    peak_memory = max(peak_memory, info["memory_bytes"])
                    if level == 'hard':
                        reason = 'host_memory'
                        stop_training(process, config, reason, hard=True)
                        break
                    if stop_sent is not None and time.monotonic() - stop_sent > 300:
                        stop_training(process, config, reason or 'stop_timeout', hard=True)
                        break
                    devices = gpu_snapshot(events=events)
                    peak_gpu = max(peak_gpu, max(x["used_bytes"] for x in devices))
                    approval = read_approval(root, model_id)
                    completed = latest_state(root, model_id)
                    if time.monotonic() - retention_checked > 60:
                        retire_states(root, model_id)
                        retention_checked = time.monotonic()
                    recovery = settle_recovery(root, bounds, recovery, completed, events)
                    if stop_requested[0] or approval:
                        if stop_sent is None:
                            stop_training(process, config, 'user_approval' if approval else 'controller_signal')
                            stop_sent = time.monotonic()
                    elif level == "soft" or max(x["used_bytes"] for x in devices) > config["limits"]["gpu_gib"] * 2**30:
                        reason = "host_memory" if level == "soft" else "gpu_memory"
                        if stop_sent is None:
                            stop_training(process, config, reason)
                            stop_sent = time.monotonic()
                    atomic_json(state_dir / "status.json", {"status": "stopping" if stop_sent else "training",
                                "model_id": model_id, "pid": process.pid, "code_commit": revision,
                                "updated_at": utc_now(), "gpu": devices, **info,
                                "peak_gpu_gib": peak_gpu / 2**30, "peak_memory_gib": peak_memory / 2**30,
                                "approval_received": bool(approval), "recovery_attempt": recovery['recoveries_scheduled']})
                    time.sleep(5)
                exit_code = process.wait()
                process = None
                tail = log_tail(log_path, start=log_offset).lower()
                stop_exhausted_trainer(root, bounds, latest_state(root, model_id), recovery, exit_code, tail)
                if stop_requested[0]:
                    atomic_json(state_dir / "status.json", {"status": "stopped", "model_id": model_id, "time": utc_now()})
                    return
                if approval:
                    atomic_json(root / "models/formal" / (model_id + ".json"), approval)
                    append_event(events, "model_approved", model_id=model_id, approval=approval, exit_code=exit_code)
                    break
                if reason is None:
                    if "gpu_memory_stop" in tail:
                        reason = "gpu_memory"
                    elif "out of memory" in tail or "resource_exhausted" in tail:
                        reason = "out_of_memory"
                    elif "non-finite" in tail or "nonfinite" in tail:
                        reason = "nonfinite"
                    elif any(token in tail for token in ('resource_limit', 'memory_stop',
                                                         'ram soft limit', 'ram hard limit', 'host-memory')):
                        reason = "host_memory"
                    else:
                        reason = "runtime_error"
                recovery_state = latest_state(root, model_id)
                recovery = settle_recovery(root, bounds, recovery, recovery_state, events)
                attempts = recovery['recoveries_scheduled'] + 1 if reason == recovery['failure_kind'] else 1
                checkpoint_runtime(root, bounds, recovery_state)
                try:
                    recovery = schedule_recovery(root, bounds, config, recovery_state, recovery, reason,
                                                 campaign['max_consecutive_recovery_failures'])
                finally:
                    append_event(events, "model_failure", model_id=model_id, reason=reason, exit_code=exit_code,
                                 consecutive_failures=attempts, log_path=str(log_path))
                append_event(events, "recovery_scheduled", model_id=model_id, reason=reason, changes=recovery['changes'],
                             attempt=recovery['recoveries_scheduled'], recovery_state=recovery_state,
                             initialization=config['paths'].get('initial_checkpoint') if not recovery_state else None)
            if stop_requested[0]:
                return
        atomic_json(state_dir / "status.json", {"status": "all_models_approved", "time": utc_now()})
    except BaseException as exc:
        if process is not None:
            stop_training(process, config, 'controller_error')
            try:
                process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                stop_training(process, config, 'stop_timeout', hard=True)
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
