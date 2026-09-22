"""Retire superseded H100 recovery states after verified K100 receipt."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time

from .common import MODEL_IDS, append_event, atomic_json, require_inside


def timestamp(identity, model):
    match = re.fullmatch(re.escape(model) + r'-\d+ep-(\d{8}T\d{6}(?:\d{6})?Z)', identity)
    if not match:
        raise ValueError('Unrecognized checkpoint identity')
    text = match.group(1)
    return datetime.strptime(text, '%Y%m%dT%H%M%S%fZ' if len(text) > 16 else '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def retire_states(root, model):
    root = Path(root).resolve()
    if model not in MODEL_IDS:
        raise ValueError(model)
    run = root / 'runs' / model
    transfer_file = root / 'reports/replication/active-transfer.json'
    pointer = run / ('latest.json' if model.startswith('MeanFlow-') else 'last.json')
    if not pointer.exists() or not transfer_file.exists():
        return []
    transfer = read(transfer_file)
    # A stale worker status gives no permission to retire a possibly active read.
    if time.time() - transfer['updated_at_unix'] > 180:
        return []
    protected = {item['identity'] for item in transfer.get('retrying', []) if item['model_id'] == model}
    if transfer.get('model_id') == model and transfer.get('status') == 'active':
        protected.add(transfer['identity'])
    latest = read(pointer)
    protected.add(latest.get('name', latest.get('checkpoint_id')))
    confirmed = []
    for receipt_path in (root / 'reports/replication/receipts').glob('*.json'):
        receipt = read(receipt_path)
        if (receipt.get('model_id') != model or receipt.get('event') != 'save'
                or receipt.get('artifact_role') != 'restore' or not receipt.get('checkpoint_received')
                or not receipt.get('received_complete')):
            continue
        source = require_inside(receipt['source_checkpoint'], run)
        identity = receipt['identity']
        if source.name != (identity if model.startswith('MeanFlow-') else identity + '.state.pt'):
            raise ValueError('K100 acknowledgement names a different source')
        confirmed.append((timestamp(identity, model), receipt))
    if not confirmed:
        return []
    # Older complete states can be retired once K100 has a newer verified restore.
    # States newer than that receipt remain local through transfer delays/failures.
    cutoff = max(item[0] for item in confirmed)
    candidates = (run / 'checkpoints').glob('*') if model.startswith('MeanFlow-') else run.glob('*.state.pt')
    retired = []
    for candidate in candidates:
        if candidate.name.startswith('.'):
            continue
        if candidate.is_symlink():
            raise ValueError('Checkpoint must not be a symbolic link')
        identity = candidate.name if candidate.is_dir() else candidate.name.removesuffix('.state.pt')
        if identity in protected or timestamp(identity, model) > cutoff:
            continue
        candidate = require_inside(candidate, run)
        if model.startswith('MeanFlow-'):
            manifest = candidate / 'manifest.json'
            if not (candidate / 'state').exists():
                continue
            complete = read(candidate / 'COMPLETE.json')
            if not complete.get('manifest_sha256'):
                raise ValueError('Retiring state has no complete marker')
            if (candidate / 'state').is_symlink():
                raise ValueError('Recovery state must not be a symlink')
            target = require_inside(candidate / 'state', candidate)
            atomic_json(candidate / 'STATE_RETIRED.json', {'reason': 'newer_K100_restore_verified', 'identity': identity,
                                                         'confirmed_restore': max(confirmed, key=lambda item: item[0])[1]['identity']})
            shutil.rmtree(target)
        else:
            metadata = read(candidate.with_name(identity + '.json'))
            if not metadata.get('state_sha256') or metadata.get('complete') is False:
                raise ValueError('Retiring state has no complete save record')
            candidate.unlink()
        retired.append(identity)
        append_event(root / 'runs/controller/events.jsonl', 'recovery_state_retired', model_id=model,
                     identity=identity, reason='newer_K100_restore_verified', ema_retained=True)
    return retired
