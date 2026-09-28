"""Quality-plan stage contracts, dispatched by the existing bounded controller.

Each model has one quality-v1 slot; LCM's branches share that slot. Legacy
registrations are read-only and are never replaced by a new recipe.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile

RECIPES = {
    'MeanFlow-B-2': ('imf_boundary_v1', 20000, 3e-5, .999, 64, 'fp32'),
    'MeanFlow-B-4': ('imf_boundary_v1', 15000, 3e-5, .999, 64, 'fp32'),
    'MeanFlow-L-2': ('imf_boundary_v1', 20000, 2e-5, .999, 64, 'fp32'),
    'Diffusion-LDM-UNet': ('min_snr_epsilon_v1', 15000, 3e-6, .999, 64, 'bf16'),
    'RectifiedFlow-NCSNpp': ('rf_batch_ot_v1', 10000, 3e-6, .999, 12, 'fp32'),
    'LatentConsistency-LDM-UNet': None,
}


def is_v2(value):
    return isinstance(value, dict) and value.get('schema_version') == 2


def validate(stage):
    fields = {'schema_version', 'stage_id', 'model_id', 'parent_checkpoint_id',
              'objective_id', 'source_step', 'stop_step', 'initialization_mode', 'recipe'}
    conditional = stage.get('model_id') in ('MeanFlow-B-4', 'MeanFlow-L-2', 'LatentConsistency-LDM-UNet') if isinstance(stage, dict) else False
    if conditional:
        fields.add('selection')
    if not is_v2(stage) or set(stage) != fields:
        raise ValueError('Invalid quality-stage fields')
    for key in ('stage_id', 'model_id', 'parent_checkpoint_id', 'objective_id'):
        if not isinstance(stage[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', stage[key]):
            raise ValueError(f'Invalid stage {key}')
    model = stage['model_id']
    if model not in RECIPES:
        raise ValueError('Unknown quality-stage model')
    expected = RECIPES[model]
    if model == 'LatentConsistency-LDM-UNet':
        expected = {'lcd_teacher_v2': ('lcd_teacher_v2', 15000, 5e-6, .95, 64, 'bf16'),
                    'lcf_real_v1': ('lcf_real_v1', 10000, 2e-6, .9999, 64, 'bf16')}.get(stage['objective_id'])
    if expected is None:
        raise ValueError('Unknown objective')
    obj, cap, peak, ema, batch, precision = expected
    if (stage['objective_id'] != obj or type(stage['source_step']) is not int or stage['source_step'] != 0 or
            type(stage['stop_step']) is not int or stage['stop_step'] != cap or
            stage['initialization_mode'] != 'ema_warm_start_new_optimizer'):
        raise ValueError('Quality-v1 stage cannot change its objective, zero origin, or fixed budget')
    recipe = stage['recipe']
    required = {'learning_rate': peak, 'end_learning_rate': peak / 10, 'warmup_steps': 500,
                'ema_decay': ema, 'microbatch': batch, 'precision': precision,
                'adam_betas': [.9, .95] if model.startswith('MeanFlow-') else [.9, .999],
                'weight_decay': 0.0}
    if not isinstance(recipe, dict) or set(recipe) != set(required):
        raise ValueError('Unknown or missing recipe fields')
    for key, value in required.items():
        actual = recipe[key]
        if isinstance(value, float):
            if type(actual) not in (int, float) or not math.isclose(actual, value, rel_tol=1e-12):
                raise ValueError(f'Quality-v1 {key} differs from authorized recipe')
        elif actual != value:
            raise ValueError(f'Quality-v1 {key} differs from authorized recipe')
    if conditional:
        from .quality.gate import select_better
        selection = stage['selection']
        if not isinstance(selection, dict) or set(selection) != {'candidate', 'baseline', 'decision'}:
            raise ValueError('Conditional stage needs actual immutable selection evidence')
        candidate, baseline = selection['candidate'], selection['baseline']
        result = select_better(candidate, baseline)
        required_model = 'Diffusion-LDM-UNet' if model == 'LatentConsistency-LDM-UNet' else 'MeanFlow-B-2'
        required_objective = 'min_snr_epsilon_v1' if model == 'LatentConsistency-LDM-UNet' else 'imf_boundary_v1'
        if (candidate.get('model_id') != required_model or candidate.get('objective_id') != required_objective or
                not candidate.get('checkpoint_id') or not candidate.get('stage_id') or
                result['decision'] != selection['decision'] or result['decision'] == 'REVIEW_REQUIRED'):
            raise ValueError('Conditional stage lacks applicable completed quality evidence')
        if stage['objective_id'] == 'lcf_real_v1':
            if result['decision'] != 'NOT_BETTER' or candidate.get('stage_complete') is not True:
                raise ValueError('LCF requires completed new Diffusion without selected improvement')
        elif result['decision'] != 'BETTER':
            raise ValueError('Promotion/re-distillation requires practical improvement')
    return copy.deepcopy(stage)


def registry_path(root, model):
    if model not in RECIPES:
        raise ValueError('Unknown model')
    return Path(root) / 'runs/controller/quality-v1' / (model + '.bounded-stage.json')


def read_registration(root, model):
    p = registry_path(root, model)
    if p.is_symlink():
        raise ValueError('Stage registration must not be symlink')
    if not p.exists():
        return None
    record = json.loads(p.read_text())
    stage = validate(record['stage'])
    if record.get('schema_version') != 2 or record.get('stop_step') != stage['stop_step']:
        raise ValueError('Invalid quality-stage registration')
    return record


def bind(root, stage):
    from .common import utc_now, fsync_directory
    validate(stage)
    p = registry_path(root, stage['model_id'])
    existing = read_registration(root, stage['model_id'])
    if existing:
        if existing['stage'] != stage:
            raise ValueError('Quality-v1 slot already registered; cannot replace branch or budget')
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.stage-', dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'schema_version': 2, 'stage': stage, 'stop_step': stage['stop_step'],
                       'registered_at_utc': utc_now()}, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        try:
            os.link(temp, p)
            fsync_directory(p.parent)
        except FileExistsError:
            if read_registration(root, stage['model_id'])['stage'] != stage:
                raise ValueError('Conflicting quality-stage registration')
    finally:
        os.unlink(temp)


def latest_path(root, stage):
    validate(stage)
    return Path(root) / 'runs' / stage['model_id'] / ('latest-' + stage['stage_id'] + '.json')


def saved_config(root, state):
    if state.get('model_id', '').startswith('MeanFlow-') or 'name' in state:
        return json.loads((Path(state['checkpoint']) / 'manifest.json').read_text())['config']
    p = Path(state['config_path'])
    return json.loads((p if p.is_absolute() else Path(root) / p).read_text())


def scheduled_learning_rate(config, step, recovery_scale=None):
    # The immutable contract is validated at admission and restore, not once per
    # optimizer update (conditional evidence includes all 256 review labels).
    stage = config['bounded_stage']
    if type(step) is not int or not 0 <= step < stage['stop_step']:
        raise ValueError('No update permitted at this absolute stage step')
    r = stage['recipe']
    scale = config.get('recovery_scale', 1.0) if recovery_scale is None else recovery_scale
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Invalid persisted recovery scale')
    if step < r['warmup_steps']:
        base = r['learning_rate'] * (step + 1) / r['warmup_steps']
    else:
        fraction = (step-r['warmup_steps'])/(stage['stop_step']-1-r['warmup_steps'])
        base = r['end_learning_rate'] + .5*(r['learning_rate']-r['end_learning_rate'])*(1+math.cos(math.pi*fraction))
    return base * scale


def resolve(root, campaign, config, state, recovery):
    stage = validate(campaign['bounded_stage'])
    if config['model_id'] != stage['model_id'] or recovery.get('status') == 'exhausted':
        raise ValueError('Stage model mismatch or exhausted recovery')
    result = copy.deepcopy(config)
    if state is not None:
        saved = saved_config(root, state)
        if saved.get('bounded_stage') != stage or not 0 <= state['step'] <= stage['stop_step']:
            raise ValueError('Resume is not a full state from this exact stage')
        if not state.get('complete', False):
            raise ValueError('Incomplete stage state')
        if not stage['model_id'].startswith('MeanFlow-'):
            result['paths']['resume'] = str(Path(root) / state['state_path'])
    else:
        if result['paths'].get('resume') or recovery.get('status') != 'idle':
            raise ValueError('Fresh stage cannot discard a recovery')
        # A removed stage pointer must not silently reset optimizer or budget.
        request_file = Path(root) / 'runs' / stage['model_id'] / 'requests.jsonl'
        if request_file.exists():
            for line in request_file.open(encoding='utf-8'):
                if json.loads(line).get('stage_id') == stage['stage_id']:
                    raise ValueError('Stage history exists without its latest full-state pointer')
    r = stage['recipe']
    scale = min(float(result.get('recovery_scale', 1)), float(result['learning_rate']) / r['learning_rate'])
    if not 0 < scale <= 1:
        raise ValueError('Runtime rate exceeds stage peak')
    result.update(bounded_stage=stage, objective_id=stage['objective_id'],
                  initialization_mode=stage['initialization_mode'], integrity_mode='metadata',
                  max_steps=stage['stop_step'], ema_decay=r['ema_decay'],
                  adam_betas=r['adam_betas'], weight_decay=r['weight_decay'], recovery_scale=scale,
                  lr_schedule={'peak': r['learning_rate'], 'end': r['end_learning_rate'],
                               'warmup': r['warmup_steps'], 'total_steps': stage['stop_step']})
    result.pop('max_epochs', None); result.pop('max_hq_epochs', None)
    bind(root, stage)
    return result


def guard_trainer(config, root):
    stage = validate(config['bounded_stage'])
    registered = read_registration(root, stage['model_id'])
    if not registered or registered['stage'] != stage or config.get('max_steps') != stage['stop_step']:
        raise ValueError('Trainer requires the controller-registered fixed stage')
    if config.get('objective_id') != stage['objective_id']:
        raise ValueError('Trainer objective differs from registration')


def guard_initialization(config, root, resume):
    """Direct trainer entry must not reset or rewind an existing stage."""
    stage = config['bounded_stage']
    pointer = latest_path(root, stage)
    if pointer.exists():
        state = json.loads(pointer.read_text())
        expected = state.get('checkpoint') if stage['model_id'].startswith('MeanFlow-') else state.get('state_path')
        if not resume or not expected:
            raise ValueError('Existing quality stage requires its latest full state')
        if resume != 'latest':
            actual = Path(resume)
            wanted = Path(expected)
            if not actual.is_absolute(): actual = Path(root) / actual
            if not wanted.is_absolute(): wanted = Path(root) / wanted
            if actual.resolve() != wanted.resolve():
                raise ValueError('Quality stage cannot rewind to an earlier state')
    else:
        if resume:
            raise ValueError('Resume requires the registered stage latest pointer')
        requests = Path(root) / 'runs' / stage['model_id'] / 'requests.jsonl'
        if requests.exists():
            with requests.open(encoding='utf-8') as stream:
                if any(json.loads(line).get('stage_id') == stage['stage_id'] for line in stream):
                    raise ValueError('Stage history exists without its latest pointer')


def validate_resume(config, payload, root):
    guard_trainer(config, root)
    stage = config['bounded_stage']
    if payload.get('config', {}).get('bounded_stage') != stage:
        raise ValueError('Cannot resume from another stage or parent optimizer')
    step = payload.get('step')
    if type(step) is not int or not 0 <= step <= stage['stop_step']:
        raise ValueError('Saved stage step outside budget')


def finish(root, campaign, state, config, events):
    from .bounded_stage import _jsonl_has, _durable_append
    from .common import atomic_json, utc_now
    stage = validate(campaign['bounded_stage'])
    if state is None or state.get('step', -1) < stage['stop_step']:
        return False
    guard_trainer(config, root)
    if (state['step'] != stage['stop_step'] or not state.get('complete') or
            saved_config(root, state).get('bounded_stage') != stage):
        raise ValueError('Stage cap requires its exact complete state')
    model = stage['model_id']
    cid = state['checkpoint_id']
    if model.startswith('MeanFlow-'):
        from .integrity import verify_meanflow_checkpoint
        verify_meanflow_checkpoint(Path(state['checkpoint']))
        if not state.get('export') or not (Path(state['export']) / 'ema.safetensors').is_file():
            raise ValueError('Final MeanFlow EMA export missing')
    else:
        for key in ('state_path', 'ema_path', 'config_path'):
            p = Path(root) / state[key]
            if not p.is_file() or p.stat().st_size <= 0:
                raise ValueError('Final full state missing ' + key)
    requests = Path(root) / 'runs' / model / 'requests.jsonl'
    rid = cid + '-review'
    expected = {'event': 'review', 'model_id': model, 'checkpoint_id': cid}
    if not _jsonl_has(requests, 'request_id', rid, expected):
        _durable_append(requests, 'review', {**state, 'request_id': rid, 'type': 'review',
                                           'codec': config['paths']['codec'], 'seed': config['seed']})
    eid = stage['stage_id'] + ':complete'
    status = dict(status='bounded_stage_complete_pending_review', model_id=model, stage=stage,
                  checkpoint_id=cid, step=state['step'], time=utc_now(), quality_approved=False,
                  next_model_started=False)
    if not _jsonl_has(events, 'bounded_event_id', eid,
                     dict(event='bounded_stage_complete', model_id=model, checkpoint_id=cid)):
        _durable_append(events, 'bounded_stage_complete', {**status, 'bounded_event_id': eid})
    atomic_json(Path(root) / 'runs/controller/status.json', status)
    return True
