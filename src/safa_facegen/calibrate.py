"""Measure full four-GPU train/save peaks and select the largest passing batch."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .common import atomic_json, append_event, memory_bytes, project_root, require_inside, utc_now, validation_identity, git_commit
from .controller import gpu_snapshot, process_environment, training_command, read_json


def measure(root, source, microbatch, steps=12):
    config=copy.deepcopy(source)
    model_id=config['model_id']
    scratch=require_inside(root/'tmp/calibration'/f'{model_id}-batch{microbatch}',root)
    if scratch.exists():
        raise FileExistsError(f'Previous calibration must be inspected before replacing: {scratch}')
    scratch.mkdir(parents=True)
    config['project_root']=str(root)
    config['paths']['output']=str(scratch/'training')
    config['microbatch']=microbatch
    config['max_steps']=steps
    config['log_interval_steps']=1
    config['required_world_size']=4
    config['calibration']=True
    config['validation_scope']=True
    config['max_recovery_failures']=0
    for key in ('save_interval_seconds','preview_interval_seconds','review_interval_seconds'):
        config[key]=1e9
    for key,value in list(config['paths'].items()):
        if value and not Path(value).is_absolute():config['paths'][key]=str(root/value)
    config['paths'].pop('resume',None)
    if not model_id.startswith('MeanFlow-'):
        config['project_root']=str(scratch)
        config['code_commit']=git_commit(root)
        config['paths']['output']=str(scratch/'runs'/model_id)
    cfg=scratch/'config.json';atomic_json(cfg,config)
    command=training_command(root,cfg,config)
    if model_id.startswith('MeanFlow-'):command+=['--max-steps',str(steps)]
    log_path=scratch/'output.log'
    before=gpu_snapshot()
    if any(item['used_bytes']>2*2**30 for item in before):
        raise RuntimeError('GPU resources occupied before calibration')
    started=time.monotonic();peak_gpu=0;peak_memory=0;reason=None;stopped=None
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,cwd=root,env=process_environment(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while process.poll() is None:
        current_gpu=max(item['used_bytes'] for item in gpu_snapshot())
        current_memory=memory_bytes()
        peak_gpu=max(peak_gpu,current_gpu);peak_memory=max(peak_memory,current_memory)
        if current_gpu>config['limits']['gpu_gib']*2**30 or current_memory>=config['limits']['memory_soft_gib']*2**30:
            reason='resource_limit'
            if stopped is None:
                os.killpg(process.pid,signal.SIGTERM);stopped=time.monotonic()
        if stopped is not None and time.monotonic()-stopped>60:
            os.killpg(process.pid,signal.SIGKILL)
        if time.monotonic()-started>1800:
            reason='calibration_timeout';os.killpg(process.pid,signal.SIGKILL)
        time.sleep(1)
    code=process.wait();elapsed=time.monotonic()-started
    text=log_path.read_text(errors='replace')
    if code and reason is None:
        reason='out_of_memory' if any(token in text.lower() for token in ('out of memory','resource_exhausted','failed to allocate')) else 'runtime_error'
    marker=Path(config['paths']['output'])/('latest.json' if model_id.startswith('MeanFlow-') else 'last.json')
    if code==0 and not marker.exists():reason='missing_completed_checkpoint'
    result={'model_id':model_id,'microbatch':microbatch,'global_batch':4*microbatch,
            'steps':steps,'elapsed_seconds':elapsed,'peak_gpu_gib':peak_gpu/2**30,
            'peak_memory_gib':peak_memory/2**30,'exit_code':code,
            'status':'passed' if code==0 and reason is None else 'failed','reason':reason,'timestamp':utc_now()}
    return result,scratch,text


def calibrate(root,config,*,start_batch=4,max_batch=1024,steps=12):
    model_id=config['model_id'];report=root/'reports/validation'/model_id
    report.mkdir(parents=True,exist_ok=True)
    passed=None;batch=start_batch;results=[];upper=None;tested=set()
    while batch<=max_batch:
        if batch in tested:break
        tested.add(batch)
        result,scratch,log=measure(root,config,batch,steps)
        results.append(result)
        atomic_json(report/'batch-attempts.json',results)
        append_event(report/'calibration.jsonl','attempt',**result)
        # Keep a small textual verification record. Temporary models and probe code
        # are removed after the measurement has been captured.
        (report/f'batch{batch}.log').write_text(log,encoding='utf-8')
        if result['status']=='passed':passed=result
        elif result['reason'] not in ('out_of_memory','resource_limit'):
            raise RuntimeError(f'Calibration implementation failed: {result}; evidence={report}')
        shutil.rmtree(require_inside(scratch,root))
        if result['status']!='passed':upper=batch
        if upper is not None:
            lower=passed['microbatch'] if passed else 0
            if upper-lower<=1:break
            batch=(upper+lower)//2
            if batch<=lower:break
        else:
            batch*=2
    if passed is None:raise RuntimeError('No batch passed the configured resource limits')
    passed={**passed,'selection':'largest_passing_integer_microbatch','attempts':len(results),
            'validation_identity':validation_identity(root,config),'code_commit':git_commit(root),
            'next_failing_microbatch':upper}
    atomic_json(report/'batch.json',passed)
    return passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--root',default=str(project_root()))
    p.add_argument('--start-batch',type=int,default=4);p.add_argument('--max-batch',type=int,default=1024)
    p.add_argument('--steps',type=int,default=12)
    a=p.parse_args();root=Path(a.root).resolve()
    print(json.dumps(calibrate(root,read_json(a.config),start_batch=a.start_batch,max_batch=a.max_batch,steps=a.steps),indent=2))


if __name__=='__main__':main()
