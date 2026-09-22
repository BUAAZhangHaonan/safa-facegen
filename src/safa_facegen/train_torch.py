"""Four-rank RF/LDM/LCD trainer with resumable state and immutable EMA exports.

Run via torchrun. The only preview/review action on this host is exporting state
and emitting callbacks; evaluation is dispatched to the storage/research host.
"""
from __future__ import annotations
import argparse
import copy
import gc
from itertools import islice
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .common import atomic_json, sha256_file, checkpoint_name, check_limits, git_commit, fsync_directory, append_event
from .data import HQDataset, CachedLatentDataset
from .torch_models.models import family, load_backbone, LDM_ID,registered_ema_sha256
from .torch_models.codec import read_checkpoint,registered_codec
from .torch_models.objectives import TrainingObjective


class ResumableDistributedSampler(DistributedSampler):
    """Skip committed samples before batching, including a changed batch size."""
    start_index=0

    def __iter__(self):
        return islice(super().__iter__(),self.start_index,None)

    def __len__(self):
        return self.num_samples-self.start_index


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as handle:
            torch.save(payload,handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(device):
    return {"python":random.getstate(), "numpy":np.random.get_state(),
            "torch_cpu":torch.get_rng_state(),
            "torch_cuda":torch.cuda.get_rng_state(device) if device.type=="cuda" else None}


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if device.type=="cuda":
        torch.cuda.set_rng_state(value["torch_cuda"],device)


@torch.no_grad()
def update_ema(target, source, decay):
    source_parameters = dict(source.named_parameters())
    for name,value in target.named_parameters():
        # Keep FP32 EMA even when the model forward uses autocast.
        value.lerp_(source_parameters[name].detach().float(),1-float(decay))
    for name,value in target.named_buffers():
        value.copy_(dict(source.named_buffers())[name])


def _emit(callbacks,event,payload):
    if callbacks is None:
        return None
    if isinstance(callbacks,dict):
        callback=callbacks.get(event)
        return callback(payload) if callback else None
    return callbacks(event,payload)


def _resolve(path,root):
    path=Path(path)
    return path if path.is_absolute() else root/path


def _reduce_flag(value,device):
    tensor=torch.tensor(int(value),device=device,dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
    return int(tensor.item())


STOP_REQUEST_REASONS = frozenset({
    "controller_signal", "user_approval", "host_memory", "gpu_memory", "controller_error"
})


def _stop_request_path(config,root,model_id):
    value=config.get("stop_request_path")
    if value is None:
        return None
    if not isinstance(value,str) or not value:
        raise ValueError("stop_request_path must be an absolute controller request path")
    path=Path(value)
    expected=root/"runs"/"controller"/(model_id+".stop.json")
    if not path.is_absolute() or path.resolve()!=expected:
        raise ValueError(f"stop_request_path must resolve to {expected}")
    return expected


def _read_stop_request(path,model_id):
    if path is None:
        return None
    if path.is_symlink():
        raise ValueError("Controller stop request must not be a symlink")
    try:
        with path.open(encoding="utf-8") as handle:
            text=handle.read(4097)
    except FileNotFoundError:
        return None
    if len(text)>4096:
        raise ValueError("Controller stop request exceeds 4096 characters")
    request=json.loads(text)
    if not isinstance(request,dict) or request.get("model_id")!=model_id:
        raise ValueError(f"Controller stop request must belong to {model_id}")
    reason=request.get("reason")
    if not isinstance(reason,str) or reason not in STOP_REQUEST_REASONS:
        raise ValueError(f"Unknown controller stop reason: {reason!r}")
    created=request.get("created_at")
    if not isinstance(created,str) or len(created)!=16:
        raise ValueError("Controller stop created_at must use YYYYMMDDTHHMMSSZ")
    datetime.strptime(created,"%Y%m%dT%H%M%SZ")
    return {"model_id":model_id,"reason":reason,"created_at":created}


def _synchronized_stop_request(path,model_id,rank,device):
    # Only rank zero reads the file. Broadcast errors too, so no rank exits while
    # its peers enter the next training collective.
    message=None
    if rank==0:
        try:
            request=_read_stop_request(path,model_id)
            if request is not None:
                message={"request":request}
        except (OSError,ValueError,TypeError) as exc:
            message={"error":f"{type(exc).__name__}: {exc}"}
    if not _reduce_flag(message is not None,device):
        return None
    messages=[message]
    if dist.is_initialized():
        dist.broadcast_object_list(messages,src=0,device=device)
    message=messages[0]
    if "error" in message:
        raise ValueError(f"Invalid controller stop request at {path}: {message['error']}")
    return message["request"]


def _recipe(config,paths,world):
    keys=("model_id","learning_rate","microbatch","gradient_accumulation_steps","precision",
          "ema_decay","weight_decay","adam_betas","gradient_clip","seed","num_workers","prefetch_factor")
    result={key:config.get(key) for key in keys}
    def identity(path):
        value=Path(path).resolve();info=value.stat()
        return {"path":str(value),"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}
    result.update(world_size=world,dataset_identity=identity(paths["dataset_manifest"]),
                  codec_sha256=registered_codec(paths["codec"])["sha256"] if paths.get("codec") else None,
                  teacher_sha256=(config.get("teacher_ema_sha256") or registered_ema_sha256(paths["teacher_checkpoint"]))
                      if paths.get("teacher_checkpoint") else None)
    for key in ("codec","latent_cache","teacher_checkpoint","initial_checkpoint"):
        if paths.get(key):result[key+"_identity"]=identity(paths[key])
    return result


def all_finite(net, optimizer=None, ema=None):
    tensors=list(net.parameters())+list(net.buffers())
    if ema is not None:
        tensors+=list(ema.parameters())+list(ema.buffers())
    if optimizer is not None:
        tensors += [value for state in optimizer.state.values() for value in state.values()
                    if isinstance(value,torch.Tensor)]
    groups={}
    for tensor in tensors:
        if tensor.is_floating_point() and tensor.numel():
            groups.setdefault(tensor.device,[]).append(tensor.detach())
    for values in groups.values():
        maxima=torch.stack([v.abs().amax() for v in values])
        if not bool(torch.isfinite(maxima).all().item()):
            return False
    return True


def run(config,callbacks=None):
    config=json.loads(json.dumps(config))
    if config.get("calibration"):
        config["validation_scope"]=True
    overrides=config.get("recovery_overrides",{})
    if set(overrides)-{"learning_rate","microbatch","precision","num_workers","prefetch_factor"}:
        raise ValueError("Unsupported recovery_overrides key")
    if overrides and not config.get("paths",{}).get("resume"):
        raise ValueError("recovery_overrides requires an explicit resume checkpoint")
    config.update(overrides)
    model_id=config["model_id"]
    kind=family(model_id)
    root=Path(config.get("project_root",Path(__file__).resolve().parents[2])).resolve()
    stop_request_path=_stop_request_path(config,root,model_id)
    world=int(os.environ.get("WORLD_SIZE","1"))
    rank=int(os.environ.get("RANK","0"))
    local_rank=int(os.environ.get("LOCAL_RANK","0"))
    device=torch.device(config.get("device",f"cuda:{local_rank}"))
    if device.type=="cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no automatic CPU fallback")
        torch.cuda.set_device(device)
        # Leave 3 GiB of the requested 72 GiB envelope for context/NCCL allocations.
        allocator_gib=max(1.,float(config["limits"].get("gpu_gib",72))-3.)
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,allocator_gib*2**30/total),device)
    if world>1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo")
    if config.get("required_world_size",4)!=world:
        raise ValueError(f"Configured world size {config.get('required_world_size',4)} differs from WORLD_SIZE={world}")
    code_revision=config.get("code_commit") or git_commit(root)
    paths={key:str(_resolve(value,root)) for key,value in config["paths"].items() if value}
    output=Path(paths["output"]).resolve()
    # Writes are confined to the deployed project; this also protects the H100 legacy bundle boundary.
    run_dir=root/"runs"/model_id
    if output!=run_dir and run_dir not in output.parents:
        raise ValueError("Training output must be runs/<model_id> or a child of it")
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    seed=int(config["seed"])
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    if device.type=="cuda":
        torch.cuda.manual_seed(seed+rank)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    microbatch=int(config["microbatch"])
    accumulation=int(config.get("gradient_accumulation_steps",1))
    if min(microbatch,accumulation)<1:
        raise ValueError("microbatch and gradient_accumulation_steps must be positive")
    if accumulation!=1:
        raise ValueError("This full-coverage trainer requires gradient_accumulation_steps=1")
    precision=config.get("precision","bf16")
    if precision not in ("fp32","bf16"):
        raise ValueError("Supported precisions: fp32,bf16")
    if precision=="bf16" and device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested on unsupported device")
    if kind=="rectified_flow":
        dataset=HQDataset(paths["dataset_manifest"],random_flip=True,seed=seed)
    else:
        dataset=CachedLatentDataset(paths["latent_cache"],random_flip=True,seed=seed)
        if dataset.manifest["representation"]!="prequant":
            raise ValueError("FFHQ LDM requires prequant VQ-f4 cache")
        dataset_metadata=json.loads(Path(paths["dataset_manifest"]).read_text(encoding="utf-8"))
        if len(dataset_metadata["records"])!=len(dataset):
            raise ValueError("Latent cache and dataset record counts differ")
        del dataset_metadata
        if dataset.manifest.get("dataset_manifest_stat"):
            info=Path(paths["dataset_manifest"]).stat()
            if dataset.manifest["dataset_manifest_stat"]!={"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}:
                raise ValueError("Dataset manifest metadata differs from its completed cache registration")
        if dataset.manifest["codec"]["sha256"]!=registered_codec(paths["codec"])["sha256"]:
            raise ValueError("Latent cache codec identity mismatch")
    if len(dataset)<microbatch*world:
        raise ValueError("Dataset smaller than one global microbatch")
    if len(dataset)%world:
        raise ValueError("Dataset count must be divisible by world size to avoid padded duplicate samples")
    sampler=ResumableDistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    workers=int(config.get("num_workers",4))
    loader_generator=torch.Generator().manual_seed(seed+rank+481516)
    loader_kwargs=dict(batch_size=microbatch,sampler=sampler,drop_last=False,num_workers=workers,
                       pin_memory=device.type=="cuda",generator=loader_generator)
    if workers:
        loader_kwargs.update(persistent_workers=True,prefetch_factor=int(config.get("prefetch_factor",2)))
    loader=DataLoader(dataset,**loader_kwargs)
    usable_batches=(len(loader)//accumulation)*accumulation
    if not usable_batches:
        raise ValueError("No complete gradient accumulation group")
    teacher=None;teacher_metadata=None
    if kind=="latent_consistency":
        if not paths.get("teacher_checkpoint"):
            raise ValueError("LCM requires project LDM teacher_checkpoint")
        payload=read_checkpoint(paths["teacher_checkpoint"])
        if payload.get("format")!="safa-facegen-ema-v1" or family(payload["model_id"])!="diffusion":
            raise ValueError("LCM teacher must be this project's LDM EMA export")
        if payload.get("codec_sha256")!=registered_codec(paths["codec"])["sha256"]:
            raise ValueError("LCM teacher and latent cache must use the same codec")
        teacher_metadata={key:payload.get(key) for key in ("model_id","state_role","step","samples_seen","codec_sha256")}
        teacher=load_backbone(LDM_ID,paths["teacher_checkpoint"],payload=payload).requires_grad_(False).eval()
        del payload
        net=(load_backbone(model_id,paths["initial_checkpoint"]) if paths.get("initial_checkpoint")
             else copy.deepcopy(teacher).requires_grad_(True))
    else:
        net=load_backbone(model_id,paths["initial_checkpoint"]).requires_grad_(True)
    # Do not unfreeze the official RF Fourier frequencies (requires_grad=False upstream).
    if kind=="rectified_flow":
        for name,param in net.named_parameters():
            if name.endswith(".W"):
                param.requires_grad_(False)
    net.to(device=device,dtype=torch.float32)
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    if teacher is not None:
        teacher.to(device=device,dtype=torch.float32)
    objective=TrainingObjective(net,kind,teacher,ema if teacher is not None else None).to(device)
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                lr=float(config["learning_rate"]),betas=tuple(config.get("adam_betas",[0.9,0.999])),
                weight_decay=float(config.get("weight_decay",0)),eps=1e-8)
    recipe=_recipe(config,paths,world)
    now=time.time()
    progress={"epoch":0,"next_batch":0,"step":0,"samples_seen":0,"samples_in_epoch_per_rank":0,
              "event_times_utc_seconds":{"save":now,"preview":now,"review":now}}
    resume=paths.get("resume")
    if resume:
        payload=read_checkpoint(resume)
        if payload.get("format")!="safa-facegen-train-v1":
            raise ValueError("Resume requires a complete training checkpoint")
        previous_recipe=payload["recipe"]
        differences={key for key in set(previous_recipe)|set(recipe) if previous_recipe.get(key)!=recipe.get(key)}
        if differences-set(overrides):
            raise ValueError(f"Resume recipe differs without explicit recovery authorization: {sorted(differences-set(overrides))}")
        net.load_state_dict(payload["model_state"],strict=True)
        ema.load_state_dict(payload["ema_state"],strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        progress=payload["progress"]
        if "event_times_utc_seconds" not in progress:
            raise ValueError("Resume checkpoint lacks absolute event schedule timestamps")
        consumed=int(progress["samples_in_epoch_per_rank"])
        if not 0<=consumed<=sampler.num_samples:
            raise ValueError("Resume sample position is outside this rank's epoch")
        if consumed==sampler.num_samples:
            progress["epoch"]+=1;progress["next_batch"]=0;progress["samples_in_epoch_per_rank"]=0
        if "learning_rate" in overrides:
            for group in optimizer.param_groups:
                group["lr"]=float(overrides["learning_rate"])
        restore_rng(payload["rng_by_rank"][rank],device)
        del payload
    train_module=DistributedDataParallel(objective,device_ids=[local_rank] if device.type=="cuda" else None,
            broadcast_buffers=False,find_unused_parameters=False) if world>1 else objective
    train_module.train()
    if _reduce_flag(not all_finite(net,optimizer,ema),device):
        raise FloatingPointError("Initial/resumed model or optimizer has non-finite values")
    request_stop=[False]
    def stop_signal(signum,frame):
        request_stop[0]=True
    old_handlers={sig:signal.signal(sig,stop_signal) for sig in (signal.SIGTERM,signal.SIGINT)}
    intervals={"save":float(config.get("save_interval_seconds",900)),
               "preview":float(config.get("preview_interval_seconds",1800)),
               "review":float(config.get("review_interval_seconds",7200))}
    latest=None
    stop_reason=[None]
    stop_without_save=[False]

    def release_prefetch():
        # Stop only this loader's own workers and release its queued host batches.
        iterator=getattr(loader,"_iterator",None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator=None
        gc.collect()
        if device.type=="cuda":torch.cuda.empty_cache()

    def publish(event,payload):
        if rank!=0:return
        append_event(run_dir/"events.jsonl",event,**payload)
        if event in ("save","preview","review") and not config.get("validation_scope"):
            request={**payload,"request_id":payload["checkpoint_id"]+"-"+event,"type":event}
            append_event(run_dir/"requests.jsonl",event,**request)
            with (run_dir/"requests.jsonl").open("rb") as handle:
                os.fsync(handle.fileno())
        _emit(callbacks,event,payload)

    def save(reason,scheduled_events=()):
        nonlocal latest
        # Calibration includes one full state/EMA serialization after optimizer
        # allocation, so its measured resource peak covers the real save path.
        if config.get("calibration") and reason!="complete":
            return True
        memory_status,memory_info=check_limits(config["limits"])
        memory_level=_reduce_flag({"ok":0,"soft":1,"hard":2}[memory_status],device)
        if memory_level==2:
            raise RuntimeError(f"Hard host-memory redline; checkpoint allocation prohibited: {memory_info}")
        if memory_level==1:
            stop_reason[0]="resource_limit"
            stop_without_save[0]=True
            release_prefetch()
            publish("stop_requested",{"model_id":model_id,"reason":"memory_soft","new_checkpoint":False,**memory_info})
            return False
        if _reduce_flag(not all_finite(net,optimizer,ema),device):
            raise FloatingPointError("Refusing to save a non-finite state as latest")
        progress["event_times_utc_seconds"]["save"]=time.time()
        for event in scheduled_events:
            progress["event_times_utc_seconds"][event]=time.time()
        rng=rng_state(device)
        rngs=[None]*world
        if world>1:
            dist.all_gather_object(rngs,rng)
        else:
            rngs[0]=rng
        if rank==0:
            name=checkpoint_name(model_id,progress["samples_seen"],len(dataset))
            if (output/(name+".state.pt")).exists():
                # Identity is immutable; wait for a distinct UTC second, never overwrite.
                time.sleep(1.05)
                name=checkpoint_name(model_id,progress["samples_seen"],len(dataset))
            state_path=output/(name+".state.pt")
            ema_path=output/(name+".ema.pt")
            config_path=output/(name+".config.json")
            metadata={"model_id":model_id,"config":config,"progress":dict(progress),
                      "step":progress["step"],"samples_seen":progress["samples_seen"],
                      "dataset_size":len(dataset),"codec_sha256":recipe["codec_sha256"],
                      "teacher_sha256":recipe["teacher_sha256"],"teacher":teacher_metadata,"code_commit":code_revision,
                      "stop_reason":stop_reason[0] if reason=="stop_requested" else None}
            atomic_torch_save({"format":"safa-facegen-train-v1",**metadata,"recipe":recipe,
                "model_state":net.state_dict(),"ema_state":ema.state_dict(),
                "optimizer_state":optimizer.state_dict(),"rng_by_rank":rngs,
                "dataloader":{"epoch":progress["epoch"],"next_batch":progress["next_batch"],
                              "samples_in_epoch_per_rank":progress["samples_in_epoch_per_rank"],
                              "sampler_seed":seed,"augmentation":"stateless_seed_epoch_index",
                              "usable_batches":usable_batches}},state_path)
            atomic_torch_save({"format":"safa-facegen-ema-v1",**metadata,
                              "state_role":"ema","model_state":ema.state_dict()},ema_path)
            atomic_json(config_path,config)
            # Production exports are hashed once for transport. Temporary numerical
            # acceptance checkpoints are compared directly and are never replicated.
            hashes={"ema":None,"state":None,"config":None} if config.get("validation_scope") else {
                "ema":sha256_file(ema_path),"state":sha256_file(state_path),"config":sha256_file(config_path)}
            latest={"checkpoint_id":name,"identity":name,"root":str(root),"complete":True,
                    "checkpoint":str(state_path.relative_to(root)),"state_path":str(state_path.relative_to(root)),
                    "ema_path":str(ema_path.relative_to(root)),"config_path":str(config_path.relative_to(root)),
                    "sha256":hashes["ema"],"state_sha256":hashes["state"],"state_role":"ema","hashes":hashes,
                    "ema_sha256":hashes["ema"],"step":progress["step"],"completed_epochs":progress["samples_seen"]//len(dataset),
                    "created_at_utc":datetime.now(timezone.utc).isoformat(),
                    "samples_seen":progress["samples_seen"],"reason":reason,"model_id":model_id,
                    "stop_reason":metadata["stop_reason"]}
            atomic_json(output/(name+".json"),latest)
            publish("save",latest)
            for event in scheduled_events:
                publish(event,latest)
            # Requests become durable before advancing the resumable pointer.
            atomic_json(run_dir/"last.json",latest)
        if world>1:
            dist.barrier()
        return True

    def needs_stop():
        limit_status,info=check_limits(config["limits"])
        level={"ok":0,"soft":1,"hard":2}[limit_status]
        if device.type=="cuda":
            allocated=torch.cuda.max_memory_allocated(device)
            info["gpu_peak_allocated_bytes"]=allocated
            if allocated>float(config["limits"].get("gpu_gib",72))*2**30:
                level=max(level,1)
        level=_reduce_flag(level,device)
        if level==2:
            raise RuntimeError(f"Hard host-memory redline; stop without allocating a new checkpoint: {info}")
        controller_request=_synchronized_stop_request(stop_request_path,model_id,rank,device)
        stop=_reduce_flag(request_stop[0] or controller_request is not None or level==1,device)
        if stop:
            stop_reason[0]=(controller_request["reason"] if controller_request else
                            "resource_limit" if level else "signal")
            stop_without_save[0]=bool(level or stop_reason[0] in {"host_memory","gpu_memory"})
            if rank==0:
                publish("stop_requested",{**info,"model_id":model_id,"reason":stop_reason[0],
                        "new_checkpoint":not stop_without_save[0],"controller_request":controller_request})
            if stop_without_save[0]:release_prefetch()
        return bool(stop)

    try:
        if overrides and rank==0:
            publish("recovery_overrides",{"model_id":model_id,"overrides":overrides,"resume":resume,
                    "epoch":progress["epoch"],"next_batch":progress["next_batch"],"recipe":recipe})
        if not resume or overrides:
            if not save("recovery" if overrides else "initialization"):
                return {"status":"stopped","reason":"memory_soft","resume_from_last_complete":True,**progress}
        while True:
            if (config.get("max_steps") is not None and progress["step"]>=int(config["max_steps"])) or (
                config.get("max_hq_epochs") is not None and progress["samples_seen"]>=float(config["max_hq_epochs"])*len(dataset)):
                return {"status":"complete",**progress}
            if needs_stop():
                if not stop_without_save[0]:save("stop_requested")
                return {"status":"stopped","reason":stop_reason[0],"resume_from_last_complete":True,**progress}
            epoch=progress["epoch"]
            sampler.set_epoch(epoch);dataset.set_epoch(epoch)
            sampler.start_index=int(progress["samples_in_epoch_per_rank"])
            optimizer.zero_grad(set_to_none=True)
            for batch_index,batch in enumerate(loader):
                boundary=(batch_index+1)%accumulation==0
                sync=train_module.no_sync() if world>1 and not boundary else __import__('contextlib').nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                        enabled=precision=="bf16" and device.type=="cuda"):
                        loss=train_module(batch.to(device,non_blocking=True))
                    if _reduce_flag(not bool(torch.isfinite(loss).item()),device):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite loss; no optimizer update or automatic retry")
                    (loss/accumulation).backward()
                if not boundary:
                    continue
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),float(config.get("gradient_clip",1.0)))
                if _reduce_flag(not bool(torch.isfinite(norm).item()),device):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite gradients; no optimizer update or automatic retry")
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if _reduce_flag(not all_finite(net,optimizer),device):
                    raise FloatingPointError("Non-finite parameter/optimizer after update; previous latest retained")
                update_ema(ema,net,float(config.get("ema_decay",0.999)))
                if _reduce_flag(not all_finite(ema),device):
                    raise FloatingPointError("Non-finite EMA after update; previous latest retained")
                progress["step"]+=1
                progress["samples_seen"]+=int(batch.shape[0])*world
                progress["samples_in_epoch_per_rank"]+=int(batch.shape[0])
                progress["next_batch"]+=1
                if rank==0 and progress["step"]%int(config.get("log_interval_steps",10))==0:
                    publish("metrics",{**progress,"model_id":model_id,"loss":float(loss.detach()),"grad_norm":float(norm),
                                               "learning_rate":optimizer.param_groups[0]["lr"]})
                now=time.time()
                due=torch.tensor([int(now-progress["event_times_utc_seconds"][event]>=intervals[event]) if rank==0 else 0
                                  for event in ("save","preview","review")],device=device)
                if world>1:
                    dist.broadcast(due,0)
                flags=dict(zip(("save","preview","review"),due.cpu().tolist()))
                completed=(config.get("max_steps") is not None and progress["step"]>=int(config["max_steps"])) or (
                    config.get("max_hq_epochs") is not None and progress["samples_seen"]>=float(config["max_hq_epochs"])*len(dataset))
                stop=needs_stop()
                if stop and stop_without_save[0]:
                    return {"status":"stopped","reason":stop_reason[0],"resume_from_last_complete":True,**progress}
                if any(flags.values()) or completed or stop:
                    if not save("stop_requested" if stop else "complete" if completed else "scheduled",
                                [event for event in ("preview","review") if flags[event]]):
                        return {"status":"stopped","reason":"memory_soft","resume_from_last_complete":True,**progress}
                if completed or stop:
                    return {"status":"stopped" if stop else "complete",
                            **({"reason":stop_reason[0]} if stop else {}),**progress}
            progress["epoch"]+=1
            progress["next_batch"]=0
            progress["samples_in_epoch_per_rank"]=0
    finally:
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    def log(event,payload):
        print(json.dumps({"event":event,**payload},allow_nan=False),flush=True)
    result=run(config,log)
    if int(os.environ.get("RANK","0"))==0:
        print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
