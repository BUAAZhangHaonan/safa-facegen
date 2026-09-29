"""Explicit legacy and quality-stage objectives; forwards return scalar loss."""
from __future__ import annotations
from ..contracts import DIFFUSION_ORIGINAL, RECTIFIED_FLOW_ORIGINAL, LATENT_CONSISTENCY_ORIGINAL, MIN_SNR_EPSILON, RF_BATCH_OT, LCF_REAL, LCD_TEACHER

import torch
from torch import distributed as dist, nn
from .models import DiffusionSchedule
from .vendor import lcd_math


LEGACY_OBJECTIVES = {
    RECTIFIED_FLOW_ORIGINAL: RECTIFIED_FLOW_ORIGINAL,
    DIFFUSION_ORIGINAL: DIFFUSION_ORIGINAL,
    LATENT_CONSISTENCY_ORIGINAL: LATENT_CONSISTENCY_ORIGINAL,
}
OBJECTIVE_FAMILIES = {
    **{value: key for key, value in LEGACY_OBJECTIVES.items()},
    MIN_SNR_EPSILON: DIFFUSION_ORIGINAL,
    RF_BATCH_OT: RECTIFIED_FLOW_ORIGINAL,
    LCF_REAL: LATENT_CONSISTENCY_ORIGINAL,
    LCD_TEACHER: LATENT_CONSISTENCY_ORIGINAL,
}


def resolve_objective(kind, objective_id=None):
    if kind not in LEGACY_OBJECTIVES:
        raise ValueError(f"Unknown training family: {kind}")
    selected = LEGACY_OBJECTIVES[kind] if objective_id is None else objective_id
    if OBJECTIVE_FAMILIES.get(selected) != kind:
        raise ValueError(f"Unknown or incompatible objective {selected!r} for {kind}")
    return selected


def min_snr_epsilon_loss(prediction, epsilon, alpha_cumprod, timesteps, gamma=5.0):
    """Weight examples before averaging: epsilon weight is min(SNR,gamma)/SNR."""
    if gamma <= 0 or prediction.shape != epsilon.shape:
        raise ValueError("Invalid Min-SNR gamma or prediction shape")
    alpha = alpha_cumprod.to(device=prediction.device, dtype=torch.float32)[timesteps]
    snr = alpha / (1 - alpha).clamp_min(1e-12)
    weights = snr.clamp_max(gamma) / snr.clamp_min(1e-12)
    per_example = (prediction.float() - epsilon.float()).square().flatten(1).mean(1)
    loss = (weights * per_example).mean()
    return loss, {"epsilon_mse": per_example.detach().mean(),
                  "weight_mean": weights.detach().mean(), "weighted_loss": loss.detach()}


@torch.no_grad()
def ot_noise_for_clean(clean, noise, *, max_global_batch=48):
    """One-to-one global current-batch OT, retaining the clean sample/rank order."""
    if clean.shape != noise.shape or clean.ndim != 4 or clean.device != noise.device:
        raise ValueError("Expected equal [B,C,H,W] clean/noise tensors on one device")
    if type(max_global_batch) is not int or max_global_batch < 1 or clean.shape[0] < 1:
        raise ValueError("Invalid OT batch limit")
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if distributed:
        local_n = torch.tensor([clean.shape[0]], device=clean.device, dtype=torch.long)
        sizes = [torch.empty_like(local_n) for _ in range(world)]
        dist.all_gather(sizes, local_n)
        if any(int(size.item()) != clean.shape[0] for size in sizes):
            raise ValueError("Global OT requires equal unpadded local batch sizes")
    if clean.shape[0] * world > max_global_batch:
        raise ValueError("Global OT batch exceeds the registered current-batch limit")
    if distributed:
        clean_parts = [torch.empty_like(clean) for _ in range(world)]
        noise_parts = [torch.empty_like(noise) for _ in range(world)]
        dist.all_gather(clean_parts, clean.contiguous())
        dist.all_gather(noise_parts, noise.contiguous())
        full_clean, full_noise = torch.cat(clean_parts), torch.cat(noise_parts)
    else:
        full_clean, full_noise = clean, noise
    count = full_clean.shape[0]
    assignment = torch.empty(count, dtype=torch.long, device=clean.device)
    ok = torch.ones(1, dtype=torch.int32, device=clean.device)
    error = None
    if rank == 0:
        try:
            from scipy.optimize import linear_sum_assignment
            with torch.autocast(device_type=clean.device.type, enabled=False):
                a, b = full_clean.float().flatten(1), full_noise.float().flatten(1)
                cost = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T) / a.shape[1]
            if not bool(torch.isfinite(cost).all()):
                raise FloatingPointError("Nonfinite minibatch transport cost")
            rows, columns = linear_sum_assignment(cost.cpu().double().numpy())
            if list(rows) != list(range(count)) or len(set(columns.tolist())) != count:
                raise RuntimeError("Transport assignment is not a complete permutation")
            assignment.copy_(torch.as_tensor(columns, device=clean.device, dtype=torch.long))
        except Exception as exc:
            error = exc
            ok.zero_()
    if distributed:
        dist.broadcast(ok, src=0)
    if not bool(ok.item()):
        raise RuntimeError("Global minibatch OT failed on rank zero") from error
    if distributed:
        dist.broadcast(assignment, src=0)
    start = rank * clean.shape[0]
    return full_noise[assignment[start:start + clean.shape[0]]].contiguous()


def rf_ot_loss(net, clean, *, max_global_batch=48):
    noise = ot_noise_for_clean(clean, torch.randn_like(clean), max_global_batch=max_global_batch)
    t = torch.rand(clean.shape[0], device=clean.device, dtype=torch.float32) * 0.999 + 0.001
    tt = t[:, None, None, None]
    prediction = net(tt * clean.float() + (1 - tt) * noise.float(), t * 999).float()
    residual = (prediction - (clean.float() - noise.float())).square().flatten(1).mean(1)
    return residual.mean(), {"velocity_mse": residual.detach().mean(), "t_mean": t.detach().mean()}


def lcm_boundary(net, noisy, timesteps, schedule):
    origin = schedule.origin(noisy, net(noisy, timesteps).float(), timesteps)
    scaled = timesteps.float() * 10.0
    skip = 0.25 / (scaled.square() + 0.25)
    out = scaled / (scaled.square() + 0.25).sqrt()
    return skip[:, None, None, None] * noisy.float() + out[:, None, None, None] * origin


def lcf_loss(student, target, clean, schedule, *, skip=20, timesteps=None, noise=None):
    """Real-data consistency: shared clean/noise at high t and t-skip, no teacher."""
    if type(skip) is not int or not 1 <= skip < 1000:
        raise ValueError("LCF skip must be in [1,999]")
    if timesteps is None:
        timesteps = torch.randint(skip, 1000, (clean.shape[0],), device=clean.device)
    if timesteps.shape != (clean.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("LCF timesteps must be long[B]")
    if bool((timesteps < skip).any()) or bool((timesteps >= 1000).any()):
        raise ValueError("LCF high timesteps out of range")
    noise = torch.randn_like(clean) if noise is None else noise
    if noise.shape != clean.shape:
        raise ValueError("LCF noise shape differs from clean shape")
    low = timesteps - skip
    high_input = schedule.add_noise(clean.float(), noise.float(), timesteps)
    low_input = schedule.add_noise(clean.float(), noise.float(), low)
    prediction = lcm_boundary(student, high_input, timesteps, schedule)
    target.eval()
    with torch.no_grad():
        target_value = lcm_boundary(target, low_input, low, schedule)
    residual = (prediction.float() - target_value.float()).square().flatten(1).mean(1)
    return residual.mean(), {"consistency_mse": residual.detach().mean(),
                             "t_high_mean": timesteps.float().detach().mean()}


class TrainingObjective(nn.Module):
    def __init__(self, net, kind, teacher=None, target=None, *, ddim_steps=50,
                 objective_id=None, gamma=5.0, lcf_skip=20, ot_global_batch=48):
        super().__init__()
        self.net = net
        self.kind = kind
        self.objective_id = resolve_objective(kind, objective_id)
        # Teacher and target are registered but frozen; DDP sees only student gradients.
        self.teacher = teacher
        self.target = target
        self.schedule = None if kind == RECTIFIED_FLOW_ORIGINAL else DiffusionSchedule()
        self.ddim_steps = ddim_steps
        self.gamma = float(gamma)
        self.lcf_skip = lcf_skip
        self.ot_global_batch = ot_global_batch
        self.last_metrics = {}
        if kind == LATENT_CONSISTENCY_ORIGINAL:
            if target is None:
                raise ValueError("LCM requires an FP32 EMA student target")
            target.requires_grad_(False).eval()
            if self.objective_id == LCF_REAL:
                if teacher is not None:
                    raise ValueError("LCF real-data objective must not load a teacher")
            else:
                if teacher is None:
                    raise ValueError("LCD requires a frozen project LDM teacher")
                teacher.requires_grad_(False).eval()
                self.solver = lcd_math().DDIMSolver(self.schedule.alphas_cumprod.numpy(), ddim_timesteps=ddim_steps)

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        if self.target is not None:
            self.target.eval()
        return self

    def forward(self, clean):
        clean = clean.float()
        self.last_metrics = {}
        if self.objective_id == RF_BATCH_OT:
            loss, metrics = rf_ot_loss(self.net, clean, max_global_batch=self.ot_global_batch)
            self.last_metrics = metrics
            return loss
        if self.objective_id == LCF_REAL:
            loss, metrics = lcf_loss(self.net, self.target, clean, self.schedule, skip=self.lcf_skip)
            self.last_metrics = metrics
            return loss
        noise = torch.randn_like(clean)
        if self.kind == RECTIFIED_FLOW_ORIGINAL:
            t = torch.rand(clean.shape[0], device=clean.device)*0.999+0.001
            tt = t[:,None,None,None]
            pred = self.net(tt*clean+(1-tt)*noise,t*999).float()
            loss = (pred-(clean-noise)).square().mean()
            self.last_metrics = {"velocity_mse": loss.detach()}
            return loss
        if self.kind == DIFFUSION_ORIGINAL:
            t = torch.randint(0,1000,(clean.shape[0],),device=clean.device)
            noisy = self.schedule.add_noise(clean,noise,t)
            pred = self.net(noisy,t).float()
            if self.objective_id == MIN_SNR_EPSILON:
                loss, metrics = min_snr_epsilon_loss(pred,noise,self.schedule.alphas_cumprod,t,self.gamma)
                self.last_metrics = metrics
                return loss
            loss = (pred-noise).square().mean()
            self.last_metrics = {"epsilon_mse": loss.detach()}
            return loss
        impl = lcd_math()
        self.solver.to(clean.device)
        index = torch.randint(0,self.ddim_steps,(clean.shape[0],),device=clean.device)
        start = self.solver.ddim_timesteps[index]
        end = (start-1000//self.ddim_steps).clamp_min(0)
        noisy = self.schedule.add_noise(clean,noise,start)
        predicted_clean = self.schedule.origin(noisy,self.net(noisy,start).float(),start)
        skip,out = impl.scalings_for_boundary_conditions(start.float())
        prediction = skip[:,None,None,None]*noisy + out[:,None,None,None]*predicted_clean
        with torch.no_grad():
            teacher_eps = self.teacher(noisy,start).float()
            teacher_origin = self.schedule.origin(noisy,teacher_eps,start)
            previous = self.solver.ddim_step(teacher_origin,teacher_eps,index).float()
            target_origin = self.schedule.origin(previous,self.target(previous,end).float(),end)
            skip,out = impl.scalings_for_boundary_conditions(end.float())
            target = skip[:,None,None,None]*previous + out[:,None,None,None]*target_origin
        loss = (prediction.float()-target.float()).square().mean()
        self.last_metrics = {"consistency_mse": loss.detach()}
        return loss
