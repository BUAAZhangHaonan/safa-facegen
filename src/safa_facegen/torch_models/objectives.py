"""RF flow matching, official FFHQ LDM epsilon MSE, unconditional LCM distillation."""
import torch
from torch import nn
from .models import DiffusionSchedule
from .vendor import lcd_math


class TrainingObjective(nn.Module):
    def __init__(self, net, kind, teacher=None, target=None, *, ddim_steps=50):
        super().__init__()
        self.net = net
        self.kind = kind
        # Teacher and target are registered but frozen; DDP sees only student gradients.
        self.teacher = teacher
        self.target = target
        self.schedule = None if kind == "rectified_flow" else DiffusionSchedule()
        self.ddim_steps = ddim_steps
        if kind == "latent_consistency":
            if teacher is None or target is None:
                raise ValueError("LCM requires frozen project LDM teacher and EMA student target")
            teacher.requires_grad_(False).eval()
            target.requires_grad_(False).eval()
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
        noise = torch.randn_like(clean)
        if self.kind == "rectified_flow":
            t = torch.rand(clean.shape[0], device=clean.device)*0.999+0.001
            tt = t[:,None,None,None]
            pred = self.net(tt*clean+(1-tt)*noise,t*999).float()
            return (pred-(clean-noise)).square().mean()
        if self.kind == "diffusion":
            t = torch.randint(0,1000,(clean.shape[0],),device=clean.device)
            noisy = self.schedule.add_noise(clean,noise,t)
            pred = self.net(noisy,t).float()
            return (pred-noise).square().mean()
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
        return (prediction.float()-target.float()).square().mean()
