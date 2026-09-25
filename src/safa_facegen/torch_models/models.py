"""Official RF/LDM backbones, and LCD student using the same FFHQ-LDM backbone."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .codec import load_codec, read_checkpoint, registered_codec
from .vendor import ldm_imports, rf_backbone, lcd_math

ldm_imports()
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import make_beta_schedule, make_ddim_timesteps, make_ddim_sampling_parameters

RF_ID = "RectifiedFlow-NCSNpp"
LDM_ID = "Diffusion-LDM-UNet"
LCM_ID = "LatentConsistency-LDM-UNet"


def registered_ema_sha256(checkpoint):
    """Read the hash already registered when an EMA was exported or transferred."""
    import json
    from pathlib import Path
    path=Path(checkpoint).resolve()
    manifest=path.with_name(path.name.removesuffix('.ema.pt')+'.json')
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        return saved.get('hashes',{}).get('ema',saved.get('ema_sha256'))
    return None


def family(model_id):
    text = model_id.lower().replace("-", "").replace("_", "")
    if text.startswith("rectifiedflow"):
        return "rectified_flow"
    if text.startswith("latentconsistency"):
        return "latent_consistency"
    if text.startswith(("latentdiffusion", "diffusion")):
        return "diffusion"
    raise ValueError(f"Unknown torch model_id: {model_id}")


def new_unet():
    # Exact models/ldm/ffhq256/config.yaml UNet parameters, no width changes.
    return UNetModel(image_size=64, in_channels=3, out_channels=3, model_channels=224,
                     attention_resolutions=[8, 4, 2], num_res_blocks=2,
                     channel_mult=[1, 2, 3, 4], num_head_channels=32, use_checkpoint=True)


def load_backbone(model_id, checkpoint, use_ema=True, *, payload=None):
    kind = family(model_id)
    net = rf_backbone() if kind == "rectified_flow" else new_unet()
    if payload is None:payload = read_checkpoint(checkpoint)
    if payload.get("format") == "safa-facegen-ema-v1":
        if payload.get("state_role")!="ema":
            raise ValueError("Project generator checkpoint must explicitly declare state_role='ema'")
        if family(payload["model_id"]) != kind:
            raise ValueError(f"Checkpoint family {payload['model_id']} does not match {model_id}")
        net.load_state_dict(payload["model_state"], strict=True)
        return net
    if kind == "rectified_flow":
        raw = payload["model"]
        net.load_state_dict({k.removeprefix("module."): v for k, v in raw.items()}, strict=True)
        if use_ema:
            shadows = payload["ema"]["shadow_params"]
            parameters = [p for p in net.parameters() if p.requires_grad]
            if len(parameters) != len(shadows):
                raise ValueError("Official RF EMA parameter count does not match NCSN++")
            with torch.no_grad():
                for param, shadow in zip(parameters, shadows):
                    if param.shape != shadow.shape:
                        raise ValueError("Official RF EMA tensor shape mismatch")
                    param.copy_(shadow)
    elif kind == "diffusion":
        raw = payload["state_dict"]
        prefix = "model.diffusion_model."
        state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        net.load_state_dict(state, strict=True)
        if use_ema:
            ema = {name: raw.get("model_ema." + ("diffusion_model." + name).replace(".", ""))
                   for name, _ in net.named_parameters()}
            if any(value is None for value in ema.values()):
                raise ValueError("Official FFHQ LDM checkpoint lacks a complete EMA; explicit raw loading is required")
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.shape != ema[name].shape:
                        raise ValueError(f"LDM EMA shape mismatch: {name}")
                    param.copy_(ema[name])
    else:
        raise ValueError("LCM must load a project LCM EMA export, never an unrelated public LCM weight")
    return net


class DiffusionSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        betas = make_beta_schedule("linear", 1000, linear_start=0.0015, linear_end=0.0195)
        # Official LDM computes cumulative products in NumPy float64 then stores float32.
        alpha = np.cumprod(1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.tensor(alpha, dtype=torch.float32))

    def coefficients(self, t, shape):
        alpha = self.alphas_cumprod[t].reshape(t.shape[0], *([1] * (len(shape)-1)))
        return alpha.sqrt(), (1-alpha).sqrt()

    def add_noise(self, clean, noise, t):
        a, s = self.coefficients(t, clean.shape)
        return a * clean.float() + s * noise.float()

    def origin(self, noisy, eps, t):
        a, s = self.coefficients(t, noisy.shape)
        return (noisy.float() - s * eps.float()) / a


class Generator(nn.Module):
    """Frozen generator; grad_enabled controls the graph, not parameter trainability."""
    def __init__(self, model_id, net, codec=None, *, steps=None, eta=1.0, ode_tol=1e-5,
                 bf16=False, checkpoint_sampling=True):
        super().__init__()
        self.model_id = model_id
        self.kind = family(model_id)
        self.net = net
        self.codec = codec
        self.noise_shape = (3, 256, 256) if self.kind == "rectified_flow" else (3, 64, 64)
        self.steps = None if self.kind=="rectified_flow" else int(steps or (4 if self.kind == "latent_consistency" else 200))
        self.eta = float(eta)
        self.ode_tol = float(ode_tol)
        self.bf16 = bool(bf16)
        self.checkpoint_sampling = bool(checkpoint_sampling)
        self.schedule = None if self.kind == "rectified_flow" else DiffusionSchedule()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # A parent SAFA module may call train(); the frozen generator keeps inference behavior.
        return super().train(False)

    @property
    def step_noise_count(self):
        if self.kind == "rectified_flow" or (self.kind == "diffusion" and self.eta == 0):
            return 0
        return self.steps - 1 if self.kind == "latent_consistency" else self.steps

    @property
    def step_noise_shape(self):
        return self.noise_shape

    def _predict(self, x, t):
        def apply(a, b):
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                                enabled=self.bf16 and x.device.type == "cuda"):
                return self.net(a, b).float()
        if torch.is_grad_enabled() and self.checkpoint_sampling:
            return activation_checkpoint(apply, x, t, use_reentrant=False)
        return apply(x, t)

    def _noise(self, noises, index, x, required):
        if not required:
            return torch.zeros_like(x)
        if noises is None:
            return torch.randn_like(x)
        out = noises[index]
        if tuple(out.shape) != tuple(x.shape) or out.device != x.device:
            raise ValueError("step_noises must have [steps,B,C,H,W] shape on the input device")
        return out.float()

    def sample(self, noise, step_noises=None, grad_enabled=False):
        if tuple(noise.shape[1:]) != self.noise_shape or noise.ndim != 4:
            raise ValueError(f"noise must have [B,{','.join(map(str,self.noise_shape))}] shape")
        if not torch.is_floating_point(noise):
            raise TypeError("noise must be floating point")
        if step_noises is not None and len(step_noises) != self.step_noise_count:
            raise ValueError(f"Expected {self.step_noise_count} step_noises, got {len(step_noises)}")
        with torch.set_grad_enabled(grad_enabled):
            x = noise.float()
            if self.kind == "rectified_flow":
                if step_noises is not None and len(step_noises):
                    raise ValueError("Deterministic RF ODE has no step noises")
                # SciPy RK45 integrates in float64 and evaluates the neural field in
                # float32. Match both precisions, retaining autograd through casts.
                from .rk45 import integrate
                self.last_ode_nfe=0
                def velocity(t, state):
                    self.last_ode_nfe+=1
                    # Match upstream: round t to FP32 before scaling its label.
                    labels = torch.full((state.shape[0],),float(t),device=state.device,dtype=torch.float32)*999.
                    return self._predict(state.float(), labels).double()
                x = integrate(velocity,x.double(),rtol=self.ode_tol,atol=self.ode_tol).float()
                return x.clamp(-1, 1)
            if self.kind == "diffusion":
                if self.steps < 1 or self.steps >= 1000 or 1000 % self.steps:
                    raise ValueError("Official uniform DDIM requires a proper divisor of 1000 below 1000")
                timesteps = make_ddim_timesteps("uniform", self.steps, 1000, verbose=False)
                sigmas, alphas, prev = make_ddim_sampling_parameters(
                    self.schedule.alphas_cumprod.detach().cpu().numpy(), timesteps, self.eta, verbose=False)
                for k, index in enumerate(reversed(range(len(timesteps)))):
                    t = torch.full((x.shape[0],), int(timesteps[index]), device=x.device, dtype=torch.long)
                    eps = self._predict(x, t)
                    a, ap, sigma = [torch.as_tensor(v[index], device=x.device, dtype=torch.float32)
                                    for v in (alphas, prev, sigmas)]
                    pred = (x - (1-a).sqrt() * eps) / a.sqrt()
                    x = ap.sqrt()*pred + (1-ap-sigma.square()).clamp_min(0).sqrt()*eps
                    x = x + sigma * self._noise(step_noises, k, x, self.eta != 0)
            else:
                if not 1 <= self.steps <= 50:
                    raise ValueError("LCM inference steps must be between 1 and 50")
                # Original official LCM Scheduler (diffusers v0.22.0), linked by
                # the pinned LCM repository: integer skipping on the teacher grid.
                origin_steps = np.arange(1,51)*20-1
                timesteps = origin_steps[::-int(50//self.steps)][:self.steps]
                math_impl = lcd_math()
                for k, step in enumerate(timesteps):
                    t = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
                    pred = self.schedule.origin(x, self._predict(x,t), t)
                    c_skip,c_out = math_impl.scalings_for_boundary_conditions(t.float())
                    denoised = c_skip[:,None,None,None]*x + c_out[:,None,None,None]*pred
                    if k == len(timesteps)-1:
                        x = denoised
                    else:
                        nxt = torch.full_like(t, int(timesteps[k+1]))
                        x = self.schedule.add_noise(denoised, self._noise(step_noises,k,x,True), nxt)
            # decode() uses the official VQ straight-through estimator; no detach/no_grad here.
            return self.codec.decode(x.float()).float().clamp(-1, 1)


def load_generator(model_id, checkpoint, device="cuda", codec_checkpoint=None,
                   allow_initialization=False, ema_sha256=None, **sampling):
    import json
    from pathlib import Path
    kind = family(model_id)
    metadata = read_checkpoint(checkpoint)
    is_export = metadata.get("format") == "safa-facegen-ema-v1"
    if not is_export and not allow_initialization:
        raise ValueError("Formal evaluation requires an EMA-only project export; set allow_initialization=True only for initialization validation")
    if is_export and metadata.get("state_role")!="ema":
        raise ValueError("Formal evaluation requires state_role='ema'")
    saved_sampling=metadata.get("config",{}).get("sampling")
    if is_export and not isinstance(saved_sampling,dict):
        raise ValueError("EMA export must contain its saved config.sampling settings")
    resolved_sampling={**(saved_sampling or {}),**sampling}
    codec_sha256=metadata.get("codec_sha256")
    net = load_backbone(model_id, checkpoint, payload=metadata)
    codec = None
    if kind != "rectified_flow":
        if codec_checkpoint is None:
            if kind == "diffusion" and not is_export:
                codec_checkpoint = checkpoint
            else:
                raise ValueError("codec_checkpoint is required for project EMA exports")
        if is_export and (not codec_sha256 or registered_codec(codec_checkpoint)['sha256']!=codec_sha256):
            raise ValueError("EMA export and supplied codec have different SHA256 identities")
        shared_payload=metadata if Path(codec_checkpoint).resolve()==Path(checkpoint).resolve() else None
        codec = load_codec(codec_checkpoint,payload=shared_payload)
    del metadata
    generator = Generator(model_id, net, codec, **resolved_sampling).to(device)
    generator.sampling_config={"steps":generator.steps,"eta":generator.eta,
        "ode_tol":generator.ode_tol,"bf16":generator.bf16,
        "checkpoint_sampling":generator.checkpoint_sampling}
    generator.state_role = "ema" if is_export else "external_initialization_ema_selected"
    if is_export and ema_sha256 is None:ema_sha256=registered_ema_sha256(checkpoint)
    generator.ema_sha256 = ema_sha256 if is_export else None
    return generator
