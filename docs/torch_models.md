# PyTorch face generators

The three adapters preserve the pinned official backbones and unconditional inputs. RF uses pixels. Diffusion and Latent Consistency use the FFHQ VQ-f4 codec with three channels at 64 × 64, prequant encoding, and scale factor 1. The training configurations share the HQ manifest and derive epoch counts from the number of actual training images consumed.

| Model ID | Pinned source | Initialization | Objective and sampling |
| --- | --- | --- | --- |
| RectifiedFlow-NCSNpp | [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow/tree/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6) | Official CelebAHQ256 NCSN++ checkpoint, EMA shadow parameters | Straight flow matching; differentiable PyTorch Dormand–Prince RK45 from 0.001 to 1, tolerance 1e-5 |
| Diffusion-LDM-UNet | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) | Official FFHQ VQ-f4 UNet EMA | Epsilon MSE; official 1,000-step square-root linear beta schedule, 0.0015–0.0195; DDIM 200 steps and eta 1 |
| LatentConsistency-LDM-UNet | [luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model/tree/a9ad79587cc8bd1e404ccd1a3056a3da969b2f62) | This project's Diffusion EMA teacher | Unconditional latent consistency distillation, 50-step teacher DDIM grid, EMA student target, official boundary scaling, stop gradient through teacher and target; four inference steps |

LCM retains the project's unconditional FFHQ UNet and omits the upstream text/CFG conditioning branches. Its teacher must be an explicit project Diffusion EMA export with the same codec hash. The upstream distillation math is retained in `vendor/latent_consistency/lcd_math.py`, extracted from the pinned training script. The provenance file records this adaptation. The teacher, target, loss, and solver run within the same schedule; the target is the FP32 student EMA.

The LCM inference grid follows the [official Diffusers 0.22.0 LCMScheduler](https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/schedulers/scheduling_lcm.py) linked by the pinned LCM repository: integer skipping on the 50-step teacher grid gives `[999, 759, 519, 279]` at four steps. The generator returns the final denoised latent; the three intermediate transitions consume explicit noises. The unused random draw after the final denoised result in that scheduler is omitted.

## Vendor boundaries

Each vendor directory contains `UPSTREAM.json`, retained source references, and available upstream license notices. The official RF pure-PyTorch FIR resampling branch is selected on all devices to avoid compiling the old custom CUDA operator. It retains the upstream operation and casts the FIR kernel to the activation dtype under BF16. The RF Fourier embedding frequencies remain frozen during training. The LDM activation-checkpoint adapter uses PyTorch's non-reentrant implementation, which supports frozen parameters and gradients with respect to the input. The frozen codec retains the official Encoder, Decoder, and VectorQuantizer2 implementation. The quantizer dependency is pinned to CompVis/taming-transformers `24268930bf1dce879235a7fddd0b2355b84d7ea6` with its license.

The differentiable RF solver translates the official SciPy RK45 coefficients, embedded error estimate, initial-step selection, RMS norm, rejection control, and endpoint truncation into PyTorch. It uses FP64 solver state and FP32 neural velocity evaluations at absolute and relative tolerance 1e-5. Casts and accepted state updates retain input gradients. Adaptive step decisions are discrete. The SciPy BSD license and source attribution are retained.

RF's source tree has no repository-wide license file at the pinned commit. Its existing file-level notices are retained; `LICENSE_NOTICE.md` records this boundary without asserting a broader license.

## Generator contract

```python
from safa_facegen.torch_models import load_generator, load_codec

generator = load_generator(
    model_id, ema_checkpoint, device="cuda",
    codec_checkpoint=codec_checkpoint,  # Required for Diffusion and LCM.
)
rgb = generator.sample(noise, step_noises=step_noises, grad_enabled=True)
```

`noise_shape` is `(3, 256, 256)` for RF and `(3, 64, 64)` for the latent models. `step_noise_shape` equals `noise_shape`. `step_noise_count` is zero for RF and deterministic DDIM, the number of DDIM steps for eta > 0, and three for four-step LCM. Each explicit step noise is a tensor shaped `(batch, *step_noise_shape)`. Providing these noises makes sample identity independent of unrelated RNG activity. The evaluation layer generates them per sample on CPU from fixed seeds. Omitting them permits fresh random noise during exploratory sampling.

All generator and codec parameters stay frozen. `grad_enabled=True` preserves the graph from the output back to the initial noise and explicit step noises, through every solver step and the decoder. RGB output is clamped to [-1, 1], so saturated output coordinates have zero clamp derivative. The VQ assignment uses the official VectorQuantizer2 straight-through estimator; its input gradient is the declared surrogate for the discrete nearest-code operation. RF gradients differentiate the executed adaptive solver computation; changes of solver control decisions are discrete.

Formal evaluation accepts only `format="safa-facegen-ema-v1"` and `state_role="ema"`. A supplied latent codec must match the registered `codec_sha256` and file size in its sibling metadata. `generator.ema_sha256` reuses the EMA-only hash registered at export/transfer; inference does not rescan the weights. External official checkpoint initialization can be checked explicitly with `allow_initialization=True`; its returned role is `external_initialization_ema_selected`, with no project EMA hash.

Project exports must contain `config.sampling`. Loading restores the saved DDIM steps/eta, RF tolerance, or LCM steps by default; explicit sampling keyword arguments override the saved values. `generator.sampling_config` exposes the resolved settings.

`load_codec(checkpoint, device)` accepts a full official LDM checkpoint or an extracted codec state dictionary. It accepts `state_dict` with `first_stage_model.` prefixes, or a bare codec dictionary. Tensor names and shapes are loaded strictly. `encode(rgb)` returns prequant latents; `decode(latent)` includes the official quantizer.

## Training and recovery

Launch the PyTorch trainer with four torchrun ranks:

```sh
PYTHONPATH=src .venv-torch/bin/torchrun --standalone --nproc_per_node=4 \
  -m safa_facegen.train_torch --config configs/rectified_flow.json
```

The initial learning rates are 2e-5 for RF and 1e-5 for Diffusion and LCM. BF16 autocast is optional; loss arithmetic and EMA remain FP32. Microbatches in the templates require the four-GPU calibration performed by the controller. The GPU allocator is capped at the configured GPU envelope minus 3 GiB for context and NCCL allocations. Allocation errors propagate to the controller; the trainer does not silently retry or change batch size. Host soft and hard redlines are 192 and 224 GiB. At the soft limit, the current loader's workers and queued batches are released and training stops using the previous complete checkpoint, without serializing new state under memory pressure. The hard limit aborts immediately without allocating another checkpoint.

The distributed sampler and loader retain the final partial batch. The manifest length must be divisible by world size, preventing sampler padding. `samples_seen` uses the actual batch size multiplied by rank count. Augmentation depends on dataset seed, epoch, and image index. Accumulation is fixed to one, with calibration increasing the microbatch within the limits.

Complete recovery state includes model, EMA, optimizer, per-rank Python/NumPy/CPU/CUDA RNG, sampler epoch, next batch, samples consumed within the current epoch, registered artifact identities, file metadata, code revision, and absolute UTC event timestamps. Ordinary resume requires the same recipe. An explicit `recovery_overrides` object may change `learning_rate`, `microbatch`, `precision`, `num_workers`, or `prefetch_factor`. The sampler resumes at the exact committed sample index before forming new batches, supporting any new positive microbatch size without repeating or dropping samples. Dataset, codec, teacher, world size, and structural recipe fields remain strict. Overrides are written to events, and the learning rate is applied to the restored optimizer.

Calibration saves one complete recovery state and EMA after its final optimizer update, covering serialization in the measured resource peak. Temporary calibration exports omit hashing and replication requests and are removed by the calibration runner.

Every optimizer update checks the loss, gradients, parameters, optimizer tensors, and EMA for finite values. A non-finite update leaves the previous complete checkpoint pointer available. Checkpoint files are atomically written, flushed, and hashed; the completed metadata and durable replication requests precede the update of `last.json`.

## Artifacts and replication

Artifacts live under `runs/<model_id>`. Names follow `<model_id>-<completed-HQ-epochs>ep-<UTC>` without experiment identifiers. The epoch component is the integer completed-pass count; metadata records exact sample exposure. A save creates `.state.pt`, `.ema.pt`, `.config.json`, and `.json`, followed by `last.json`. Timestamps make each identity immutable even when multiple checkpoints fall within the same epoch.

The full state uses `format="safa-facegen-train-v1"`. The separate EMA export contains `format="safa-facegen-ema-v1"`, `state_role="ema"`, `model_state`, `model_id`, config/progress, dataset size, codec/teacher hashes, and code revision. It excludes raw model and optimizer state.

`events.jsonl` and `requests.jsonl` are in the run root. Save requests are emitted every 900 seconds; preview and review requests use 1,800 and 7,200 seconds. Scheduled preview/review only export and queue state on H100. The K100 worker performs image generation and evaluation. Callback and request payloads include:

```text
model_id, checkpoint_id, identity, complete=true,
checkpoint/state_path, ema_path, config_path,
hashes={state,ema,config}, state_sha256, sha256/ema_sha256,
state_role=ema, step, samples_seen, completed_epochs,
created_at_utc, reason
```

Paths are relative to the project root. `sha256` and `ema_sha256` both refer to the EMA-only file. Each JSONL request also has `event`, `type`, and `request_id=<checkpoint_id>-<event>`.

Implementation acceptance scripts and their reports are kept outside the delivered Git source. Production training and inference reuse registered artifact metadata and avoid repeated full-file integrity scans.
