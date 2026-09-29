"""Sequential inference/gradient acceptance: two samples per model, zero updates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time
from runtime import read_release, load_selected


def save(path, value):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume-failed', action='store_true', help='Continue the inspected partial acceptance record')
    a = p.parse_args()
    root = a.release.resolve()
    manifest = read_release(root)
    report_path = root / 'release-smoke.json'
    previous = None
    if report_path.exists():
        if not a.resume_failed:
            raise FileExistsError(f'Inspect the existing acceptance record: {report_path}')
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        expected = {(r['model_id'],r['checkpoint_id']) for r in manifest['models']}
        completed = {(r['model_id'],r['checkpoint_id']) for r in previous.get('models',[])}
        if (previous.get('status') not in ('failed','running') or
            previous.get('source_commit') != manifest['source_commit'] or not completed.issubset(expected) or
            any(r.get('status') != 'passed' for r in previous.get('models',[]))):
            raise ValueError('Inspect the acceptance identity and source version before continuing')
    import torch
    from PIL import Image
    def save_image(tensor, model_id, name):
        target = root / 'interface-output' / model_id / (name + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = tensor[0].detach().float().cpu().add(1).mul(127.5).round().clamp(0,255).to(torch.uint8)
        Image.fromarray(pixels.permute(1,2,0).numpy()).save(target)
        return target.relative_to(root).as_posix()
    report = {'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
              'device':a.device,'source_commit':manifest['source_commit'],
              'generation_protocol':'interface_checks_seed42_current_runtime; historical evaluations retained',
              'training_updates':0,'models':[],
              'relocated_release_path':str(root),'working_directory':str(Path.cwd()),
              'python_executable':os.sys.executable,'torch_version':torch.__version__,
              'cuda_version':torch.version.cuda}
    if previous is not None:
        report['models'] = previous['models']
        report['previous_attempts'] = [*previous.get('previous_attempts', []),
            {k:v for k,v in previous.items() if k not in ('models','previous_attempts')}]
    save(report_path, report)
    completed_ids = {r['model_id'] for r in report['models']}
    try:
        for row in manifest['models']:
            if row['model_id'] in completed_ids:
                continue
            started = time.monotonic()
            generator, selected = load_selected(root, row['model_id'], a.device)
            report['active_model'] = row['model_id']
            save(report_path, report)
            if any(p.requires_grad for p in generator.parameters()):
                raise RuntimeError('Generator parameters must remain frozen')
            if tuple(generator.noise_shape) != tuple(row['noise_shape']):
                raise RuntimeError('Noise shape differs from the release specification')
            if generator.step_noise_count != row['step_noise_count']:
                raise RuntimeError('Step-noise count differs from the release specification')
            with torch.inference_mode():
                generated = generator.generate(num_images=1, seed=42)
            if (tuple(generated.shape) != (1,3,256,256) or generated.dtype != torch.float32
                    or not bool(torch.isfinite(generated).all())):
                raise RuntimeError('High-level generation returned invalid data')
            lo, hi = float(generated.min()), float(generated.max())
            if lo < -1.000001 or hi > 1.000001:
                raise RuntimeError('Output range differs from [-1,1]')
            generated_path = save_image(generated, row['model_id'], 'generate-seed42')
            del generated
            rng = torch.Generator(device='cpu').manual_seed(42)
            shape = (1,*generator.noise_shape)
            noise = torch.randn(shape, generator=rng, dtype=torch.float32).to(a.device).requires_grad_(True)
            extra = [torch.randn(shape, generator=rng, dtype=torch.float32).to(a.device).requires_grad_(True)
                     for _ in range(generator.step_noise_count)]
            image = generator.sample(noise, step_noises=extra, grad_enabled=True)
            if (tuple(image.shape) != (1,3,256,256) or image.dtype != torch.float32
                    or not bool(torch.isfinite(image).all())):
                raise RuntimeError('Differentiable generation returned invalid data')
            grad_lo, grad_hi = float(image.detach().min()), float(image.detach().max())
            if grad_lo < -1.000001 or grad_hi > 1.000001:
                raise RuntimeError('Differentiable output range differs from [-1,1]')
            gradient_path = save_image(image, row['model_id'], 'sample-seed42')
            direction = torch.randn(tuple(image.shape), generator=rng, dtype=torch.float32).to(a.device)
            scalar = (image.float()*direction).mean()
            gradients = torch.autograd.grad(scalar, [noise,*extra], allow_unused=False)
            if not all(bool(torch.isfinite(v).all()) for v in gradients):
                raise RuntimeError('Nonfinite input gradient')
            norm = float(gradients[0].float().norm())
            if norm <= 0:
                raise RuntimeError('Initial-noise gradient is zero')
            if any(p.grad is not None for p in generator.parameters()):
                raise RuntimeError('Frozen model parameter gradients were populated')
            report['models'].append({'model_id':row['model_id'],'checkpoint_id':row['checkpoint_id'],
                   'status':'passed','generated_samples':2,'initial_noise_gradient_norm':norm,
                   'explicit_step_gradients_finite':len(gradients)-1,'output_min':lo,'output_max':hi,
                   'parameters_frozen':True,'parameter_gradients_empty':True,
                   'loaded_ema_identity':generator.ema_identity,
                   'sampling':getattr(generator,'sampling_config',row['sampling']),
                   'codec':row.get('codec'),'dtype':str(image.dtype),
                   'sample_output_min':grad_lo,'sample_output_max':grad_hi,
                   'output_files':[generated_path,gradient_path],
                   'step_gradient_norms':[float(g.float().norm()) for g in gradients[1:]],
                   'cuda_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                   'matmul_precision':torch.get_float32_matmul_precision(),
                   'seconds':time.monotonic()-started})
            save(report_path, report)
            print(f"Passed: {row['model_id']}", flush=True)
            del gradients, scalar, direction, image, extra, noise, generator
            gc.collect()
            if a.device.startswith('cuda'):
                torch.cuda.empty_cache()
        report.update(status='passed',finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(report_path, report)
        manifest.update(runtime_smoke='passed',release_status='PRETRAINED_READY')
        save(root / 'release.json', manifest)
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(report_path, report)
        raise

if __name__ == '__main__':
    main()
