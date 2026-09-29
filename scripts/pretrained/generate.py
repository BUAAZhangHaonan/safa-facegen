"""Generate an unconditional image from a selected pretrained generator."""
import argparse
from pathlib import Path
from runtime import load_selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--model')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('face.png'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    from PIL import Image
    generator, row = load_selected(a.release, a.model, a.device)
    with torch.inference_mode():
        images = generator.generate(num_images=1, seed=a.seed)
    if tuple(images.shape) != (1,3,256,256) or not bool(torch.isfinite(images).all()):
        raise RuntimeError('Invalid generated tensor')
    pixels = images[0].detach().float().cpu().clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.permute(1,2,0).numpy()).save(a.output)
    print(f"{row['model_id']} | {row['checkpoint_id']} | {a.output}")

if __name__ == '__main__':
    main()
