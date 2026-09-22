"""Verify official initialization archives and register their exact local contents."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile

from .common import atomic_json, require_inside, sha256_file, utc_now


def prepare_ldm(root, archive):
    root = Path(root).resolve()
    archive = require_inside(archive, root)
    destination = root / 'models/initialization/Diffusion-LDM-UNet-FFHQ-official'
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        for item in members:
            require_inside(destination / item.filename, destination)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains a symbolic link')
        checkpoints = [item for item in members if item.filename.endswith('/model.ckpt') or item.filename == 'model.ckpt']
        if len(checkpoints) != 1:
            raise ValueError(f'Expected one official checkpoint, got {len(checkpoints)}')
        destination.mkdir(parents=True)
        for item in members:
            if item.is_dir():
                continue
            # Preserve the official configuration and checkpoint; image examples
            # and redundant training logs have no role in this project.
            if item != checkpoints[0] and Path(item.filename).suffix not in ('.yaml', '.yml'):
                continue
            output = destination / ('model.ckpt' if item == checkpoints[0] else Path(item.filename).name)
            if output.exists():
                raise FileExistsError(output)
            with package.open(item) as source, output.open('xb') as target:
                # ZipExtFile verifies CRC while the selected member is extracted.
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                target.flush(); os.fsync(target.fileno())
    import torch
    checkpoint = destination / 'model.ckpt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload['state_dict']
    prefix = 'first_stage_model.'
    codec = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not codec:
        raise ValueError('Official checkpoint has no VQ codec')
    codec_path = root / 'models/codecs/LDM-VQ4.pt'
    codec_path.parent.mkdir(parents=True, exist_ok=True)
    if codec_path.exists():
        raise FileExistsError(codec_path)
    torch.save(codec, codec_path)
    del payload, state, codec
    from .torch_models.codec import load_codec
    decoder = load_codec(codec_path, 'cpu')
    del decoder
    record = {'model_id': 'Diffusion-LDM-UNet', 'source': 'https://ommer-lab.com/files/latent-diffusion/ffhq.zip',
              'upstream_commit': 'a506df5756472e2ebaf9078affdde2c4f1502cd4',
              'checkpoint_sha256': sha256_file(checkpoint),
              'codec_sha256': sha256_file(codec_path), 'created_at': utc_now(), 'strict_codec_load': 'passed'}
    atomic_json(destination / 'provenance.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--ldm-archive', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_ldm(args.root, args.ldm_archive), indent=2))


if __name__ == '__main__':
    main()
