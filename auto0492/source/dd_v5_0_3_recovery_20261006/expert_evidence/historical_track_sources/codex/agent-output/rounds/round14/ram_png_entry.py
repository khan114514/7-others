"""Official sampler with exact torchvision PNG bytes buffered in host RAM.

Only the PNG backing store changes. The official image dataset, transforms,
Inception batches, statistics, and Frechet calculation remain in use.
"""
import faulthandler
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, 'historical-codex-workspace')
from runtime_guard import apply_guard


def main():
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, str(Path.cwd()))
    print(json.dumps({'execution_guard': apply_guard()}), flush=True)
    import torch
    import torchvision.utils
    from PIL import Image
    from evaluation import fid_score

    torch.cuda.set_per_process_memory_fraction(.30)
    original_save = torchvision.utils.save_image
    original_fid = fid_score.calculate_fid_given_paths
    original_activations = fid_score.get_activations
    evidence = Path(os.environ['DD_RAM_EVIDENCE'])
    assert evidence.is_relative_to(Path('historical-codex-workspace/agent-output'))
    evidence.mkdir(parents=True, exist_ok=True)
    probe = os.environ.get('DD_RAM_EQUIVALENCE_PROBE') == '1'
    payloads = {}
    checks = 0
    last_base = None
    byte_total = 0
    manifest = (evidence / 'sample_manifest.jsonl').open('w')

    def retained_activations(*args, **kwargs):
        result = original_activations(*args, **kwargs)
        feature_path = os.environ.get('DD_RAM_FEATURE_PATH')
        if feature_path and len(result) == 50000:
            import numpy as np
            destination = Path(feature_path)
            assert destination.is_relative_to(Path('historical-codex-models'))
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, result, allow_pickle=False)
            (evidence/'inception_features.json').write_text(json.dumps({
                'path':str(destination), 'shape':list(result.shape), 'dtype':str(result.dtype),
                'image_order':'lexicographic PNG filenames; official scorer order',
                'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                'purpose':'Post-score paired uncertainty diagnostics; official metric unchanged'},indent=2)+'\n')
        return result

    fid_score.get_activations = retained_activations

    def checked_save(tensor, path, **kwargs):
        nonlocal checks, last_base, byte_total
        base = tensor._base if tensor._base is not None else tensor
        if base is not last_base:
            if not bool(torch.isfinite(base).all()):
                raise FloatingPointError('Nonfinite decoded output')
            checks += 1
            last_base = base
        path = Path(path)
        index = int(path.stem)
        assert index == len(payloads) and path.suffix == '.png'
        buffer = io.BytesIO()
        original_save(tensor, buffer, format='png', **kwargs)
        raw = buffer.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            assert decoded.size == (256, 256)
        payloads[path.name] = raw
        byte_total += len(raw)
        manifest.write(json.dumps({'file': path.name, 'bytes': len(raw), 'sha256': digest}) + '\n')
        if index < 64:
            original_save(tensor, path, **kwargs)
            assert path.read_bytes() == raw, 'RAM and disk PNG bytes differ'
        if index % 1000 == 0:
            manifest.flush()
            print(json.dumps({'ram_png_count': len(payloads), 'ram_png_bytes': byte_total,
                              'finite_decoded_batch_checks': checks,
                              'peak_torch_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                              'peak_torch_reserved_mib': torch.cuda.max_memory_reserved()/2**20}), flush=True)

    def calculate(paths, **kwargs):
        manifest.flush()
        manifest.close()
        names = sorted(payloads)  # Matches the scorer's lexicographic Path order.
        assert [int(name[:-4]) for name in sorted(names, key=lambda x: int(x[:-4]))] == list(range(len(names)))
        assert kwargs['batch_size'] == 256 and kwargs['dims'] == 2048 and kwargs['num_workers'] == 8
        transforms = fid_score.TF.ToTensor()
        dataset = fid_score.ImagePathDataset([io.BytesIO(payloads[name]) for name in names], transforms=transforms)
        disk_paths = sorted(Path(paths[1]).glob('*.png'))
        for path in disk_paths:
            expected = transforms(Image.open(io.BytesIO(payloads[path.name])).convert('RGB'))
            actual = transforms(Image.open(path).convert('RGB'))
            assert torch.equal(expected, actual)
        grid = Image.new('RGB', (2048, 2048))
        for i in range(min(64, len(names))):
            with Image.open(io.BytesIO(payloads[f'{i}.png'])) as image:
                grid.paste(image.convert('RGB'), ((i % 8)*256, (i // 8)*256))
        grid.save(evidence / 'sample_grid.jpg', quality=90)
        result = original_fid((paths[0], dataset), **kwargs)
        report = {'storage': 'exact torchvision PNG bytes in host RAM', 'sample_count': len(names),
                  'encoded_bytes': byte_total, 'disk_pngs_verified_exact': len(disk_paths),
                  'disk_and_ram_transforms_verified_exact': True,
                  'official_image_dataset_and_scorer_used': True,
                  'lexicographic_image_order_preserved': True, 'fid_ram': float(result),
                  'seed_set': [0], 'batch_size': 8, 'inception_batch_size': 256,
                  'finite_decoded_batch_checks': checks, 'completed_unix': time.time()}
        if probe:
            assert len(disk_paths) == len(names)
            disk_result = original_fid(paths, **kwargs)
            report.update(fid_disk=float(disk_result), fid_exact_match=float(result) == float(disk_result))
            assert report['fid_exact_match'], 'RAM and disk FID differ'
        (evidence / 'ram_png_storage.json').write_text(json.dumps(report, indent=2) + '\n')
        return result

    torchvision.utils.save_image = checked_save
    fid_score.calculate_fid_given_paths = calculate
    runpy.run_path('sample.py', run_name='__main__')
    print(json.dumps({'finished_unix': time.time(), 'ram_png_count': len(payloads),
                      'peak_torch_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                      'peak_torch_reserved_mib': torch.cuda.max_memory_reserved()/2**20}), flush=True)


if __name__ == '__main__':
    main()
