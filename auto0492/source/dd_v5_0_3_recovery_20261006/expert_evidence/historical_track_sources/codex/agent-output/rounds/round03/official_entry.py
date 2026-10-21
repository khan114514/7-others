"""Guarded official sampler entry with non-mutating finite-output checks."""
import json
import faulthandler
import os
import runpy
import sys
import time
from pathlib import Path

from runtime_guard import apply_guard

if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, str(Path.cwd()))
    print(json.dumps({'execution_guard': apply_guard()}), flush=True)
    import torch
    import torchvision.utils

    fraction = float(os.environ.get('DD_CUDA_MEMORY_FRACTION', '.42'))
    if not 0 < fraction <= .42:
        raise ValueError('Memory cap must be within the track limit')
    torch.cuda.set_per_process_memory_fraction(fraction)
    original_save = torchvision.utils.save_image
    checks = 0
    last_base = None

    def checked_save(tensor, *args, **kwargs):
        global checks, last_base
        base = tensor._base if tensor._base is not None else tensor
        if base is not last_base:
            if not bool(torch.isfinite(base).all()):
                raise FloatingPointError('Official decoded output contains nonfinite values')
            checks += 1
            last_base = base
            if checks == 1 or checks % 25 == 0:
                print(json.dumps({'finite_decoded_batch_checks': checks,
                      'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}), flush=True)
        return original_save(tensor, *args, **kwargs)

    torchvision.utils.save_image = checked_save
    runpy.run_path('sample.py', run_name='__main__')
    print(json.dumps({'finite_decoded_batch_checks': checks, 'finished_unix': time.time(),
          'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
          'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}), flush=True)
