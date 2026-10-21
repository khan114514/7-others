"""Shared offline DD reimplementation support, frozen across candidates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from config import ENGINE, deployment
DEPLOYMENT = deployment()
WORK = Path(os.environ['DD_RUN'])
REPO = Path(DEPLOYMENT['repo'])
ASSETS = Path(DEPLOYMENT['assets'])
VAE_PATH = ASSETS / 'VAR/vae_ch160v4096z32.pth'
TEACHER_PATH = ASSETS / 'VAR-teacher/var_d16.pth'
TORCH_HOME = Path(DEPLOYMENT['torch_home'])
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def require_phase_one():
    # Supervisor independently verifies hashes before spawning a trusted stage.
    if os.environ.get('DD_SUPERVISED_STAGE') != '1':
        raise RuntimeError('Use the task supervisor; direct stages are unsupported')


def load_method(path):
    from method_proxy import MethodProxy
    proxy = MethodProxy(path)
    return proxy, proxy.spec


def setup_runtime(seed=1234):
    sys.path.insert(0, str(REPO))
    for directory in DEPLOYMENT.get('extra_site_packages', []):
        sys.path.insert(0, directory)
    import random
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.set_per_process_memory_fraction(.36)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.hub.set_dir(str(TORCH_HOME / 'hub'))
    return torch


def budget_check(deadline, reserve=0):
    if time.time() >= deadline - reserve:
        raise TimeoutError('Candidate generation and training budget exhausted')


def decode_tokens(tokens, vae):
    """Differentiable equivalent of the released multiscale latent decoder.

    sample.transfer_BLCv_to_B3HW is decorated no_grad and cannot train a head.
    Frozen VAE parameters still permit gradients with respect to its input.
    Returns images in [-1,1], the input range expected by LPIPS.
    """
    import torch.nn.functional as F
    f_hat = tokens.new_zeros(tokens.shape[0], 32, 16, 16)
    begin = 0
    for scale, pn in enumerate(PATCH_NUMS):
        part = tokens[:, begin:begin + pn * pn].transpose(1, 2).reshape(-1, 32, pn, pn)
        if pn != 16:
            part = F.interpolate(part, size=(16, 16), mode='bicubic')
        f_hat = f_hat + vae.quantize.quant_resi[scale / 9](part)
        begin += pn * pn
    return vae.fhat_to_img(f_hat)


def parameter_telemetry():
    import torch
    return {'peak_torch_allocated_mib':torch.cuda.max_memory_allocated() / 2**20,
            'peak_torch_reserved_mib':torch.cuda.max_memory_reserved() / 2**20}
