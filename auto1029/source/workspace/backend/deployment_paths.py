"""Root-only nonsecret deployment bindings; never inherited by candidate jail."""
import json
import os
import stat
from pathlib import Path

if os.geteuid() != 0:
    raise PermissionError('GPU deployment backend requires trusted root')
CONFIG_PATH = Path(os.environ['HEIMA_DEPLOYMENT_CONFIG']).absolute()
if CONFIG_PATH.resolve() != CONFIG_PATH:
    raise ValueError('Deployment config must not traverse symlinks')
fd = os.open(CONFIG_PATH, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077 or info.st_size > 65536:
        raise ValueError('Deployment config must be a bounded root-owned private file')
    CONFIG = json.loads(os.read(fd, 65537))
finally:
    os.close(fd)
REQUIRED = {'project_root', 'scoring_python', 'jail_python', 'stdlib_root', 'torchtune_package', 'gpu_lock_files', 'nvidia_smi'}
if set(CONFIG) != REQUIRED or not isinstance(CONFIG['gpu_lock_files'], list) or not CONFIG['gpu_lock_files']:
    raise ValueError('Explicit deployment paths and shared GPU mutexes required')


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('Trusted deployment path must be absolute')
    return Path(value)


PROJECT_ROOT = absolute(CONFIG['project_root'])
PRIVATE_ROOT = PROJECT_ROOT / 'private'
SCORING_PYTHON = absolute(CONFIG['scoring_python'])
JAIL_PYTHON = absolute(CONFIG['jail_python'])
STDLIB_ROOT = absolute(CONFIG['stdlib_root'])
TORCHTUNE_PACKAGE = absolute(CONFIG['torchtune_package'])
GPU_LOCK_FILES = [absolute(p) for p in CONFIG['gpu_lock_files']]
NVIDIA_SMI = absolute(CONFIG['nvidia_smi'])
