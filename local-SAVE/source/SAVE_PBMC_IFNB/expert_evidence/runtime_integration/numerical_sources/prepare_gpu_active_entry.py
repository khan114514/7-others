"""Create a task-owned active runner receipt before numerical freezing.

This preparation entry point must never rewrite a live frozen runtime.
The runtime builder and any reviewed runner update precede this step.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(root):
    root = root.resolve()
    if (root / 'private/numerical-runtime-binding.json').exists():
        raise RuntimeError('Numerical binding already exists; no runtime or receipt may be rewritten')
    target = root / 'private/gpu-jail-runtime-v2-active-entry.json'
    if target.exists() or target.is_symlink():
        raise RuntimeError('Active entry already exists; preserve its original bytes')
    source = root / 'preparation/gpu_jail_entry.py'
    runner = root / 'private/gpu-jail-runtime-v2/runner.py'
    if source.is_symlink() or runner.is_symlink() or source.read_bytes() != runner.read_bytes():
        raise RuntimeError('Runtime runner must equal the reviewed trusted entry source before activation')
    value = {'entry_sha256': hashlib.sha256(runner.read_bytes()).hexdigest()}
    with target.open('xb') as out:
        out.write(json.dumps(value, indent=2).encode())
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root)))
