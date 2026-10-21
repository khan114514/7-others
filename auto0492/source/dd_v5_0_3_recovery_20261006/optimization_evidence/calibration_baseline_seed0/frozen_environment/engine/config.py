"""Deployment paths; this module never loads candidate code."""
import hashlib
import json
import os
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
TASK = ENGINE.parent.parent
DEPLOYMENT = Path(os.environ.get('DD_DEPLOYMENT', '/opt/dd/deployment.json'))


def deployment():
    value = json.loads(DEPLOYMENT.read_text())
    for name in ('assets', 'repo', 'python', 'torch_home', 'runs'):
        if not isinstance(value.get(name), str) or not Path(value[name]).is_absolute():
            raise ValueError('Invalid deployment field: ' + name)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.stage')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def verify_assets():
    root = Path(deployment()['assets'])
    manifest = json.loads((ENGINE.parent / 'assets.json').read_text())
    for row in manifest['files']:
        path = root / row['file']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Frozen asset mismatch: ' + row['file'])
    return manifest
