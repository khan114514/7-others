"""Root-only portable source/injection guards; candidate code remains data."""
import fcntl,hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path

def regular(path, limit=64 * 1024 * 1024):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('Private input cannot traverse a symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022 or info.st_size > limit:
            raise ValueError('Bounded immutable-operator regular input required')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        final = os.fstat(fd)
        if len(raw) != info.st_size or (final.st_mtime_ns, final.st_size) != (info.st_mtime_ns, info.st_size):
            raise ValueError('Private input changed during read')
        return raw
    finally:
        os.close(fd)

def sha(path):
    return hashlib.sha256(regular(path)).hexdigest()

def atomic(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)

def wait_for_lease(config):
    # Only the resource-wait policy changes relative to the old smoke helper;
    # scorer/decoder/metric/source bytes remain immutable. No wall cutoff.
    while True:
        paths = {Path(p) for p in config['gpu_lock_files']}
        paths.update(p for p in Path('/tmp').glob('*gpu*lock') if p.is_file())
        held = []
        try:
            for path in sorted(paths):
                stream = path.open('a+')
                held.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = {k: v for k, v in os.environ.items() if k != 'LD_PRELOAD'}
            value = subprocess.check_output([config['nvidia_smi'], '--query-gpu=memory.used',
                         '--format=csv,noheader,nounits'], env=env, text=True)
            if int(value.strip().splitlines()[0]) < 256:
                return held
        except BlockingIOError:
            pass
        except Exception:
            for stream in held:
                stream.close()
            raise
        for stream in held:
            stream.close()
        time.sleep(10)


def verify_installed_sources(root):
    from deployment_paths import PROJECT_ROOT, PRIVATE_ROOT
    if os.geteuid() != 0 or root.resolve() != root or root != PROJECT_ROOT:
        raise ValueError('Exact trusted configured deployment required')
    info = PRIVATE_ROOT.stat()
    if info.st_uid != 0 or info.st_mode & 0o077:
        raise PermissionError('Backend private directory must be root-only')
    manifest = json.loads(regular(PRIVATE_ROOT / 'portable-private-route-source-manifest.json'))
    if manifest['hidden_rows_in_bundle'] != 0:
        raise ValueError('Public bundle must contain no private evaluation rows')
    for row in manifest['code_and_frozen_inputs']:
        path = Path(row['deployment_path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'private':
            raise ValueError('Fixed private relative deployment paths required')
        raw = regular(root / path)
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Installed trusted source differs from frozen bundle')
    return manifest


def verify_backend(root):
    from deployment_paths import CONFIG, PRIVATE_ROOT
    manifest = verify_installed_sources(root)
    receipt = json.loads(regular(PRIVATE_ROOT / 'portable-full-install-receipt.json'))
    if (receipt['phase'] != 'FULL_BACKEND_INPUT_HASHES_PASS_GPU_NATIVE_VALIDATION_PENDING'
            or receipt['asset_manifest_sha256'] != manifest['asset_manifest_sha256']):
        raise ValueError('Actual full installation/hash verification required before private GPU work')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(PRIVATE_ROOT))
    return root, CONFIG


def injection(root, canonical):
    # Platform/expert supplies the exact frozen private dataset directly into
    # this root-only directory. No author machine path or public row fallback.
    from deployment_paths import PRIVATE_ROOT, PROJECT_ROOT
    if root != PROJECT_ROOT or canonical != root:
        raise ValueError('Configured private injection root required')
    target = PRIVATE_ROOT / 'hidden-injected-v1'
    info = target.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        raise PermissionError('Private inputs must be root-only')
    contract = json.loads(regular(PRIVATE_ROOT / 'hidden-validation-protocol-v2.json'))
    if sha(PRIVATE_ROOT / 'policy-score-protocol-v2.json') != contract['dev_protocol_sha256']:
        raise ValueError('Frozen protocol binding differs')
    if sha(target / 'samples.json') != contract['hidden_samples_sha256']:
        raise ValueError('Exact private samples must be externally injected')
    rows = json.loads(regular(target / 'samples.json'))
    if len(rows) != contract['hidden_samples_per_seed'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Private corpus row count or identifiers differ')
    for row in rows:
        relative = Path(row['image'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'images':
            raise ValueError('Bounded injected image path required')
        if sha(target / relative) != row['image_sha256']:
            raise ValueError('Injected private image hash differs')
    return {'private_rows': len(rows), 'rows_sha256': contract['hidden_samples_sha256'],
            'private_payload_bytes_copied': 0, 'public_hidden_payload_present': False}
