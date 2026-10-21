"""Root-side installer and full hash checker; never loads candidate code."""
import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


def relative(value):
    path = Path(value)
    if not isinstance(value, str) or path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe logical deployment path')
    return path


def checked_file(path, record, allow_trusted_asset_link=False):
    if allow_trusted_asset_link:
        path = path.resolve(strict=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('Input must be a trusted root-owned regular file')
        if info.st_size != record['bytes']:
            raise ValueError('Input size differs from frozen manifest')
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError('Input hash differs from frozen manifest')
    finally:
        os.close(fd)


def install(source, target, record):
    checked_file(source, record)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        checked_file(target, record)
        return 'ALREADY_IDENTICAL'
    raw = source.read_bytes()
    # Inputs are trusted, small code/data files; no weight copy or evaluation.
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError('Frozen lightweight input too large')
    fd, name = tempfile.mkstemp(prefix='.heima-install-', dir=target.parent)
    pending = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if another writer won.
        os.link(pending, target, follow_symlinks=False)
    finally:
        pending.unlink(missing_ok=True)
    checked_file(target, record)
    return 'INSTALLED'


def main():
    if os.geteuid() != 0:
        raise PermissionError('Backend installation requires trusted root')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--code-only', action='store_true', help='Install small inputs; NOT READY for scoring')
    parser.add_argument('--receipt-name', default='portable-install-receipt.json')
    args = parser.parse_args()
    receipt_name = relative(args.receipt_name)
    if len(receipt_name.parts) != 1 or receipt_name.suffix != '.json':
        raise ValueError('Receipt must be a fresh JSON filename')
    os.environ['HEIMA_DEPLOYMENT_CONFIG'] = str(args.config.absolute())
    from deployment_paths import PRIVATE_ROOT, PROJECT_ROOT, TORCHTUNE_PACKAGE
    bundle = Path(__file__).resolve().parent
    manifest_path = bundle / 'asset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['hidden_rows_in_bundle'] != 0:
        raise ValueError('Unexpected Hidden rows in public deployment bundle')
    installed = []
    for record in manifest['code'] + manifest['frozen_inputs']:
        source = bundle / relative(record['bundle_path'])
        path = relative(record['deployment_path'])
        if path.parts[0] != 'private':
            raise ValueError('Lightweight backend inputs must remain private')
        target = PROJECT_ROOT / path
        installed.append({'path': path.as_posix(), 'status': install(source, target, record)})
    receipt = {'phase': 'CODE_AND_DEV_INSTALLED_ASSETS_NOT_VERIFIED', 'files': installed,
               'asset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'candidate_code_loaded': False, 'hidden_rows_installed': 0,
               'gpu_used': False, 'model_research_seconds': 0,
               'portable_gpu_or_native_score_claim': False}
    if not args.code_only:
        verified = []
        for record in manifest['external_assets']:
            path = relative(record['deployment_path'])
            checked_file(PROJECT_ROOT / path, record, allow_trusted_asset_link=True)
            verified.append(path.as_posix())
        for record in manifest['trusted_torchtune_package']['files']:
            checked_file(TORCHTUNE_PACKAGE / relative(record['package_path']), record)
        receipt.update(phase='FULL_BACKEND_INPUT_HASHES_PASS_GPU_NATIVE_VALIDATION_PENDING',
                       verified_external_assets=verified,
                       trusted_torchtune_files_verified=len(manifest['trusted_torchtune_package']['files']))
    PRIVATE_ROOT.mkdir(mode=0o700, exist_ok=True)
    source_snapshot = {
        'asset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'hidden_rows_in_bundle': manifest['hidden_rows_in_bundle'],
        'code_and_frozen_inputs': manifest['code'] + manifest['frozen_inputs']}
    source_path = PRIVATE_ROOT / 'portable-private-route-source-manifest.json'
    source_raw = (json.dumps(source_snapshot, indent=2) + '\n').encode()
    if source_path.exists():
        if source_path.is_symlink() or source_path.read_bytes() != source_raw:
            raise ValueError('Preserve nonidentical installed source snapshot')
    else:
        with source_path.open('xb') as stream:
            stream.write(source_raw)
        source_path.chmod(0o600)

    target = PRIVATE_ROOT / receipt_name
    # A previous proof is retained; installation cannot rewrite its receipt.
    with target.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'phase': receipt['phase'], 'installed_files': len(installed),
                      'candidate_code_loaded': False, 'gpu_used': False, 'model_research_seconds': 0}))


if __name__ == '__main__':
    main()
