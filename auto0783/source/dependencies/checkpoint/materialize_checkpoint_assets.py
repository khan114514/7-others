"""Restore checksum-bound public model assets outside candidate-visible paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import urllib.request
import uuid

REVISION = '60bbb8b163827b9324d72bd84d6ddc75ebb902bb'
REPOSITORY = 'hanlincs/Bifrost-1'
TENSOR_MANIFEST_SHA = '61d7fabc0298197c99c63317cfc906360d3dfd06e78741cdf4ac728ecf5d94fe'


def verify_descriptor(descriptor, tensor_bytes):
    if (descriptor['schema_version'] != 1 or descriptor['repo_id'] != REPOSITORY
            or descriptor['revision'] != REVISION or descriptor['license'] != 'apache-2.0'
            or descriptor['manifest_sha256'] != TENSOR_MANIFEST_SHA
            or hashlib.sha256(tensor_bytes).hexdigest() != TENSOR_MANIFEST_SHA):
        raise ValueError('Public checkpoint identity or tensor manifest differs')
    tensor = json.loads(tensor_bytes)
    if (tensor['checkpoint_revision'] != REVISION or tensor['tensor_count'] != 677
            or descriptor['decoder_tensor_count'] != 677
            or descriptor['decoder_tensor_bytes'] != tensor['tensor_bytes']
            or set(descriptor['files']) != set(tensor['checkpoint_shards'])):
        raise ValueError('Declared public checkpoint tensor set differs')
    for name, item in descriptor['files'].items():
        if not re.fullmatch(r'model-0000[1-6]-of-00006\.safetensors', name):
            raise ValueError('Unexpected public shard name')
        bound = tensor['checkpoint_shards'][name]
        expected_url = 'https://huggingface.co/' + REPOSITORY + '/resolve/' + REVISION + '/' + name
        if (item['url'] != expected_url or item['sha256'] != bound['sha256']
                or item['bytes'] != bound['bytes'] or not 0 < item['bytes'] <= 5 * 1024**3):
            raise ValueError('Public shard URL, size or checksum differs')
    return tensor


def verify_file(path, specification):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != specification['bytes']:
            raise ValueError('Existing public asset is not the declared regular file')
        digest = hashlib.sha256(); size = 0
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block: break
            size += len(block); digest.update(block)
            if size > specification['bytes']: raise ValueError('Public asset grew beyond its declared size')
        after = os.fstat(descriptor)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Public asset changed during checksum verification')
        if size != specification['bytes'] or digest.hexdigest() != specification['sha256']:
            raise ValueError('Existing public asset checksum differs')
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tensor-manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--cache-only', action='store_true', help='Verify existing shards; never access the network')
    parser.add_argument('--plan-only', action='store_true', help='Validate the immutable descriptor without reading or downloading shards')
    args = parser.parse_args()
    descriptor = json.loads(args.manifest.read_text())
    tensor = verify_descriptor(descriptor, args.tensor_manifest.read_bytes())
    if args.plan_only:
        print(json.dumps({'status': 'PUBLIC_CHECKPOINT_DESCRIPTOR_VERIFIED_ONLY',
                          'materialization_complete': False, 'shards': len(descriptor['files']),
                          'tensor_count': tensor['tensor_count'], 'new_download': False}))
        return
    if args.destination.is_symlink(): raise ValueError('Asset destination must be a real directory')
    args.destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, specification in descriptor['files'].items():
        target = args.destination / name
        if target.exists() or target.is_symlink():
            verify_file(target, specification)
        elif args.cache_only:
            raise ValueError('Declared cached public shard is missing')
        else:
            pending = args.destination / ('.public-shard-' + uuid.uuid4().hex)
            try:
                digest = hashlib.sha256(); size = 0
                with urllib.request.urlopen(specification['url'], timeout=180) as incoming, pending.open('xb') as out:
                    while True:
                        block = incoming.read(8 * 1024 * 1024)
                        if not block: break
                        size += len(block)
                        if size > specification['bytes']: raise ValueError('Public download exceeded declared size')
                        digest.update(block); out.write(block)
                    out.flush(); os.fsync(out.fileno())
                if size != specification['bytes'] or digest.hexdigest() != specification['sha256']:
                    raise ValueError('Downloaded public shard binding differs')
                os.link(pending, target, follow_symlinks=False)
                verify_file(target, specification)
            finally:
                if pending.exists(): pending.unlink()
        rows.append({'name': name, 'bytes': specification['bytes'], 'sha256': specification['sha256']})
    print(json.dumps({'status': 'PUBLIC_CHECKPOINT_ASSETS_MATERIALIZED_AND_ALL_SHA_VERIFIED',
                      'materialization_complete': True, 'files': rows}))


if __name__ == '__main__':
    main()
