"""Materialize pinned public CLI dependencies before the frozen Docker build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected_url = 'https://registry.npmjs.org/@openai/codex/-/codex-0.160.0-linux-arm64.tgz'
    if manifest['url'] != expected_url:
        raise ValueError('The public dependency URL differs from the pinned release')
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='bifrost-public-cli-') as scratch:
        archive = args.archive
        if archive is None:
            archive = Path(scratch) / 'public-cli.tgz'
            with urllib.request.urlopen(expected_url, timeout=60) as incoming, archive.open('wb') as output:
                while True:
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        if digest(archive) != manifest['archive_sha256']:
            raise ValueError('The public archive SHA256 differs from the pinned release')
        with tarfile.open(archive, 'r:gz') as source:
            for name, specification in manifest['binaries'].items():
                if name not in {'codex', 'codex-code-mode-host', 'rg'}:
                    raise ValueError('Unexpected executable name')
                target = destination / name
                if target.is_symlink():
                    raise ValueError('A dependency destination is a symbolic link')
                if target.exists():
                    if target.stat().st_size != specification['bytes'] or digest(target) != specification['sha256']:
                        raise ValueError('An existing dependency differs from the pinned bytes: ' + name)
                else:
                    matches = [item for item in source.getmembers() if item.name == specification['member']]
                    if len(matches) != 1 or not matches[0].isfile() or matches[0].size != specification['bytes']:
                        raise ValueError('The pinned archive member is missing or has an unexpected type or size')
                    incoming = source.extractfile(matches[0])
                    temporary = Path(scratch) / name
                    with incoming, temporary.open('wb') as output:
                        while True:
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                    if digest(temporary) != specification['sha256']:
                        raise ValueError('The public executable SHA256 differs: ' + name)
                    with temporary.open('rb') as incoming, target.open('xb') as output:
                        while True:
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                os.chmod(target, 0o755)
                rows.append({'name': name, 'bytes': target.stat().st_size, 'sha256': digest(target)})
    print(json.dumps({'status': 'PINNED_PUBLIC_CLI_ASSETS_MATERIALIZED', 'files': rows}))


if __name__ == '__main__':
    main()
