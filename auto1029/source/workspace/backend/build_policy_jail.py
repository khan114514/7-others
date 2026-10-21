"""Build task-owned immutable Python stdlib and ELF runtime; no site packages."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from deployment_paths import PRIVATE_ROOT as ROOT, STDLIB_ROOT, JAIL_PYTHON
BASE = ROOT / 'policy-jail-base-v1'
BASE.mkdir(mode=0o700)
source = STDLIB_ROOT
dest = BASE / source.relative_to('/')
shutil.copytree(source, dest, ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'tests', 'idlelib', 'ensurepip', 'tkinter'))
files = [JAIL_PYTHON, Path('/lib/x86_64-linux-gnu/libseccomp.so.2')]
files += list(source.glob('lib-dynload/*.so'))
environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
seen = set()
while files:
    file = files.pop()
    name = str(file)
    if name in seen:
        continue
    seen.add(name)
    target = BASE / file.relative_to('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(file, target)
    value = subprocess.run(['/usr/bin/ldd', name], env=environment, capture_output=True, text=True)
    for match in re.finditer(r'(?:=>\s+)?(/[^\s]+)\s+\(', value.stdout):
        files.append(Path(match.group(1)))
shutil.copyfile(ROOT / 'policy_jail_runner.py', BASE / 'runner.py')
for path in BASE.rglob('*'):
    if path.is_dir():
        path.chmod(0o755)
    else:
        path.chmod(0o755 if path.name == 'python3.12' or 'ld-linux' in path.name else 0o644)
BASE.chmod(0o755)
manifest = {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(BASE.rglob('*')) if path.is_file()}
(ROOT / 'policy-jail-base-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'runtime_files': len(manifest), 'bytes': sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()), 'site_packages_copied': False, 'secrets_copied': False}))
