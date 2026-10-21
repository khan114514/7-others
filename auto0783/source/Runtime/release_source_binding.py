"""Source identities for freshly materialized Task and portable native runtime."""
import hashlib
import json
import stat
from pathlib import Path

RUNTIME=Path(__file__).resolve().parent
TASK_SHA='2abb0bb70731458dbf320b26be53b0f219a1f3807838d73201219e83333f386e'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block=stream.read(1024*1024)
            if not block:break
            value.update(block)
    return value.hexdigest()


def file_manifest(root):
    output={}
    if root.is_symlink() or not root.is_dir():raise ValueError('Expected a real frozen source root')
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Frozen source symlinks are forbidden')
        mode=path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise ValueError('Frozen source special files are forbidden')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':output[path.relative_to(root).as_posix()]=digest(path)
    return output


def current_binding(task_path):
    task_path=Path(task_path);manifest=file_manifest(task_path)
    task_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if len(manifest)!=48 or task_sha!=TASK_SHA:raise ValueError('Task differs from the materialized release source identity')
    sources={}
    for directory,prefix in [(RUNTIME,'Runtime'),(RUNTIME.parent/'dependencies','dependencies')]:
        for name,sha in file_manifest(directory).items():
            if name.endswith(('.py','.json')) or name.endswith('requirements.txt') or name.endswith('requirements-worker.txt'):sources[prefix+'/'+name]=sha
    package=RUNTIME.parent
    for name in ['selected-method-identities.json','Trajectories/GPT/selected-method/method.py','Trajectories/GPT/selected-method/program.json','Trajectories/Seed/selected-method/method.py','Trajectories/Seed/selected-method/program.json']:
        path=package/name
        if path.is_symlink() or not path.is_file():raise ValueError('Frozen selected artifact is missing or indirect')
        sources[name]=digest(path)
    return {'task_source_manifest_sha256':task_sha,'runtime_sources':sources}


def require_current_isolation(task_path,proof_path):
    proof=json.loads(Path(proof_path).read_text());expected=current_binding(task_path)
    if proof.get('status')!='PASS' or any(proof.get(k)!=v for k,v in expected.items()):raise RuntimeError('Fresh native isolation proof does not bind the current materialized Task and public runtime')
    return expected
