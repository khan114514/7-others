"""Explicit teaching-task-root-v1 -> Harbor0.23 native contexts adapter.

No private assets are injected by export. The private test mount must be
provisioned into the separate verifier by a verified platform provider.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',type=Path)
    parser.add_argument('out',type=Path)
    a=parser.parse_args()
    if a.out.exists():
        raise RuntimeError('Use an empty new native task directory')
    shutil.copytree(a.task,a.out,ignore=shutil.ignore_patterns('__pycache__'))
    source=(a.task/'environment/Dockerfile').read_text()
    native=source.replace('COPY environment/requirements.txt','COPY requirements.txt')
    native=native.replace('COPY environment/starter','COPY starter')
    native=native.replace('COPY environment/public_assets','COPY public_assets')
    native=native.replace('COPY environment/public_eval','COPY public_eval')
    native=native.replace('COPY solution /workspace/solution','COPY initial_solution /workspace/solution')
    (a.out/'environment/Dockerfile').write_text(native)
    shutil.copytree(a.task/'solution',a.out/'environment/initial_solution')
    config=(a.task/'task.toml').read_text().replace('teaching-task-root-v1','harbor-environment-v1')
    (a.out/'task.toml').write_text(config)
    manifest={'adapter':'teaching-to-native-v1','target_harbor_version':'0.23.0',
              'source_task_digest':{},'native_task_digest':{},
              'agent_context':'environment','verifier_context':'tests',
              'hidden_injected':False,'runtime_verified':False}
    for directory,key in [(a.task,'source_task_digest'),(a.out,'native_task_digest')]:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                manifest[key][str(p.relative_to(directory))]=sha(p)
    (a.out.parent/(a.out.name+'-adapter.json')).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
