"""Inject trusted assets into a newly exported separate Verifier context only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def digest_tree(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native-task',type=Path,required=True)
    p.add_argument('--private-assets',type=Path,required=True)
    p.add_argument('--private-contract',type=Path,required=True)
    a=p.parse_args()
    task=a.native_task.resolve()
    contract=json.loads(a.private_contract.read_text())
    before=digest_tree(task/'environment')
    hidden=task/'tests/hidden_assets'
    hidden.mkdir(exist_ok=True)
    if any(hidden.iterdir()):
        raise RuntimeError('Private asset destination must be empty')
    for name in ['controls.h5ad','targets.h5ad']:
        path=a.private_assets/name
        if path.is_symlink() or sha(path)!=contract['assets'][name]['sha256']:
            raise RuntimeError('Trusted private asset fingerprint mismatch')
        shutil.copyfile(path,hidden/name)
    anchors=json.loads((task/'tests/anchors.json').read_text())
    anchors['hidden']=contract['hidden_anchor']
    (task/'tests/anchors.json').write_text(json.dumps(anchors,indent=2))
    frozen=json.loads((task/'tests/frozen_assets.json').read_text())
    frozen['/tests/anchors.json']=sha(task/'tests/anchors.json')
    (task/'tests/frozen_assets.json').write_text(json.dumps(frozen,indent=2))
    (task/'tests/private_asset_manifest.json').write_text(json.dumps({name:sha(hidden/name) for name in ['controls.h5ad','targets.h5ad']},indent=2))
    assert digest_tree(task/'environment')==before
    receipt={'private_verifier_assets_injected':True,'agent_context_unchanged':True,
             'official_hidden_verified':False,'provider_isolation_verified':False,
             'verifier_private_asset_sha256':{name:sha(hidden/name) for name in ['controls.h5ad','targets.h5ad']}}
    (task.parent/(task.name+'-private-injection.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
