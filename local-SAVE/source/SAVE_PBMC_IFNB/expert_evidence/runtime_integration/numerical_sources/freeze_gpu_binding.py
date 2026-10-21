"""Freeze trusted numerical sources and input fingerprints, without credentials."""
import argparse,hashlib,json
from pathlib import Path
FILES=['freeze_gpu_binding.py','gpu_jail_entry.py','gpu_jail_launcher.py','gpu_jail_supervisor.py',
       'gpu_fixed_evaluator.py','gpu_service.py','gpu_lease.py','independent_metrics.py']
def current(root):
    files={name:hashlib.sha256((root/'preparation'/name).read_bytes()).hexdigest() for name in FILES}
    files['task_frozen_assets']=hashlib.sha256((root/'submission/workspace/harbor_task/tests/frozen_assets.json').read_bytes()).hexdigest()
    files['private_contract']=hashlib.sha256((root/'private/private_verifier_contract_v1.json').read_bytes()).hexdigest()
    files['active_entry']=hashlib.sha256((root/'private/gpu-jail-runtime-v2-active-entry.json').read_bytes()).hexdigest()
    return files
def verify(root):
    binding=json.loads((root/'private/numerical-runtime-binding.json').read_text())
    actual=current(root)
    if actual!=binding['files']:raise RuntimeError('Frozen numerical sources changed; evaluation stopped')
    return binding['sha256']
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    files=current(a.root);digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    out=a.root/'private/numerical-runtime-binding.json';assert not out.exists()
    out.write_text(json.dumps({'files':files,'sha256':digest},indent=2));print(digest)
if __name__=='__main__':main()
