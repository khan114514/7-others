"""Run non-destructive attacks on the declared Linux deployment, not on macOS."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'environment/engine'))
from config import ENGINE, deployment
from method_proxy import MethodProxy
from supervisor import verify_code


def main():
    import torch
    root = Path(deployment()['runs'])
    os.environ['DD_RUN'] = str(root)
    system_canary = Path('/etc')/('dd-kernel-probe-' + uuid.uuid4().hex)
    run_canary = root/('kernel-canary-' + uuid.uuid4().hex)
    # These contain only synthetic text, never a real host credential.
    for path in [system_canary, run_canary]:
        path.write_text('private synthetic canary\n')
        path.chmod(0o600)
    source = """from types import SimpleNamespace
import socket, os, stat
from pathlib import Path
outcomes={}
def fork_probe():
 pid=os.fork()
 if pid==0:os._exit(0)
 os.waitpid(pid,0)
for name,action in [
 ('frozen_write',lambda:Path(FROZEN).write_text('tampered')),
 ('frozen_trainer_read',lambda:Path(FROZEN).read_text()),
 ('other_run_read',lambda:Path(RUN_CANARY).read_text()),
 ('system_private_canary_read',lambda:Path(SYSTEM_CANARY).read_text()),
 ('network',lambda:socket.socket(socket.AF_INET,socket.SOCK_STREAM)),
 ('unix_socket_creation',lambda:socket.socket(socket.AF_UNIX)),
 ('parent_signal_zero',lambda:os.kill(os.getppid(),0)),
 ('new_process',fork_probe),
 ('device_creation',lambda:os.mknod(Path(os.environ['TMPDIR'])/'synthetic_null',stat.S_IFCHR|0o600,os.makedev(1,3))),
 ('gpu_device',lambda:os.open(GPU_DEVICE,os.O_RDONLY))]:
 try:
  value=action()
  if isinstance(value,socket.socket):value.close()
  outcomes[name]='ALLOWED'
 except PermissionError:outcomes[name]='DENIED'
def get_method():return SimpleNamespace(name='boundary_probe',teacher=SimpleNamespace(num_samples=0),train_notes=str(outcomes))
def fm_time_sampler(batch_shape,device,generator=None):
 if batch_shape==tuple([1]):
  import time
  time.sleep(60)
 import torch
 return .5+.5*torch.rand(batch_shape,device=device,generator=generator)
def select_teacher_candidates(agreement,multiplier):
 import torch
 return torch.arange(len(agreement)//multiplier)*multiplier+agreement.reshape(-1,multiplier).argmin(1)
def pair_weights(class_labels,target_ids):return 1+(class_labels%2).float()
""".replace('FROZEN',repr(str(ENGINE/'framework/train_student.py'))).replace(
            'RUN_CANARY',repr(str(run_canary))).replace('SYSTEM_CANARY',repr(str(system_canary))).replace(
            'GPU_DEVICE',repr(str(next(p for p in Path('/dev').glob('nvidia*') if p.name[6:].isdigit()))))
    try:
        with tempfile.TemporaryDirectory(prefix='probe-source-',dir=root) as folder:
            path=Path(folder)/'method.py'
            path.write_text(source)
            frozen=(ENGINE/'framework/train_student.py').read_bytes()
            proxy=MethodProxy(path)
            try:
                outcomes=ast.literal_eval(proxy.spec.train_notes)
                if len(outcomes)!=10 or set(outcomes.values())!={'DENIED'}:
                    raise RuntimeError('A forbidden operation was permitted')
                if (ENGINE/'framework/train_student.py').read_bytes()!=frozen:
                    raise RuntimeError('Frozen trainer changed')
                generator=torch.Generator().manual_seed(123)
                times=proxy.fm_time_sampler([16],'cpu',generator)
                expected=.5+.5*torch.rand([16],generator=torch.Generator().manual_seed(123))
                indices=proxy.select_teacher_candidates(torch.tensor([4.,2.,1.,3.]),2)
                weights=proxy.pair_weights(torch.tensor([0,1,2,3]),torch.tensor([1,2,3,4]))
                if not torch.equal(times,expected) or indices.tolist()!=[1,2] or abs(weights.mean().item()-1)>1e-6:
                    raise RuntimeError('Legitimate tensor RPC failed')
                try:
                    proxy.request({'op':'time','shape':[1],'rng':torch.Generator().get_state().tolist()},timeout=.5)
                except TimeoutError:
                    if proxy.process.poll() is None:raise RuntimeError('Timed-out worker survived')
                else:raise RuntimeError('Timed-out method was accepted')
                print(json.dumps({'kernel_attacks':outcomes,'timeout_process_killed':True,
                    'legitimate_cpu_tensor_rpcs_passed':True,'frozen_code_preserved':True,
                    'frozen_manifest_sha256':verify_code(),'guard':proxy.guard}))
            finally:proxy.close()
    finally:
        for path in [system_canary,run_canary]:path.unlink(missing_ok=True)


if __name__=='__main__':main()
