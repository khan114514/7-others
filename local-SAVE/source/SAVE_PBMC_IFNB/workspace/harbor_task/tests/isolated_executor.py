"""Namespace executor: no labels, scoring code, secrets or peer directories.

Fail closed on a namespace failure. GPU/cgroup isolation is also a platform
prerequisite; this task-level executor does not certify an unverified provider.
"""
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys

class IsolationUnavailable(RuntimeError):
    pass

def limits():
    resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def run_candidate(method, train, controls, backend, runner, output, seed, log, reload=False):
    bwrap=shutil.which('bwrap')
    if not bwrap or not Path('/opt/runtime/bin/python').exists():
        raise IsolationUnavailable('Requires built task image with bubblewrap and /opt/runtime')
    args=[bwrap,'--die-with-parent','--new-session','--unshare-all',
          '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin',
          '--symlink','usr/sbin','/sbin','--symlink','usr/lib','/lib',
          '--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev',
          '--tmpfs','/tmp','--dir','/opt','--dir','/workspace','--dir','/data',
          '--ro-bind','/opt/runtime','/opt/runtime',
          '--ro-bind',str(backend),'/opt/backend',
          '--ro-bind',str(runner),'/opt/train_eval.py',
          '--ro-bind',str(method),'/workspace/method.py',
          '--ro-bind',str(train),'/data/train.h5ad',
          '--ro-bind',str(controls),'/data/controls.h5ad',
          '--bind',str(output),'/out','--chdir','/workspace','--clearenv',
          '--setenv','PATH','/opt/runtime/bin:/usr/bin',
          '--setenv','PYTHONPATH','/opt/backend',
          '--setenv','HOME','/tmp','--setenv','MPLCONFIGDIR','/tmp/mpl',
          '--setenv','NUMBA_CACHE_DIR','/tmp/numba',
          '--setenv','XDG_CACHE_HOME','/tmp/cache',
          '--setenv','OMP_NUM_THREADS','4','--setenv','MKL_NUM_THREADS','4',
          '--setenv','PYTHONHASHSEED',str(seed)]
    for device in sorted(Path('/dev').glob('nvidia*')):
        if device.is_char_device():
            args+=['--dev-bind',str(device),str(device)]
    if Path('/proc/driver/nvidia').exists():
        args+=['--ro-bind','/proc/driver/nvidia','/proc/driver/nvidia']
    probe=subprocess.run(args+['/bin/true'],capture_output=True,text=True,timeout=20)
    if probe.returncode:
        raise IsolationUnavailable('Namespace prerequisite failed: '+probe.stderr[-1200:])
    # Directory /out exists as a mount. A child path stays absent for fresh-run guard.
    command=args+['/opt/runtime/bin/python','/opt/train_eval.py',
                  '--method','/workspace/method.py','--train','/data/train.h5ad',
                  '--controls','/data/controls.h5ad','--out','/out/run','--seed',str(seed)]
    if reload:
        command+=['--reload']
    with log.open('w') as stream:
        return subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                              timeout=120 if reload else 900,preexec_fn=limits,
                              env={'PATH':'/usr/bin:/bin'}).returncode
