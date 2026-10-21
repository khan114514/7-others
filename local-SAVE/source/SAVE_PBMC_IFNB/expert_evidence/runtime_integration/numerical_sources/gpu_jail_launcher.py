"""Trusted launcher. Candidate code is never imported by the GPU host process."""
import argparse,ctypes,os,resource,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--jail',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
p.add_argument('--uid',type=int,required=True);p.add_argument('--mode',choices=['probe','run'],default='probe')
p.add_argument('--seed',type=int,default=17);p.add_argument('--reload',action='store_true')
p.add_argument('--checkpoint-fd',type=int);p.add_argument('--prediction-fd',type=int);p.add_argument('--receipt-fd',type=int);a=p.parse_args()
jail=a.jail.resolve();jail.relative_to(a.root.resolve()/'private/gpu-jails');assert 54000<=a.uid<64000
resource.setrlimit(resource.RLIMIT_DATA,(32*1024**3,)*2)
resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,)*2)
resource.setrlimit(resource.RLIMIT_NOFILE,(256,256));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
runtime=a.root.resolve()/'preparation/remote_venv/bin/python'
argv=[str(runtime),'-I','-B',str(jail/'runner.py'),'--uid',str(a.uid),'--mode',a.mode,'--jail',str(jail),'--seed',str(a.seed)]
if a.reload:argv+=['--reload']
if a.mode=='run':
    assert all(isinstance(f,int) and 3<=f<256 for f in [a.checkpoint_fd,a.prediction_fd,a.receipt_fd])
    argv+=['--checkpoint-fd',str(a.checkpoint_fd),'--prediction-fd',str(a.prediction_fd),'--receipt-fd',str(a.receipt_fd)]
os.execve(str(runtime),argv,
 {'PATH':'/opt/runtime/bin:/usr/bin:/bin','HOME':str(jail/'tmp'),'LANG':'C.UTF-8','TMPDIR':str(jail/'tmp'),
  'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
  'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(jail/'tmp/mpl'),'NUMBA_CACHE_DIR':str(jail/'tmp/numba'),
  'XDG_CACHE_HOME':str(jail/'tmp/cache'),'CUDA_CACHE_DISABLE':'1',
  'LD_LIBRARY_PATH':'/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/opt/runtime/lib/python3.10/site-packages/torch/lib'})
