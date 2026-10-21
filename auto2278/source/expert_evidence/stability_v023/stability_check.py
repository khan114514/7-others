"""Twelve-hour environment liveness with a real repeated numerical workload.

This is never counted as Coding Agent optimization time.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
spec=importlib.util.spec_from_file_location('baseline','/workspace/environment/starter/method.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
g=torch.Generator().manual_seed(211)
x=torch.randn(1,128,640,generator=g)
w=[torch.randn(10,64,640,generator=g)/640**.5 for _ in range(3)]
start=time.monotonic();iteration=0
while time.monotonic()-start < 43200:
 iteration+=1
 with torch.inference_mode():y=m.attention(x,*w,1.1,5.)
 assert torch.isfinite(y).all()
 status=Path('/proc/self/status').read_text()
 memory=[line for line in status.splitlines() if line.startswith(('VmRSS:','VmSize:'))]
 print(json.dumps({'iteration':iteration,'unix':time.time(),'observed_seconds':time.monotonic()-start,
                   'numerical_finite':True,'fd_count':len(list(Path('/proc/self/fd').iterdir())),
                   'memory':memory,'free_bytes':__import__('shutil').disk_usage('/workspace').free,
                   'agent_effective_seconds':0}),flush=True)
 time.sleep(60)
print(json.dumps({'status':'completed_12h_environment_check','observed_seconds':time.monotonic()-start,
                 'agent_effective_seconds':0}),flush=True)
