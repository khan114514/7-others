"""Trusted durable A100 worker. Only bounded tensor-graph requests enter stdin."""
import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import re
import sys
import subprocess
import time
import uuid
from pathlib import Path
import torch
from checkpoint_loader import load_public_tensors
from full_decoder_evaluator import FullDecoderEvaluator

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--lock',action='append',required=True)
args=parser.parse_args();root=Path(args.root);worker_id=uuid.uuid4().hex
run_dir=root/'numerical-workers'/worker_id;run_dir.mkdir(parents=True,exist_ok=False)
state={'worker_id':worker_id,'pid':os.getpid(),'formal_trajectory_effective_seconds':0}
def status(phase,**fields):
    value={**state,'worker_pid':state['pid'],'pid':os.getpid(),'phase':phase,'epoch':time.time(),**fields}
    pending=run_dir/'status.pending.json';pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(run_dir/'status.json')
    print(json.dumps({'type':'status',**value}),flush=True)

status('INITIALIZING_TRUSTED_WORKER')
assert torch.__version__=='2.5.1+cu121','Both workers must use the same actual pinned Torch version'
weights,binding=load_public_tensors('http://127.0.0.1:19195',args.manifest_sha256,status)
(run_dir/'checkpoint-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
baseline=json.loads((root/'baseline-program.json').read_text())
source_binding={'checkpoint_binding_sha256':hashlib.sha256((run_dir/'checkpoint-binding.json').read_bytes()).hexdigest(),
    'manifest_sha256':args.manifest_sha256,'baseline_program_sha256':hashlib.sha256(json.dumps(baseline,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'source_sha256s':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')},
    'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,'gpu_uuid':args.gpu_uuid}
(run_dir/'source-binding.json').write_text(json.dumps(source_binding,indent=2)+'\n')
evaluator=FullDecoderEvaluator(weights,root/'full-scout-inputs.safetensors',baseline,args.gpu_uuid,args.lock,status)
assert not torch.cuda.is_initialized(),'Trusted persistent CPU worker must never retain a CUDA context'
status('READY_FOR_BOUNDED_EVALUATION',tensor_count=len(weights),source_binding='source-binding.json')
def evaluate_one(request,directory,queued):
    # Fixed trusted compilation resources; each short-lived CUDA job has its own
    # private executable cache, with no cross-role or cross-job cache reuse.
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(directory/'compiler-cache')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
    try:
        value=evaluator.evaluate(request['program'],request['profile'],directory,preacquired_queue_seconds=queued)
        result={'execution_status':'COMPLETED',**value}
    except Exception as error:
        result={'execution_status':'FAILED','error_type':type(error).__name__,'error':str(error)[-1500:],
                'reward':0.0,'quality_pass':False,'formal_trajectory_effective_seconds':0}
    pending=directory/'child-result.pending.json';pending.write_text(json.dumps(result,indent=2)+'\n');pending.replace(directory/'child-result.json')
    # CUDA runs only in this short-lived child. The CPU parent owns the lease
    # until child exit AND observed GPU memory return to idle, so another task
    # never acquires our lock while a persistent idle CUDA context remains.
for line in sys.stdin:
    if len(line.encode())>512*1024:
        print(json.dumps({'type':'protocol_error','error':'Request size exceeded','worker_id':worker_id}),flush=True);continue
    try:
        request=json.loads(line)
        if set(request)!={'action','run_id','profile','program','artifact_sha256'} or request['action']!='evaluate':
            raise ValueError('Unsupported trusted request schema')
        if not isinstance(request['run_id'],str) or not re.fullmatch(r'[0-9a-f]{32}',request['run_id']):
            raise ValueError('Invalid run identity')
        canonical=json.dumps(request['program'],sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest()!=request['artifact_sha256']:raise ValueError('Candidate artifact hash mismatch')
        directory=run_dir/'jobs'/request['run_id'];directory.mkdir(parents=True,exist_ok=False)
        (directory/'program.json').write_bytes(canonical+b'\n')
        (directory/'request.json').write_text(json.dumps(request,indent=2)+'\n')
        status('STARTING_BOUNDED_EVALUATION',run_id=request['run_id'],profile=request['profile'])
        started=time.time();queued=evaluator.acquire()
        assert not torch.cuda.is_initialized(),'Trusted CPU parent must not own a CUDA context'
        child=multiprocessing.get_context('fork').Process(target=evaluate_one,args=(request,directory,queued))
        child.start();status('CUDA_CHILD_RUNNING',run_id=request['run_id'],child_pid=child.pid,queue_seconds=queued)
        child.join(timeout=7200)
        if child.is_alive():
            child.terminate();child.join(timeout=30)
            if child.is_alive():child.kill();child.join()
        child_result=directory/'child-result.json'
        result=json.loads(child_result.read_text()) if child_result.exists() else {
            'execution_status':'FAILED','error_type':'ChildExitedWithoutResult','child_exit_code':child.exitcode,
            'reward':0.0,'quality_pass':False,'formal_trajectory_effective_seconds':0}
        deadline=time.monotonic()+60
        while True:
            observed=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip()
            fields=observed.split(',')
            if fields[0].strip()==args.gpu_uuid and int(fields[1])<=64:break
            if time.monotonic()>deadline:raise RuntimeError('CUDA child exited but GPU memory did not return to idle; preserve the held lease')
            time.sleep(1)
        evaluator.close_job()
        result.update(run_id=request['run_id'],worker_id=worker_id,artifact_sha256=request['artifact_sha256'],
            started_epoch=started,completed_epoch=time.time(),source_binding_sha256=hashlib.sha256((run_dir/'source-binding.json').read_bytes()).hexdigest(),
            checkpoint_manifest_sha256=args.manifest_sha256,gpu_after_release=observed,
            child_pid=child.pid,child_exit_code=child.exitcode,parent_cuda_context_initialized=torch.cuda.is_initialized())
        pending=directory/'result.pending.json';pending.write_text(json.dumps(result,indent=2)+'\n');pending.replace(directory/'result.json')
        print(json.dumps({'type':'result',**result}),flush=True)
        status('READY_FOR_BOUNDED_EVALUATION',last_run_id=request['run_id'])
    except Exception as error:
        print(json.dumps({'type':'protocol_error','worker_id':worker_id,'error_type':type(error).__name__,'error':str(error)[-1000:]}),flush=True)
status('TRUSTED_WORKER_INPUT_CLOSED')
