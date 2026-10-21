"""Task-owned durable controller for a trusted CPU parent and leased CUDA children.

Inputs in this private inbox are authored by the trusted gateway/controller. The
candidate containers never mount this directory or receive SSH/provider secrets.
"""
import hashlib
import json
import os
import selectors
import shlex
import subprocess
import time
import uuid
from numerical_controller_config import load_config, worker_command, probe_command, MANIFEST_SHA
import fcntl
from pathlib import Path

CONFIG=load_config()
PRIVATE=Path(CONFIG['state_directory'])
if PRIVATE.is_symlink():raise ValueError('Controller state root must not be a symlink')
PRIVATE.mkdir(parents=True,exist_ok=True)
controller_guard=(PRIVATE/'controller.guard').open('a')
fcntl.flock(controller_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
run_id=uuid.uuid4().hex
root=PRIVATE/'numerical-controller'/run_id;root.mkdir(parents=True)
(root/'inbox').mkdir();(root/'results').mkdir()
manifest_sha=MANIFEST_SHA
command=worker_command(CONFIG)
process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
    stderr=(root/'ssh-and-worker-stderr.log').open('a'),text=True,bufsize=1)
state={'controller_id':run_id,'pid':os.getpid(),'ssh_pid':process.pid,'command':command,
       'root':str(root),'started_epoch':time.time(),'formal_trajectory_effective_seconds':0,
       'checkpoint_manifest_sha256':manifest_sha}
def status(phase,**fields):
    state.update(phase=phase,updated_epoch=time.time(),**fields)
    pending=root/'status.pending.json';pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(root/'status.json')
    pending=PRIVATE/'numerical-controller-latest.pending.json';pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(PRIVATE/'numerical-controller-latest.json')
    print(json.dumps({'phase':phase,'controller_id':run_id,**fields}),flush=True)
status('LOADING_TRUSTED_CPU_CHECKPOINT')
selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
ready=False;inflight=None;sent=set()
raw=(root/'worker-events.jsonl').open('a')
buffer=bytearray()
last_resource_probe=0.0
def handle_frame(frame):
    global ready,inflight
    if frame.get('type')=='status':
        phase=frame['phase']
        state['remote_worker_id']=frame['worker_id'];state['remote_worker_pid']=frame.get('worker_pid',frame['pid'])
        if phase=='READY_FOR_BOUNDED_EVALUATION':
            ready=True;inflight=None
        status(phase,remote_progress={key:value for key,value in frame.items() if key in ['loaded','total','run_id','profile','child_pid','queue_seconds','case','repetition','steps','quality']})
    elif frame.get('type')=='result':
        identifier=frame['run_id']
        pending=root/'results'/(identifier+'.pending.json');pending.write_text(json.dumps(frame,indent=2)+'\n');pending.replace(root/'results'/(identifier+'.json'))
        status('EVALUATION_RESULT_SAVED',last_run_id=identifier,last_execution_status=frame.get('execution_status'),
               last_quality_pass=frame.get('quality_pass'),last_profile=frame.get('profile'))
    elif frame.get('type')=='protocol_error':
        status('TRUSTED_PROTOCOL_ERROR',error=frame)
        ready=False
try:
    while process.poll() is None:
        for key,_ in selector.select(timeout=0.2):
            data=os.read(key.fileobj.fileno(),65536)
            if not data:break
            raw.write(data.decode(errors='replace'));raw.flush();buffer.extend(data)
            while b'\n' in buffer:
                line,_,rest=buffer.partition(b'\n');buffer=bytearray(rest)
                try:frame=json.loads(line)
                except ValueError:continue
                handle_frame(frame)
        if ready and inflight is None:
            pending_requests=[path for path in sorted((root/'inbox').glob('*.json')) if path.stem not in sent]
            if pending_requests:
                if time.monotonic()-last_resource_probe < 15:
                    continue
                last_resource_probe=time.monotonic()
                probe_argv=probe_command(CONFIG)
                observed=subprocess.run(probe_argv,capture_output=True,text=True,timeout=30)
                resource=json.loads(observed.stdout) if observed.returncode==0 else {'available':False,'probe_failed':True,'exit_code':observed.returncode}
                with (root/'actual-resource-probes.jsonl').open('a') as record:
                    record.write(json.dumps({'epoch':time.time(),'observation':resource,'GPU_allocated':False,'model_called':False})+'\n')
                if resource.get('available') is not True:
                    status('WAITING_FOR_EXISTING_SHARED_RESOURCES',resource_probe=resource,queued_requests=len(pending_requests))
                    continue
            for path in pending_requests:
                if path.stem in sent:continue
                value=json.loads(path.read_text())
                if path.stem!=value.get('run_id'):raise ValueError('Private inbox lineage mismatch')
                process.stdin.write(json.dumps(value,separators=(',',':'))+'\n');process.stdin.flush()
                sent.add(path.stem);inflight=path.stem;ready=False
                status('BOUNDED_REQUEST_SENT',inflight_run_id=inflight,profile=value['profile'])
                break
    status('TRUSTED_WORKER_EXITED',exit_code=process.poll())
finally:
    raw.close()
    if process.poll() is None:
        process.stdin.close();process.terminate()
