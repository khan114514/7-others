"""Trusted model/evaluation role relay. No credential or SSH reaches Docker."""
import base64,hashlib,json,os,subprocess,threading,time,uuid
from pathlib import Path
from .seed_transport import SaveSeedTransportV2 as Relay
from .deployment import read
ROOT=Path(read()['controller_task_root']).resolve()

REMOTE=read()['remote_root']
HOSTS=read()['gpu_hosts']

class SaveRelay(Relay):
    def __init__(self,*args,**kwargs):
        self.eval_lock=threading.Lock();self.eval_slots=threading.BoundedSemaphore(2);super().__init__(*args,**kwargs)
    def read_requests(self):
        import re
        while not self.dead.is_set():
            line=self.process.stdout.readline(2*1024**2+1)
            if not line:return
            try:
                if len(line.encode())>2*1024**2:raise ValueError('Queue frame limit')
                frame=json.loads(line)
                if set(frame)!={'role','id','operation','payload'} or frame['role']!=self.role or not re.fullmatch('[0-9a-f]{32}',frame['id']):raise ValueError('Physical queue identity')
                if frame['operation']=='evaluate':
                    if self.eval_slots.acquire(blocking=False):threading.Thread(target=self.respond,args=(frame,),daemon=True).start()
                    else:
                        self.send(frame['id'],0,'header',status=429,content_type='application/json')
                        self.send(frame['id'],1,'chunk',data=base64.b64encode(b'{"status":"QUEUE_LIMIT","error":"At most two evaluations may be pending per physical lane"}').decode())
                        self.send(frame['id'],2,'end')
                else:self.pool.submit(self.respond,frame)
            except (ValueError,TypeError,KeyError):continue
    def respond(self,frame):
        if frame['operation']!='evaluate':return super().respond(frame)
        # Keep model requests independent of a long-running numerical tool.
        sequence=0
        def send(kind,**fields):
            nonlocal sequence
            self.send(frame['id'],sequence,kind,**fields);sequence+=1
        with self.eval_lock:
            try:
                result=self.evaluate(frame)
                send('header',status=200,content_type='application/json')
                send('chunk',data=base64.b64encode(json.dumps(result).encode()).decode())
            except Exception as e:
                send('header',status=502,content_type='application/json')
                send('chunk',data=base64.b64encode(json.dumps({'status':'INFRASTRUCTURE_ERROR','error':self.redact(str(e))[:1000]}).encode()).decode())
            finally:
                send('end');self.eval_slots.release()
    def evaluate(self,frame):
        payload=frame['payload']
        if not isinstance(payload,dict) or set(payload)!={'method_source','method_sha256'}:raise ValueError('Only source text and its fingerprint are accepted')
        if not isinstance(payload['method_source'],str) or len(payload['method_source'].encode())>1024**2:raise ValueError('Method source budget')
        if hashlib.sha256(payload['method_source'].encode()).hexdigest()!=payload['method_sha256']:raise ValueError('Method fingerprint mismatch')
        split='hidden' if self.role=='verifier' else 'dev'
        folder=self.folder/('evaluation-'+frame['id']);folder.mkdir(mode=0o700)
        (folder/'method.py').write_text(payload['method_source'])
        queued=time.monotonic();queued_epoch=time.time();first_running=None;attempt=0
        # No paid model retry or effective research credit is generated here.
        while not self.dead.is_set():
            for host in HOSTS:
                attempt+=1;rid=uuid.uuid4().hex
                import shlex
                base=[REMOTE+'/preparation/remote_venv/bin/python',REMOTE+'/preparation/gpu_service_resource.py',
                  '--root',REMOTE,'--job',rid,'--split',split,'--port',str(host['port'])]
                configuration=read()
                ssh=['ssh','-p',str(host['port']),'-i',configuration['ssh_key_file'],'-o','BatchMode=yes',
                  '-o','ConnectTimeout=15',configuration['ssh_user']+'@'+configuration['ssh_host']]
                start=time.time();operation='submit';known=False
                while not self.dead.is_set():
                    argv=[*ssh,' '.join(shlex.quote(x) for x in [*base,operation])]
                    try:
                        run=subprocess.run(argv,input=json.dumps(payload).encode() if operation=='submit' else b'',
                                           capture_output=True,timeout=40)
                        with (folder/f'attempt_{attempt}-transport.original.jsonl').open('a') as log:
                            log.write(json.dumps({'operation':operation,'exit_code':run.returncode,
                             'stdout':run.stdout.decode(errors='replace'),'stderr':run.stderr.decode(errors='replace'),
                             'epoch':time.time()})+'\n')
                        if run.returncode==0:
                            response=json.loads(run.stdout.decode().splitlines()[-1]);state=response['status']
                            if state=='UNKNOWN':
                                if known:raise RuntimeError('A dispatched durable job lost its state; reconciliation required')
                                operation='submit'
                            elif state=='BUSY':break
                            elif state=='COMPLETE':
                                result=response['result'];break
                            elif state in ['FAILED','LOST_WORKER']:raise RuntimeError('GPU durable worker '+state+'; original records retained')
                            elif state in ['RUNNING','DISPATCHED']:
                                known=True;operation='status'
                                if state=='RUNNING' and first_running is None:first_running=time.monotonic()
                            else:raise RuntimeError('Unknown durable worker state')
                        else:operation='status'
                    except subprocess.TimeoutExpired:
                        operation='status'
                    if time.time()-start>6000:raise RuntimeError('Durable job reconciliation timed out; no duplicate dispatched')
                    self.dead.wait(15)
                if self.dead.is_set():raise RuntimeError('Owned relay stopped; dispatched job retained for reconciliation')
                if response['status']=='BUSY':continue
                if result.get('method_sha256')!=payload['method_sha256'] or result.get('split')!=split or result.get('status') not in ['COMPLETE','CANDIDATE_ERROR','HARD_GATE','TIMEOUT','RESOURCE_LIMIT']:raise RuntimeError('Incomplete or mismatched evaluation')
                result['physical_gpu_uuid']=host['gpu'];result['dispatch_total_seconds']=time.monotonic()-queued;result['queue_seconds_excluded']=max(0,result['dispatch_total_seconds']-min(result['dispatch_total_seconds'],float(result['elapsed_seconds'])))
                result['requested_epoch_local']=queued_epoch;result['completed_epoch_local']=time.time()
                result['durable_job_id']=rid;result['physical_gpu_port']=host['port']
                (folder/'result.json').write_text(json.dumps(result,indent=2))
                return result
            self.dead.wait(60)
        raise RuntimeError('Owned relay stopped while evaluation was queued')
