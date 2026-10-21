"""Protected file IPC. The only accepted action is the fixed Dev suite."""
import hashlib,json,os,pathlib,re,stat,sys,time,uuid
sys.path.insert(0,'/opt/public_eval')
from evaluator import evaluate,atomic_json
ROOT=pathlib.Path('/opt/public_eval');QUEUE=pathlib.Path('/tmp/.public-valuation')
def process(path):
 nonce=path.name.removesuffix('.request')
 try:
  fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  with os.fdopen(fd,'r') as f:
   info=os.fstat(f.fileno())
   if not stat.S_ISREG(info.st_mode) or info.st_uid!=1000 or info.st_size>1000:raise ValueError('FORMAT: invalid request file')
   request=json.loads(f.read(1001))
  if request!={'evaluate':True}:raise ValueError('FORMAT: unsupported request')
  protocol=json.loads((ROOT/'protocol.json').read_text())
  solve=pathlib.Path('/workspace/solution/solve.sh')
  managed_dirs={'.git','.agents','.codex','.aws','__pycache__'}
  unexpected=[p.name for p in solve.parent.iterdir() if p.name not in {'method.py','solve.sh'} and not (p.name in managed_dirs and p.is_dir() and not p.is_symlink())]
  if unexpected:raise ValueError('HARD_GATE: unexpected submission assets')
  if solve.is_symlink() or not solve.is_file() or hashlib.sha256(solve.read_bytes()).hexdigest()!=protocol['solve_sha256']:raise ValueError('HARD_GATE: frozen solve.sh changed')
  out=ROOT/'results'/(uuid.uuid4().hex+'.json')
  method=pathlib.Path('/workspace/solution/method.py');source=method.read_bytes()
  result=evaluate(method,pathlib.Path('/workspace/environment/public_assets/dev'),ROOT/'targets',protocol,ROOT/'candidate_worker.py',out)
  if hashlib.sha256(source).hexdigest()==result['method_sha256']:out.with_suffix('.method.py').write_bytes(source)
  response={'status':result['status'],'raw_metric':result['raw_metric'],'score':result['score'],'quality_valid':result['quality_valid'],'method_sha256':result['method_sha256'],'result_path':str(out)}
 except Exception as error:
  kind=str(error).split(':')[0] if ':' in str(error) else 'INFRASTRUCTURE'
  response={'status':kind,'failure_reason':str(error),'score':-1 if kind=='HARD_GATE' else None}
 atomic_json(QUEUE/(nonce+'.response'),response)
 path.unlink(missing_ok=True)
if __name__=='__main__':
 QUEUE.mkdir(exist_ok=False);QUEUE.chmod(0o1777)
 ready=QUEUE/'ready';ready.write_text('PUBLIC_SERVICE_READY');ready.chmod(0o444)
 print('PUBLIC_SERVICE_READY',flush=True)
 while True:
  for path in sorted(QUEUE.glob('*.ack')):
   if re.fullmatch('[a-f0-9]{32}\.ack',path.name):
    (QUEUE/(path.stem+'.response')).unlink(missing_ok=True);path.unlink(missing_ok=True)
  for path in sorted(QUEUE.glob('*.request')):
   if re.fullmatch('[a-f0-9]{32}\.request',path.name):process(path)
  time.sleep(.1)
