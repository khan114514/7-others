import hashlib,json,pathlib,sys,uuid
sys.path.insert(0,'/tests')
from evaluator import evaluate,atomic_json
root=pathlib.Path('/tests');logs=pathlib.Path('/logs/verifier');logs.mkdir(parents=True,exist_ok=True)
for filename in ['reward.txt','reward.json','result.json']:
 (logs/filename).unlink(missing_ok=True)
protocol=json.loads((root/'protocol.json').read_text());solve=pathlib.Path('/workspace/solution/solve.sh')
result=logs/'result.json'
try:
 managed_dirs={'.git','.agents','.codex','.aws','__pycache__'}
 unexpected=[p.name for p in solve.parent.iterdir() if p.name not in {'method.py','solve.sh'} and not (p.name in managed_dirs and p.is_dir() and not p.is_symlink())]
 if unexpected:raise ValueError('HARD_GATE: unexpected submission assets')
 if solve.is_symlink() or not solve.is_file() or hashlib.sha256(solve.read_bytes()).hexdigest()!=protocol['solve_sha256']:
  raise ValueError('HARD_GATE: frozen solve.sh changed or absent')
 res=evaluate(pathlib.Path('/workspace/solution/method.py'),root/'private_cases',root/'private_targets',
              protocol,root/'candidate_worker.py',result)
 if res['score'] is None:raise ValueError('INFRASTRUCTURE: scoring anchor not calibrated')
 reward=res['score']
except Exception as e:
 kind=str(e).split(':')[0] if ':' in str(e) else 'INFRASTRUCTURE'
 atomic_json(result,{'status':kind,'failure_reason':str(e),'score':-1 if kind=='HARD_GATE' else None})
 if kind!='HARD_GATE':print(str(e),flush=True);raise SystemExit(1)
 reward=-1
tmp=logs/('.stage-'+uuid.uuid4().hex+'.txt');tmp.write_text(str(float(reward))+'\n');tmp.replace(logs/'reward.txt')
print(json.dumps({'reward':float(reward),'result':'result.json'}),flush=True)
