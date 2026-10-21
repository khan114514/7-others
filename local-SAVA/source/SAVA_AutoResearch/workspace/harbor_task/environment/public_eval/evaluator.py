"""Frozen evaluator. Method timing is taken by this parent, never self reported."""
import argparse,ast,datetime,hashlib,json,os,pathlib,select,signal,subprocess,sys,tempfile,time,uuid
import numpy as np
from scipy.stats import spearmanr

def guard(path):
 if path.is_symlink() or not path.is_file():raise ValueError('FORMAT: method.py must be a regular file')
 raw=path.read_bytes()
 if len(raw)>1048576:raise ValueError('RESOURCE: method exceeds 1 MiB')
 tree=ast.parse(raw.decode())
 allowed={'numpy','scipy','torch','ot','geomloss','math','collections','itertools','functools','typing','dataclasses'}
 forbidden={'open','eval','exec','compile','__import__','input','breakpoint','globals','locals','getattr','setattr','delattr','vars','dir','__builtins__'}
 forbidden_attrs={'load','save','loadtxt','savetxt','memmap','fromfile','tofile','system','popen','ctypes','ctypeslib','__dict__','__subclasses__','__globals__','__code__','__closure__','_getframe','read','write'}
 forbidden_attrs.update({'open','fdopen','savez','savez_compressed','genfromtxt','fromregex','NpzFile','DataSource','read_array','open_memmap','loadmat','savemat','mmread','mmwrite','npyio','_npyio_impl','_datasource','os','sys','argv','environ','builtins','importlib','inspect','pickle','zipfile','socket','subprocess','io'})
 unsafe_modules=('numpy.lib.npyio','numpy.lib._npyio_impl','numpy.lib._datasource','numpy.lib.format','scipy.io','torch.serialization','torch.package','torch.utils.cpp_extension')
 def unsafe_module(name):return any(name==p or name.startswith(p+'.') for p in unsafe_modules)
 def unsafe_member(name):return name in forbidden_attrs or name.lstrip('_') in {'os','sys','builtins','importlib','inspect','pickle','zipfile','socket','subprocess','io'}
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):
   if any(n.name.split('.')[0] not in allowed or unsafe_module(n.name) for n in node.names):raise ValueError('HARD_GATE: unsupported or file/process/introspection import')
  if isinstance(node,ast.ImportFrom):
   if node.level or not node.module or node.module.split('.')[0] not in allowed:raise ValueError('HARD_GATE: unsupported import')
   if unsafe_module(node.module) or any(unsafe_member(n.name) for n in node.names):raise ValueError('HARD_GATE: file/process/introspection import alias')
  if isinstance(node,ast.Name) and (node.id in forbidden-{'input'} or unsafe_member(node.id)):raise ValueError('HARD_GATE: file/process/introspection builtin or API reference')
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in forbidden:raise ValueError('HARD_GATE: file/process/introspection operation')
  if isinstance(node,ast.Attribute) and (unsafe_member(node.attr) or node.attr.startswith('__')):raise ValueError('HARD_GATE: file/process/introspection attribute')
 return hashlib.sha256(raw).hexdigest()

def drop_identity():
 if os.getuid()==0:
  os.setgroups([]);os.setgid(1001);os.setuid(1001)

def invoke(method,case,worker):
 # No truth, targets, hidden paths or credentials enter the child input/environment.
 with tempfile.TemporaryDirectory(prefix='valuation-case-') as td:
  d=pathlib.Path(td);os.chmod(d,0o1777)
  inp=d/'input.npz';out=d/'values.npy';stderr=d/'stderr.log'
  np.savez(inp,**{k:case[k] for k in ['x','y','r','ry','seed']});os.chmod(inp,0o444)
  marker='READY_'+uuid.uuid4().hex
  env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':td,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
       'MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','PYTHONHASHSEED':'0'}
  started=time.monotonic()
  with stderr.open('wb') as err:
   proc=subprocess.Popen([sys.executable,'-I',str(worker),'--method',str(method),'--input',str(inp),
       '--output',str(out),'--ready',marker],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,
       env=env,preexec_fn=drop_identity,start_new_session=True)
   if not select.select([proc.stdout],[],[],60)[0]:
    os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('TIMEOUT: method import did not become ready')
   line=proc.stdout.readline().decode().strip()
   if line!=marker:
    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise ValueError('FORMAT: unexpected output during method import: '+line[:200])
   t0=time.perf_counter()
   proc.stdin.write(b'GO\n');proc.stdin.flush();proc.stdin.close()
   output_bytes=0
   while True:
    remaining=60-(time.perf_counter()-t0)
    if remaining<=0 or not select.select([proc.stdout],[],[],remaining)[0]:
     os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('TIMEOUT: method call exceeds 60 seconds')
    message=proc.stdout.readline();output_bytes+=len(message)
    if output_bytes>1048576:
     os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise ValueError('RESOURCE: method stdout exceeds 1 MiB')
    if message.decode().strip()=='DONE_'+marker:
     elapsed=time.perf_counter()-t0;break
    if not message:
     proc.wait();raise ValueError(f'EXECUTION: method exited before completing output ({proc.returncode})')
    print(message.decode(errors='replace').rstrip(),flush=True)
   proc.stdout.read();proc.wait(timeout=60)
  log=stderr.read_text(errors='replace')
  if log:print(log,flush=True)
  if proc.returncode!=0:raise ValueError(f'EXECUTION: method exited {proc.returncode}; {log[-400:]}')
  if not out.is_file() or out.is_symlink() or out.stat().st_size>1048576:raise ValueError('FORMAT: missing/oversize output')
  values=np.load(out,allow_pickle=False)
  if values.shape!=(len(case['x']),) or values.dtype.kind!='f' or not np.isfinite(values).all():raise ValueError('FORMAT: invalid output array')
  return values,elapsed,time.monotonic()-started

def atomic_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name('.stage-'+uuid.uuid4().hex+'.json')
 with tmp.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
 os.chmod(tmp,0o644);os.replace(tmp,path)

def _evaluate_snapshot(method,case_dir,target_dir,protocol,worker,out,seed=None,repeats=9):
 mh=guard(method);case_paths=sorted(case_dir.glob('*.npz'))
 rows=[]
 for path in case_paths:
  case=dict(np.load(path,allow_pickle=False));s=int(case['seed'])
  if seed is not None and s!=seed:continue
  gold=np.load(target_dir/(path.stem+'.npy'),allow_pickle=False)
  observed=[];quality=[];wall=[]
  for rep in range(repeats):
   values,elapsed,end_to_end=invoke(method,case,worker)
   dispersion=float(np.std(gold))
   if dispersion<=0:raise ValueError('INFRASTRUCTURE: degenerate target dispersion')
   rmse=float(np.sqrt(np.mean((values-gold)**2))/dispersion)
   rho=float(spearmanr(values,gold).statistic)
   k=len(gold)//4;top=np.argsort(-gold,kind='stable')[:k];chosen=np.argsort(-values,kind='stable')[:k]
   overlap=float(len(np.intersect1d(top,chosen))/k)
   valid=rmse<=.01 and rho>=.999 and overlap>=.98
   quality.append({'normalized_rmse':rmse,'spearman':rho,'top_quartile_overlap':overlap,'valid':valid})
   observed.append(elapsed);wall.append(end_to_end)
   print(json.dumps({'case_id':path.stem,'seed':s,'repeat':rep,'method_seconds':elapsed,'process_seconds':end_to_end,**quality[-1]}),flush=True)
  rows.append({'case_id':path.stem,'seed':s,'method_seconds':float(np.median(observed)),
               'samples_seconds':observed,'process_seconds':wall,'quality':quality,
               'valid':all(q['valid'] for q in quality),'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 if not rows:raise ValueError('INFRASTRUCTURE: no evaluation cases selected')
 expected=6*(1 if seed is not None else 5)
 if len(rows)!=expected:raise ValueError(f'INFRASTRUCTURE: expected {expected} cases, found {len(rows)}')
 raw=float(np.mean([r['method_seconds'] for r in rows]));valid=all(r['valid'] for r in rows)
 B=protocol['score']['B'];score=None if B is None else (B-raw)/B
 result={'schema_version':'valuation-result.v1','status':'ok' if valid else 'HARD_GATE',
         'seed':seed,'method_sha256':mh,'protocol_id':protocol['id'],'raw_metric':raw,
         'score':score if valid else -1,'quality_valid':valid,'cases':rows,
         'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 atomic_json(out,result);print(json.dumps({'status':result['status'],'raw_metric':raw,'score':result['score'],'result':out.name}),flush=True)
 return result

def evaluate(method,case_dir,target_dir,protocol,worker,out,seed=None,repeats=9):
 original_hash=guard(method);raw=method.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=original_hash:raise ValueError('FORMAT: method changed during snapshot')
 with tempfile.TemporaryDirectory(prefix='valuation-method-') as td:
  d=pathlib.Path(td);os.chmod(d,0o755);stable=d/'method.py'
  stable.write_bytes(raw);os.chmod(stable,0o444)
  return _evaluate_snapshot(stable,case_dir,target_dir,protocol,worker,out,seed,repeats)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--cases',required=True)
 p.add_argument('--targets',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
 p.add_argument('--worker',required=True);p.add_argument('--seed',type=int);p.add_argument('--repeats',type=int,default=9)
 a=p.parse_args()
 try:evaluate(pathlib.Path(a.method),pathlib.Path(a.cases),pathlib.Path(a.targets),json.loads(pathlib.Path(a.protocol).read_text()),pathlib.Path(a.worker),pathlib.Path(a.out),a.seed,a.repeats)
 except Exception as e:
  kind=str(e).split(':')[0] if ':' in str(e) else 'INFRASTRUCTURE'
  atomic_json(pathlib.Path(a.out),{'status':kind,'failure_reason':str(e),'score':-1 if kind=='HARD_GATE' else None})
  print(json.dumps({'status':kind,'failure_reason':str(e)}),flush=True);raise SystemExit(1)
