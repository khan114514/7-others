"""Root-only private Verifier API; public Dev endpoint is unchanged."""
import argparse,fcntl,hashlib,json,math,os,re,stat,subprocess,sys,time
from pathlib import Path
SEEDS=[17,29,43,71,101];LIMIT=2*1024*1024
UUID=re.compile(r'^[0-9a-f]{32}$')

def digest(raw):return hashlib.sha256(raw).hexdigest()
def validate_submission(role,payload):
    if role not in ['gpt','seed'] or not isinstance(payload,dict):raise ValueError('Fixed physical role required')
    if set(payload)!={'action','job_id','files','method_notes','seeds'} or payload['action']!='submit':
        raise ValueError('Only fixed private submission fields allowed')
    if not UUID.fullmatch(payload.get('job_id','')) or payload['seeds']!=SEEDS:
        raise ValueError('Fresh owned job and all five formal seeds required')
    files=payload['files'];notes=payload['method_notes']
    if not isinstance(files,dict) or 'method.py' not in files or not 1<=len(files)<=64:raise ValueError('Source bundle required')
    for name,value in files.items():
        if not isinstance(name,str) or not isinstance(value,str):raise ValueError('Source names/content must be text')
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix()!=name or path.suffix!='.py' or len(value.encode())>1024*1024:
            raise ValueError('Only bounded relative Python source data allowed')
    encoded=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    if len(encoded)>LIMIT:raise ValueError('Source bundle budget exceeded')
    if not isinstance(notes,dict) or not notes or len(json.dumps(notes).encode())>65536:
        raise ValueError('Bounded nonempty method metadata required')
    return dict(role=role,job_id=payload['job_id'],files=files,method_notes=notes,seeds=SEEDS,
        policy_sha256=digest(encoded),method_notes_sha256=digest(json.dumps(notes,sort_keys=True,separators=(',',':')).encode()),
        source_kind='native_verified_artifact',submitted_epoch=time.time())

def regular(path,limit=LIMIT):
    path=Path(path)
    if path.resolve()!=path:raise ValueError('Trusted data cannot traverse symlinks')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_mode&0o022 or info.st_size>limit:
            raise ValueError('Bounded root-owned regular data required')
        raw=os.read(fd,limit+1)
        if len(raw)!=info.st_size:raise ValueError('Incomplete bounded data read')
        return raw
    finally:os.close(fd)
def atomic(path,value):
    tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.chmod(0o600);tmp.replace(path)

def status(jobs,job,role):
    if not UUID.fullmatch(job):raise ValueError('Invalid private job')
    folder=jobs/job;request=json.loads(regular(folder/'request.json',4*1024*1024))
    if request['role']!=role:raise ValueError('Private ticket belongs to another physical role')
    result=dict(job_id=job,role=role,remote_observed_epoch=time.time(),submitted_epoch=request['submitted_epoch'],
        policy_sha256=request['policy_sha256'],method_notes_sha256=request['method_notes_sha256'],
        phase='WAITING_FOR_GPU_EXCLUDED',private_verifier_route=True)
    progress=folder/'worker-status.json';execution=folder/'execution.json'
    if progress.exists():
        raw=json.loads(regular(progress))
        for key in ['phase','gpu_started_epoch','seed','completed_questions']:
            if key in raw:result[key]=raw[key]
    if execution.exists():
        receipt=json.loads(regular(execution));result.update(exit_code=receipt['exit_code'],completed_epoch=receipt['completed_epoch'])
        summary=folder/'verifier-summary.json'
        if receipt['exit_code']==0 and summary.exists():
            value=json.loads(regular(summary))
            if not value['complete_private_protocol'] or value['policy_sha256']!=request['policy_sha256']:
                raise ValueError('Actual full private result identity differs')
            result.update(phase='COMPLETE',result=value)
        else:result.update(phase='FAILED',error='Private scorer failure retained; no reward claim')
    return result

def main():
    from deployment_paths import PRIVATE_ROOT,SCORING_PYTHON
    if os.geteuid()!=0:raise PermissionError('Root-only private Verifier service required')
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['gpt','seed'],required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    raw=sys.stdin.buffer.read(LIMIT+1)
    if len(raw)>LIMIT:raise ValueError('Private transport budget exceeded')
    payload=json.loads(raw);action=payload.get('action') if isinstance(payload,dict) else None
    jobs=PRIVATE_ROOT/'private-verifier-jobs';jobs.mkdir(mode=0o700,exist_ok=True)
    if action=='active' and set(payload)=={'action'}:
        active=None
        for folder in jobs.iterdir():
            if folder.is_dir() and (folder/'request.json').exists() and not (folder/'execution.json').exists():
                req=json.loads(regular(folder/'request.json'))
                if req['role']==args.role:active=status(jobs,folder.name,args.role);break
        print(json.dumps(dict(active=active,role=args.role,remote_observed_epoch=time.time())));return
    if action=='status' and set(payload)=={'action','job_id'}:
        print(json.dumps(status(jobs,payload['job_id'],args.role)));return
    request=validate_submission(args.role,payload)
    if args.preflight_only:
        print(json.dumps(dict(phase='PRIVATE_VERIFIER_SOURCE_DATA_PREFLIGHT_ONLY',policy_sha256=request['policy_sha256'],
          candidate_imported_by_host=False,gpu_jobs=0,model_calls=0)));return
    worker=PRIVATE_ROOT/'harbor_private_worker.py'
    worker_sha=digest(regular(worker));service_sha=digest(regular(Path(__file__).resolve()))
    support_sha=digest(regular(PRIVATE_ROOT/'trusted_private_backend_support.py'))
    lock=(jobs/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    folder=jobs/request['job_id']
    if folder.exists():
        old=json.loads(regular(folder/'request.json',4*1024*1024))
        if any(old[k]!=request[k] for k in ['role','policy_sha256','method_notes_sha256','seeds']):
            raise ValueError('Private idempotency key reused for different inputs')
        print(json.dumps(status(jobs,folder.name,args.role)));return
    for old in jobs.iterdir():
        if old.is_dir() and (old/'request.json').exists() and not (old/'execution.json').exists():
            if json.loads(regular(old/'request.json'))['role']==args.role:raise ValueError('Existing own private ticket unfinished')
    if os.statvfs('/').f_bavail*os.statvfs('/').f_frsize<2*1024**3:
        raise RuntimeError('Root reserve below validated minimum; no private job launched')
    folder.mkdir(mode=0o700);request.update(worker_sha256=worker_sha,service_sha256=service_sha,backend_support_sha256=support_sha)
    atomic(folder/'request.json',request)
    argv=[str(SCORING_PYTHON),str(worker),'--job',folder.name,'--expected-worker-sha256',worker_sha,'--expected-service-sha256',service_sha,'--expected-support-sha256',support_sha]
    with (folder/'execution.log').open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            close_fds=True,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
    atomic(folder/'launch.json',dict(pid=child.pid,launched_epoch=time.time(),command=argv,no_wall_clock_cutoff=True))
    print(json.dumps(status(jobs,folder.name,args.role)))

if __name__=='__main__':main()
