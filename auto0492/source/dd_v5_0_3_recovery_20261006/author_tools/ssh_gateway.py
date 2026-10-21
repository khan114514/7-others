"""Restricted SSH transport for an externally controlled GPU deployment.

The forced SSH command ignores shell commands. It accepts only scoped method
submissions and status requests, and never returns deployment paths or keys.
Install this on the GPU host, outside the agent container.
"""
import argparse
import contextlib
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

SCOPE = re.compile(r'^[a-z0-9_-]{1,64}$')
JOB = re.compile(r'^[a-f0-9]{32}$')
MAX_INPUT = 262144
FAILURE_ADVICE = {
    'HARD_GATE': 'Correct the method configuration or values to satisfy the published interface.',
    'FORMAT_ERROR': 'Submit a readable UTF-8 method source conforming to the published interface.',
    'BUDGET_TIMEOUT': 'Reduce legal teacher/training work within the fixed budget.',
    'RESOURCE_MEMORY': 'Reduce legal proposal or sample work to fit the fixed memory limit.',
    'INFRASTRUCTURE_STORAGE': 'Archive and verify completed artifacts; restore storage before a fresh run.',
    'INFRASTRUCTURE_RUNTIME': 'Inspect trusted runtime logs and repair the execution service before retrying.',
    'INVALID_NUMERICAL_OUTPUT': 'Inspect finite training/evaluation checks; retain this invalid run.',
    'METHOD_RUNTIME_FAILURE': 'Inspect the method worker log and correct the submitted executable method.',
}


def classify_failure(data, run=None):
    """Emit categories/advice only; never export raw tracebacks or private paths."""
    error_type = data.get('error_type')
    error = data.get('error', '')
    # The trusted supervisor wraps stage exits. Read only its own stage logs
    # to retain the specific infrastructure cause instead of a generic exit.
    text = ''
    wrapped_stage = (error_type == 'RuntimeError' and isinstance(error, str)
                     and re.fullmatch(r'(teacher|training|evaluation) failed with exit code -?\d+', error))
    if error_type == 'OSError':
        text = str(error)
    if run is not None and wrapped_stage:
        for name in ('teacher.log', 'training.log', 'evaluation.log'):
            path = run / name
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    text += '\n' + stream.read(65536).decode('utf-8', errors='replace')
    if 'No space left on device' in text:
        category = 'INFRASTRUCTURE_STORAGE'
    elif error_type == 'MemoryError' or 'CUDA out of memory' in text or 'OutOfMemoryError:' in text:
        category = 'RESOURCE_MEMORY'
    elif error_type == 'TimeoutError' or 'TimeoutError:' in text:
        category = 'BUDGET_TIMEOUT'
    elif (error_type == 'FloatingPointError' or 'FloatingPointError:' in text
          or error == 'Invalid independently computed FID'
          or error == 'Evaluation sample or finite-output coverage mismatch'):
        category = 'INVALID_NUMERICAL_OUTPUT'
    elif error_type == 'ValueError':
        category = 'HARD_GATE'
    elif wrapped_stage and any('ValueError: ' + message in text for message in (
            'FM sampler returned the wrong shape', 'FM times must be finite and in [0,1]',
            'Teacher selector returned the wrong shape',
            'Teacher selector must preserve each original label group',
            'Pair weights must have the requested shape and lie in [0,16]',
            'Pair weights must have positive mass',
            'Candidate returned an invalid random-generator state')):
        category = 'HARD_GATE'
    elif error_type in ('SyntaxError', 'UnicodeDecodeError'):
        category = 'FORMAT_ERROR'
    elif (isinstance(error, str) and error.startswith(('Candidate process exited', 'Candidate method call'))
          or 'Candidate process exited' in text or 'Candidate method call' in text):
        category = 'METHOD_RUNTIME_FAILURE'
    else:
        category = 'INFRASTRUCTURE_RUNTIME'
    # These categories do not grant any successful quality or score status.
    safe_type = error_type if error_type in {
        'RuntimeError', 'ValueError', 'TimeoutError', 'OSError', 'FloatingPointError',
        'SyntaxError', 'UnicodeDecodeError', 'WorkerExited', 'MemoryError'} else 'TrustedPipelineFailure'
    return {'failure_category': category, 'error_type': safe_type,
            'repair_action': FAILURE_ADVICE[category]}


def configuration():
    path = Path(os.environ.get('DD_DEPLOYMENT', '/opt/dd/deployment.json'))
    data = json.loads(path.read_text())
    # The deployed repo is environment/vendor/var.
    engine = Path(data['repo']).parents[1]/'engine'
    return data, engine, path


def formal_policy_binding(config, engine):
    # Only trusted code is imported; no submitted source is loaded here.
    sys.path.insert(0,str(engine))
    from evaluation_policy import load_formal_policy
    from method_contract import PROTOCOL
    return load_formal_policy(config,engine.parent.parent,PROTOCOL)[1]


def jobs_root():
    config, _, _ = configuration()
    root = Path(config['runs'])/'gateway_jobs'
    root.mkdir(exist_ok=True)
    return root


def atomic(path, value):
    stage = path.with_suffix(path.suffix+'.stage')
    stage.write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')
    stage.replace(path)


def identity(value):
    scope = value.get('scope')
    if not isinstance(scope,str) or not SCOPE.fullmatch(scope):
        raise ValueError('Invalid scope')
    return scope


def submit(value):
    scope=identity(value)
    root=jobs_root()
    with (root/('scope-'+scope+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return submit_locked(value)


def submit_locked(value):
    scope = identity(value)
    source = value.get('source')
    mode = value.get('mode')
    if not isinstance(source,str) or not 1 <= len(source.encode()) <= 131072:
        raise ValueError('Invalid method source size')
    if mode not in ('calibration', 'formal'):
        raise ValueError('Unsupported execution mode')
    config, engine, deployment = configuration()
    if not engine.is_dir():
        raise RuntimeError('GPU engine is unavailable')
    private_binding = formal_policy_binding(config,engine) if mode=='formal' else None
    root = jobs_root()
    # Prevent a model from building an unbounded private queue. Different
    # scopes keep their own methods, outputs and trajectory identities.
    for directory in root.iterdir():
        info=directory/'request.json'
        if info.is_file() and json.loads(info.read_text()).get('scope')==scope:
            if status({'scope':scope,'job_id':directory.name})['status']=='PENDING':
                raise ValueError('This scope already has a pending job')
    job = uuid.uuid4().hex
    directory = root/job
    directory.mkdir(mode=0o700)
    (directory/'method.py').write_text(source)
    request={'scope':scope,'mode':mode,'job_id':job,
        'submitted_unix':time.time(),'method_sha256':hashlib.sha256(source.encode()).hexdigest()}
    if private_binding is not None:
        request['formal_policy_sha256']=private_binding
    atomic(directory/'request.json',request)
    env = {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','DD_DEPLOYMENT':str(deployment)}
    with (directory/'driver.log').open('wb') as log:
        process = subprocess.Popen([config['python'],str(Path(__file__).resolve()),
                '--worker',job],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=env,start_new_session=True,close_fds=True)
    atomic(directory/'process.json',{'pid':process.pid,'started_unix':time.time(),
                                   'start_ticks':process_identity(process.pid)})
    return {'job_id':job,'scope':scope,'status':'ACCEPTED'}


@contextlib.contextmanager
def shared_gpu_lease(config):
    """Respect existing author GPU mutexes without changing their files."""
    paths=config.get('gpu_shared_leases',[])
    if not paths:
        yield
        return
    expected=config.get('gpu_uuid')
    if (not isinstance(paths,list) or len(paths)>4 or len(paths)!=len(set(paths))
            or any(not isinstance(p,str) or not Path(p).is_absolute() for p in paths)
            or not isinstance(expected,str) or not expected.startswith('GPU-')):
        raise ValueError('Unsafe shared GPU lease configuration')
    deadline=time.monotonic()+12*3600
    with contextlib.ExitStack() as stack:
        leases=[stack.enter_context(Path(p).open('r')) for p in paths]
        while True:
            if time.monotonic()>deadline:
                raise TimeoutError('Existing GPU lease did not become available')
            acquired=[]
            ready=False
            try:
                for lease in leases:
                    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    acquired.append(lease)
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used',
                        '--format=csv,noheader,nounits'],text=True,timeout=20)
                rows=[line.split(',') for line in raw.strip().splitlines()]
                if len(rows)!=1 or len(rows[0])!=2 or rows[0][0].strip()!=expected:
                    raise RuntimeError('Configured GPU identity changed')
                ready=int(rows[0][1])<=64
            except BlockingIOError:
                pass
            finally:
                if not ready:
                    for lease in reversed(acquired):
                        fcntl.flock(lease,fcntl.LOCK_UN)
            if ready:
                try:
                    yield
                finally:
                    for lease in reversed(acquired):
                        fcntl.flock(lease,fcntl.LOCK_UN)
                return
            time.sleep(20)


def worker(job):
    if not JOB.fullmatch(job):
        raise ValueError('Invalid job identity')
    directory=jobs_root()/job
    request=json.loads((directory/'request.json').read_text())
    config,engine,deployment=configuration()
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','DD_DEPLOYMENT':str(deployment)}
    if request['mode']=='formal':
        if formal_policy_binding(config,engine)!=request.get('formal_policy_sha256'):
            raise ValueError('Private formal policy changed after submission')
        env['DD_EXPECTED_FORMAL_POLICY_SHA256']=request['formal_policy_sha256']
    with shared_gpu_lease(config):
        if request['mode']=='formal' and formal_policy_binding(config,engine)!=request.get('formal_policy_sha256'):
            raise ValueError('Private formal policy changed while waiting for GPU')
        with (directory/'pipeline.log').open('wb') as log:
            code=subprocess.run([config['python'],str(engine/'supervisor.py'),
                    '--method',str(directory/'method.py'),
                    *(['--calibration'] if request['mode']=='calibration' else [])],
                    env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    lines=(directory/'pipeline.log').read_text(errors='replace').splitlines()
    marker=None
    for line in reversed(lines):
        try:
            value=json.loads(line)
        except ValueError:
            continue
        if isinstance(value,dict) and isinstance(value.get('run'),str):
            marker=value;break
    response={'job_id':job,'scope':request['scope'],
              'status':'FAILED', 'returncode':code,
              'finished_unix':time.time(),'method_sha256':request['method_sha256']}
    if marker and re.fullmatch(r'run-[a-f0-9]{32}',marker['run']):
        run=Path(config['runs'])/marker['run']
        result=run/'result.json'
        response['run_id']=marker['run']
        if (run/'failure.json').is_file():
            response.update(classify_failure(json.loads((run/'failure.json').read_text()), run))
        if result.is_file():
            data=json.loads(result.read_text())
            expected_kind='CALIBRATION' if request['mode']=='calibration' else 'FORMAL'
            if (code!=0 or data.get('status')!='SUCCESS' or data.get('run_id')!=marker['run']
                    or data.get('kind')!=expected_kind or data.get('sample_count')!=50000
                    or data.get('method_sha256')!=request['method_sha256']):
                raise ValueError('Trusted fresh result binding failed')
            metric=data.get('metric')
            if type(metric) not in (int,float) or not math.isfinite(metric) or metric<0:
                raise ValueError('Invalid trusted metric')
            if expected_kind=='FORMAL':
                reward=data.get('reward')
                if type(reward) not in (int,float) or not math.isfinite(reward):
                    raise ValueError('Invalid trusted reward')
            response['status']='SUCCESS'
            for name in ('protocol','kind','metric','reward','sample_count','checkpoint_sha256',
                         'sample_manifest_sha256','teacher_and_training_seconds',
                         'teacher_seed','training_seed','evaluation_seed','frozen_manifest_sha256'):
                # Formal sampling seed stays in the author-owned raw result.
                # The separate verifier needs the digest bindings and score,
                # never the private sampling stream itself.
                if name in data and not (expected_kind=='FORMAL' and name=='evaluation_seed'):
                    response[name]=data[name]
    if response['status'] == 'FAILED' and 'failure_category' not in response:
        response.update(classify_failure({}))
    atomic(directory/'response.json',response)


def process_identity(pid):
    try:
        # A /proc PID start timestamp prevents a reused PID satisfying liveness.
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else int(fields[19])
    except (OSError,ValueError,IndexError):
        return None


def status(value):
    scope=identity(value)
    job=value.get('job_id')
    if not isinstance(job,str) or not JOB.fullmatch(job):
        raise ValueError('Invalid job identity')
    directory=jobs_root()/job
    request=json.loads((directory/'request.json').read_text())
    if request['scope']!=scope:
        raise ValueError('Job belongs to another scope')
    path=directory/'response.json'
    if path.is_file():return json.loads(path.read_text())
    process_file=directory/'process.json'
    if process_file.exists():
        process=json.loads(process_file.read_text())
        if process.get('start_ticks') is None or process_identity(process['pid'])!=process['start_ticks']:
            response={'job_id':job,'scope':scope,'status':'FAILED',
                      'error_type':'WorkerExited','finished_unix':time.time(),
                      'method_sha256':request['method_sha256']}
            response.update(classify_failure({'error_type':'WorkerExited'}))
            atomic(path,response)
            return response
    return {'job_id':job,'scope':scope,'status':'PENDING'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker);return
    raw=sys.stdin.buffer.read(MAX_INPUT+1)
    if len(raw)>MAX_INPUT:raise ValueError('Request is too large')
    value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError('Request must be a JSON object')
    operation=value.get('operation')
    if operation=='submit':response=submit(value)
    elif operation=='status':response=status(value)
    elif operation=='health':response={'status':'OK','mode':'restricted-dd-gateway'}
    else:raise ValueError('Unsupported operation')
    print(json.dumps(response,allow_nan=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'REJECTED','error_type':type(error).__name__,
                          'error':'Request rejected; inspect trusted gateway logs.'}))
        raise SystemExit(2)
