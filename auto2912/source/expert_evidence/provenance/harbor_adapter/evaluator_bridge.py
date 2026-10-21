"""Owner-specific local CPU evaluator; actor code is never imported on the host."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

PRIVATE=Path(__file__).resolve().parent
DOCKER='<AUTHOR_HOME>/Applications/Docker.app/Contents/Resources/bin/docker'


def docker(*args,timeout=1800,input=None,check=True):
    result=subprocess.run([DOCKER,*args],input=input,capture_output=True,text=True,timeout=timeout)
    if check and result.returncode:raise RuntimeError('Own evaluator container operation failed: '+result.stderr[-1200:])
    return result


class EvaluatorBridge:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.locks={role:threading.Lock() for role in ['gpt','seed']}

    def __call__(self,role,payload):
        if role not in self.locks or set(payload)!={'files'}:raise ValueError('Unexpected evaluation operation')
        files=payload['files']
        if not isinstance(files,dict) or 'method.py' not in files or not 1<=len(files)<=16:
            raise ValueError('Invalid method source bundle')
        for name,value in files.items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.py' or not isinstance(value,str) or len(value.encode())>1024**2:
                raise ValueError('Invalid source path or content')
        encoded=json.dumps(files)
        if len(encoded.encode())>1536*1024:raise ValueError('Method source bundle too large')
        with self.locks[role]:
            identifier=uuid.uuid4().hex
            folder=self.folder/role/identifier
            folder.mkdir(parents=True,mode=0o700)
            (folder/'source.json').write_text(encoded+'\n')
            name='forecast-eval-'+identifier[:16]
            started=time.time()
            try:
                docker('run','-d','--name',name,'--network','none','--cap-drop','ALL',
                       '--cap-add','SETUID','--cap-add','SETGID','--cap-add','CHOWN',
                       '--security-opt','no-new-privileges','--user','0:0','--memory','4g','--cpus','2',
                       '--entrypoint','sleep','--label','codex.task=auto2912',
                       'auto2912-task:20261006','infinity')
                writer="import json,sys;from pathlib import Path;files=json.load(sys.stdin);base=Path('/workspace/solution');\nfor path in base.rglob('*.py'):path.unlink()\nfor name,value in files.items():\n p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value);p.chmod(0o444)\n"
                docker('exec','-i','-u','0:0',name,'python','-I','-c',writer,input=encoded)
                scored=docker('exec','-u','0:0',name,'python','-I','/opt/forecast-engine/grader.py',
                              '--output','/logs/verifier/evaluation',timeout=3600,check=False)
                (folder/'run.log').write_text(scored.stdout+scored.stderr)
                fetched=docker('exec','-u','0:0',name,'python','-I','-c',
                    "from pathlib import Path;print(Path('/logs/verifier/evaluation/result.json').read_text())" if scored.returncode==0 else
                    "from pathlib import Path;print(Path('/logs/verifier/evaluation/failure.json').read_text())",check=False)
                value=json.loads(fetched.stdout) if fetched.returncode==0 else {'status':'infrastructure_error','raw_mean':None,'normalized_score':None}
                docker('cp',name+':/logs/verifier/evaluation',str(folder/'evidence'))
                receipt=dict(role=role,started_epoch=started,completed_epoch=time.time(),
                             exit_code=scored.returncode,source_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                             candidate_code_imported_on_host=False,status=value['status'])
                (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                public={k:value.get(k) for k in ['status','raw_metric','raw_direction','raw_mean','normalized_score','seeds','quality_gates_pass','method_sha256']}
                public['evaluation_id']=identifier
                public['fit_reload_seconds']=time.time()-started
                if value['status']!='ok':public['message']='Method execution or quality gate failed; inspect your method and retry.'
                return public
            finally:
                docker('rm','-f',name,check=False)
