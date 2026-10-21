"""Bounded real Docker regression probe; no task data, credentials or GPU."""
import ast,json,pathlib,subprocess,time,uuid
from safe_native_shutdown import CODE
ROOT=pathlib.Path(__file__).resolve().parents[1]
DOCKER='<CONTROLLER_HOME>/.docker/bin/docker'
IMAGE='sha256:030938d1a78446342c8ad77c813df6d36f8422a48ab7bdc16a3da7bbc368ebbc'
tree=ast.parse((ROOT/'preparation/save_resilient_research_agent.py').read_text())
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='stop_owned_native')
legacy=next(n.value.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
out={}
def docker(*args):return subprocess.run([DOCKER,*args],capture_output=True,text=True,timeout=30)
for name,code in [('legacy',legacy),('fixed',CODE)]:
 cid=None
 try:
  run=docker('run','-d','--name','save-shutdown-probe-'+uuid.uuid4().hex[:12],'--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','128m','--cpus','0.25','--mount','type=bind,source='+str(ROOT/'preparation/runtime-build/agent')+',target=/opt/save-cli,readonly',IMAGE,'sh','-c','sleep infinity')
  assert run.returncode==0,run.stderr;cid=run.stdout.strip()
  spawn=docker('exec','-d',cid,'/opt/runtime/bin/python','/opt/save-cli/local_model_proxy.py');assert spawn.returncode==0
  time.sleep(.5)
  before=docker('exec',cid,'/opt/runtime/bin/python','-I','-c',"from pathlib import Path; import json;print(json.dumps({p.name:(p/'cmdline').read_bytes().decode(errors='replace').replace(chr(0),' ') for p in Path('/proc').glob('[0-9]*') if (p/'cmdline').exists()}))")
  result=docker('exec',cid,'/opt/runtime/bin/python','-I','-c',code);time.sleep(.5)
  state=json.loads(docker('inspect',cid).stdout)[0]['State']
  probe=docker('exec',cid,'/opt/runtime/bin/python','-I','-c','print("artifact_io_still_available")') if state['Running'] else None
  out[name]={'before':json.loads(before.stdout),'shutdown_exit':result.returncode,'shutdown_stdout':result.stdout,'shutdown_stderr':result.stderr,'state':state,'artifact_exec_exit':probe.returncode if probe else None}
 finally:
  if cid:docker('rm','-f',cid)
assert not out['legacy']['state']['Running'] and out['legacy']['state']['ExitCode']==137
assert out['fixed']['state']['Running'] and out['fixed']['artifact_exec_exit']==0
assert json.loads(out['fixed']['shutdown_stdout'])['roots']
(ROOT/'private/native-shutdown-regression.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'legacy_killed_keepalive':True,'fixed_preserved_container_and_artifact_io':True,'fixed_stopped_actual_proxy':True}))
