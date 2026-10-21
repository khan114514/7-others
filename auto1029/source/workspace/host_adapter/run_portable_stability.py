"""Separate final portable source43200s; never designated model time."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import signal
import time
import uuid
from pathlib import Path
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from host_paths import PRIVATE_ROOT


async def main(task):
    root=PRIVATE_ROOT/'portable-stability';root.mkdir(exist_ok=True)
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=asyncio.Event();loop=asyncio.get_running_loop()
    for sig in [signal.SIGTERM,signal.SIGINT]:loop.add_signal_handler(sig,stopped.set)
    identifier=uuid.uuid4().hex;out=root/identifier;out.mkdir()
    runtime=Path(__file__).parent
    def manifests():
        return ({str(p.relative_to(task)):hashlib.sha256(p.read_bytes()).hexdigest() for p in task.rglob('*') if p.is_file()},
                {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.glob('*.py')})
    task_hashes,host_hashes=manifests()
    state={'phase':'STARTING','id':identifier,'started_epoch':time.time(),'required_seconds':43200,
           'task_source_manifest':task_hashes,'host_source_manifest':host_hashes,'observations':0,
           'model_research_seconds':0,'model_calls':0,'gpu_jobs_submitted':0,
           'new_machine_or_full_native_reward_claim':False}
    def save():
        f=out/'status.pending.json';f.write_text(json.dumps(state,indent=2)+'\n');f.replace(out/'status.json')
    save();trial=await Trial.create(TrialConfig(task={'path':str(task)},trial_name='heima-portable-stability-'+identifier[:12],
        trials_dir=out,agent={'name':'nop'},environment={'import_path':'heima_harbor_environment:HeimaDockerEnvironment','kwargs':{'role':'gpt'}}))
    env=trial.agent_environment
    try:
        await env.start(False);start=time.monotonic();state.update(phase='PORTABLE_FINAL_SOURCE_STABILITY_RUNNING',container_started_epoch=time.time());save()
        while time.monotonic()-start<43200:
            if stopped.is_set():state.update(phase='STOPPED_WITHOUT_12H_CLAIM');save();return
            if manifests()!=(task_hashes,host_hashes):raise RuntimeError('Portable final task or host source changed')
            result=await env.exec(command="codex sandbox -c 'sandbox_mode=\"workspace-write\"' python -c \"from pathlib import Path;p=Path('/workspace/solution/portable-stability.txt');p.write_text('verified');assert p.read_text()=='verified'\"",user='1000:1000',timeout_sec=30)
            assert result.return_code==0
            state.update(observations=state['observations']+1,observed_epoch=time.time(),elapsed_monotonic_seconds=time.monotonic()-start)
            save()
            with (out/'observations.jsonl').open('a') as stream:stream.write(json.dumps({'observed_epoch':state['observed_epoch'],'elapsed_monotonic_seconds':state['elapsed_monotonic_seconds'],'actual_native_sandbox_write_exit_code':0,'model_research_seconds':0})+'\n')
            try:await asyncio.wait_for(stopped.wait(),timeout=60)
            except asyncio.TimeoutError:pass
        assert manifests()==(task_hashes,host_hashes)
        state.update(phase='COMPLETE_PORTABLE_FINAL_SOURCE_12H_STABILITY',completed_epoch=time.time(),elapsed_monotonic_seconds=time.monotonic()-start);save()
    except Exception as error:
        state.update(phase='FAILED_PRESERVE_EVIDENCE',error_type=type(error).__name__,error=str(error)[:500]);save();raise
    finally:await env.stop(True)
    print(json.dumps({'phase':state['phase'],'folder':str(out),'model_research_seconds':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True);args=parser.parse_args()
    asyncio.run(main(args.task.resolve()))
