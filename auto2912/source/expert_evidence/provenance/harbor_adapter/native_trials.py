import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

PRIVATE=Path(__file__).resolve().parent
ROOT=PRIVATE.parent


async def trial(kind,role='gpt',task=None):
    task=Path(task) if task else ROOT/'workspace/harbor_task'
    target=PRIVATE/'native-harbor'/kind/(role+'-'+uuid.uuid4().hex)
    target.mkdir(parents=True)
    hashes={str(p.relative_to(task)):hashlib.sha256(p.read_bytes()).hexdigest() for p in task.rglob('*') if p.is_file()}
    (target/'task-source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    agent={'name':'nop'} if kind in ['nop','contract-nop'] else {
        'import_path':'research_agent:ForecastResearchAgent','kwargs':{'role':role},'override_timeout_sec':108000}
    if kind=='model-probe':agent={'import_path':'model_probe_agent:ForecastModelProbe','kwargs':{'role':role},'override_timeout_sec':600}
    if kind=='contract-handoff':agent={'import_path':'contract_agent:FixtureHandoffAgent'}
    configuration=TrialConfig(task={'path':str(task)},trial_name='forecast-'+kind+'-'+role+'-'+uuid.uuid4().hex[:8],
        trials_dir=target,agent=agent,
        environment={'import_path':'forecast_harbor_environment:ForecastDockerEnvironment',
                     'force_build':kind in ['nop','contract-nop'],
                     'kwargs':{'role':role,'enable_models':kind in ['research','model-probe']}},
        verifier={'override_timeout_sec':1800})
    (target/'launch.json').write_text(json.dumps({'kind':kind,'role':role,'started_epoch':time.time(),
        'harbor_version':importlib.metadata.version('harbor'),'model_research':kind=='research'},indent=2)+'\n')
    running=await Trial.create(configuration)
    result=await running.run()
    (target/'returned-result.json').write_text(result.model_dump_json(indent=2))
    (target/'completed.json').write_text(json.dumps({'completed_epoch':time.time()},indent=2)+'\n')
    return target,result.model_dump(mode='json')


async def nop(task=None,kind='nop'):
    path,value=await trial(kind,task=task)
    rewards=(value.get('verifier_result') or {}).get('rewards',{})
    exception=value.get('exception_info')
    if exception or not rewards or abs(float(next(iter(rewards.values()))))>1e-7:
        raise RuntimeError('Actual native Harbor NOP did not produce a valid zero baseline reward; '+str(path))
    return path


async def research():
    # Resume may find a successfully completed role from an earlier controller
    # or a role-only recovery process. Its valid clean native result is reused;
    # never launch another ten-hour run simply because its peer failed startup.
    ready={}
    for role in ['gpt','seed']:
        status=PRIVATE/'campaign'/role/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('effective_seconds',0)<36000:continue
        for result_path in sorted((PRIVATE/'native-harbor/research').glob(role+'-*/returned-result.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            value=json.loads(result_path.read_text())
            if value.get('exception_info') or not (value.get('verifier_result') or {}).get('rewards'):continue
            ready[role]=(result_path.parent,value)
            break
    roles=[role for role in ['gpt','seed'] if role not in ready]
    pending=await asyncio.gather(*(trial('research',role) for role in roles),return_exceptions=True)
    outputs=[ready[role] for role in ['gpt','seed'] if role in ready]+pending
    failures=[str(value) for value in outputs if isinstance(value,BaseException)]
    if failures:raise RuntimeError('Actual model track failure: '+'; '.join(failures))
    for path,value in outputs:
        if value.get('exception_info') or not (value.get('verifier_result') or {}).get('rewards'):
            raise RuntimeError('Clean native final verification failed: '+str(path))
    return outputs


async def probes():
    values=await asyncio.gather(trial('model-probe','gpt'),trial('model-probe','seed'),return_exceptions=True)
    for value in values:
        if isinstance(value,BaseException):raise value
        path,result=value
        if result.get('exception_info'):raise RuntimeError('Bounded exact-model preflight failed: '+str(path))
    return values


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['nop','contract-nop','contract-handoff','research'],required=True)
    parser.add_argument('--task')
    args=parser.parse_args()
    if args.kind=='contract-handoff':
        path,result=asyncio.run(trial(args.kind,task=args.task))
        rewards=(result.get('verifier_result') or {}).get('rewards',{})
        if result.get('exception_info') or not rewards or not .99<float(next(iter(rewards.values())))<1.01:
            raise RuntimeError('Modified-source native artifact handoff failed: '+str(path))
        print(path)
    else:print(asyncio.run(research() if args.kind=='research' else nop(args.task,args.kind)))
