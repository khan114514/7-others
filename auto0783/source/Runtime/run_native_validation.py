"""Native release NOP/Oracle/selected reruns; never a research author or CPU audit."""
import sys
sys.dont_write_bytecode=True
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from release_source_binding import current_binding,require_current_isolation


def canonical(path):
    return hashlib.sha256(json.dumps(json.loads(path.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_gate(path,binding):
    receipt=json.loads(path.read_text())
    if receipt.get('status')!='PASS_NATIVE_FULL_GPU_CPU_AUDIT_PENDING' or any(receipt.get(k)!=v for k,v in binding.items()):raise RuntimeError('Required native gate does not match the current Task and runtime')


async def run(args):
    task=args.task.resolve();state=args.runtime_state.resolve();runtime=Path(__file__).resolve().parent
    if state.is_relative_to(task) or state.is_relative_to(runtime):raise ValueError('Evidence state must be outside Task and Runtime')
    binding=current_binding(task)
    if not args.execute:
        print(json.dumps({'status':'DESCRIBE_ONLY_NO_NATIVE_TRIAL_OR_NUMERICAL_REQUEST','mode':args.mode,'lane':args.lane,**binding,'Model_GPU_called':False}));return
    require_current_isolation(task,state/'harbor-isolation-preflight-latest.json')
    if args.mode in ['oracle','selected']:require_gate(state/'native-nop-latest.json',binding)
    if args.mode=='selected':require_gate(state/'native-oracle-latest.json',binding)
    state.mkdir(parents=True,exist_ok=True)
    # A single native validation per state namespace; no scheduled duplicate run.
    lock=(state/'native-validation.guard').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identifier=uuid.uuid4().hex;folder=state/'native-validations'/identifier;folder.mkdir(parents=True,exist_ok=False)
    receipt={'status':'RUNNING',**binding,'mode':args.mode,'run_id':identifier,'pid':os.getpid(),'started_epoch':time.time(),'Model_called':False,'research_credit':0,'candidate_code_executed_on_host':False,'independent_CPU_audit_complete':False,'final_delivery_complete':False}
    def save():
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        pending=state/('native-'+args.mode+'-latest.pending.json');pending.write_text(json.dumps(receipt,indent=2)+'\n');pending.replace(state/('native-'+args.mode+'-latest.json'))
    save()
    agent={'name':'nop'};author=args.mode=='oracle';expected=canonical(task/'environment/public_eval/baseline-program.json') if args.mode=='nop' else None
    kwargs={'lane':args.lane,'numerical_status':str(args.numerical_status.resolve()),'runtime_root':str(state)}
    def config(name,disable=False):
        return TrialConfig(task={'path':task},trial_name=name,trials_dir=folder,agent=agent,verifier={'disable':disable},environment={'import_path':('oracle_exec_environment:OracleExecDockerEnvironment' if author else 'hybrid_harbor_environment:HybridDockerEnvironment'),'kwargs':{**kwargs,**({'author_oracle':True} if author else {})}})
    try:
        if author:
            agent={'name':'oracle'}
            pre=await Trial.create(config('release-oracle-exec-'+identifier[:12],True))
            pre_result=json.loads((await pre.run()).model_dump_json());(folder/'oracle-exec-native-result.json').write_text(json.dumps(pre_result,indent=2)+'\n')
            if pre_result.get('exception_info') is not None or not pre_result.get('finished_at'):raise RuntimeError('Fresh repaired native Oracle exec failed')
            graph=pre.paths.artifacts_dir/'workspace/solution/program.json'
            if graph.is_symlink() or not graph.is_file() or graph.stat().st_size>524288:raise RuntimeError('Native Oracle did not collect its real bounded graph')
            expected=canonical(graph);receipt['native_oracle_exec_preflight_id']=str(pre.id);save()
        if args.mode=='selected':
            title='GPT' if args.lane=='gpt' else 'Seed';package=runtime.parent
            selection=json.loads((package/'selected-method-identities.json').read_text())[title]
            snapshot=package/'Trajectories'/title/'selected-method';expected=selection['artifact_sha256']
            agent={'import_path':'frozen_artifact_agent:FrozenArtifactAgent','kwargs':{'lane':args.lane,'source_snapshot':str(snapshot),'expected_files':selection['source_files'],'expected_artifact':expected,'frozen_binding':binding,'evidence_root':str(folder/'copy-evidence'),'task_path':str(task)}}
        trial=await Trial.create(config('release-'+args.mode+'-'+identifier[:12]));receipt['native_trial_id']=str(trial.id);receipt['native_trial_dir']=str(trial.paths.trial_dir);save()
        result=json.loads((await trial.run()).model_dump_json());(folder/'native-returned-result.json').write_text(json.dumps(result,indent=2)+'\n')
        if current_binding(task)!=binding:raise RuntimeError('Protected release sources changed during the native trial')
        if result.get('exception_info') is not None or not result.get('finished_at') or result.get('verifier_environment_mode')!='separate':raise RuntimeError('Native separate Verifier branch did not close')
        if (trial.paths.verifier_dir/'grader-error.json').exists():raise RuntimeError('Native reward came from a grading error')
        numerical=json.loads((trial.paths.verifier_dir/'numerical-result.json').read_text());reward=result.get('verifier_result',{}).get('rewards',{}).get('reward')
        if (numerical.get('execution_status')!='COMPLETED' or numerical.get('quality_pass') is not True or numerical.get('profile')!='formal' or numerical.get('steps')!=256
                or len(numerical.get('records',[]))!=18 or len(numerical.get('final_quality',[]))!=6 or numerical.get('artifact_sha256')!=expected
                or reward!=numerical.get('reward') or numerical.get('parent_cuda_context_initialized') is not False or numerical.get('child_exit_code')!=0):raise RuntimeError('Native full numerical result failed artifact/protocol/quality/lifecycle checks')
        if args.mode=='nop':
            if numerical.get('baseline_identity') is not True:raise RuntimeError('NOP did not validate baseline identity and observed normalization')
        else:
            if numerical.get('baseline_identity') is not False or numerical.get('noise_3sigma_pass') is not True or not 0<reward<0.8:raise RuntimeError('Non-baseline method failed its measured improvement gates')
            if args.mode=='oracle' and reward<0.15:raise RuntimeError('Published Reference improvement is insufficient')
        receipt.update(status='PASS_NATIVE_FULL_GPU_CPU_AUDIT_PENDING',reward=reward,numerical_run_id=numerical['run_id'],artifact_sha256=expected,numerical_source_binding_sha256=numerical['source_binding_sha256'],original_full_numerical_result_SHA=hashlib.sha256((trial.paths.verifier_dir/'numerical-result.json').read_bytes()).hexdigest(),full_hidden_numerical_evaluation=True)
    except Exception as error:
        receipt.update(status='FAIL_PRESERVED',error_type=type(error).__name__,error=str(error)[-1800:])
    finally:
        receipt['completed_epoch']=time.time();save();print(json.dumps(receipt,indent=2));lock.close()
    if receipt['status']=='FAIL_PRESERVED':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['nop','oracle','selected'],required=True);parser.add_argument('--lane',choices=['gpt','seed'],default='gpt')
    parser.add_argument('--task',type=Path,required=True);parser.add_argument('--runtime-state',type=Path,required=True);parser.add_argument('--numerical-status',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    asyncio.run(run(parser.parse_args()))
if __name__=='__main__':main()
