"""One bounded fixed-source closure, zero additional research/model calls."""
import asyncio,fcntl,json,os,subprocess,time,uuid
from pathlib import Path
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from audit_research_clock import audit
ROOT=Path(__file__).resolve().parents[1]
async def main():
 lock=(ROOT/'private/seed-closure-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pointer=json.loads((ROOT/'private/seed-recovery-latest.json').read_text());old=Path(pointer['formal_evidence'])/'status.json'
 assert pointer['status']=='FORMAL_RECOVERY_FAILED'
 assert not subprocess.run(['ps','-p',str(pointer['controller_pid']),'-o','command='],capture_output=True,text=True).stdout.strip()
 state=json.loads(old.read_text());clock=audit(old);assert clock['minimum_10h_passed']
 cause=json.loads((ROOT/'private/seed-closure-shutdown-root-cause-latest.json').read_text());assert cause['classification']=='CONTROLLER_SHUTDOWN_IMPLEMENTATION_FAILURE_AFTER_12H_OBSERVATION'
 assert state['stability_passed'] and state['container_stability_seconds']>=43200
 rid=uuid.uuid4().hex;folder=ROOT/'private/track-closure-recovery'/rid;folder.mkdir(parents=True);evidence=folder/'seed-research-evidence'
 record={'run_id':rid,'controller_pid':os.getpid(),'lane':'seed','status':'FORMAL_FIXED_SHA_CLOSURE_RUNNING',
  'folder':str(folder),'formal_evidence':str(evidence),'started_epoch':time.time(),'previous_status':str(old),
  'model':state['model'],'reasoning_effort':state['reasoning_effort'],'research_credit_seconds':0,'model_calls':0,
  'physical_container_rebuilt_after_confirmed_controller_shutdown_failure':True,'old_continuous_12h_observation_preserved':True}
 (folder/'previous-controller-pointer.original.json').write_text(json.dumps(pointer,indent=2))
 def save():
  (folder/'controller-status.json').write_text(json.dumps(record,indent=2));p=ROOT/'private/seed-recovery-latest.pending.json';p.write_text(json.dumps(record,indent=2));p.replace(ROOT/'private/seed-recovery-latest.json')
 save();best=state['best_valid_dev']
 cfg=TrialConfig(task={'path':ROOT/'private/native_task_v7'},trial_name='save-fixed-sha-closure-'+rid[:12],trials_dir=folder,
  agent={'import_path':'save_closure_recovery_agent:SaveClosureRecoveryAgent','model_name':state['model'],
   'kwargs':{'previous_status':str(old),'private_root':str(evidence),'method_source':best['source'],'method_sha256':best['method_sha256']}},
  environment={'import_path':'portable_runtime_v4.environment:SaveHybridDockerEnvironment',
   'kwargs':{'lane':'seed','numerical_status':str(ROOT/'private/no_numerical_controller.json'),'runtime_root':str(folder/'seed')}})
 try:
  trial=await Trial.create(cfg);value=json.loads((await trial.run()).model_dump_json())
  (folder/'seed-native-result.json').write_text(json.dumps(value,indent=2))
  assert value.get('exception_info') is None,value.get('exception_info')
  assert value['verifier_environment_mode']=='separate'
  for path in [trial.paths.agent_dir/'dev-result.original.json',trial.paths.verifier_dir/'result.json']:
   v=json.loads(path.read_text());assert v['method_sha256']==best['method_sha256'] and v['status']=='COMPLETE' and v['all_seeds_complete'] and v['quality_gates_passed'] and v['independent_fresh_reload_verified'] and len(v['rows'])==5
  record.update(status='FORMAL_RECOVERY_COMPLETE_REQUIRES_FINAL_QA',native_exception=None,completed_epoch=time.time());save()
 except Exception as error:
  record.update(status='FORMAL_RECOVERY_FAILED',failure_type=type(error).__name__,failure_reason=str(error)[:1500],completed_epoch=time.time());save();raise
if __name__=='__main__':asyncio.run(main())
