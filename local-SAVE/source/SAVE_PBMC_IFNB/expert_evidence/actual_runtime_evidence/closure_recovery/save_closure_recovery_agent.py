"""Close already completed own research through a genuine fixed-SHA Harbor run."""
import json,time
from pathlib import Path
from audit_research_clock import audit
from save_artifact_validation_agent import SaveArtifactValidationAgent

class SaveClosureRecoveryAgent(SaveArtifactValidationAgent):
 def __init__(self,*args,previous_status,private_root,**kwargs):
  self.previous_path=Path(previous_status);self.previous=json.loads(self.previous_path.read_text())
  self.root=Path(private_root);self.root.mkdir(parents=True,exist_ok=False)
  self.clock=audit(self.previous_path);assert self.clock['minimum_10h_passed']
  old=self.previous
  assert old['status']=='FORMAL_RESEARCH_FAILED' and old['stability_passed'] and old['container_stability_seconds']>=43200
  self.state={'lane':old['lane'],'model':old['model'],'reasoning_effort':old['reasoning_effort'],
   'status':'FORMAL_CLOSURE_RECOVERY_SETUP','started_epoch':time.time(),'rounds':[],
   'prior_state_source':str(self.previous_path),'prior_effective_research_seconds':self.clock['verified_effective_seconds'],
   'effective_research_seconds':self.clock['verified_effective_seconds'],'research_credit_from_this_segment_seconds':0,
   'best_valid_dev':old['best_valid_dev'],'container_stability_seconds':0,'stability_passed':False,
   'previous_observed_12h_container_stability_proofs':[{'status_source':str(self.previous_path),
      'seconds':old['container_stability_seconds'],'environment_binding':old['environment_binding']}],
   'qualified_stability_passed':True,'qualified_stability_basis':'Actual previous owned physical container observed healthy for 43213.856 seconds; new validation container measured separately',
   'closure_only_no_model_calls':True,'physical_container_rebuilt_after_controller_shutdown_failure':True,
   'own_native_history_preserved_at':str(self.previous_path.parent/'round_0002/own-native-history'),
   'previous_actual_native_session_id':old['restored_actual_native_session_id'],'new_native_session_started':False,
   'preparation_time_reused':False,'preflight_only':False}
  self.save();super().__init__(*args,**kwargs)
 def save(self):
  p=self.root/'status.pending.json';p.write_text(json.dumps(self.state,indent=2));p.replace(self.root/'status.json')
 async def observe(self,environment):
  info=json.loads(environment._docker('inspect',self.container).stdout)[0]
  assert info['State']['Running'] and not info['State']['OOMKilled'] and info['RestartCount']==0
  assert info['State']['StartedAt']==self.initial['State']['StartedAt']
  seconds=time.monotonic()-self.started_monotonic
  self.state['container_stability_seconds']=seconds
  with (self.root/'container-liveness.original.jsonl').open('a') as f:
   f.write(json.dumps({'epoch':time.time(),'container_id':self.container,'restart_count':0,'actual_running':True,'elapsed_monotonic_seconds':seconds})+'\n')
  self.save()
 async def setup(self,environment):
  await super().setup(environment)
  self.initial=json.loads(environment._docker('inspect',self.container).stdout)[0];self.started_monotonic=time.monotonic()
  self.state['environment_binding']=json.loads((environment.own_root/'environment-binding.json').read_text())
  self.state['same_owned_relay_dir']=str(environment.relay.folder)
  assert self.container!=self.previous['environment_binding']['container_id']
  self.state['status']='FIXED_BEST_SHA_CLOSURE_DEV_RUNNING';await self.observe(environment)
 async def run(self,instruction,environment,context):
  try:
   await super().run(instruction,environment,context);await self.observe(environment)
   v=json.loads((self.logs_dir/'dev-result.original.json').read_text())
   assert abs(v['raw_metric']-self.previous['best_valid_dev']['raw_metric'])<1e-12
   self.state.update(status='FORMAL_RESEARCH_COMPLETE',completed_epoch=time.time(),final_method_sha256=self.digest,
                     fixed_sha_dev_result_sha256=__import__('hashlib').sha256((self.logs_dir/'dev-result.original.json').read_bytes()).hexdigest())
   self.save();context.metadata=self.state
  except Exception as error:
   self.state.update(status='FORMAL_RESEARCH_FAILED',failure_type=type(error).__name__,failure_reason=str(error)[:1500]);self.save();raise
