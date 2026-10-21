"""Real Harbor0.23 NOP using the separately provisioned trusted deployment."""
import argparse,asyncio,json
from pathlib import Path
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
async def main():
 p=argparse.ArgumentParser();p.add_argument('--native-task',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('Use a fresh output directory')
 a.out.mkdir(parents=True)
 cfg=TrialConfig(task={'path':a.native_task.resolve()},trial_name='save-nop',trials_dir=a.out.resolve(),
  agent={'name':'nop'},environment={'import_path':'portable_runtime_v4.environment:SaveHybridDockerEnvironment',
   'kwargs':{'lane':'gpt','numerical_status':str(a.out.resolve()/'numerical-unused.json'),'runtime_root':str(a.out.resolve()/'runtime')}})
 trial=await Trial.create(cfg);result=await trial.run();value=json.loads(result.model_dump_json())
 (a.out/'returned-result.json').write_text(json.dumps(value,indent=2))
 if value.get('exception_info') is not None:raise RuntimeError('NOP failed; inspect retained native Trial evidence')
 if value.get('verifier_environment_mode')!='separate' or not value.get('verifier_result',{}).get('rewards'):raise RuntimeError('Independent Verifier reward missing')
 print(json.dumps({'trial_id':str(trial.id),'rewards':value['verifier_result']['rewards'],'verifier_environment_mode':value['verifier_environment_mode']}))
if __name__=='__main__':asyncio.run(main())
