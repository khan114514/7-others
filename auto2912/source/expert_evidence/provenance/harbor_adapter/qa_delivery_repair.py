"""Run only frozen artifact checks after both independent research tracks end."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import fcntl

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

PRIVATE=Path(__file__).resolve().parent
ROOT=PRIVATE.parent
TASK=ROOT/'workspace/harbor_task'
OUTPUT=PRIVATE/'post-selection-native'


async def trial(kind,role='gpt'):
    hashes={str(p.relative_to(TASK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in TASK.rglob('*') if p.is_file()}
    # Reuse only a genuinely successful check on precisely this task version.
    # Failure logs remain intact; successful NOP/Dev fits are not duplicated.
    for f in sorted(OUTPUT.glob(kind+'-'+role+'-*/returned-result.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        value=json.loads(f.read_text())
        if value.get('exception_info') or not (value.get('verifier_result') or {}).get('rewards'):continue
        if json.loads((f.parent/'task-source-manifest.json').read_text())!=hashes:continue
        return f.parent,value
    target=OUTPUT/(kind+'-'+role+'-'+uuid.uuid4().hex)
    target.mkdir(parents=True)
    (target/'task-source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    agent={'name':'nop'} if kind=='nop' else {'import_path':'post_selection_harbor:FrozenCandidateAgent','kwargs':{'role':role}}
    configuration=TrialConfig(task={'path':str(TASK)},trial_name='forecast-final-'+kind+'-'+role+'-'+uuid.uuid4().hex[:8],
        trials_dir=target,agent=agent,environment={'import_path':'post_selection_harbor:FrozenEvaluationEnvironment',
        'force_build':True,'kwargs':{'role':role,'hidden_evaluation':kind=='hidden'}},verifier={'override_timeout_sec':1800})
    (target/'launch.json').write_text(json.dumps({'role':role,'kind':kind,'harbor_version':'0.23.0',
        'model_research':False,'credited_model_seconds':0,'started_epoch':time.time()},indent=2)+'\n')
    run=await Trial.create(configuration)
    result=(await run.run()).model_dump(mode='json')
    (target/'returned-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result.get('exception_info') or not (result.get('verifier_result') or {}).get('rewards'):
        raise RuntimeError('Frozen native check failed; exact evidence retained: '+str(target))
    return target,result


def main():
    lock=(PRIVATE/'post-selection-repair.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (PRIVATE/'post-selection-checks-complete.json').exists():return
    assert all(json.loads((PRIVATE/'campaign'/r/'status.json').read_text())['effective_seconds']>=36000 for r in ('gpt','seed'))
    assert json.loads((PRIVATE/'stability-status.json').read_text())['passed']
    kit=PRIVATE/'post-selection-hidden-kit'
    kit.mkdir(exist_ok=True)
    for name in ('hidden_features.npz','hidden_targets.npz'):shutil.copy2(PRIVATE/'forecast-export'/name,kit/name)
    python='<AUTHOR_HOME>/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    code="import numpy as n,json;from pathlib import Path;p=Path("+repr(str(kit))+");x=n.load(p/'hidden_features.npz')['base'].astype(n.float64);y=n.load(p/'hidden_targets.npz')['target'];m=n.isfinite(y)&(y>0);print(json.dumps({'b':float(n.abs(x-y)[m].mean()),'shape':list(x.shape)}))"
    observed=json.loads(subprocess.check_output([python,'-I','-c',code],text=True))
    comparison=json.loads((ROOT/'optimization_evidence/comparison_summary.json').read_text())
    protocol=json.loads((TASK/'environment/verifier_assets/protocol.json').read_text())
    ratio=(comparison['baseline_mean']-comparison['upper_estimate'])/comparison['baseline_mean']
    protocol.update(prediction_shape=observed['shape'],baseline_mae=observed['b'],
        estimated_attainable_mae=observed['b']*(1-ratio),
        dev_target_sha256=hashlib.sha256((kit/'hidden_targets.npz').read_bytes()).hexdigest(),
        evaluation_split='externally_injected_hidden',upper_estimation_source='same independent pilot reduction fraction, frozen before formal Reference; no held-out Reference used')
    (kit/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    nop,result=asyncio.run(trial('nop'))
    assert abs(result['verifier_result']['rewards']['reward'])<1e-7
    outputs=[]
    for role in ('gpt','seed'):
        path,result=asyncio.run(trial('dev',role))
        evaluation=json.loads(next(path.glob('*/verifier/evaluation/result.json')).read_text())
        expected=json.loads((PRIVATE/'campaign'/role/'status.json').read_text())['best_raw_mae']
        assert abs(evaluation['raw_mean']-expected)<1e-10 and evaluation['quality_gates_pass']
        outputs.append((path,result))
    # The unique method was selected and frozen on Dev before these labels are used.
    selected=max(outputs,key=lambda x:x[1]['verifier_result']['rewards']['reward'])
    selected_role=json.loads((selected[0]/'launch.json').read_text())['role']
    hidden,hidden_result=asyncio.run(trial('hidden',selected_role))
    hidden_evaluation=json.loads(next(hidden.glob('*/verifier/evaluation/result.json')).read_text())
    if not hidden_evaluation['quality_gates_pass'] or hidden_evaluation['evaluation_split']!='externally_injected_hidden':
        raise RuntimeError('Actual post-selection Hidden quality gates failed; do not claim completion')
    (PRIVATE/'post-selection-checks-complete.json').write_text(json.dumps({'completed_epoch':time.time(),
        'nop':str(nop),'dev':{json.loads((p/'launch.json').read_text())['role']:str(p) for p,_ in outputs},
        'selected_on_dev_before_hidden':selected_role,'hidden':str(hidden),'model_seconds_added':0,
        'research_clocks_or_methods_changed':False,'official_platform_hidden_acceptance_claimed':False},indent=2)+'\n')


if __name__=='__main__':main()
