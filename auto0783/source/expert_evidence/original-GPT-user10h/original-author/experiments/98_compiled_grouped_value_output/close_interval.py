"""Verify actual96-98 feedback and restore highest valid own Dev reward97."""
from pathlib import Path
import hashlib
import json
import runpy

root=Path('/workspace/solution')
names=['96_compiled_prefix_text_qkv','97_compiled_frequency_prefix','98_compiled_grouped_value_output']
digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
best=root/'experiments/97_compiled_frequency_prefix'
saved=json.loads((best/'program.json').read_text())
actual=json.loads((root/'program.json').read_text())
feedback=json.loads((best/'dev-result.json').read_text())
assert actual==saved==runpy.run_path(str(root/'method.py'))['build_program']()
assert (root/'method.py').read_bytes()==(best/'method.py').read_bytes()
assert feedback['quality_pass'] and digest(actual)==feedback['artifact_sha256']
classifications={
    names[0]:'valid_non_selected_cached_biased_QKV_region',
    names[1]:'selected_highest_valid_Dev_reward_with_baseline_outlier_caution',
    names[2]:'valid_non_selected_compiled_grouped_value_output'}
new=[]
for name in names:
    directory=root/'experiments'/name
    p=json.loads((directory/'program.json').read_text())
    r=json.loads((directory/'dev-result.json').read_text())
    c=json.loads((directory/'cli-receipt.json').read_text())
    assert p==runpy.run_path(str(directory/'method.py'))['build_program']()
    assert digest(p)==r['artifact_sha256']
    assert c['exit_code']==0 and c['foreground']
    assert json.loads((directory/'dev-stdout.txt').read_text())==r
    assert r['execution_status']=='COMPLETED' and r['quality_pass']
    assert all(x['worst_relative_rmse']==x['worst_scaled_max_abs']==0 for x in r['records'])
    assert all(x['relative_rmse']==x['scaled_max_abs']==0 for x in r['final_quality']+r['source_alignment'])
    entry={key:r[key] for key in ['artifact_sha256','execution_status','quality_pass','reward',
                 'baseline_mean_ms','candidate_mean_ms','baseline_sample_sd_ms','peak_cuda_allocated_bytes']}
    entry.update(experiment=name,compiled_variant_count=r['compilation_observations']['compiled_variant_count'],
                 feedback=f'experiments/{name}/dev-result.json',classification=classifications[name],
                 foreground_exit_code=c['exit_code'],original_worker_started_epoch=r['started_epoch'],
                 original_worker_completed_epoch=r['completed_epoch'])
    new.append(entry)
valid=[]
new_interface_count=0
for path in (root/'experiments').glob('*/dev-result.json'):
    r=json.loads(path.read_text())
    if int(path.parent.name.split('_')[0])>=51:new_interface_count+=1
    if r.get('quality_pass') and r.get('execution_status')=='COMPLETED':
        valid.append((r['reward'],path.parent.name))
assert new_interface_count==48 and max(valid)[1]=='97_compiled_frequency_prefix'
receipt={
    'restored_experiment':best.name,'builder_regeneration_exit_code':0,
    'builder_bytes_equal_saved_selection':True,'program_equal_saved_selection':True,
    'canonical_sha256':digest(actual),'selection_matches_actual_valid_worker_feedback':True,
    'highest_actual_valid_own_Dev_reward':max(valid)[0],
    'selection_caution':'97baseline case0is unusually high; its candidate mean is above historical53. Highest actual valid reward does not establish a candidate-paired speedup or generated-kernel cause.',
    'new_experiment_graphs_match_actual_worker_digests':True,
    'all_actual_stdout_JSON_matches_results':True,'all_foreground_exit_codes':[0,0,0],
    'all_disclosed_new_numerical_errors_zero':True,
    'unchanged_evaluation_rerun':False,'human_cumulative_target_claimed_complete':False}
relative='experiments/98_compiled_grouped_value_output/restoration-receipt.json'
(root/relative).write_text(json.dumps(receipt,indent=2)+'\n')
status_path=root/'research-status.json'
status=json.loads(status_path.read_text())
assert status['total_new_interface_evaluation_count']==45
status['continued_interval_history'].append({
    'experiments':status['latest_interval_experiments'],'next_decision':status['next_decision'],
    'restoration_receipt':status['restoration_receipt']})
status.update(latest_interval_experiments=new,new_evaluation_count=3,
              total_new_interface_evaluation_count=new_interface_count,restoration_receipt=relative)
status['best_valid_method']=new[1].copy()
status['selection_caution']=receipt['selection_caution']
status['next_decision']='Implement exactly the prospective97-based grouped score construction c244df417096809938e4445e8d107e0b1105bc692625f99b8ac20cfa5e18109b: remove key_repeat, group seven queries per KVhead, restore canonical BF16scores and compile key concat/transpose/query reshape/GEMM/score reshape. Three runtime inputs/sole score output; unchanged eager scale/mask/FP32softmax/BF16cast/value path, old three regions and3871096320prefixbytes. Exact14336row-base/symbolic128feature556key index proof and valid eager46supply a concrete argument, not compiled CUDA qualification. Run one distinct foreground Dev next interval, preserving failure/worse outcome. No arbitrary boundaries, flags or unchanged repeats; retain97until actual valid reward is higher.'
status['duration_target_accounting']='Controller reported3432seconds remaining at96-98 interval start. Controller alone accounts observed meaningful work; no local credit, target completion or global-optimality claim.'
for entry in [new[0],new[2]]:
    status['failed_methods_preserved'].append(entry['experiment']+': '+entry['classification'])
for name in ['compiled-text-projection-argument.json','frequency-prefix-compilation-argument.json',
             'compiled-grouped-value-output-argument.json','compiled-grouped-score-argument.json']:
    file='research-diagnostics/'+name
    assert (root/file).exists()
    if file not in status['source_diagnostics']:status['source_diagnostics'].append(file)
status_path.write_text(json.dumps(status,indent=2)+'\n')
with (root/'research-notes.md').open('a') as f:
    f.write('\n96-98restoration verified: python method.py actual exit0; builder bytes/regenerated graph equal actual selected97and canonical'+digest(actual)+'. All48actual new-interface results plus historical own valid results rank97highest measured valid reward0.7388630857582987. Saved96/97/98builders equal programs/worker digests; stdout JSON equals feedback; all foreground exits0and disclosed stage/final/source errorszero. Receipt beside98;statusarchives93-95and records96-98,97selection/baseline-outlier caution and exact grouped-score next decision. Controller target remains ongoing without local durationcredit or completion claim.\n')
print(json.dumps({'restored_sha256':digest(actual),'actual_best':max(valid)[1],
                 'completed_new_interface_evaluations':new_interface_count,'foreground_exits':[0,0,0]}))
