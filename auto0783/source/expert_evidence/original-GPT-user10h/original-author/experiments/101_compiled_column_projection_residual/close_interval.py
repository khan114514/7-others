"""Preserve actual99-101 and verify highest actual valid own97 restoration."""
from pathlib import Path
import hashlib
import json
import runpy

root=Path('/workspace/solution')
names=['99_compiled_grouped_score','100_compiled_value_columns_producer',
       '101_compiled_column_projection_residual']
digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
best=root/'experiments/97_compiled_frequency_prefix'
saved=json.loads((best/'program.json').read_text())
actual=json.loads((root/'program.json').read_text())
feedback=json.loads((best/'dev-result.json').read_text())
assert actual==saved==runpy.run_path(str(root/'method.py'))['build_program']()
assert (root/'method.py').read_bytes()==(best/'method.py').read_bytes()
assert feedback['quality_pass'] and digest(actual)==feedback['artifact_sha256']
classification={
    names[0]:'valid_non_selected_grouped_score_compilation',
    names[1]:'valid_non_selected_branchwise_value_column_producer',
    names[2]:'valid_non_selected_column_projection_residual_compilation'}
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
                 feedback=f'experiments/{name}/dev-result.json',classification=classification[name],
                 foreground_exit_code=c['exit_code'],original_worker_started_epoch=r['started_epoch'],
                 original_worker_completed_epoch=r['completed_epoch'])
    new.append(entry)
valid=[]
count=0
for path in (root/'experiments').glob('*/dev-result.json'):
    r=json.loads(path.read_text())
    if int(path.parent.name.split('_')[0])>=51:count+=1
    if r.get('quality_pass') and r.get('execution_status')=='COMPLETED':
        valid.append((r['reward'],path.parent.name))
assert count==51 and max(valid)[1]==best.name
receipt={
    'restored_experiment':best.name,'builder_regeneration_exit_code':0,
    'builder_bytes_equal_saved_selection':True,'program_equal_saved_selection':True,
    'canonical_sha256':digest(actual),'selection_matches_actual_valid_worker_feedback':True,
    'highest_actual_valid_own_Dev_reward':max(valid)[0],
    'selection_caution':'97baseline case0is unusually high; separate jobs do not establish a candidate-paired speedup over53 or generated-kernel cause.',
    'new_experiment_graphs_match_actual_worker_digests':True,
    'all_actual_stdout_JSON_matches_results':True,'all_foreground_exit_codes':[0,0,0],
    'all_disclosed_new_numerical_errors_zero':True,
    'unchanged_evaluation_rerun':False,'human_cumulative_target_claimed_complete':False}
relative='experiments/101_compiled_column_projection_residual/restoration-receipt.json'
(root/relative).write_text(json.dumps(receipt,indent=2)+'\n')
status_path=root/'research-status.json'
status=json.loads(status_path.read_text())
assert status['total_new_interface_evaluation_count']==48
status['continued_interval_history'].append({
    'experiments':status['latest_interval_experiments'],'next_decision':status['next_decision'],
    'restoration_receipt':status['restoration_receipt']})
status.update(latest_interval_experiments=new,new_evaluation_count=3,
              total_new_interface_evaluation_count=count,restoration_receipt=relative)
assert status['best_valid_method']['experiment']==best.name
status['next_decision']='Assess the concrete prospective97-based eager-unit-post-RMS/compiled-weight-and-MLP tradeoff8063ecb7936acdea0f1d3fbd91b47d165ce2e33ad128b6e56eec23aaddf0656e; if pursued implement exactly and run one distinct foreground Dev. Keep published RMS reduction/rsqrt/casts entirely eager; only original weight-first BF16mul joins the existing image MLP region.111nodes/three regions,3871096544prefixbytes(+224), no weight folding or compiled reduction. Extra unit multiply costs3670016outputbytes/layer and51380224elements/forward, so benefit is uncertain despite65280finite scalar identity checks. No emitted profile proves fusion/latency. Preserve every failure/worse result; retain97until actual valid reward exceeds it, with baseline-outlier caution. Plain dead-node pruning has no opportunity: all107nodes are live.'
status['duration_target_accounting']='Controller reported2457seconds remaining at99-101 interval start. Controller alone accounts observed meaningful work; no local credit, target completion or global-optimality claim.'
for entry in new:status['failed_methods_preserved'].append(entry['experiment']+': '+entry['classification'])
for name in ['value-columns-producer-argument.json','compiled-column-projection-argument.json',
             'eager-unit-rms-weight-mlp-argument.json','selected97-backward-liveness.json']:
    file='research-diagnostics/'+name
    assert (root/file).exists()
    if file not in status['source_diagnostics']:status['source_diagnostics'].append(file)
status_path.write_text(json.dumps(status,indent=2)+'\n')
with (root/'research-notes.md').open('a') as f:
    f.write('\n99-101 restoration verified: python method.py actual exit0; root builder bytes and regenerated graph equal saved actual selected97and worker canonical'+digest(actual)+'. All51actual new-interface results plus historical own valid results still rank97highest actual valid reward0.7388630857582987. Saved99/100/101builders equal their programs and worker digests; stdout JSON equals actual feedback; everyforeground exit0and disclosedstage/final/sourceerrorzero. Receipt beside101;statusarchives96-98and records new results, retained97baseline caution, negative liveness source result and exact eager-normalization/compiled-weight next decision. Human target ongoing/controller-accounted; no local durationcredit or completion claim.\n')
print(json.dumps({'restored_sha256':digest(actual),'actual_best':max(valid)[1],
                 'completed_new_interface_evaluations':count,'foreground_exits':[0,0,0]}))
