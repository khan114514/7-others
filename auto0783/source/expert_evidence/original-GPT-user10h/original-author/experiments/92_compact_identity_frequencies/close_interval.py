"""Verify selected restoration and preserve actual90-92 outcomes/status."""
from pathlib import Path
import hashlib
import json
import runpy

root=Path('/workspace/solution')
names=['90_prerotated_opaque_text_attention','91_prerotated_query_without_copy','92_compact_identity_frequencies']
digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
best=root/'experiments/53_compiled_image_rotary'
saved=json.loads((best/'program.json').read_text())
actual=json.loads((root/'program.json').read_text())
feedback=json.loads((best/'dev-result.json').read_text())
assert actual==saved==runpy.run_path(str(root/'method.py'))['build_program']()
assert (root/'method.py').read_bytes()==(best/'method.py').read_bytes()
assert feedback['quality_pass'] and digest(actual)==feedback['artifact_sha256']
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
                 'baseline_mean_ms','candidate_mean_ms','peak_cuda_allocated_bytes']}
    entry.update(experiment=name,compiled_variant_count=r['compilation_observations']['compiled_variant_count'],
                 feedback=f'experiments/{name}/dev-result.json',classification='valid_lower_memory_below_best_reward',
                 foreground_exit_code=c['exit_code'],original_worker_started_epoch=r['started_epoch'],
                 original_worker_completed_epoch=r['completed_epoch'])
    new.append(entry)
valid=[]
count=0
for path in (root/'experiments').glob('*/dev-result.json'):
    if int(path.parent.name.split('_')[0])<51:continue
    r=json.loads(path.read_text())
    count+=1
    if r.get('quality_pass') and r.get('execution_status')=='COMPLETED':
        valid.append((r['reward'],path.parent.name))
assert count==42 and max(valid)[1]=='53_compiled_image_rotary'
receipt={
    'restored_experiment':'53_compiled_image_rotary','builder_regeneration_exit_code':0,
    'builder_bytes_equal_saved_selection':True,'program_equal_saved_selection':True,
    'canonical_sha256':digest(actual),'selection_matches_actual_valid_worker_feedback':True,
    'highest_actual_valid_own_new_interface_reward':max(valid)[0],
    'new_experiment_graphs_match_actual_worker_digests':True,
    'all_new_actual_stdout_JSON_matches_results':True,'all_foreground_exit_codes':[0,0,0],
    'all_disclosed_new_numerical_errors_zero':True,
    'unchanged_evaluation_rerun':False,'human_cumulative_target_claimed_complete':False}
relative='experiments/92_compact_identity_frequencies/restoration-receipt.json'
(root/relative).write_text(json.dumps(receipt,indent=2)+'\n')
status_path=root/'research-status.json'
status=json.loads(status_path.read_text())
assert status['total_new_interface_evaluation_count']==39
status['continued_interval_history'].append({
    'experiments':status['latest_interval_experiments'],'next_decision':status['next_decision'],
    'restoration_receipt':status['restoration_receipt']})
status.update(latest_interval_experiments=new,new_evaluation_count=3,
              total_new_interface_evaluation_count=count,restoration_receipt=relative)
status['next_decision']='Assess the concrete53-based singleton image-query mask broadcast construction, then pursue one changed foreground Dev if justified. Only image_attention_mask.stop556->301; full score/mask/softmax/value arithmetic and two compiled image regions equal.854016 public Float32 positions prove all disclosed image mask rows equal, but producer body/withheld masks are unprovided, so the general argument remains conditional.107nodes/3855216960prefixbytes; distinct repeated-add address reuse can be measured without claiming a speed/peak guarantee. Preserve every outcome; no CFG-batch compression or arbitrary row sweeps. Keep53until an actual higher valid reward.'
status['duration_target_accounting']='Controller reported5156seconds remaining at90-92 interval start. Controller alone accounts observed meaningful work; no local credit, target completion or global-optimality claim.'
for item in new:
    status['failed_methods_preserved'].append(item['experiment']+': '+item['classification'])
for name in ['prerotated-query-copy-audit.json','compact-identity-frequency-argument.json','diagnostic-image-query-broadcast.json']:
    file='research-diagnostics/'+name
    assert (root/file).exists()
    if file not in status['source_diagnostics']:status['source_diagnostics'].append(file)
status_path.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'restored_sha256':digest(actual),'actual_best':max(valid)[1],
                 'completed_new_interface_evaluations':count,'foreground_exits':[0,0,0]}))
