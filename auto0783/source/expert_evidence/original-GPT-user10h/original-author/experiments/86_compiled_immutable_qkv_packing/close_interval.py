"""Verify selected restoration and preserve actual84-86 outcomes/status."""
from pathlib import Path
import hashlib
import json
import runpy

root = Path('/workspace/solution')
names = ['84_compiled_packed_down_mlp', '85_full_state_cached_text_rms', '86_compiled_immutable_qkv_packing']
digest = lambda p: hashlib.sha256(json.dumps(p, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
best = root/'experiments/53_compiled_image_rotary'
saved = json.loads((best/'program.json').read_text())
actual = json.loads((root/'program.json').read_text())
feedback = json.loads((best/'dev-result.json').read_text())
assert actual == saved == runpy.run_path(str(root/'method.py'))['build_program']()
assert (root/'method.py').read_bytes() == (best/'method.py').read_bytes()
assert feedback['quality_pass'] and digest(actual) == feedback['artifact_sha256']
new = []
stdout_status = {}
for name in names:
    directory = root/'experiments'/name
    p = json.loads((directory/'program.json').read_text())
    r = json.loads((directory/'dev-result.json').read_text())
    c = json.loads((directory/'cli-receipt.json').read_text())
    assert p == runpy.run_path(str(directory/'method.py'))['build_program']()
    assert digest(p) == r['artifact_sha256']
    exit_code = c.get('exit_code', c.get('foreground_exit_code'))
    assert exit_code == 0
    if (directory/'dev-stdout.txt').exists():
        assert json.loads((directory/'dev-stdout.txt').read_text()) == r
        stdout_status[name] = 'saved_actual_stdout_JSON_matches_result'
    else:
        assert name == '85_full_state_cached_text_rms'
        stdout_status[name] = 'actual_foreground_stdout_observed; not_separately_saved; actual_result_retained'
    entry = {key: r[key] for key in ['artifact_sha256','execution_status','quality_pass','reward',
               'baseline_mean_ms','candidate_mean_ms','peak_cuda_allocated_bytes']}
    entry.update(experiment=name, compiled_variant_count=r['compilation_observations']['compiled_variant_count'],
                 feedback=f'experiments/{name}/dev-result.json', classification='valid_non_improving_method',
                 foreground_exit_code=exit_code, original_worker_started_epoch=r['started_epoch'],
                 original_worker_completed_epoch=r['completed_epoch'])
    new.append(entry)
valid = []
for path in (root/'experiments').glob('*/dev-result.json'):
    if int(path.parent.name.split('_')[0]) < 51:
        continue
    r = json.loads(path.read_text())
    if r.get('quality_pass') and r.get('execution_status') == 'COMPLETED':
        valid.append((r['reward'], path.parent.name))
assert max(valid)[1] == '53_compiled_image_rotary'
receipt = {
    'restored_experiment':'53_compiled_image_rotary', 'builder_regeneration_exit_code':0,
    'builder_bytes_equal_saved_selection':True, 'program_equal_saved_selection':True,
    'canonical_sha256':digest(actual), 'selection_matches_actual_valid_worker_feedback':True,
    'highest_actual_valid_own_new_interface_reward':max(valid)[0],
    'new_experiment_graphs_match_actual_worker_digests':True,
    'stdout_retention_by_experiment':stdout_status, 'all_foreground_exit_codes':[0,0,0],
    'unchanged_evaluation_rerun':False, 'human_cumulative_target_claimed_complete':False}
relative = 'experiments/86_compiled_immutable_qkv_packing/restoration-receipt.json'
(root/relative).write_text(json.dumps(receipt,indent=2)+'\n')
status_path = root/'research-status.json'
status = json.loads(status_path.read_text())
status['continued_interval_history'].append({
    'experiments':status['latest_interval_experiments'], 'next_decision':status['next_decision'],
    'restoration_receipt':status['restoration_receipt']})
status.update(latest_interval_experiments=new, new_evaluation_count=3,
              total_new_interface_evaluation_count=36, restoration_receipt=relative)
status['next_decision'] = 'Implement the prospective eager300x300 text attention sharing existing selected53 text_key_rotary. Preserve original text projections/image executables and all eager rounding; shape/cache checks pass122nodes/6562321920bytes. Added persistent score/probability storage and cache bookkeeping may dominate small first-build duplicate rotary savings. Run one changed foreground Dev and preserve all outcomes; no unqualified ranking.'
status['duration_target_accounting'] = 'Controller reported7319seconds remaining at84-86 interval start; updated to6158seconds at next boundary. Controller alone accounts observed meaningful work; no local credit or cumulative completion claim.'
for item in new:
    status['failed_methods_preserved'].append(item['experiment']+': '+item['classification'])
for name in ['full-state-text-rms-argument.json','immutable-qkv-pack-profile.json',
             'shared-text-key-attention-feasibility.json','shared-text-key-storage-argument.json']:
    file = 'research-diagnostics/'+name
    assert (root/file).exists()
    if file not in status['source_diagnostics']:
        status['source_diagnostics'].append(file)
status_path.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'restored_sha256':digest(actual),'actual_best':max(valid)[1],
                  'completed_new_interface_evaluations':36,'foreground_exits':[0,0,0]}))
