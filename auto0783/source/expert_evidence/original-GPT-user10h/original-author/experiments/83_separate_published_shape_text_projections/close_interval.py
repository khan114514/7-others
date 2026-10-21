"""Verify actual preserved feedback and restored selection, then record status."""
import hashlib
import json
from pathlib import Path

root=Path('/workspace/solution')
def read(path):return json.loads((root/path).read_text())
def digest(program):return hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest()
selected='53_compiled_image_rotary'
saved=read(f'experiments/{selected}/program.json')
current=read('program.json')
feedback=read(f'experiments/{selected}/dev-result.json')
assert (root/'method.py').read_bytes()==(root/f'experiments/{selected}/method.py').read_bytes()
assert current==saved
assert digest(current)==feedback['artifact_sha256']=='0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196'
assert feedback['quality_pass']
for record in feedback['records']:
    assert record['worst_relative_rmse']==record['worst_scaled_max_abs']==0
for field in ('final_quality','source_alignment'):
    assert all(x['relative_rmse']==x['scaled_max_abs']==0 for x in feedback[field])

latest=[]
for experiment in ('81_separate_cached_text_projections','82_isolated_text_projection_packing','83_separate_published_shape_text_projections'):
    directory=f'experiments/{experiment}'
    result=read(directory+'/dev-result.json')
    receipt=read(directory+'/cli-receipt.json')
    assert receipt['foreground'] and receipt['exit_code']==0
    assert digest(read(directory+'/program.json'))==result['artifact_sha256']
    assert json.loads((root/directory/'dev-stdout.txt').read_text())==result
    valid=result['quality_pass']
    classification='hard_gate_numerical_failure' if not valid else (
        'valid_lower_memory_below_best_reward' if experiment.startswith('83_') else 'valid_non_improving_method')
    latest.append({
        'experiment':experiment,'artifact_sha256':result['artifact_sha256'],
        'execution_status':result['execution_status'],'quality_pass':valid,
        'reward':result['reward'],'baseline_mean_ms':result['baseline_mean_ms'],
        'candidate_mean_ms':result['candidate_mean_ms'],
        'peak_cuda_allocated_bytes':result['peak_cuda_allocated_bytes'],
        'compiled_variant_count':result['compilation_observations']['compiled_variant_count'],
        'feedback':directory+'/dev-result.json','classification':classification,
        'foreground_exit_code':receipt['exit_code'],
        'original_worker_started_epoch':result['started_epoch'],
        'original_worker_completed_epoch':result['completed_epoch']})

# Rank only actual quality-valid own Dev. Never rescue invalid latency.
ranking=[]
for path in sorted((root/'experiments').glob('*/dev-result.json')):
    if not path.parent.name[:2].isdigit() or int(path.parent.name[:2])<51:continue
    result=json.loads(path.read_text())
    if result.get('quality_pass'):
        ranking.append((result['reward'],path.parent.name,result['artifact_sha256']))
assert max(ranking)[1]==selected

receipt={
    'restored_experiment':selected,'builder_regeneration_exit_code':0,
    'builder_bytes_equal_saved_selection':True,
    'program_equal_saved_selection':True,
    'canonical_sha256':digest(current),
    'selection_matches_actual_valid_worker_feedback':True,
    'highest_actual_valid_own_new_interface_reward':feedback['reward'],
    'new_experiment_graphs_match_actual_worker_digests':True,
    'actual_cli_stdout_matches_each_actual_result':True,
    'all_foreground_exit_codes':[x['foreground_exit_code'] for x in latest],
    'unchanged_evaluation_rerun':False,
    'human_cumulative_target_claimed_complete':False}
receipt_path='experiments/83_separate_published_shape_text_projections/restoration-receipt.json'
(root/receipt_path).write_text(json.dumps(receipt,indent=2)+'\n')
status=read('research-status.json')
status.setdefault('continued_interval_history',[]).append({
    'experiments':status['latest_interval_experiments'],
    'next_decision':status['next_decision'],
    'restoration_receipt':status['restoration_receipt']})
status['latest_interval_experiments']=latest
status['new_evaluation_count']=3
status['total_new_interface_evaluation_count']+=3
assert status['total_new_interface_evaluation_count']==33
status['restoration_receipt']=receipt_path
status['status']='ongoing_bounded_research'
status['duration_target_accounting']='Human/controller reported8143seconds remaining at interval start. Controller alone accounts observed meaningful remaining work; no local credit or cumulative completion claim.'
status['next_decision']='Implement the specific packed-down-only compiled row-major MLP on valid83. Stage frozen-weight transpose view, prefix contiguous copied down weight, original row-major gate/up-SiLU-product and full18944-wide matmul inside one stage region. Prospective112nodes and6828183040prefixbytes fit unchanged limits. Eager50 and feature-leading compiled80 do not measure this composition; first-copy cost, actual memory/rounding/latency require distinct Dev. Keep53 until an actual valid higher reward.'
for x in latest:
    status['failed_methods_preserved'].append(x['experiment']+': '+x['classification'])
for diagnostic in ('research-diagnostics/separate-text-projection-audit.json',
                   'research-diagnostics/published-projection-geometry-argument.json',
                   'research-diagnostics/full-shape-text-projection-audit.json',
                   'research-diagnostics/compiled-packed-down-feasibility.json'):
    assert (root/diagnostic).exists(),diagnostic
    if diagnostic not in status['source_diagnostics']:status['source_diagnostics'].append(diagnostic)
(root/'research-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'restoration':receipt,'latest_experiments':latest,
                  'total_new_interface_evaluation_count':status['total_new_interface_evaluation_count']},indent=2))
