"""Reconcile completed own Dev jobs, choose within current GPU, preserve history.

Run after actual115 worker result completes. Its foreground handle was lost at
the steering boundary, so preserve unknown clientexit/console, not a fake0.
No evaluator invocation
or claim of controller work credit, cross-job paired gains or global optimum.
"""
from pathlib import Path
import json,hashlib,runpy,statistics,shutil,math
names=['108_isolated_v_compiled_qk','109_packed_compiled_qkv','110_compiled_key_score_composition','111_compiled_packed_value_contraction','112_compiled_pre_softmax_partition','113_compiled_post_softmax_partition','114_compiled_down_only_columns','115_compiled_complete_post_softmax']
rows=[]
for name in names:
    folder=Path('experiments')/name;r=json.loads((folder/'dev-result.json').read_text());p=json.loads((folder/'program.json').read_text());c=json.loads((folder/'foreground-cli.json').read_text())
    assert runpy.run_path(str(folder/'method.py'))['build_program']()==p
    digest=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest==r['artifact_sha256']
    assert r['execution_status']=='COMPLETED'
    if name.startswith('115_'):
        assert c['exit_code'] is None and r['child_exit_code']==0
    else:
        assert c['exit_code']==0
    assert r['steps']==32 and r['profile']=='dev' and not r['is_formal_protocol'] and len(r['records'])==3
    b=statistics.mean(x['baseline_ms'] for x in r['per_seed']);t=statistics.mean(x['candidate_ms'] for x in r['per_seed']);sd=statistics.stdev(x['baseline_ms'] for x in r['per_seed'])
    assert abs(b-r['baseline_mean_ms'])<1e-9 and abs(t-r['candidate_mean_ms'])<1e-9 and abs(sd-r['baseline_sample_sd_ms'])<1e-9
    assert r['reward']==(-1 if not r['quality_pass'] else (b-t)/b)
    if r['quality_pass']:
        assert all(x['worst_relative_rmse']==x['worst_scaled_max_abs']==0 for x in r['records'])
        assert all(x['relative_rmse']==x['scaled_max_abs']==0 for x in r['final_quality']+r['source_alignment'])
        assert r['peak_cuda_allocated_bytes']<=34*1024**3
    full_console=not name.startswith(('109_','115_'))
    if full_console:assert json.loads((folder/'stdout.txt').read_text())==r
    rows.append({'experiment':name,'artifact_sha256':digest,'run_id':r['run_id'],'gpu_uuid':r['gpu_uuid'],'execution_status':r['execution_status'],'quality_pass':r['quality_pass'],'reward':r['reward'],'baseline_mean_ms':b,'candidate_mean_ms':t,'baseline_sample_sd_ms':sd,'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],'noise_3sigma_pass':r['noise_3sigma_pass'],'feedback':str(folder/'dev-result.json'),'foreground_exit_code':c['exit_code'],'worker_child_exit_code':r['child_exit_code'],'CLI_clocks':{'started':c['started'],'finished':c['finished']},'worker_clocks':[r['started_epoch'],r['completed_epoch']],'compiled_variant_count':r['compilation_observations']['compiled_variant_count'],'full_console_equals_result':True if full_console else None,'console_limit':('115clienthandle/exit/console unavailable at steering; actual complete worker JSON preserved' if name.startswith('115_') else '109console tool-truncated; full evaluator JSON preserved') if not full_console else None,'classification':'eligible_current_GPU_Dev' if r['quality_pass'] else 'numerical_hard_gate_failure_ineligible'})
assert len({x['gpu_uuid'] for x in rows})==1
selected=max((r for r in rows if r['quality_pass']),key=lambda r:r['reward'])
report={'status':'bounded_interval_completed_research_target_controller_accounted','latest_controller_remaining_seconds_at_message':165,'remaining_seconds_at_boundary':'not inferred; trusted controller to report','selected_current_GPU':selected,'historical_selected_method':'97_compiled_frequency_prefix, reward0.7388630857582987, priorGPU; all artifacts retained','selection_rule':'highest eligible actual own Dev reward in eight changed currentGPU jobs; historical rewards excluded from this comparison','cross_candidate_paired_gain_established':False,'global_optimum_established':False,'formal_QA_claimed':False,'results':rows,'limits':['Every actual result is public3case32stage1repetition Dev; final withheld256stage3repeat evaluation remains untested.','Separate Baseline-paired jobs do not establish candidate-paired incremental improvement.','Current compiled selected method has no supplied generatedcode/operator/kernelprofile; mechanisms remain hypotheses.','Historical eager41profiles and priorGPU scores cannot identify current compiled execution costs.','Finite-min mask topology lemma is conditional on scaled-logit bounds, not established universally from tensor manifest.','Originalround14fullnativeexportgap remains preserved unchanged in research-closure/actual-own-last-interval.'],'diagnostics':['research-resume-current-gpu/bf16-dot-tree-argument.json','research-resume-current-gpu/public-attention_dependency-audit-v3.json','research-resume-current-gpu/finite-attention_suppression-argument.json'],'next_decisions':['Do not sweep/repeat failed splitwidth108 or valid slower value/down orientations and fragment boundaries unchanged.','Do not use public25/32textlengths as fixed unseenprefix lengths; catalog has no dynamiccompaction.','Any subsequent method must supply a new preserving producer/consumer or resource argument under unchanged precision/cache/backend guards; actual compiled layout and universal finite-mask bound remain open source/diagnostic questions.'],'ordinary_work_credit':'Controller only; no local sum or invented clock/credit.'}
Path('research-resume-current-gpu/completed-interval.json').write_text(json.dumps(report,indent=2)+'\n')
s=json.loads(Path('research-status.json').read_text())
s.setdefault('historical_prior_GPU_best_valid_method',s['best_valid_method'])
s.setdefault('historical_duration_target_accounting',s['duration_target_accounting'])
s.setdefault('historical_latest_interval_experiments',s.get('latest_interval_experiments'))
s['status']='original_session_research_boundary_current_GPU'
s['best_valid_method']=selected;s['current_GPU_completed_experiments']=rows;s['latest_interval_experiments']=rows
s['current_GPU_evaluation_count']=8;s['new_evaluation_count']=8;s['total_new_interface_evaluation_count']=65
s['duration_target_accounting']='Latest trusted controller steering says165seconds remain at its last message. Boundary remaining not inferred here; prior13057.821882record belongs to previous target. No exact inherited hours or QA qualification claimed.'
s['selection_caution']=report['limits'][1]+' Historical97score remains historical; no unchanged currentGPU97repeat.'
s['restoration_receipt']='research-resume-current-gpu/restoration-receipt.json';s['next_decision']=report['next_decisions'];s['open_questions']=report['limits'][2:]
s['source_diagnostics']=s.get('source_diagnostics',[])+report['diagnostics']
s['failed_methods_preserved'].append('108_isolated_v_compiled_qk: actual numericalhardgate failure, currentGPU; all nonselected109/110/111/112/113/114 retained; selected115clientcapturegap separately preserved')
Path('research-status.json').write_text(json.dumps(s,indent=2)+'\n')
e={'status':'interim_bounded_research_boundary_not_global_exhaustion','prior_stopping_judgments':'preserved before-storage-only-human-resume and research-resume-current-gpu/before-resume files','latest_controller_remaining_seconds_at_message':165,'boundary_credit':'trusted controller only; no inferred target completion','selected_current_GPU':selected,'global_optimum_or_universal_exhaustion_established':False,'ongoing_scope':'Meaningful changed-method work and evidence are preserved; do not infer completion from a slower candidate or historical stopping notes.','limits':report['limits'],'next_decisions':report['next_decisions'],'formal_QA_claimed':False}
Path('research-exhausted.json').write_text(json.dumps(e,indent=2)+'\n')
shutil.copyfile(Path('experiments')/selected['experiment']/'method.py','method.py')
print(json.dumps({'selected':selected['experiment'],'reward':selected['reward'],'valid_current_GPU_methods':sum(r['quality_pass'] for r in rows),'current_GPU_results':len(rows),'builder_copied':'method.py','target_completion_claimed':False},indent=2))
