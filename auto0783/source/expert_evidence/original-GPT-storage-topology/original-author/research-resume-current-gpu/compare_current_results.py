"""Own current-GPU result audit; rank only eligible current-GPU Dev jobs.

No candidate-paired gain can be inferred from different jobs or from historical
baselines on another GPU. Timing_scope's legacy descriptive256 string is not
used to reinterpret actual steps32 records.
"""
from pathlib import Path
import json,hashlib,statistics
names=['108_isolated_v_compiled_qk','109_packed_compiled_qkv','110_compiled_key_score_composition','111_compiled_packed_value_contraction']
rows=[]
for name in names:
    folder=Path('experiments')/name;r=json.loads((folder/'dev-result.json').read_text());p=json.loads((folder/'program.json').read_text());c=json.loads((folder/'foreground-cli.json').read_text())
    assert r['artifact_sha256']==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert c['exit_code']==0 and r['profile']=='dev' and not r['is_formal_protocol'] and r['steps']==32 and all(x['steps']==32 for x in r['records'])
    assert r['execution_status']=='COMPLETED'
    b=statistics.mean(x['baseline_ms'] for x in r['per_seed']);t=statistics.mean(x['candidate_ms'] for x in r['per_seed']);sd=statistics.stdev(x['baseline_ms'] for x in r['per_seed'])
    assert abs(b-r['baseline_mean_ms'])<1e-9 and abs(t-r['candidate_mean_ms'])<1e-9 and abs(sd-r['baseline_sample_sd_ms'])<1e-9
    assert r['reward']==(-1 if not r['quality_pass'] else (b-t)/b)
    rows.append({'experiment':name,'artifact_sha256':r['artifact_sha256'],'run_id':r['run_id'],'current_gpu':r['gpu_uuid'],'quality_pass':r['quality_pass'],'reward':r['reward'],'baseline_mean_ms':b,'candidate_mean_ms':t,'baseline_sd_ms':sd,'own_baseline_reduction_ms':b-t,'own_baseline_3sigma_pass':r['noise_3sigma_pass'],'peak_bytes':r['peak_cuda_allocated_bytes'],'peak_margin_bytes':34*1024**3-r['peak_cuda_allocated_bytes'],'worst_stage_rmse':max(x['worst_relative_rmse'] for x in r['records']),'worst_stage_scaled_max':max(x['worst_scaled_max_abs'] for x in r['records']),'worst_final_rmse':max(x['relative_rmse'] for x in r['final_quality']),'worker_clocks':[r['started_epoch'],r['completed_epoch']],'full_console_capture':'limited/truncated' if name.startswith('109_') else 'actual console parsed equals result'})
    if not name.startswith('109_'):assert json.loads((folder/'stdout.txt').read_text())==r
assert len({r['current_gpu'] for r in rows})==1
valid=[r for r in rows if r['quality_pass']]
selected=max(valid,key=lambda r:r['reward'])
report={'scope':'four distinct current-GPU own Dev jobs; no formal/Hidden/QA material','eligible_current_gpu_selection':selected,'results':rows,'historical_selection':'97_compiled_frequency_prefix; separately preserved, not current-GPU comparator','candidate_comparison_limit':'Different jobs have separate Baseline measurements. Ranking eligible current-GPU Dev reward does not establish statistical candidate-paired incremental improvement or kernel cause. No unchanged selected97rerun.','formal_extrapolation':'Public Dev has32stages/1repetition/3cases; formal256stages/3repetitions/6withheldcases not measured. First-prefix build cost is included but its current-GPU per-stage share is unavailable.'}
Path('research-resume-current-gpu/current-results-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'current_gpu':rows[0]['current_gpu'],'eligible_selected':selected['experiment'],'results':[{'experiment':r['experiment'],'quality':r['quality_pass'],'reward':r['reward'],'baseline_sd_ms':r['baseline_sd_ms'],'peak_margin_bytes':r['peak_margin_bytes']} for r in rows]},indent=2))
