from pathlib import Path
import runpy,json,hashlib
base=Path('experiments/97_compiled_frequency_prefix/method.py').read_text()
s=base.replace('Experiment 97: compile the complete original immutable frequency producer chain.','Experiment 112: full image projection through canonical BF16 masked scores in one region.')
old="""    # Trusted fullgraph compilation may fuse SiLU and gating while retaining
    # checkpoint GEMMs and required precision casts. No tensor lifetime changes.
"""
new="""    # Whole pre-softmax image attention partition. Original4608 projection
    # and score GEMM geometry, BF16 pointwise boundaries and cached producers
    # remain unchanged. Return only raw V and masked scores; keep all FP32
    # softmax/reduction and the original value/output path outside/eager.
    pre_members=['image_qkv','image_v','image_qk','image_qk_shape','image_qk_heads',
                 'image_qk_rotary_first','image_qk_rotary_second',
                 'image_qk_rotary_half','image_qk_rotary_direct',
                 'image_qk_rotary_rotated','image_qk_rotary',
                 'image_query','image_key','key_rotary','key_repeat',
                 'key_transpose','image_scores','image_scaled','image_masked']
    value_members=['v','value_shape','value_heads','value_repeat']
    by_id={n['id']:n for n in nodes}
    nodes=[n for n in nodes if n['id'] not in set(pre_members+value_members)]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_prob_fp32')
    nodes[at:at]=[by_id[name] for name in pre_members+value_members]
"""
assert old in s;s=s.replace(old,new)
start=s.index("                {'id':'image_rotary_compiled','nodes':[");end=s.index("                {'id':'image_mlp_compiled'",start)
s=s[:start]+"                {'id':'image_pre_softmax_compiled','nodes':pre_members},\n"+s[end:]
folder=Path('experiments/112_compiled_pre_softmax_partition');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
assert p==json.loads(Path('research-resume-current-gpu/prospective112-program.json').read_text())
(folder/'argument.json').write_bytes(Path('research-resume-current-gpu/pre-softmax-boundary-argument.json').read_bytes())
h=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for q in Path('experiments').glob('*/program.json'):
    if q.parent!=folder:assert h!=hashlib.sha256(json.dumps(json.loads(q.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
print({'prospective_equals_implemented':True,'new_canonical':h})
