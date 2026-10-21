"""Prospective full-projection/rotary/canonical-score composition.

No root mutation or numerical execution. Compare actual public matched node
objects and derive boundaries before choosing a changed Dev experiment.
"""
from pathlib import Path
import runpy,json,ast,hashlib
p=runpy.run_path('experiments/97_compiled_frequency_prefix/method.py')['build_program']();old=json.loads(json.dumps(p));byid={n['id']:n for n in p['layer_body']}
projection=['image_qkv','image_v','image_qk','image_qk_shape','image_qk_heads','image_qk_rotary_first','image_qk_rotary_second','image_qk_rotary_half','image_qk_rotary_direct','image_qk_rotary_rotated','image_qk_rotary','image_query','image_key']
score=['key_rotary','key_repeat','key_transpose','image_scores','image_scaled','image_masked']
values=['v','value_shape','value_heads','value_repeat']
body=p['layer_body'];members=projection+score
body=[n for n in body if n['id'] not in set(members+values)]
at=next(i for i,n in enumerate(body) if n['id']=='image_prob_fp32')
body[at:at]=[byid[x] for x in members+values]
p['layer_body']=body;p['compile_regions'][0]={'id':'image_pre_softmax_compiled','nodes':members}
assert {n['id']:n for n in old['layer_body']}=={n['id']:n for n in body}
assert [n['id'] for n in old['layer_body'] if n['id'] not in set(members+values)]==[n['id'] for n in body if n['id'] not in set(members+values)]
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
r=next(r for r in scope['validate_regions'](p) if r['id']=='image_pre_softmax_compiled')
assert r['outputs']==('image_v','image_masked')
report={'source_argument':'Return only original unrotatedV rows and BF16masked scores from an entire full-projection/QKrotary/keyrepeat/canonicalscore stage region. No padded identity frequencies or split biased GEMM widths.','region':{k:r[k] for k in ['id','inputs','outputs','prefix']},'every_original97node_object_equal':True,'region_nodes':members,'GEMM_geometry_unchanged':{'image_projection_M_N_K':[512,4608,3584],'score_batch_M_N_K':[56,256,556,128]},'all_BF16_rotary_score_scale_mask_boundaries':'original nodes intact under unchanged compiler policy','softmax_and_value_output_path':'original97 eager path','current110_comparison':'110returns full packed36-head QKV and invokes score region separately; proposed region returns rawV plusmaskedscore without exposing rotatedQK/packedQKV boundary.','potential_savings':['No original rotatedQK or full packed36-head result is observable outside region.','Projection/rotary and score producer/consumer use one dispatcher boundary.'],'counterarguments':['OriginalV slice can retain full4608 projection storage; eliminating raw QKV allocation is not assumed.','A100 GEMMs may still need internal rotary/replication outputs; no measured fusion/code layout.','Fullgraph may change accumulation/precision despite intact nodes; Dev must qualify.','Past historical58 and current109/110 are arithmetic-family evidence, not a paired timing forecast.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
Path('research-resume-current-gpu/prospective112-program.json').write_text(json.dumps(p,indent=2)+'\n')
Path('research-resume-current-gpu/pre-softmax-boundary-argument.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
