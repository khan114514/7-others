"""Prospective complete post-softmax partition; no root mutation or Dev."""
from pathlib import Path
import runpy,json,ast,hashlib
p=runpy.run_path('experiments/113_compiled_post_softmax_partition/method.py')['build_program']();old=json.loads(json.dumps(p));byid={n['id']:n for n in p['layer_body']}
r=next(r for r in p['compile_regions'] if r['id']=='image_post_softmax_compiled')
original_members=r['nodes'][:];members=original_members+['image_attention_out','image_residual_rows']
body=[n for n in p['layer_body'] if n['id'] not in set(members)]
# Original postregion location is immediately before text-residual prefix
# construction; insert before that, after all softmax/runtime inputs exist.
at=next(i for i,n in enumerate(body) if n['id']=='text_state_rows')
body[at:at]=[byid[name] for name in members]
p['layer_body']=body;r['nodes']=members;r['id']='image_post_softmax_output_compiled'
assert {n['id']:n for n in old['layer_body']}=={n['id']:n for n in body}
assert byid['image_residual_rows']['inputs']==['image_input_rows','image_attention_out']
assert byid['image_attention_out']['inputs']==['image_attended_rows','w:self_attn.vision_branch_o_proj.weight']
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
region=next(r for r in scope['validate_regions'](p) if r['id']=='image_post_softmax_output_compiled');assert region['outputs']==('image_residual_rows',)
report={'status':'prospective_only_not_implemented_as_root_or_evaluated','argument':'Extend cast/value-preparation/contraction region through original3584feature outputprojection and original state+projection residual. Return only residual rows to unchanged eagerRMS. This hides canonicalattended-row boundary and preserves original entire postsoftmax phase.','region':{k:region[k] for k in ['id','inputs','outputs','prefix']},'every_original_node_object_equal_actual113':True,'preserved_numerical_boundaries':'FP32softmax entirelyeager; originalBF16cast/contraction/outputprojection/residual nodes intact; RMS entirelyeager','public_evidence':'113qualified cast/valuefamily but slower;57qualified contraction/outputprojection but slower;68qualified projection/residual but slower. No measured completeproducer-to-residual partition or emittedlayout.','specific_boundary_distinction':'113returns image_attended_rows before eagerlinear, while prospective phase can lower producerpacking with its immediateprojectionconsumer. This is not a gamma/bias/residual reassociation.','counterevidence':['All three overlapping historical/current fragments were nonselected. Combining them need not improve.','No actual compiledstrides/copies/fusedepilogues are measured.','No candidate-paired incremental evidence; unchangedFLOPs.','A future Dev must be a new concretegraph; do not rerun any oldmethod merely to occupy time.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
Path('research-resume-current-gpu/prospective115-program.json').write_text(json.dumps(p,indent=2)+'\n');Path('research-resume-current-gpu/post-softmax-output-boundary-argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
