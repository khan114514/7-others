from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/112_compiled_pre_softmax_partition/method.py').read_text()
s=base.replace('Experiment 112: full image projection through canonical BF16 masked scores in one region.','Experiment 113: eager softmax followed by compiled BF16 cast/value preparation and contraction.')
needle="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
add="""    # Keep the sensitive FP32 softmax entirely eager. Its BF16 cast, original
    # value preparation and full556-term reversed contraction form one region
    # whose sole output is canonical image attended rows. No output projection
    # or RMS reduction is compiled in this added region.
    post_members=['image_prob','v','value_shape','value_heads','value_repeat',
                  'value_transpose','probability_transpose','channel_query_output',
                  'image_output_heads','image_output_rows','image_attended_rows']
    by_post={n['id']:n for n in nodes if n['id'] in post_members}
    nodes=[n for n in nodes if n['id'] not in by_post]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_attention_out')
    nodes[at:at]=[by_post[name] for name in post_members]
"""
assert needle in s;s=s.replace(needle,add+needle)
s=s.replace("            'compile_regions':[","            'compile_regions':[\n                {'id':'image_post_softmax_compiled','nodes':post_members},")
folder=Path('experiments/113_compiled_post_softmax_partition');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
old=runpy.run_path('experiments/112_compiled_pre_softmax_partition/method.py')['build_program']()
a={n['id']:n for n in old['layer_body']};b={n['id']:n for n in p['layer_body']};assert a==b
members=p['compile_regions'][0]['nodes'];assert old['compile_regions']==p['compile_regions'][1:]
assert [n['id'] for n in old['layer_body'] if n['id'] not in members]==[n['id'] for n in p['layer_body'] if n['id'] not in members]
assert b['image_prob']=={'id':'image_prob','op':'cast','inputs':['image_prob_fp32'],'dtype':'bf16'}
assert b['image_prob_fp32']=={'id':'image_prob_fp32','op':'softmax','inputs':['image_masked'],'dim':-1}
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
r=next(r for r in scope['validate_regions'](p) if r['id']=='image_post_softmax_compiled')
assert r['outputs']==('image_attended_rows',)
assert r['inputs']==('image_prob_fp32','text_v_rows','image_v')
report={'hypothesis':'Keep sensitive FP32 softmax eager, but hide BF16probabilitycast and value preparation/transposedcontraction behind one canonical attended-row output. Compiler may jointly handle cast/packing and consumer layout.','all_original112node_objects_equal':True,'all_other_relative_order_and_three_regions_equal':True,'region':{k:r[k] for k in ['id','inputs','outputs','prefix']},'contraction_geometry_batch_M_N_K':[56,128,256,556],'BF16_probability_conversion':'Original cast node intact under fixed emulate_precision_casts policy; GEMM still requires BF16 operands.','output_feature_order':'Original28heads*128features in canonical batch/token/feature order','eager_operators':['FP32softmax','image output projection','both RMS norms'],'historical_distinction':'55/65compiled softmax failed; this does not compile softmax.57consumed alreadyBF16probabilities and included outputprojection, whereas this includes cast and originalvalueproducer but excludes projection.111returned channel layout from packedV; this consumes originalrawV and returns canonical rows.','no_asserted_buffer_elimination':'Compiler may still allocateBF16 probabilities and repeated values; no code/profile supplied.','canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n')
for q in Path('experiments').glob('*/program.json'):
    if q.parent!=folder:assert report['canonical_sha256']!=hashlib.sha256(json.dumps(json.loads(q.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(report,indent=2))
