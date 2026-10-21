from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/110_compiled_key_score_composition/method.py').read_text()
s=base.replace('Experiment 110: canonical key repetition and score GEMM/epilogue share a region.','Experiment 111: packed V restoration/repetition shares its original reversed contraction.')
needle="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
add="""    # Keep packed V restoration, text/image value concat, repetition and
    # unchanged reversed value GEMM in one region with a sole channel result.
    # BF16 probabilities stay an eager runtime input; full556-term reduction
    # and original56-batch contraction geometry are unchanged.
    value_members=['image_v_packed_heads','image_v_packed_rows','image_v','v',
                   'value_shape','value_heads','value_repeat','value_transpose',
                   'probability_transpose','channel_query_output']
    selected_values={n['id']:n for n in nodes if n['id'] in value_members}
    nodes=[n for n in nodes if n['id'] not in selected_values]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_output_heads')
    nodes[at:at]=[selected_values[name] for name in value_members]
"""
assert needle in s;s=s.replace(needle,add+needle)
s=s.replace("            'compile_regions':[","            'compile_regions':[\n                {'id':'packed_value_producer_consumer_compiled','nodes':value_members},")
folder=Path('experiments/111_compiled_packed_value_contraction');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
basep=runpy.run_path('experiments/110_compiled_key_score_composition/method.py')['build_program']();a={n['id']:n for n in basep['layer_body']};b={n['id']:n for n in p['layer_body']};assert a==b
members=p['compile_regions'][0]['nodes']
assert [n['id'] for n in p['layer_body'] if n['id'] not in members]==[n['id'] for n in basep['layer_body'] if n['id'] not in members]
assert basep['compile_regions']==p['compile_regions'][1:]
# Independent inverse index trace, preserving original KV,G,D head order and
# all556 source positions. Values and BF16 probabilities are unmodified.
checked=0
for batch in range(2):
    for head in range(28):
        kv=head//7
        for position in range(556):
            for d in range(128):
                if position<300:
                    source=('text_v_rows',batch,position,kv*128+d)
                    expected=('text_v_rows',batch,position,kv*128+d)
                else:
                    source=('image_packed_qkv',batch,32+kv,position-300,d)
                    expected=('image_packed_qkv',batch,32+head//7,position-300,d)
                assert source==expected
                checked+=1
for bidx in range(2):
    for h in range(28):
        for q in range(256):
            assert ('image_prob',bidx,h,q,'position')==('image_prob',bidx,h,q,'position')
# Trusted allocation-free boundary contract.
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
r=next(r for r in scope['validate_regions'](p) if r['id']=='packed_value_producer_consumer_compiled')
assert r['outputs']==('channel_query_output',)
assert r['inputs']==('image_packed_qkv','text_v_rows','image_prob')
report={'hypothesis':'Let one region jointly lower packedV row restoration, concatenation, repetition and its original reversed value consumer; eliminate exposed intermediate boundaries, not mathematically required work.','same_every_node_object':True,'same_other_relative_order_and_four_regions':True,'region':{k:r[k] for k in ['id','inputs','outputs','prefix']},'value_source_indices_checked':checked,'contraction_geometry':[56,128,256,556],'ordered_reduction_terms':556,'score_softmax_probability_and_output_projection':'unchanged from actual valid110, eager softmax/cast/linear','logical_repeated_value_bytes':2*28*556*128*2,'packedV_source':'four last heads in image_packed_qkv; original coefficient order, no extra arithmetic','limitations':['No current compiled109/110 storage strides or kernels are measured. Region may still copy and repeat.','Original56-batch GEMM is preserved; dimensions not changed to grouped98.','Region excludes original output projection unlike57, retaining its existing eager boundary.','All numerical/resource gates remain decisive despite unchanged node objects.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
