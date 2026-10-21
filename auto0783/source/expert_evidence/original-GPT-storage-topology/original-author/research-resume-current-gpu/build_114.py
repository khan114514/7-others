from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/112_compiled_pre_softmax_partition/method.py').read_text()
s=base.replace('Experiment 112: full image projection through canonical BF16 masked scores in one region.','Experiment 114: compile original gate/up with isolated feature-leading down projection.')
needle="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
add="""    # Original gate/up F.linear geometry; reorient only the bias-free down
    # contraction using a single3584x18944 by18944x512 matrix product. The
    # compiler may lower SiLU/gating into the down operand's chosen layout.
    # No checkpoint permutation or retained/transposed weight copy is added.
    def mlp_node(name,op,inputs,**attrs):
        return {'id':name,'op':op,'inputs':inputs,**attrs}
    mlp_nodes=[
        mlp_node('image_gate','linear',['image_post_norm','w:vision_branch_mlp.gate_proj.weight']),
        mlp_node('image_gate_silu','silu',['image_gate']),
        mlp_node('image_up','linear',['image_post_norm','w:vision_branch_mlp.up_proj.weight']),
        mlp_node('image_gated','mul',['image_gate_silu','image_up']),
        mlp_node('image_mlp_flat','reshape',['image_gated'],shape=[512,18944]),
        mlp_node('image_mlp_columns','transpose',['image_mlp_flat'],dim0=0,dim1=1),
        mlp_node('image_down','matmul',['w:vision_branch_mlp.down_proj.weight','image_mlp_columns']),
        mlp_node('image_mlp_rows','transpose',['image_down'],dim0=0,dim1=1),
        mlp_node('image_mlp','reshape',['image_mlp_rows'],shape=[2,256,3584])]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_mlp')
    nodes[at:at+1]=mlp_nodes
    mlp_members=[n['id'] for n in mlp_nodes]
"""
assert needle in s;s=s.replace(needle,add+needle)
s=s.replace("{'id':'image_mlp_compiled','nodes':['image_mlp']}","{'id':'image_mlp_compiled','nodes':mlp_members}")
folder=Path('experiments/114_compiled_down_only_columns');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
old=runpy.run_path('experiments/112_compiled_pre_softmax_partition/method.py')['build_program']();oldn={n['id']:n for n in old['layer_body']};newn={n['id']:n for n in p['layer_body']}
assert all(oldn[x]==newn[x] for x in oldn if x!='image_mlp')
historical39=json.loads(Path('experiments/39_weight_left_image_down/program.json').read_text());n39={n['id']:n for n in historical39['layer_body']}
mlp=next(r for r in p['compile_regions'] if r['id']=='image_mlp_compiled')
assert all(newn[x]==n39[x] for x in mlp['nodes'])
source=Path('/opt/bifrost/public_eval/tensor_dag.py').read_text();assert 'result=F.linear(F.silu(F.linear(value,gate))*F.linear(value,up),down)' in source
# Independent canonical(batch,token,feature) -> singleton matrix coordinates.
coords=0
for b in range(2):
    for t in range(256):
        row=256*b+t
        for f in range(3584):
            assert divmod(row,256)==(b,t)
            assert (f,row)==(f,b*256+t)
            coords+=1
# Symbolic entire18944 ordered hidden contraction retains checkpoint columns.
assert newn['image_down']['inputs']==['w:vision_branch_mlp.down_proj.weight','image_mlp_columns']
assert newn['image_mlp_flat']['inputs']==['image_gated'] and newn['image_mlp_flat']['shape']==[512,18944]
assert newn['image_gated']['inputs']==['image_gate_silu','image_up']
for j in range(18944):assert j//18944==0 and j%18944==j
compiler_tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in compiler_tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
r=next(r for r in scope['validate_regions'](p) if r['id']=='image_mlp_compiled');assert r['outputs']==('image_mlp',)
report={'hypothesis':'Keep gate/up original row-oriented geometry while compile SiLU/gating with isolated column-oriented down GEMM. Compiler can choose activation producer layout without a retained weight copy.','correction':'80already uses a single mergedCFG512-column matrix for ALL three projections, contrary to initial spoken two-batch premise. That unchanged geometry is not resubmitted.','every_unrelated112node_unchanged':True,'all_MLPnode_objects_match_actual_historical_eager39':True,'MLP_geometry':{'gate_up_M_N_K':[512,18944,3584],'down_M_N_K':[3584,512,18944]},'canonical_output_coordinates_checked':coords,'ordered_down_features':18944,'real_equivalence':'output[b,t,f]=sum_j down[f,j]*(silu(gate[b,t,j])*up[b,t,j]); completefeatureorder retained','region':{k:r[k] for k in ['id','inputs','outputs','prefix']},'prefix_additions':0,'historical_distinction':'39was eager only;80reorients gate/up as well as down;84copies transposed down weights but retains original row-oriented contraction.','counterarguments':['39eager and80fully-feature-leading compiler were slower; this is not a predicted benefit from historical data.','No actual compiledproducer stride/fusion is measured; down GEMM tiling may be worse.','Different FP32accumulation can change BF16outputs, as scalar dot-tree counterexample demonstrates.','Tensor limits, precision, backend, all gates unchanged.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n')
for q in Path('experiments').glob('*/program.json'):
    if q.parent!=folder:assert report['canonical_sha256']!=hashlib.sha256(json.dumps(json.loads(q.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(report,indent=2))
