from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/97_compiled_frequency_prefix/method.py').read_text()
s=base.replace('Experiment 97: compile the complete original immutable frequency producer chain.','Experiment 108: isolate V; compile original-basis image QK projection and rotary.')
s=s.replace("'vision_qkv_weight'","'vision_qk_weight'").replace("'vision_qkv_bias'","'vision_qk_bias'")
s=s.replace("'w:self_attn.vision_branch_k_proj.weight',\n        'w:self_attn.vision_branch_v_proj.weight'","'w:self_attn.vision_branch_k_proj.weight'")
s=s.replace("'w:self_attn.vision_branch_k_proj.bias',\n        'w:self_attn.vision_branch_v_proj.bias'","'w:self_attn.vision_branch_k_proj.bias'")
s=s.replace("    image_qkv=add('image_qkv','linear',[image_norm,qkv_weight,qkv_bias])","    # Independent V stays eager; QK is projected immediately before rotary.\n    image_v=add('image_v','linear',[image_norm,\n        'w:self_attn.vision_branch_v_proj.weight',\n        'w:self_attn.vision_branch_v_proj.bias'])")
s=s.replace("            image_rows=add('image_v','slice',[image_qkv],dim=2,start=start,stop=stop)","            image_rows=image_v")
s=s.replace("    image_qk=add('image_qk','slice',[image_qkv],dim=2,start=0,stop=4096)","    image_qk=add('image_qk','linear',[image_norm,qkv_weight,qkv_bias])")
old="""    # Trusted fullgraph compilation may fuse SiLU and gating while retaining
    # checkpoint GEMMs and required precision casts. No tensor lifetime changes.
"""
new="""    # Contiguous QK producer/rotary region returns only rotated Q and K.
    # Moving V's slice outside a fused projection instead would expose QKV.
    # Dot products/biases retain checkpoint feature order; Dev qualifies BF16.
    qk_members=['image_qk','image_qk_shape','image_qk_heads',
                'image_qk_rotary_first','image_qk_rotary_second',
                'image_qk_rotary_half','image_qk_rotary_direct',
                'image_qk_rotary_rotated','image_qk_rotary',
                'image_query','image_key']
    moved={n['id']:n for n in nodes if n['id'] in qk_members}
    nodes=[n for n in nodes if n['id'] not in moved]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_signed_sin')+1
    nodes[at:at]=[moved[name] for name in qk_members]
"""
assert old in s;s=s.replace(old,new)
start=s.index("                {'id':'image_rotary_compiled','nodes':[")
end=s.index("                {'id':'image_mlp_compiled'",start)
s=s[:start]+"                {'id':'image_qk_projection_rotary_compiled','nodes':qk_members},\n"+s[end:]
folder=Path('experiments/108_isolated_v_compiled_qk');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s)
p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n')
assert json.loads(Path('program.json').read_text())==runpy.run_path('experiments/97_compiled_frequency_prefix/method.py')['build_program']()
# Independent symbolic coefficient/bias mapping, all output channels,
# retaining the entire original ordered3584-feature contraction.
original=[('q',j) for j in range(3584)]+[('k',j) for j in range(512)]+[('v',j) for j in range(512)]
newqk=[('q',j) for j in range(3584)]+[('k',j) for j in range(512)]
newv=[('v',j) for j in range(512)]
assert original==newqk+newv
nodes={n['id']:n for n in p['layer_body']}
assert nodes['vision_qk_weight']['inputs']==['w:self_attn.vision_branch_q_proj.weight','w:self_attn.vision_branch_k_proj.weight']
assert nodes['vision_qk_bias']['inputs']==['w:self_attn.vision_branch_q_proj.bias','w:self_attn.vision_branch_k_proj.bias']
assert nodes['image_v']['inputs']==['image_input_norm','w:self_attn.vision_branch_v_proj.weight','w:self_attn.vision_branch_v_proj.bias']
assert nodes['image_qk']['inputs']==['image_input_norm','vision_qk_weight','vision_qk_bias']
# Each Q/K rotary head still indexes the same adjacent128 features.
for head in range(32):
    for d in range(128):assert original[head*128+d]==newqk[head*128+d]
# Use actual published allocation-free boundary derivation.
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text())
scope={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
regions=scope['validate_regions'](p)
qk=next(r for r in regions if r['id']=='image_qk_projection_rotary_compiled')
assert qk['outputs']==('image_query','image_key')
report={'hypothesis':'Separate V removes unrotated V from compiled QK boundary, allowing different producer/rotary lowering. Benefits unknown; adds one GEMM.','symbolic_output_coefficients_and_biases_checked':len(original),'ordered_reduction_features':3584,'qk_head_coordinates_checked':4096,'input_norm_and_rotary_rounding_nodes':'unchanged','region':{k:qk[k] for k in ['id','inputs','outputs','prefix']},'real_equivalence':'all checkpoint coefficients/biases and rotary indices preserved','numerical_limit':'Changed GEMM output width/strides and compiler may alter BF16; no numerical guarantee from symbolic audit','old_58_boundary':['image_v','image_query','image_key'],'all_other_regions':'selected97 frequency prefix and atomic MLP','canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n')
Path('method.py').write_text(s)
print(json.dumps(report,indent=2))
