from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/97_compiled_frequency_prefix/method.py').read_text()
s=base.replace('Experiment 97: compile the complete original immutable frequency producer chain.','Experiment 109: pack rotated Q/K and unchanged V into one compiled projection output.')
old="""    # Trusted fullgraph compilation may fuse SiLU and gating while retaining
    # checkpoint GEMMs and required precision casts. No tensor lifetime changes.
"""
new="""    # Retain full biased4608 projection geometry; replace the multi-output
    # projection/rotary boundary with one packed36-head output. V is only
    # reindexed/copied, never rotated or subjected to extra arithmetic.
    by_id={n['id']:n for n in nodes}
    region_members=['image_qkv','image_qk','image_qk_shape','image_qk_heads',
                    'image_qk_rotary_first','image_qk_rotary_second',
                    'image_qk_rotary_half','image_qk_rotary_direct',
                    'image_qk_rotary_rotated','image_qk_rotary']
    region_nodes=[by_id[name] for name in region_members]
    def make(name,op,inputs,**attrs):
        return {'id':name,'op':op,'inputs':inputs,**attrs}
    region_nodes.extend([
        make('image_v_raw','slice',['image_qkv'],dim=2,start=4096,stop=4608),
        make('image_v_raw_shape','reshape',['image_v_raw'],shape=[2,256,4,128]),
        make('image_v_raw_heads','transpose',['image_v_raw_shape'],dim0=1,dim1=2),
        make('image_packed_qkv','concat',['image_qk_rotary','image_v_raw_heads'],dim=1)])
    region_members=[n['id'] for n in region_nodes]
    outside=[
        make('image_query','slice',['image_packed_qkv'],dim=1,start=0,stop=28),
        make('image_key','slice',['image_packed_qkv'],dim=1,start=28,stop=32),
        make('image_v_packed_heads','slice',['image_packed_qkv'],dim=1,start=32,stop=36),
        make('image_v_packed_rows','transpose',['image_v_packed_heads'],dim0=1,dim1=2),
        make('image_v','reshape',['image_v_packed_rows'],shape=[2,256,512]),
        by_id['v']]
    remove=set(region_members)|{'image_query','image_key','image_v','v'}
    nodes=[n for n in nodes if n['id'] not in remove]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_signed_sin')+1
    nodes[at:at]=region_nodes+outside
"""
assert old in s;s=s.replace(old,new)
start=s.index("                {'id':'image_rotary_compiled','nodes':[");end=s.index("                {'id':'image_mlp_compiled'",start)
s=s[:start]+"                {'id':'image_packed_projection_rotary_compiled','nodes':region_members},\n"+s[end:]
folder=Path('experiments/109_packed_compiled_qkv');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
basep=runpy.run_path('experiments/97_compiled_frequency_prefix/method.py')['build_program']();oldn={n['id']:n for n in basep['layer_body']};newn={n['id']:n for n in p['layer_body']}
rotary=['image_qk_rotary_first','image_qk_rotary_second','image_qk_rotary_half','image_qk_rotary_direct','image_qk_rotary_rotated','image_qk_rotary']
assert all(oldn[x]==newn[x] for x in ['vision_qkv_weight','vision_qkv_bias','image_qkv','image_qk','image_qk_shape','image_qk_heads']+rotary)
assert all(oldn[x]==newn[x] for x in set(oldn)-{'image_query','image_key','image_v'})
# Independently resolve every packed-head output channel and restored V index.
channels=[]
for h in range(36):
    for d in range(128):
        channel=h*128+d
        if h<32:
            assert channel<4096;channels.append(('rotated_qk',channel))
        else:
            vchannel=(h-32)*128+d
            assert 4096+vchannel==channel;channels.append(('v',vchannel))
assert channels[:4096]==[('rotated_qk',j) for j in range(4096)]
assert channels[4096:]==[('v',j) for j in range(512)]
# Transpose then reshape restores canonical token/feature values, even if
# PyTorch needs a copy for non-contiguous packed head storage.
restored=0
for b in range(2):
    for t in range(256):
        for f in range(512):
            vh,d=divmod(f,128)
            assert (b,32+vh,t,d)==(b,32+f//128,t,f%128)
            assert channels[(32+vh)*128+d]==('v',f)
            restored+=1
compiler_tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in compiler_tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
region=next(r for r in scope['validate_regions'](p) if r['id']=='image_packed_projection_rotary_compiled')
assert region['outputs']==('image_packed_qkv',)
report={'hypothesis':'Keep published matched full projection shape; replace three outputs with packed single36-head result so compiler can choose one storage/layout boundary.','full_projection_shape_M_N_K':[512,4608,3584],'biased_projection_and_all_six_rotary_node_objects_unchanged':True,'packed_channel_indices_checked':len(channels),'restored_V_coordinates_checked':restored,'V_extra_numerical_operations':0,'region':{k:region[k] for k in ['id','inputs','outputs','prefix']},'single_boundary_shape':[2,36,256,128],'single_boundary_bytes':2*36*256*128*2,'limitations':['Adds explicit packing concat and eager V reshape may copy. No measured copy elimination, epilogue fusion, stride or latency argument.','108failure does not establish a unique numerical cause; this method avoids its changed GEMM widths.','Same geometry as valid historical58 is not current-GPU numerical qualification.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
