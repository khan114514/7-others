"""Compile the original cached biased text projection and its packing boundary."""
import hashlib
import json
from pathlib import Path
import runpy

directory=Path('experiments/96_compiled_prefix_text_qkv')
base=Path('experiments/53_compiled_image_rotary')
before=json.loads((base/'program.json').read_text())
assert Path('method.py').read_bytes()==(base/'method.py').read_bytes()
assert json.loads(Path('program.json').read_text())==before
source=(base/'method.py').read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 96: compile the original prefix text QKV projection and packing.')
anchor="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
assert source.count(anchor)==1
insertion='''    # Bring the three existing prefix views next to their projection source.
    # Every operation, coefficient, bias, shape and lifetime remains original.
    row_ids=['text_q_rows','text_k_rows','text_v_rows']
    rows={node['id']:node for node in nodes if node['id'] in row_ids}
    nodes=[node for node in nodes if node['id'] not in row_ids]
    position=next(i for i,node in enumerate(nodes) if node['id']=='text_qkv')+1
    nodes[position:position]=[rows[name] for name in row_ids]
'''
source=source.replace(anchor,insertion+anchor)
old="{'id':'image_mlp_compiled','nodes':['image_mlp']}]"
new="""{'id':'image_mlp_compiled','nodes':['image_mlp']},
                {'id':'text_projection_compiled','nodes':[
                    'text_qkv_weight','text_qkv_bias','text_input_flat',
                    'text_flat_qkv','text_qkv','text_q_rows','text_k_rows','text_v_rows']}]"""
assert source.count(old)==1
source=source.replace(old,new)
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
assert after==json.loads(Path('research-diagnostics/compiled-text-projection-prospective-program.json').read_text())
assert {n['id']:n for n in before['layer_body']}=={n['id']:n for n in after['layer_body']}
moved=['text_q_rows','text_k_rows','text_v_rows']
assert [n for n in before['layer_body'] if n['id'] not in moved]==[n for n in after['layer_body'] if n['id'] not in moved]
assert after['compile_regions'][:2]==before['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'all_original107node_objects_equal':True,
    'moved_nodes':moved,'relative_order_of_all_other_nodes_equal':True,
    'both_original_image_regions_equal':True,
    'prefix_projection_M_N_K':[600,4608,3584],
    'all_weight_bias_shape_lifetime_and_input_constants_unchanged':True,
    'all_eight_prefix_outputs_separately_observable_cached_and_counted':True,
    'first_construction_after_reset_included':True,
    'canonical_sha256':hashlib.sha256(json.dumps(after,sort_keys=True,separators=(',',':')).encode()).hexdigest()
},indent=2)+'\n')
