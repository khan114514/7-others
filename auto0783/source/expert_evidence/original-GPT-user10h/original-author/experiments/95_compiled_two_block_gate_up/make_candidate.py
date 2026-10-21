"""Two original-shape paired projections with stage-internal weight packing."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/95_compiled_two_block_gate_up')
source=Path('experiments/62_text_view_common_subexpressions/method.py').read_text()
source=source.replace('"""','"""Experiment 95: two compiled paired gate/up blocks with stage weight packing.\n\n',1)
old="""    image_mlp=add('image_mlp','mlp',[image_post_norm,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])"""
assert source.count(old)==1
replacement='''    # Each paired weight has original N18944 and fits the100M tensor limit.
    # Both copies cannot fit the prefix budget, so they are internal stage
    # values; their actual reconstruction is included in every forward.
    image_gate_first_weight=add('image_gate_first_weight','slice',[
        'w:vision_branch_mlp.gate_proj.weight'],dim=0,start=0,stop=9472)
    image_up_first_weight=add('image_up_first_weight','slice',[
        'w:vision_branch_mlp.up_proj.weight'],dim=0,start=0,stop=9472)
    image_first_fused_weight=add('image_first_fused_weight','concat',[
        image_gate_first_weight,image_up_first_weight],dim=0)
    image_gate_second_weight=add('image_gate_second_weight','slice',[
        'w:vision_branch_mlp.gate_proj.weight'],dim=0,start=9472,stop=18944)
    image_up_second_weight=add('image_up_second_weight','slice',[
        'w:vision_branch_mlp.up_proj.weight'],dim=0,start=9472,stop=18944)
    image_second_fused_weight=add('image_second_fused_weight','concat',[
        image_gate_second_weight,image_up_second_weight],dim=0)
    image_first_fused=add('image_first_fused','linear',[image_post_norm,image_first_fused_weight])
    image_second_fused=add('image_second_fused','linear',[image_post_norm,image_second_fused_weight])
    image_gate_first=add('image_gate_first','slice',[image_first_fused],dim=2,start=0,stop=9472)
    image_up_first=add('image_up_first','slice',[image_first_fused],dim=2,start=9472,stop=18944)
    image_gate_second=add('image_gate_second','slice',[image_second_fused],dim=2,start=0,stop=9472)
    image_up_second=add('image_up_second','slice',[image_second_fused],dim=2,start=9472,stop=18944)
    image_gate=add('image_gate','concat',[image_gate_first,image_gate_second],dim=2)
    image_up=add('image_up','concat',[image_up_first,image_up_second],dim=2)
    image_gate_silu=add('image_gate_silu','silu',[image_gate])
    image_gated_product=add('image_gated_product','mul',[image_gate_silu,image_up])
    image_mlp=add('image_mlp','linear',[image_gated_product,'w:vision_branch_mlp.down_proj.weight'])'''
source=source.replace(old,replacement)
members=['image_gate_first_weight','image_up_first_weight','image_first_fused_weight',
         'image_gate_second_weight','image_up_second_weight','image_second_fused_weight',
         'image_first_fused','image_second_fused','image_gate_first','image_up_first',
         'image_gate_second','image_up_second','image_gate','image_up','image_gate_silu',
         'image_gated_product','image_mlp']
old_region="{'id':'image_mlp_compiled','nodes':['image_mlp']}"
assert source.count(old_region)==1
source=source.replace(old_region,repr({'id':'image_mlp_compiled','nodes':members}))
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/62_text_view_common_subexpressions/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys()&new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'image_mlp'} and not old_nodes.keys()-new_nodes.keys()
assert before['compile_regions'][0]==after['compile_regions'][0]
assert [n for n in after['layer_body'] if n['id'] in old_nodes and n['id']!='image_mlp']==[n for n in before['layer_body'] if n['id']!='image_mlp']
assert not [n for n in after['layer_body'] if n.get('lifetime')=='prefix' and n['id'] not in old_nodes]
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'62_text_view_common_subexpressions',
    'changed_existing_nodes':sorted(changed),
    'all_other_existing_node_objects_and_relative_order_equal':True,
    'original_image_rotary_region_equal':True,
    'MLP_region_members':members,'all_new_MLP_nodes_stage':True,
    'no_new_numerical_prefix_values':True,
    'each_packed_weight_shape':[18944,3584],
    'first_rows_order':['gate0:9472','up0:9472'],
    'second_rows_order':['gate9472:18944','up9472:18944'],
    'both_projection_shapes_equal_original_M_N_K':[512,18944,3584],
    'full_ordered_gate_up_and_down_K_preserved':True,
    'packing_reconstruction_is_timed_on_every_forward':True,
    'compiler_elimination_of_weight_copies_unmeasured':True
},indent=2)+'\n')
