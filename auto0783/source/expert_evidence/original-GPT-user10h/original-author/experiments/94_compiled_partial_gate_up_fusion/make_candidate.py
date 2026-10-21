"""Half gate/up fusion preserves one original N dimension within cache bounds."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/94_compiled_partial_gate_up_fusion')
source=Path('experiments/62_text_view_common_subexpressions/method.py').read_text()
source=source.replace('"""','"""Experiment 94: compiled partially fused gate/up projection.\n\n',1)
old="""    image_mlp=add('image_mlp','mlp',[image_post_norm,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])"""
assert source.count(old)==1
replacement='''    # Full gate/up concatenation violates the per-tensor/cache bounds.
    # Pair the first9472 rows of each projection: combined N=18944 matches
    # an original complete projection. Keep full ordered down reduction.
    # Only the immutable combined copy is retained; slices are stage views.
    image_gate_first_weight=add('image_gate_first_weight','slice',[
        'w:vision_branch_mlp.gate_proj.weight'],dim=0,start=0,stop=9472)
    image_up_first_weight=add('image_up_first_weight','slice',[
        'w:vision_branch_mlp.up_proj.weight'],dim=0,start=0,stop=9472)
    image_fused_gate_up_weight=add('image_fused_gate_up_weight','concat',[
        image_gate_first_weight,image_up_first_weight],dim=0,lifetime='prefix')
    image_gate_remaining_weight=add('image_gate_remaining_weight','slice',[
        'w:vision_branch_mlp.gate_proj.weight'],dim=0,start=9472,stop=18944)
    image_up_remaining_weight=add('image_up_remaining_weight','slice',[
        'w:vision_branch_mlp.up_proj.weight'],dim=0,start=9472,stop=18944)
    image_fused_gate_up=add('image_fused_gate_up','linear',[
        image_post_norm,image_fused_gate_up_weight])
    image_gate_fused=add('image_gate_fused','slice',[image_fused_gate_up],dim=2,start=0,stop=9472)
    image_up_fused=add('image_up_fused','slice',[image_fused_gate_up],dim=2,start=9472,stop=18944)
    image_gate_remaining=add('image_gate_remaining','linear',[image_post_norm,image_gate_remaining_weight])
    image_up_remaining=add('image_up_remaining','linear',[image_post_norm,image_up_remaining_weight])
    image_gate=add('image_gate','concat',[image_gate_fused,image_gate_remaining],dim=2)
    image_up=add('image_up','concat',[image_up_fused,image_up_remaining],dim=2)
    image_gate_silu=add('image_gate_silu','silu',[image_gate])
    image_gated_product=add('image_gated_product','mul',[image_gate_silu,image_up])
    image_mlp=add('image_mlp','linear',[image_gated_product,'w:vision_branch_mlp.down_proj.weight'])'''
source=source.replace(old,replacement)
old_region="{'id':'image_mlp_compiled','nodes':['image_mlp']}"
members=['image_gate_remaining_weight','image_up_remaining_weight',
         'image_fused_gate_up','image_gate_fused','image_up_fused',
         'image_gate_remaining','image_up_remaining','image_gate','image_up',
         'image_gate_silu','image_gated_product','image_mlp']
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
assert [n['id'] for n in after['layer_body'] if n.get('lifetime')=='prefix' and n['id'] not in old_nodes]==['image_fused_gate_up_weight']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'62_text_view_common_subexpressions',
    'base_retains_original53_arithmetic_after_two_known_view_cleanups':True,
    'changed_existing_nodes':sorted(changed),
    'new_nodes':sorted(new_nodes.keys()-old_nodes.keys()),
    'all_other_existing_node_objects_and_relative_order_equal':True,
    'original_image_rotary_region_equal':True,
    'immutable_fused_weight_shape':[18944,3584],
    'one_new_prefix_node':'image_fused_gate_up_weight',
    'new_stage_MLP_region_members':members,
    'original_hidden_and_down_reduction_sizes':[3584,18944],
    'chosen_block_reason':'Pair two9472-output halves so fused GEMM N=18944 equals an original gate/up projection; full fusion exceeds fixed limits.',
    'performance_tradeoff':'Three input projections replace two, plus retained copy first-construction; no operation-count/input-read gain promised.',
    'numerical_GPU_layout_dispatch_and_peak_unqualified':True
},indent=2)+'\n')
