"""Separate cached text projections with original full2x556row geometry."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/83_separate_published_shape_text_projections')
source=Path('experiments/53_compiled_image_rotary/method.py').read_text()
source=source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                      'Experiment 83: separate cached full-shaped text projections with selected image path.')
start=source.index('    # Fuse the cached biased text projections')
stop=source.index("    qkv_weight=add('vision_qkv_weight'",start)
source=source[:start]+"""    # Restore published text-projection row geometry while avoiding fused
    # text weight/bias copies. Auxiliary image rows are sliced away before
    # every cross-token text operation, so their prefix reuse is harmless.
    text_projection_input=add('text_projection_input','concat',[
        text_norm_rows,image_norm],dim=1)
    text_full_projections={}
    for kind in ['q','k','v']:
        text_full_projections[kind]=add('text_'+kind+'_full','linear',[
            text_projection_input,'w:self_attn.'+kind+'_proj.weight',
            'w:self_attn.'+kind+'_proj.bias'])
"""+source[stop:]
old="        text_rows=add('text_'+kind+'_rows','slice',[text_qkv],dim=2,start=start,stop=stop)"
new="        text_rows=add('text_'+kind+'_rows','slice',[text_full_projections[kind]],dim=1,start=0,stop=300)"
assert source.count(old)==1
source=source.replace(old,new)
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/53_compiled_image_rotary/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'text_q_rows','text_k_rows','text_v_rows'}
removed=old_nodes.keys()-new_nodes.keys()
added=new_nodes.keys()-old_nodes.keys()
assert removed=={'text_qkv_weight','text_qkv_bias','text_input_flat','text_flat_qkv','text_qkv'}
assert added=={'text_projection_input','text_q_full','text_k_full','text_v_full'}
assert before['compile_regions']==after['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary','changed_existing_nodes':sorted(changed),
    'removed_nodes':sorted(removed),'added_nodes':sorted(added),
    'all_other_existing_nodes_and_compile_regions_equal':True,
    'separate_text_linear_input_shape':[2,556,3584],
    'cache_safety_argument_requires_all_auxiliary_image_rows_discarded':True},indent=2)+'\n')
