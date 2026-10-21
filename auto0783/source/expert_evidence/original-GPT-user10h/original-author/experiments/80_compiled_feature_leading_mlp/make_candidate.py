"""Internalize feature-leading gated MLP in one trusted compiled stage region."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/80_compiled_feature_leading_mlp')
source=Path('experiments/53_compiled_image_rotary/method.py').read_text()
source=source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                      'Experiment 80: compile feature-leading image MLP with selected rotary.')
old="""    image_mlp=add('image_mlp','mlp',[image_post_norm,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])"""
new="""    # Feature-leading gated MLP. Trusted fullgraph compilation internalizes
    # all views and BF16 pointwise intermediates; weights stay unchanged.
    image_mlp_flat=add('image_mlp_flat','reshape',[image_post_norm],shape=[512,3584])
    image_mlp_columns=add('image_mlp_columns','transpose',[image_mlp_flat],dim0=0,dim1=1)
    image_gate=add('image_gate','matmul',['w:vision_branch_mlp.gate_proj.weight',image_mlp_columns])
    image_up=add('image_up','matmul',['w:vision_branch_mlp.up_proj.weight',image_mlp_columns])
    image_gate_silu=add('image_gate_silu','silu',[image_gate])
    image_gated=add('image_gated','mul',[image_gate_silu,image_up])
    image_down=add('image_down','matmul',['w:vision_branch_mlp.down_proj.weight',image_gated])
    image_mlp_rows=add('image_mlp_rows','transpose',[image_down],dim0=0,dim1=1)
    image_mlp=add('image_mlp','reshape',[image_mlp_rows],shape=[2,256,3584])"""
assert source.count(old)==1
source=source.replace(old,new)
old_region="{'id':'image_mlp_compiled','nodes':['image_mlp']}"
new_ids=['image_mlp_flat','image_mlp_columns','image_gate','image_up',
         'image_gate_silu','image_gated','image_down','image_mlp_rows','image_mlp']
assert source.count(old_region)==1
source=source.replace(old_region,"{'id':'image_mlp_compiled','nodes':"+repr(new_ids)+"}")
source=source.replace('trusted compiled image rotary region.', 'compiled feature-leading image MLP.')
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/53_compiled_image_rotary/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
assert {name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}=={'image_mlp'}
assert old_nodes.keys()-new_nodes.keys()==set()
assert new_nodes.keys()-old_nodes.keys()==set(new_ids)-{'image_mlp'}
assert before['compile_regions'][0]==after['compile_regions'][0]
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary','compiled_mlp_nodes':new_ids,
    'only_changed_existing_node':'image_mlp',
    'all_other_nodes_equal':True,'rotary_region_equal':True,
    'extra_prefix_tensors':0,'new_weight_copies':0},indent=2)+'\n')
