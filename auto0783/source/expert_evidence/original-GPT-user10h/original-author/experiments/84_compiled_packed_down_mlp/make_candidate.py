"""Compile row-major gate/up MLP with a prefix-packed frozen down weight."""
from pathlib import Path
import hashlib
import json
import runpy

directory=Path('experiments/84_compiled_packed_down_mlp')
source=Path('experiments/83_separate_published_shape_text_projections/method.py').read_text()
source=source.replace('Experiment 83: separate cached full-shaped text projections with selected image path.',
                      'Experiment 84: compiled row-major MLP with cached contiguous down weight.')
old="""    image_mlp=add('image_mlp','mlp',[image_post_norm,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])"""
new="""    # Frozen down coefficients are copied in transposed physical storage
    # once per prefix. The view stays stage-local; copy construction is timed.
    image_down_view=add('image_down_weight_view','transpose',[
        'w:vision_branch_mlp.down_proj.weight'],dim0=0,dim1=1)
    image_down_weight=add('image_down_weight_contiguous','concat',[
        image_down_view],dim=0,lifetime='prefix')
    # Keep original row-major projections and rounded SiLU/gating sequence.
    image_gate=add('image_gate','linear',[
        image_post_norm,'w:vision_branch_mlp.gate_proj.weight'])
    image_gate_silu=add('image_gate_silu','silu',[image_gate])
    image_up=add('image_up','linear',[
        image_post_norm,'w:vision_branch_mlp.up_proj.weight'])
    image_gated=add('image_gated','mul',[image_gate_silu,image_up])
    image_mlp=add('image_mlp','matmul',[image_gated,image_down_weight])"""
assert source.count(old)==1
source=source.replace(old,new)
old_region="{'id':'image_mlp_compiled','nodes':['image_mlp']}"
stage_ids=['image_gate','image_gate_silu','image_up','image_gated','image_mlp']
assert source.count(old_region)==1
source=source.replace(old_region,"{'id':'image_mlp_compiled','nodes':"+repr(stage_ids)+"}")
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/83_separate_published_shape_text_projections/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'image_mlp'}
assert not old_nodes.keys()-new_nodes.keys()
assert new_nodes.keys()-old_nodes.keys()=={'image_down_weight_view','image_down_weight_contiguous','image_gate','image_gate_silu','image_up','image_gated'}
assert before['compile_regions'][0]==after['compile_regions'][0]
digest=hashlib.sha256(json.dumps(after,sort_keys=True,separators=(',',':')).encode()).hexdigest()
expected=json.loads(Path('research-diagnostics/compiled-packed-down-feasibility.json').read_text())
assert digest==expected['prospective_canonical_sha256']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'83_separate_published_shape_text_projections',
    'only_changed_existing_node':'image_mlp',
    'all_other_nodes_and_original_rotary_region_equal':True,
    'new_dynamic_region_nodes':stage_ids,'canonical_sha256':digest,
    'matches_preceding_static_proposal_digest':True,
    'new_prefix_copy_bytes_all28layers':3802136576,
    'old_prefix_lifetimes_preserved':True},indent=2)+'\n')
