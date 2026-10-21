"""Keep selected fused text GEMM, copy only its cached Q/K/V output views."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/82_isolated_text_projection_packing')
source=Path('experiments/53_compiled_image_rotary/method.py').read_text()
source=source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                      'Experiment 82: isolated cached text Q/K/V output packing.')
old="        text_rows=add('text_'+kind+'_rows','slice',[text_qkv],dim=2,start=start,stop=stop)"
new="""        text_view=add('text_'+kind+'_rows_view','slice',[text_qkv],dim=2,start=start,stop=stop)
        text_rows=add('text_'+kind+'_rows','concat',[text_view],dim=2)"""
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
assert new_nodes.keys()-old_nodes.keys()=={'text_q_rows_view','text_k_rows_view','text_v_rows_view'}
assert not old_nodes.keys()-new_nodes.keys()
assert before['compile_regions']==after['compile_regions']
for kind in ('q','k','v'):
    original=old_nodes['text_'+kind+'_rows'].copy()
    original['id']='text_'+kind+'_rows_view'
    assert new_nodes[original['id']]==original
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary','changed_existing_nodes':sorted(changed),
    'only_new_nodes_are_original_slice_copies':True,
    'all_other_nodes_and_regions_equal':True,
    'original_text_GEMM_weight_bias_and_geometry_equal':True,
    'new_operation_is_bit_preserving_copy_before_downstream_consumers':True,
    'no_GPU_downstream_dispatch_qualification_from_local_audit':True},indent=2)+'\n')
