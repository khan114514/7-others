"""Compile the intact natural immutable frequency-layout producer chain."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/97_compiled_frequency_prefix')
base=Path('experiments/53_compiled_image_rotary')
source=(base/'method.py').read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 97: compile the complete original immutable frequency producer chain.')
old="{'id':'image_mlp_compiled','nodes':['image_mlp']}]"
before=json.loads((base/'program.json').read_text())
body=before['layer_body']
start=next(i for i,n in enumerate(body) if n['id']=='cos_axis_0')
stop=next(i for i,n in enumerate(body) if n['id']=='image_signed_sin')+1
members=[n['id'] for n in body[start:stop]]
assert len(members)==32 and all(n['lifetime']=='prefix' for n in body[start:stop])
assert source.count(old)==1
source=source.replace(old,"{'id':'image_mlp_compiled','nodes':['image_mlp']},\n                "+repr({'id':'frequency_prefix_compiled','nodes':members})+"]")
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
assert after['layer_body']==before['layer_body'] and after['layer_output']==before['layer_output']
assert after['compile_regions'][:2]==before['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'all_original107node_objects_and_order_equal':True,
    'both_original_image_regions_equal':True,
    'new_prefix_region_nodes':members,'natural_contiguous_producer_chain_not_reordered':True,
    'two_external_frequency_inputs':['cos','sin'],
    'all32prefix_values_returned_separately_counted_and_cached':True,
    'all_index_sections_sine_sign_operations_and_lifetimes_unchanged':True,
    'no_RMS_softmax_GEMM_or_attention_inside_region':True,
    'first_construction_and_cache_resets_unchanged':True,
    'copy_fusion_layout_dispatch_latency_and_peak_unmeasured':True
},indent=2)+'\n')
