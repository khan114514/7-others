"""Compile original-basis column projection plus its residual consumer."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/101_compiled_column_projection_residual')
base=Path('experiments/97_compiled_frequency_prefix')
before=json.loads((base/'program.json').read_text())
source=(base/'method.py').read_text().replace(
    'Experiment 97: compile the complete original immutable frequency producer chain.',
    'Experiment 101: compile original-basis column projection and residual addition.')
start=source.index("    image_output_heads=add('image_output_heads'")
stop=source.index("    text_state=add('text_state_rows'",start)
old42=Path('experiments/42_direct_channel_output_projection/method.py').read_text()
first=old42.index('    # Heads and channels are already adjacent')
last=old42.index("    text_state=add('text_state_rows'",first)
residual="    image_residual=add('image_residual_rows','add',[image_state,image_out])\n"
assert source.count(residual)==1
source=source[:start]+old42[first:last]+residual+source[stop:]
# The new adjacent region includes this one original add before text nodes.
assert source.count(residual)==2
position=source.index(residual,source.index("    text_state=add('text_state_rows'"))
source=source[:position]+source[position+len(residual):]
members=['image_attended_columns','image_projected_columns','image_attention_out','image_residual_rows']
region={'id':'column_projection_residual_compiled','nodes':members}
old=repr(before['compile_regions'][-1])+']}'
assert source.count(old)==1
source=source.replace(old,repr(before['compile_regions'][-1])+',\n                '+repr(region)+']}')
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
original={n['id']:n for n in before['layer_body']}
new={n['id']:n for n in after['layer_body']}
removed={'image_output_heads','image_output_rows','image_attended_rows'}
added={'image_attended_columns','image_projected_columns'}
assert set(original)-set(new)==removed and set(new)-set(original)==added
assert {k for k in new if k in original and new[k]!=original[k]}=={'image_attention_out'}
assert [n for n in after['layer_body'] if n['id'] not in added|{'image_attention_out','image_residual_rows'}]==[
    n for n in before['layer_body'] if n['id'] not in removed|{'image_attention_out','image_residual_rows'}]
assert len(after['layer_body'])==106
assert after['compile_regions'][:-1]==before['compile_regions']
assert new['image_residual_rows']==original['image_residual_rows']
assert all(new[x].get('lifetime','stage')=='stage' for x in members)
Path('program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'removed_nodes':sorted(removed),'added_nodes':sorted(added),
    'changed_original_node':'image_attention_out',
    'original_residual_add_moved_adjacent_without_expression_change':True,
    'all_other_node_objects_relative_order_lifetimes_equal':True,
    'three_original_regions_equal':True,'new_stage_region_nodes':members,
    'projection_uses_exact_original_unpermuted_checkpoint_weight':True,
    'full_original3584_reduction_order_and_KV_G_D_basis_preserved':True,
    'probability_value_GEMM_remains_outside_unchanged':True,
    'no_new_prefix_values_or_GEMM_input_packing_claim':True,
    'column_GEMM_dispatch_transpose_residual_lowering_numerics_latency_and_peak_unmeasured':True
},indent=2)+'\n')
print('Constructed106node column-projection/residualgraph;fourregions')
