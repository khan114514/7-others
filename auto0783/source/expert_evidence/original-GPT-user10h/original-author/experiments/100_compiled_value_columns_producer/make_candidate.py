"""Construct changing value columns directly from both unreplicated branches.

Only indexing/copy operations change; full probability/value GEMM stays eager.
"""
import json
from pathlib import Path
import runpy

directory=Path('experiments/100_compiled_value_columns_producer')
base=Path('experiments/97_compiled_frequency_prefix')
before=json.loads((base/'program.json').read_text())
source=(base/'method.py').read_text().replace(
    'Experiment 97: compile the complete original immutable frequency producer chain.',
    'Experiment 100: compile branchwise channel/key-major value construction.')
old="            projections.append(add(kind,'concat',[text_rows,image_rows],dim=1))\n"
assert source.count(old)==1
source=source.replace(old,'''            # Copy both branches into channel/key-major order before repetition.
            value_text_shape=add('value_text_shape','reshape',[text_rows],shape=[2,300,4,128])
            value_text_heads=add('value_text_heads','transpose',[value_text_shape],dim0=1,dim1=2)
            value_text_columns=add('value_text_columns','transpose',[value_text_heads],dim0=2,dim1=3)
            value_image_shape=add('value_image_shape','reshape',[image_rows],shape=[2,256,4,128])
            value_image_heads=add('value_image_heads','transpose',[value_image_shape],dim0=1,dim1=2)
            value_image_columns=add('value_image_columns','transpose',[value_image_heads],dim0=2,dim1=3)
            value_columns=add('value_columns','concat',[value_text_columns,value_image_columns],dim=3)
            value_transpose=add('value_transpose','repeat_heads',[value_columns],count=7)
            projections.append(value_transpose)
''')
for line in [
    "    value_shape=add('value_shape','reshape',[projections[2]],shape=[2,556,4,128])\n",
    "    value_heads=add('value_heads','transpose',[value_shape],dim0=1,dim1=2)\n",
    "    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)\n",
    "    value_transpose=add('value_transpose','transpose',[value_repeat],dim0=2,dim1=3)\n"]:
    assert source.count(line)==1
    source=source.replace(line,'')
members=['image_v','value_text_shape','value_text_heads','value_text_columns',
         'value_image_shape','value_image_heads','value_image_columns','value_columns','value_transpose']
region={'id':'value_columns_producer_compiled','nodes':members}
old=repr(before['compile_regions'][-1])+']}'
assert source.count(old)==1
source=source.replace(old,repr(before['compile_regions'][-1])+',\n                '+repr(region)+']}')
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
oldnodes={n['id']:n for n in before['layer_body']}
newnodes={n['id']:n for n in after['layer_body']}
removed={'v','value_shape','value_heads','value_repeat'}
added=set(members)-{'image_v','value_transpose'}
assert set(oldnodes)-set(newnodes)==removed
assert set(newnodes)-set(oldnodes)==added
assert {k for k in newnodes if k in oldnodes and newnodes[k]!=oldnodes[k]}=={'value_transpose'}
assert [n for n in after['layer_body'] if n['id'] not in added|{'value_transpose'}]==[
    n for n in before['layer_body'] if n['id'] not in removed|{'value_transpose'}]
assert after['compile_regions'][:-1]==before['compile_regions']
assert len(after['layer_body'])==110
assert all(newnodes[x].get('lifetime','stage')=='stage' for x in members)
assert newnodes['channel_query_output']==oldnodes['channel_query_output']
assert newnodes['image_scores']==oldnodes['image_scores']
Path('program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'removed_nodes':sorted(removed),'added_nodes':sorted(added),
    'changed_original_node':'value_transpose','all_other_node_objects_relative_order_equal':True,
    'three_original_regions_equal':True,'new_contiguous_stage_region':members,
    'no_new_prefix_values_or_lifetime_changes':True,
    'no_GEMM_score_scale_mask_softmax_cast_projection_RMS_or_MLP_changes':True,
    'full_original_28_head_128_by556_value_operand_logical_shape':True,
    'branchwise_transposed_concat_then_repeat_not_token_concat_then_repeat_transpose':True,
    'actual_compiled_strides_copies_GEMM_dispatch_numerics_latency_and_peak_unmeasured':True
},indent=2)+'\n')
print('Constructed110node branchwise value-column producer;fourregions')
