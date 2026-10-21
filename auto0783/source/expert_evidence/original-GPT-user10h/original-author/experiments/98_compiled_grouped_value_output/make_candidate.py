"""Group value contraction then compile canonical packing/output projection.

Frozen output weight and its original feature reduction order stay unchanged.
"""
import json
from pathlib import Path
import runpy

directory=Path('experiments/98_compiled_grouped_value_output')
base=Path('experiments/97_compiled_frequency_prefix')
before=json.loads((base/'program.json').read_text())
source=(base/'method.py').read_text().replace(
    'Experiment 97: compile the complete original immutable frequency producer chain.',
    'Experiment 98: compile grouped value contraction with canonical output packing.')
removed="    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)\n"
assert source.count(removed)==1
source=source.replace(removed,'')
start=source.index('    # Reverse only the probability/value contraction;')
stop=source.index('    text_state=add(',start)
old44=(Path('experiments/44_grouped_value_product')/'method.py').read_text()
first=old44.index('    # Merge seven query heads sharing each KV head into query rows.')
last=old44.index('    text_state=add(',first)
source=source[:start]+old44[first:last]+source[stop:]
members=['value_transpose','probability_groups','probability_transpose',
         'channel_query_output','channel_group_output','image_token_first',
         'image_output_rows','image_attended_rows','image_attention_out']
region={'id':'grouped_value_output_compiled','nodes':members}
old=repr(before['compile_regions'][-1])+']}'
assert source.count(old)==1
source=source.replace(old,repr(before['compile_regions'][-1])+',\n                '+repr(region)+']}')
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={'value_transpose','probability_transpose','image_output_rows'}
removed_ids={'value_repeat','image_output_heads'}
added_ids={'probability_groups','channel_group_output','image_token_first'}
assert set(old_nodes)-set(new_nodes)==removed_ids
assert set(new_nodes)-set(old_nodes)==added_ids
assert {k for k in new_nodes if k in old_nodes and new_nodes[k]!=old_nodes[k]}==changed
assert [n for n in after['layer_body'] if n['id'] not in changed|added_ids]==[
    n for n in before['layer_body'] if n['id'] not in changed|removed_ids]
assert after['compile_regions'][:-1]==before['compile_regions']
assert after['layer_output']==before['layer_output']
assert len(after['layer_body'])==108
assert all(new_nodes[x].get('lifetime','stage')=='stage' for x in members)
assert new_nodes['image_attention_out']==old_nodes['image_attention_out']
assert new_nodes['image_attended_rows']==old_nodes['image_attended_rows']
Path('program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'removed_nodes':sorted(removed_ids),'added_nodes':sorted(added_ids),
    'changed_original_nodes':sorted(changed),'all_other_node_objects_relative_order_equal':True,
    'old_three_regions_equal':True,'new_stage_region_nodes':members,
    'original_projection_weight_and_reduction_order_unchanged':True,
    'original_scores_mask_softmax_cast_and_probability_values_unchanged':True,
    'unrepeated_four_KV_heads_each_share_seven_query_heads':True,
    'full556_term_value_contraction_then_original3584_term_projection':True,
    'no_new_prefix_values_or_changed_cache_lifetimes':True,
    'no_emitted_kernel_latency_peak_or_numerical_certification':True
},indent=2)+'\n')
print('Constructed experiment98;108nodes;original canonical output projection')
