"""Isolate experiment77's paired projection basis on unchanged value GEMM.

Keep selected53's exact 56-batch value product. Transform only its already
computed output into KV,D,H features and pair original weight columns.
This diagnostic adds materialization and is not a promised speed method.
"""
import hashlib
import json
import runpy
from pathlib import Path

folder=Path(__file__).resolve().parent
base=Path('experiments/53_compiled_image_rotary/method.py')
source=base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 78: isolate paired projection feature basis, unchanged values.')
anchor="""    image_output_heads=add('image_output_heads','transpose',[channel_query_output],dim0=2,dim1=3)
    image_output_rows=add('image_output_rows','transpose',[image_output_heads],dim0=1,dim1=2)
    image_attended=add('image_attended_rows','reshape',[image_output_rows],shape=[2,256,3584])
    image_out=add('image_attention_out','linear',[image_attended,'w:self_attn.vision_branch_o_proj.weight'])"""
assert source.count(anchor)==1
source=source.replace(anchor,"""    # Repack the original completed product from KV,H,D to KV,D,H; no
    # probability/value contraction, arithmetic or grouping change upstream.
    image_head_groups=add('image_head_groups','reshape',[channel_query_output],shape=[2,4,7,128,256])
    image_channel_groups=add('image_channel_groups','transpose',[image_head_groups],dim0=2,dim1=3)
    image_group_columns=add('image_group_columns','reshape',[image_channel_groups],shape=[2,3584,256])
    image_attended=add('image_attended_rows','transpose',[image_group_columns],dim0=1,dim1=2)
    output_weight_group_shape=add('output_weight_group_shape','reshape',
        ['w:self_attn.vision_branch_o_proj.weight'],shape=[3584,4,7,128],lifetime='prefix')
    output_weight_channel_first=add('output_weight_channel_first','transpose',
        [output_weight_group_shape],dim0=2,dim1=3,lifetime='prefix')
    output_weight_group_basis=add('output_weight_group_basis','reshape',
        [output_weight_channel_first],shape=[3584,3584],lifetime='prefix')
    image_out=add('image_attention_out','linear',[image_attended,output_weight_group_basis])""")
Path('method.py').write_text(source)
(folder/'method.py').write_text(source)
old_program=runpy.run_path(str(base))['build_program']()
new_program=runpy.run_path('method.py')['build_program']()
grouped=runpy.run_path('experiments/77_grouped_value_feature_basis/method.py')['build_program']()
old={n['id']:n for n in old_program['layer_body']};new={n['id']:n for n in new_program['layer_body']};prior={n['id']:n for n in grouped['layer_body']}
changed=sorted(k for k in old.keys() & new.keys() if old[k]!=new[k])
assert changed==['image_attended_rows','image_attention_out']
assert old_program['compile_regions']==new_program['compile_regions']
end=next(i for i,n in enumerate(old_program['layer_body']) if n['id']=='channel_query_output')+1
assert old_program['layer_body'][:end]==new_program['layer_body'][:end]
weight_nodes=['output_weight_group_shape','output_weight_channel_first','output_weight_group_basis']
assert all(new[name]==prior[name] for name in weight_nodes)
(folder/'equivalence-receipt.json').write_text(json.dumps({
    'base':str(base),
    'candidate_canonical_sha256':hashlib.sha256(json.dumps(new_program,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'changed_existing_nodes':changed,'added_nodes':sorted(new.keys()-old.keys()),'removed_nodes':sorted(old.keys()-new.keys()),
    'entire_graph_through_value_product_identical':True,
    'selected_rotary_and_MLP_regions_identical':True,
    'paired_weight_permutation_nodes_identical_to77':weight_nodes,
    'argument':'Pure output feature copy KV,H,D -> KV,D,H, paired with identical frozen weight-column copy. Same projection products in changed order; no guarantee for BF16 reduction.',
    'diagnostic_goal':'Test whether paired projection basis fails on unchanged selected53 value GEMM, removing grouped-GEMM interaction. Added stage copy means no expected direct speed gain.'
},indent=2)+'\n')
