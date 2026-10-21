"""Pair a grouped value-output basis with a matching frozen-weight view/copy.

Grouped reversed product naturally stores B,KV,D,H,T. Flatten KV,D,H as a
feature basis and permute output projection columns in the same order. This
avoids its KV,H,D,T regrouping copy but changes projection reduction order.
Mathematical equivalence is exact; actual BF16 numerical gates decide.
"""
import hashlib
import json
import runpy
from pathlib import Path

folder=Path(__file__).resolve().parent
base=Path('experiments/53_compiled_image_rotary/method.py')
source=base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 77: grouped value features with paired projection basis.')
anchor="    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)\n"
assert source.count(anchor)==1
source=source.replace(anchor,'')
anchor="    value_transpose=add('value_transpose','transpose',[value_repeat],dim0=2,dim1=3)\n    probability_transpose=add('probability_transpose','transpose',[image_prob],dim0=2,dim1=3)"
assert source.count(anchor)==1
source=source.replace(anchor,"""    value_transpose=add('value_transpose','transpose',[value_heads],dim0=2,dim1=3)
    probability_groups=add('probability_groups','reshape',[image_prob],shape=[2,4,1792,556])
    probability_transpose=add('probability_transpose','transpose',[probability_groups],dim0=2,dim1=3)""")
anchor="""    image_output_heads=add('image_output_heads','transpose',[channel_query_output],dim0=2,dim1=3)
    image_output_rows=add('image_output_rows','transpose',[image_output_heads],dim0=1,dim1=2)
    image_attended=add('image_attended_rows','reshape',[image_output_rows],shape=[2,256,3584])
    image_out=add('image_attention_out','linear',[image_attended,'w:self_attn.vision_branch_o_proj.weight'])"""
assert source.count(anchor)==1
source=source.replace(anchor,"""    # Grouped reversed product stores [B,KV,D,H*T]. Its feature basis KV,D,H
    # can flatten directly; original KV,H,D packing would need a copy.
    image_group_columns=add('image_group_columns','reshape',[channel_query_output],shape=[2,3584,256])
    image_attended=add('image_attended_rows','transpose',[image_group_columns],dim0=1,dim1=2)
    # Preserve each frozen column's coefficient by pairing the same bijection.
    output_weight_group_shape=add('output_weight_group_shape','reshape',
        ['w:self_attn.vision_branch_o_proj.weight'],shape=[3584,4,7,128],lifetime='prefix')
    output_weight_channel_first=add('output_weight_channel_first','transpose',
        [output_weight_group_shape],dim0=2,dim1=3,lifetime='prefix')
    output_weight_group_basis=add('output_weight_group_basis','reshape',
        [output_weight_channel_first],shape=[3584,3584],lifetime='prefix')
    image_out=add('image_attention_out','linear',[image_attended,output_weight_group_basis])""")
Path('method.py').write_text(source)
(folder/'method.py').write_text(source)
original=runpy.run_path(str(base))['build_program']()
candidate=runpy.run_path('method.py')['build_program']()
old={n['id']:n for n in original['layer_body']}
new={n['id']:n for n in candidate['layer_body']}
changed=sorted(k for k in old.keys() & new.keys() if old[k]!=new[k])
assert changed==['image_attended_rows','image_attention_out','probability_transpose','value_transpose']
assert original['compile_regions']==candidate['compile_regions']
# Verify activation packing address and an independently inverted weight-column
# bijection over all 3584 features. No checkpoint values or GPU assumptions.
permutation=[]
for column in range(3584):
    kv, inner=divmod(column,128*7)
    channel, subhead=divmod(inner,7)
    original_column=kv*(7*128)+subhead*128+channel
    permutation.append(original_column)
    old_kv,old_inner=divmod(original_column,7*128)
    old_head,old_channel=divmod(old_inner,128)
    assert (old_kv*128+old_channel)*7+old_head==column
    for batch in range(2):
        for token in (0,1,127,255):
            grouped_product_address=((batch*4+kv)*128+channel)*1792+subhead*256+token
            direct_columns_address=(batch*3584+column)*256+token
            assert grouped_product_address==direct_columns_address
assert sorted(permutation)==list(range(3584))
(folder/'equivalence-receipt.json').write_text(json.dumps({
    'base':str(base),
    'candidate_canonical_sha256':hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'changed_existing_nodes':changed,
    'removed_nodes':sorted(old.keys()-new.keys()),'added_nodes':sorted(new.keys()-old.keys()),
    'selected_rotary_and_MLP_regions_identical':True,
    'feature_bijection_checked':len(permutation),
    'paired_basis':'Original features (KV,H,D); new features (KV,D,H). W_new[j,(kv*128+d)*7+h]=W_old[j,(kv*7+h)*128+d]. Same coefficient paired with same activation.',
    'real_arithmetic_equivalence':'Sum over the identical 3584 products in a bijective order; group value product preserves original 556-key dot product.',
    'numerical_limit':'Projection reduction order changes; generic real-valued equality does not prove BF16 agreement. No precision or guard changes; actual Dev determines validity.',
    'layout_argument':'Grouped reversed product B,KV,D,H*T can view as B,3584,T with the new basis; original KV,H,D row packing is not an affine view.',
    'cache_argument':'Three 3584x3584 BF16 prefix graph nodes are conservatively counted, including two views. First weight permutation reconstruction remains timed.',
    'distinct_from':'44 restores original feature order before projection and must repack; 42/43 use original ungrouped features and no paired column permutation.'
},indent=2)+'\n')
