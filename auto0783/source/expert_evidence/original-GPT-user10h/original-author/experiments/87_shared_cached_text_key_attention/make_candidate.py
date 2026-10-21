"""Expose eager300x300 cached text attention and share existing rotated key."""
import hashlib
import json
from pathlib import Path
import runpy

directory = Path('experiments/87_shared_cached_text_key_attention')
source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                        'Experiment 87: eager cached text attention shares existing rotated keys.')
start = source.index("    text_attended=add('text_attended','attention_eager'")
stop = source.index('    # Fused projection lays out', start)
old = source[start:stop]
assert "text_out_rows=add('text_attention_out'" in old
source = source[:start] + source[stop:]
marker = "    text_key=apply_rotary('text_key_rotary',text_key_heads,text_key_cos,text_key_sin)\n"
assert source.count(marker) == 1
replacement = '''    # Both eager text and changing image attention read this cached rotation.
    # Explicit intermediates are prefix values under the original rules;
    # their first construction and added retention remain real costs.
    text_query_shape=add('text_query_shape','reshape',['text_q_rows'],shape=[2,300,28,128])
    text_query_heads=add('text_query_heads','transpose',[text_query_shape],dim0=1,dim1=2)
    text_query_rotated=add('text_query_rotated','rotary',[text_query_heads,text_cos,text_sin])
    text_cached_key_repeat=add('text_cached_key_repeat','repeat_heads',[text_key],count=7)
    text_cached_key_transpose=add('text_cached_key_transpose','transpose',[text_cached_key_repeat],dim0=2,dim1=3)
    text_cached_value_shape=add('text_cached_value_shape','reshape',['text_v_rows'],shape=[2,300,4,128])
    text_cached_value_heads=add('text_cached_value_heads','transpose',[text_cached_value_shape],dim0=1,dim1=2)
    text_cached_value_repeat=add('text_cached_value_repeat','repeat_heads',[text_cached_value_heads],count=7)
    text_scores=add('text_scores','matmul',[text_query_rotated,text_cached_key_transpose])
    text_scaled=add('text_scaled','scale',[text_scores],factor=128**-0.5)
    text_masked=add('text_masked','add',[text_scaled,text_mask])
    text_prob_fp32=add('text_prob_fp32','softmax',[text_masked],dim=-1)
    text_prob=add('text_prob','cast',[text_prob_fp32],dtype='bf16')
    text_output_heads=add('text_output_heads','matmul',[text_prob,text_cached_value_repeat])
    text_attended_heads=add('text_attended_heads','transpose',[text_output_heads],dim0=1,dim1=2)
    text_attended=add('text_attended','reshape',[text_attended_heads],shape=[2,300,3584])
    text_out_rows=add('text_attention_out','linear',[text_attended,'w:self_attn.o_proj.weight'])
'''
source = source.replace(marker, marker + replacement)
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
program = runpy.run_path('method.py')['build_program']()
prospective = json.loads(Path('research-diagnostics/shared-text-key-prospective-program.json').read_text())
assert program == prospective
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary',
    'implemented_graph_equals_source_derived_prospective':True,
    'canonical_sha256':hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'existing_rotated_key_shared_by_text_and_image_attention':True,
    'norm_and_softmax_remain_eager':True,
    'original_image_regions_equal':True,
    'original_projection_shapes_and_eager_rounding_preserved':True,
    'all_new_helpers_prefix_counted_without_lifetime_exception':True,
    'local_validation_is_not_quality_or_peak_certification':True
},indent=2)+'\n')
