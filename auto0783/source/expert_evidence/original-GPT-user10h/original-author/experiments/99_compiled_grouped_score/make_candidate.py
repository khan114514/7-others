"""Implement the previously audited grouped-score region on selected97."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/99_compiled_grouped_score')
base=Path('experiments/97_compiled_frequency_prefix')
before=json.loads((base/'program.json').read_text())
prospective=json.loads(Path('research-diagnostics/compiled-grouped-score-prospective-program.json').read_text())
source=(base/'method.py').read_text().replace(
    'Experiment 97: compile the complete original immutable frequency producer chain.',
    'Experiment 99: compile grouped score construction with eager epilogue unchanged.')
start=source.index("    key_rotary=add('key_rotary'")
stop=source.index("    value_shape=add('value_shape'",start)
source=source[:start]+'''    # Seven query heads share each full KV key; keep all128 reduction terms.
    key_rotary=add('key_rotary','concat',[text_key,image_key],dim=2)
    key_transpose=add('key_transpose','transpose',[key_rotary],dim0=2,dim1=3)
    query_groups=add('query_groups','reshape',[image_query],shape=[2,4,1792,128])
    grouped_scores=add('grouped_scores','matmul',[query_groups,key_transpose])
    image_scores=add('image_scores','reshape',[grouped_scores],shape=[2,28,256,556])
'''+source[stop:]
removed="    image_scores=add('image_scores','matmul',[image_query,key_transpose])\n"
assert source.count(removed)==1
source=source.replace(removed,'')
old=repr(before['compile_regions'][-1])+']}'
assert source.count(old)==1
source=source.replace(old,repr(before['compile_regions'][-1])+',\n                '+repr(prospective['compile_regions'][-1])+']}')
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
after=runpy.run_path('method.py')['build_program']()
assert after==prospective
assert len(after['layer_body'])==108
Path('program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'program.json').write_text(json.dumps(after,indent=2)+'\n')
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':base.name,'equals_preceding_interval_audited_prospective_program':True,
    'sole_removed_node':'key_repeat','added_nodes':['query_groups','grouped_scores'],
    'changed_original_nodes':['key_transpose','image_scores'],
    'five_stage_region_nodes':prospective['compile_regions'][-1]['nodes'],
    'all_other_original_node_objects_relative_order_lifetimes_equal':True,
    'all_three_original_compiled_regions_equal':True,
    'canonical_score_BF16_boundary_eager_scale_mask_softmax_and_value_path_equal':True,
    'full128_feature_contraction_full556keys_and_canonical_query_order_preserved':True,
    'no_prefix_or_cache_budget_changes':True,
    'compiled_kernel_numerics_strides_materialization_latency_and_peak_unmeasured':True
},indent=2)+'\n')
print('Implemented exact prospective grouped-score graph;108nodes;fourregions')
