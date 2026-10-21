"""Remove the old standalone query copy after eager native attention repacking."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/91_prerotated_query_without_copy')
source=Path('experiments/90_prerotated_opaque_text_attention/method.py').read_text()
source=source.replace('Experiment 90: eager opaque text attention on pre-rotated Q/K and identity frequencies.',
                      'Experiment 91: pre-rotated eager attention consumes the direct joint query slice.')
start=source.index("    text_query_view=add('text_query_rotated_view'")
stop=source.index('    # Pack the already-rotated heads',start)
source=source[:start]+'''    # Native attention receives rows after transpose/reshape. Its old direct
    # score-GEMM consumer is gone, so keep the joint query as a counted view.
    text_query_rotated=add('text_query_rotated','slice',[text_qk_rotary],dim=1,start=0,stop=28)
'''+source[stop:]
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/90_prerotated_opaque_text_attention/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old={n['id']:n for n in before['layer_body']}
new={n['id']:n for n in after['layer_body']}
changed={name for name in old.keys()&new.keys() if old[name]!=new[name]}
assert changed=={'text_query_rotated'}
assert old.keys()-new.keys()=={'text_query_rotated_view'}
assert not new.keys()-old.keys()
replacement={**old['text_query_rotated_view'],'id':'text_query_rotated'}
assert new['text_query_rotated']==replacement
assert before['compile_regions']==after['compile_regions']
users=[n['id'] for n in before['layer_body'] if 'text_query_rotated' in n['inputs']]
assert users==['text_prerotated_query_heads_to_rows']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'90_prerotated_opaque_text_attention',
    'changed_existing_nodes':sorted(changed),'removed_nodes':['text_query_rotated_view'],
    'all_other_existing_node_objects_and_regions_equal':True,
    'removed_copy_original_sole_consumer':users,
    'direct_query_slice_metadata_equals_original_view':True,
    'row_reshape_may_copy_or_alias_actual_layout_unmeasured':True,
    'no_GEMM_reduction_constant_attention_norm_or_precision_change':True
},indent=2)+'\n')
