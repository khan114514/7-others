"""Use singleton batch/token identity frequencies in eager native attention."""
import json
from pathlib import Path
import runpy

directory=Path('experiments/92_compact_identity_frequencies')
source=Path('experiments/91_prerotated_query_without_copy/method.py').read_text()
source=source.replace('Experiment 91: pre-rotated eager attention consumes the direct joint query slice.',
                      'Experiment 92: compact identity frequencies broadcast over native text attention.')
old="    text_identity_sin=add('text_identity_sin','scale',[text_cos],factor=0.0)"
assert source.count(old)==1
source=source.replace(old,'''    # Every raw frequency is identity; only one batch/token is needed.
    # Slice token first to avoid counting a large intermediate batch view.
    text_identity_token=add('text_identity_token','slice',[text_cos],dim=2,start=0,stop=1)
    text_identity_template=add('text_identity_template','slice',[text_identity_token],dim=1,start=0,stop=1)
    text_identity_sin=add('text_identity_sin','scale',[text_identity_template],factor=0.0)''')
(directory/'method.py').write_text(source)
Path('method.py').write_text(source)
before=json.loads(Path('experiments/91_prerotated_query_without_copy/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys()&new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'text_identity_sin'}
assert not old_nodes.keys()-new_nodes.keys()
assert new_nodes.keys()-old_nodes.keys()=={'text_identity_token','text_identity_template'}
assert before['compile_regions']==after['compile_regions']
assert new_nodes['text_identity_sin']=={**old_nodes['text_identity_sin'],'inputs':['text_identity_template']}
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'91_prerotated_query_without_copy',
    'changed_existing_nodes':sorted(changed),
    'added_prefix_views':['text_identity_token','text_identity_template'],
    'raw_frequency_shape':[3,1,1,128],
    'rotary_assembled_frequency_shape':[1,1,1,128],
    'all_other_existing_nodes_and_regions_equal':True,
    'no_attention_reduction_mask_projection_norm_or_precision_change':True,
    'finite_values_required_and_signedzero_can_differ':True,
    'native_rotate_output_strides_and_latency_unmeasured':True
},indent=2)+'\n')
