"""Joint eager text Q/K rotary; retain original contiguous query geometry."""
import json
from pathlib import Path
import runpy

directory = Path('experiments/88_joint_eager_text_qk_rotary')
source = Path('experiments/87_shared_cached_text_key_attention/method.py').read_text()
source = source.replace('Experiment 87: eager cached text attention shares existing rotated keys.',
                        'Experiment 88: jointly rotate cached text Q/K with a contiguous query output.')
start = source.index("    text_key_shape=add('text_key_shape'")
stop = source.index("    text_cached_key_repeat=add('text_cached_key_repeat'", start)
source = source[:start] + '''    # Joint Q/K columns are adjacent in the existing fused projection.
    # Eager rotary is head-local and preserves the original BF16 operations.
    text_qk_rows=add('text_qk_rows','slice',[text_qkv],dim=2,start=0,stop=4096)
    text_qk_shape=add('text_qk_shape','reshape',[text_qk_rows],shape=[2,300,32,128])
    text_qk_heads=add('text_qk_heads','transpose',[text_qk_shape],dim0=1,dim1=2)
    text_qk_rotary=add('text_qk_rotary','rotary',[text_qk_heads,text_cos,text_sin])
    text_key=add('text_key_rotary','slice',[text_qk_rotary],dim=1,start=28,stop=32)
    text_query_view=add('text_query_rotated_view','slice',[text_qk_rotary],dim=1,start=0,stop=28)
    # The joint slice has a32-head batch stride; a counted real copy retains
    # the separate contiguous28-head score-input geometry of valid87.
    text_query_rotated=add('text_query_rotated','concat',[text_query_view],dim=1)
''' + source[stop:]
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before = json.loads(Path('experiments/87_shared_cached_text_key_attention/program.json').read_text())
after = runpy.run_path('method.py')['build_program']()
old = {n['id']:n for n in before['layer_body']}
new = {n['id']:n for n in after['layer_body']}
changed = {name for name in old.keys() & new.keys() if old[name] != new[name]}
assert changed == {'text_key_rotary','text_query_rotated'}
removed = old.keys() - new.keys()
assert removed == {'text_key_shape','text_key_heads','text_key_cos','text_key_sin',
                   'text_key_rotary_first','text_key_rotary_second','text_key_rotary_negative',
                   'text_key_rotary_half','text_key_rotary_direct','text_key_rotary_rotated',
                   'text_query_shape','text_query_heads'}
assert new.keys() - old.keys() == {'text_qk_rows','text_qk_shape','text_qk_heads','text_qk_rotary','text_query_rotated_view'}
assert before['compile_regions'] == after['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'87_shared_cached_text_key_attention', 'changed_existing_nodes':sorted(changed),
    'removed_obsolete_nodes':sorted(removed), 'new_nodes':sorted(new.keys()-old.keys()),
    'all_other_existing_nodes_and_image_regions_equal':True,
    'query_copy_preserves_standalone_contiguous_batch_stride':True,
    'rotary_remains_eager':True,
    'original_QKV_projection_score_and_softmax_geometry_equal':True
},indent=2)+'\n')
