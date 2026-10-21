"""Use direct frozen text projections over the selected600normalized rows."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/81_separate_cached_text_projections')
source=Path('experiments/53_compiled_image_rotary/method.py').read_text()
source=source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                      'Experiment 81: separate cached600-row text projections with selected image path.')
start=source.index('    # Fuse the cached biased text projections')
stop=source.index("    qkv_weight=add('vision_qkv_weight'",start)
source=source[:start]+"""    # Text projections are built once per prefix. Preserve original600rows,
    # but avoid retaining fused text weight/bias copies across all28layers.
    text_flat=add('text_input_flat','reshape',[text_norm_rows],shape=[600,3584])
    text_flat_projections={}
    for kind in ['q','k','v']:
        text_flat_projections[kind]=add('text_'+kind+'_flat','linear',[
            text_flat,'w:self_attn.'+kind+'_proj.weight',
            'w:self_attn.'+kind+'_proj.bias'])
"""+source[stop:]
old="        text_rows=add('text_'+kind+'_rows','slice',[text_qkv],dim=2,start=start,stop=stop)"
new="        text_rows=add('text_'+kind+'_rows','reshape',[text_flat_projections[kind]],shape=[2,300,stop-start])"
assert source.count(old)==1
source=source.replace(old,new)
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/53_compiled_image_rotary/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'text_q_rows','text_k_rows','text_v_rows'}
removed=old_nodes.keys()-new_nodes.keys()
added=new_nodes.keys()-old_nodes.keys()
assert removed=={'text_qkv_weight','text_qkv_bias','text_flat_qkv','text_qkv'}
assert added=={'text_q_flat','text_k_flat','text_v_flat'}
assert before['compile_regions']==after['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary','changed_existing_nodes':sorted(changed),
    'removed_nodes':sorted(removed),'added_nodes':sorted(added),
    'all_other_nodes_equal':True,'all_compile_regions_equal':True,
    'original600flat_text_input_equal':True},indent=2)+'\n')
