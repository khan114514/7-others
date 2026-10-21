"""Use eager published-shape cached text RMS directly in full-shaped projections."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/85_full_state_cached_text_rms')
source=Path('experiments/83_separate_published_shape_text_projections/method.py').read_text()
source=source.replace('Experiment 83: separate cached full-shaped text projections with selected image path.',
                      'Experiment 85: full-state eager cached text RMS feeding full-shaped projections.')
start=source.index('    # RMS reduces only the hidden dimension;')
stop=source.index("    image_state=add('image_input_rows'",start)
source=source[:start]+"""    # Original eager full-shaped text RMS. Cached auxiliary image rows are
    # discarded by every full projection's text slice before any attention.
    text_norm=add('text_input_norm','rms_norm',['state','w:input_layernorm.weight'])
"""+source[stop:]
old="""    text_projection_input=add('text_projection_input','concat',[
        text_norm_rows,image_norm],dim=1)
"""
assert source.count(old)==1
source=source.replace(old,'')
source=source.replace("            text_projection_input,'w:self_attn.'+kind+'_proj.weight',",
                      "            text_norm,'w:self_attn.'+kind+'_proj.weight',")
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/83_separate_published_shape_text_projections/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
changed={name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
assert changed=={'text_input_norm','text_q_full','text_k_full','text_v_full'}
removed=old_nodes.keys()-new_nodes.keys()
assert removed=={'input_text_rows','text_input_norm_rows','text_projection_input'}
assert not new_nodes.keys()-old_nodes.keys()
assert before['compile_regions']==after['compile_regions']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'83_separate_published_shape_text_projections',
    'changed_existing_nodes':sorted(changed),'removed_nodes':sorted(removed),
    'all_other_existing_nodes_and_regions_equal':True,
    'norm_compilation_introduced':False,
    'original_full_projection_geometry_unchanged':True},indent=2)+'\n')
