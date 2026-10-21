"""Restrict cached packed image frequencies to their consumed token positions."""
from pathlib import Path
import json
import runpy

directory=Path('experiments/79_image_only_packed_frequency')
source=Path('experiments/70_atomic_compiled_text_key_rotary/method.py').read_text()
source=source.replace('Experiment 70: atomic compiled prefix text-key rotary composition.',
                      'Experiment 79: image-only packed frequency cache with atomic text-key rotary.')
source=source.replace('    def rotary_layout(source):', '    def rotary_layout(name,source):')
source=source.replace("part=add(source+'_axis_'", "part=add(name+'_axis_'")
source=source.replace("add(source+'_axis_shape_'", "add(name+'_axis_shape_'")
source=source.replace('shape=[2,1,556,128]', 'shape=[2,1,256,128]')
source=source.replace("add(source+'_section_'", "add(name+'_section_'")
source=source.replace("return add(source+'_layout','concat'", "return add('image_'+name,'concat'")
old="""    full_cos=rotary_layout('cos')
    full_sin=rotary_layout('sin')
    image_cos=add('image_cos','slice',[full_cos],dim=2,start=300,stop=556,lifetime='prefix')
    image_sin=add('image_sin','slice',[full_sin],dim=2,start=300,stop=556,lifetime='prefix')"""
new="""    # Token slicing commutes with the fixed axis/channel section assembly.
    # Packed image frequencies stay prefix values; changing Q/K arithmetic
    # and its original six-node compiler boundary are unchanged.
    raw_cos=add('image_raw_cos','slice',['cos'],dim=2,start=300,stop=556,lifetime='prefix')
    raw_sin=add('image_raw_sin','slice',['sin'],dim=2,start=300,stop=556,lifetime='prefix')
    image_cos=rotary_layout('cos',raw_cos)
    image_sin=rotary_layout('sin',raw_sin)"""
assert source.count(old)==1
source=source.replace(old,new)
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
old_program=json.loads(Path('experiments/70_atomic_compiled_text_key_rotary/program.json').read_text())
new_program=runpy.run_path('method.py')['build_program']()
assert old_program['compile_regions']==new_program['compile_regions']
old_nodes={n['id']:n for n in old_program['layer_body']}
new_nodes={n['id']:n for n in new_program['layer_body']}
changed={name for name in old_nodes.keys() & new_nodes.keys() if old_nodes[name]!=new_nodes[name]}
allowed={'image_cos','image_sin'} | {f'{kind}_axis_{axis}' for kind in ('cos','sin') for axis in range(3)} | {f'{kind}_axis_shape_{axis}' for kind in ('cos','sin') for axis in range(3)}
assert changed==allowed
assert old_nodes.keys()-new_nodes.keys()=={'cos_layout','sin_layout'}
assert new_nodes.keys()-old_nodes.keys()=={'image_raw_cos','image_raw_sin'}
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'70_atomic_compiled_text_key_rotary',
    'changed_existing_nodes':sorted(changed),
    'removed_nodes':sorted(old_nodes.keys()-new_nodes.keys()),
    'added_nodes':sorted(new_nodes.keys()-old_nodes.keys()),
    'all_other_nodes_equal':True,'all_compilation_regions_equal':True},indent=2)+'\n')
