"""Compile only four unchanged frozen QKV packing nodes from actual selected53."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

directory = Path('experiments/86_compiled_immutable_qkv_packing')
source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace('Experiment 53: compile pointwise image rotary alongside image MLP.',
                        'Experiment 86: compile immutable QKV packing with selected image regions.')
text_start = source.index("    text_qkv_weight=add('text_qkv_weight'")
text_stop = source.index("    text_flat=add('text_input_flat'", text_start)
text_pack = source[text_start:text_stop]
source = source[:text_start] + source[text_stop:]
vision_start = source.index("    qkv_weight=add('vision_qkv_weight'")
vision_stop = source.index("    image_qkv=add('image_qkv'", vision_start)
vision_pack = source[vision_start:vision_stop]
source = source[:vision_start] + source[vision_stop:]
insertion = source.index('    # RMS reduces only the hidden dimension;')
source = source[:insertion] + '''    # Frozen packing is independent of all state. Every output remains a
    # separately retained prefix result; resets still require reconstruction.
''' + text_pack + vision_pack + source[insertion:]
marker = "            'compile_regions':[\n"
assert source.count(marker) == 1
source = source.replace(marker, marker + '''                {'id':'immutable_qkv_packing_compiled','nodes':[
                    'text_qkv_weight','text_qkv_bias',
                    'vision_qkv_weight','vision_qkv_bias']},
''')
Path('method.py').write_text(source)
(directory / 'method.py').write_text(source)
before = json.loads(Path('experiments/53_compiled_image_rotary/program.json').read_text())
after = runpy.run_path('method.py')['build_program']()
old_nodes = {n['id']: n for n in before['layer_body']}
new_nodes = {n['id']: n for n in after['layer_body']}
assert old_nodes == new_nodes
members = after['compile_regions'][0]['nodes']
assert [n['id'] for n in after['layer_body'][:4]] == members
assert after['compile_regions'][1:] == before['compile_regions']
assert after['layer_output'] == before['layer_output']
assert [n['id'] for n in before['layer_body'] if n['id'] not in members] == [n['id'] for n in after['layer_body'][4:]]
assert all(n['op'] == 'concat' and n['lifetime'] == 'prefix' and all(s.startswith('w:') for s in n['inputs']) for n in after['layer_body'][:4])

# Execute only the pure validation function extracted from trusted source;
# this locally verifies the actual compiled input/output/lifetime boundary.
compiler_path = Path('/opt/bifrost/public_eval/bounded_graph_compiler.py')
compiler_data = compiler_path.read_bytes()
tree = ast.parse(compiler_data)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_regions')
namespace = {'MAX_REGIONS': 16}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(compiler_path), 'exec'), namespace)
region = namespace['validate_regions'](after)[0]
assert region['outputs'] == tuple(members)
assert region['prefix'] is True and len(region['inputs']) == 12
(directory / 'construction-receipt.json').write_text(json.dumps({
    'base': '53_compiled_image_rotary',
    'all_original_node_objects_equal': True,
    'only_reordered_four_independent_weight_only_nodes': True,
    'all_other_node_order_and_original_regions_equal': True,
    'public_region_validator_source_sha256': hashlib.sha256(compiler_data).hexdigest(),
    'new_prefix_region_inputs': list(region['inputs']),
    'new_prefix_region_outputs': list(region['outputs']),
    'no_numerical_arithmetic_introduced': True,
    'cache_budget_and_reset_semantics_unchanged': True
}, indent=2) + '\n')
