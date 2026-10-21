"""Compile only image residual add and full-layer concat; keep MLP separate.

Original experiment56 includes MLP in this region. Here its exact selected
executable/output boundary stays intact; no GEMM or reduction in new region.
"""
import ast
import hashlib
import json
import runpy
from pathlib import Path

folder=Path(__file__).resolve().parent
base=Path('experiments/53_compiled_image_rotary/method.py')
source=base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 76: separate final image residual/concatenation region.')
anchor="                {'id':'image_mlp_compiled','nodes':['image_mlp']}]}"
assert source.count(anchor)==1
source=source.replace(anchor,"""                {'id':'image_mlp_compiled','nodes':['image_mlp']},
                {'id':'layer_tail_compiled','nodes':['image_output','output']}]}""")
Path('method.py').write_text(source)
(folder/'method.py').write_text(source)
original=runpy.run_path(str(base))['build_program']()
candidate=runpy.run_path('method.py')['build_program']()
assert original['layer_body']==candidate['layer_body']
assert original['compile_regions']==candidate['compile_regions'][:-1]
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text())
scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_region_validator','exec'),scope)
region=scope['validate_regions'](candidate)[-1]
assert region['id']=='layer_tail_compiled'
assert region['outputs']==('output',)
assert region['inputs']==('image_residual_rows','image_mlp','text_output')
assert [n['id'] for n in candidate['layer_body'] if 'image_output' in n['inputs']]==['output']
(folder/'equivalence-receipt.json').write_text(json.dumps({
    'base':str(base),
    'candidate_canonical_sha256':hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'all_graph_nodes_and_order_identical':True,
    'original_compiled_regions_identical':True,
    'region_inputs':region['inputs'],'region_outputs':region['outputs'],
    'argument':'Only original BF16 add and pure concat in added contiguous stage region. Rounded image sum copied into original layer output. No folding into GEMM/down projection or reduction.',
    'hypothesis':'Internal image_output buffer has sole concat consumer; direct output production may reduce storage and launches. No generated fusion or timing claim.',
    'distinct_from':'56 combines opaque MLP and both tail nodes into one region, eliminating original MLP boundary; this leaves selected53 MLP region and output untouched.'
},indent=2)+'\n')
