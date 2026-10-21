"""Internalize the original value concatenation and KV-repetition copies.

Move only pure stage views/copies to their earliest legal producer. Compile
the sole-consumer chain without the score, softmax or value contraction.
"""
import ast
import hashlib
import json
import runpy
from pathlib import Path

folder = Path(__file__).resolve().parent
base = Path('experiments/53_compiled_image_rotary/method.py')
source = base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 75: compile original value concatenation and materialization.')
lines = [
    "    value_shape=add('value_shape','reshape',[projections[2]],shape=[2,556,4,128])\n",
    "    value_heads=add('value_heads','transpose',[value_shape],dim0=1,dim1=2)\n",
    "    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)\n",
    "    value_transpose=add('value_transpose','transpose',[value_repeat],dim0=2,dim1=3)\n",
]
for line in lines:
    assert source.count(line) == 1
    source = source.replace(line, '')
anchor = "            projections.append(add(kind,'concat',[text_rows,image_rows],dim=1))\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor +
    "            # This original sole-consumer copy chain can execute now.\n" +
    ''.join('        ' + line for line in lines))
anchor = "            'compile_regions':[\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + """                {'id':'value_materialization_compiled','nodes':[
                    'image_v','v','value_shape','value_heads',
                    'value_repeat','value_transpose']},
""")
Path('method.py').write_text(source)
(folder / 'method.py').write_text(source)
original = runpy.run_path(str(base))['build_program']()
candidate = runpy.run_path('method.py')['build_program']()
assert {n['id']:n for n in original['layer_body']} == {n['id']:n for n in candidate['layer_body']}
scope = {}
tree = ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.Assign)
     or isinstance(n, ast.FunctionDef) and n.name == 'validate_regions'],type_ignores=[]),'public_region_validator','exec'),scope)
region = next(r for r in scope['validate_regions'](candidate) if r['id']=='value_materialization_compiled')
assert region['inputs'] == ('image_qkv','text_v_rows')
assert region['outputs'] == ('value_transpose',)
consumers = {name:[n['id'] for n in candidate['layer_body'] if name in n['inputs']]
             for name in ['image_v','v','value_shape','value_heads','value_repeat','value_transpose']}
assert all(len(v)==1 for v in consumers.values())
(folder / 'equivalence-receipt.json').write_text(json.dumps({
    'base':str(base),
    'candidate_canonical_sha256':hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'all_original_nodes_identical_by_id':True,
    'reorder':'Original four value view/copy nodes move just after v, before independent text attention and image rotary. No dependency or cache lifetime changes.',
    'region_inputs':region['inputs'],'region_outputs':region['outputs'],
    'sole_consumer_chain':consumers,
    'argument':'Only exact slice/concat/reshape/transpose/head repetition in region. Original repeat-before-transpose order and all arithmetic remain. Full unrepeated v is internal; fixed compiler may pack directly from fused QKV inputs.',
    'layout_limit':'No candidate stride contract exists. Retaining original operator order does not prove selected compiled output strides or generated copy elimination. Actual Dev fidelity/latency required.',
    'distinct_from':'57 starts after value_repeat; 60 compiles rank4 key materialization; 74 changes transpose/repeat order and is not the base.'
},indent=2)+'\n')
