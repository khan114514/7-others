"""Compile original score contraction with scale/mask sole consumers."""
import ast
import hashlib
import json
import runpy
from pathlib import Path

folder = Path(__file__).resolve().parent
base = Path('experiments/53_compiled_image_rotary/method.py')
source = base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 73: compiled original score matmul with scale/mask epilogue.')
mask = "    image_mask_rows=add('image_attention_mask','slice',['attention_mask'],dim=2,start=300,stop=556,lifetime='prefix')\n"
score = "    image_scores=add('image_scores','matmul',[image_query,key_transpose])\n"
assert source.count(mask) == source.count(score) == 1
source = source.replace(mask, '').replace(score, mask + score)
anchor = "                {'id':'image_mlp_compiled','nodes':['image_mlp']}]}"
replacement = """                # Preserve each BF16 operator boundary; do not compile softmax.
                {'id':'image_score_epilogue_compiled','nodes':[
                    'image_scores','image_scaled','image_masked']},
                {'id':'image_mlp_compiled','nodes':['image_mlp']}]}"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
Path('method.py').write_text(source)
(folder / 'method.py').write_text(source)
original = runpy.run_path(str(base))['build_program']()
candidate = runpy.run_path('method.py')['build_program']()
assert {n['id']:n for n in candidate['layer_body']} == {n['id']:n for n in original['layer_body']}
ctree = ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text())
scope = {}
exec(compile(ast.Module(body=[n for n in ctree.body if isinstance(n, ast.Assign) or isinstance(n, ast.FunctionDef) and n.name == 'validate_regions'], type_ignores=[]), 'public_region_validator', 'exec'), scope)
region = next(r for r in scope['validate_regions'](candidate) if r['id'] == 'image_score_epilogue_compiled')
assert region['outputs'] == ('image_masked',)
consumers = {key: [n['id'] for n in candidate['layer_body'] if key in n['inputs']]
             for key in ['image_scores','image_scaled','image_masked']}
assert consumers == {'image_scores':['image_scaled'], 'image_scaled':['image_masked'], 'image_masked':['image_prob_fp32']}
(folder / 'equivalence-receipt.json').write_text(json.dumps({
    'base': str(base),
    'candidate_canonical_sha256': hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'all_nodes_identical_by_id': True,
    'reorder': 'Independent prefix image_attention_mask moves before image_scores to make three stage nodes contiguous.',
    'new_region_inputs': region['inputs'], 'new_region_outputs': region['outputs'],
    'sole_consumer_chain': consumers,
    'precision_argument': 'No arithmetic replacement: original rounded matmul, rounded BF16 scale, rounded BF16 add. Fixed compiler precision emulation plus actual Dev gates; no bias/addmm folding.',
    'distinct_from': '54 compiled only scale/mask with score returned eager; 57 and68 use different K and consumers; 55/65 compile softmax and are excluded.'
}, indent=2) + '\n')
