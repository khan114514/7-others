"""Demand separately contiguous Q/K outputs at the compiled rotary boundary.

Single-input concat is the catalog copy composition; no arithmetic is changed.
The compiler may internalize the joint rotated tensor and emit split layouts.
"""
import hashlib
import json
import runpy
from pathlib import Path

folder = Path(__file__).resolve().parent
original = Path('experiments/53_compiled_image_rotary/method.py')
source = original.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 72: contiguous split Q/K outputs from compiled rotary.')
anchor = """    image_query=add('image_query','slice',[image_qk_rotary],dim=1,start=0,stop=28)
    image_key=add('image_key','slice',[image_qk_rotary],dim=1,start=28,stop=32)"""
replacement = """    # Single-input cat preserves values and demands a contiguous allocation.
    # Return these split layouts from the stage region; keep full Q/K internal.
    image_query_view=add('image_query_view','slice',[image_qk_rotary],dim=1,start=0,stop=28)
    image_query=add('image_query','concat',[image_query_view],dim=0)
    image_key_view=add('image_key_view','slice',[image_qk_rotary],dim=1,start=28,stop=32)
    image_key=add('image_key','concat',[image_key_view],dim=0)"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = "                    'image_qk_rotary_rotated','image_qk_rotary']},"
replacement = """                    'image_qk_rotary_rotated','image_qk_rotary',
                    'image_query_view','image_query','image_key_view','image_key']},"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
Path('method.py').write_text(source)
(folder / 'method.py').write_text(source)
baseline = runpy.run_path(str(original))['build_program']()
candidate = runpy.run_path('method.py')['build_program']()
old = {node['id']: node for node in baseline['layer_body']}
new = {node['id']: node for node in candidate['layer_body']}
changed = [key for key in old if old[key] != new[key]]
assert changed == ['image_query', 'image_key']
assert [key for key in new if key not in old] == ['image_query_view', 'image_key_view']
for key in ['image_query', 'image_key']:
    assert new[key + '_view'] == dict(old[key], id=key + '_view')
    assert new[key] == {'id': key, 'op': 'concat', 'inputs': [key + '_view'], 'dim': 0}
(folder / 'equivalence-receipt.json').write_text(json.dumps({
    'base': str(original),
    'base_canonical_sha256': hashlib.sha256(json.dumps(baseline,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'candidate_canonical_sha256': hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'changed_existing_nodes': changed,
    'unchanged_other_nodes': len(old) - len(changed),
    'argument': 'Original Q/K slices copied with single-input concat. Tensor values and all arithmetic nodes are unchanged; output stride and compiler boundary change. Numerical Dev qualification remains required.',
    'hypothesis_scope': 'Historical eager 41 score-copy observation plus source-derived batch gap; selected compiled method has no supplied kernel profile.'
}, indent=2) + '\n')
