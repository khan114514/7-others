"""Use the existing mandatory KV repetition copy to pack channel-major V.

T23(repeat_heads(V,7)) == repeat_heads(T23(V),7) by exact index permutation.
Same logical left operand, same 556-key BF16 dot product, different strides.
"""
import hashlib
import json
import runpy
from pathlib import Path

folder = Path(__file__).resolve().parent
base = Path('experiments/53_compiled_image_rotary/method.py')
source = base.read_text().replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 74: transpose values before KV replication to change packing.')
old = "    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)"
new = """    # Repeat the exact same values after swapping token/channel axes. The
    # mandatory replication copy now packs the left GEMM operand by channel.
    value_head_channels=add('value_head_channels','transpose',[value_heads],dim0=2,dim1=3)
    value_transpose=add('value_transpose','repeat_heads',[value_head_channels],count=7)"""
assert source.count(old) == 1
source = source.replace(old, new)
old = "    value_transpose=add('value_transpose','transpose',[value_repeat],dim0=2,dim1=3)\n"
assert source.count(old) == 1
source = source.replace(old, '')
Path('method.py').write_text(source)
(folder / 'method.py').write_text(source)
original = runpy.run_path(str(base))['build_program']()
candidate = runpy.run_path('method.py')['build_program']()
old_nodes = {n['id']:n for n in original['layer_body']}
new_nodes = {n['id']:n for n in candidate['layer_body']}
assert set(old_nodes) - set(new_nodes) == {'value_repeat'}
assert set(new_nodes) - set(old_nodes) == {'value_head_channels'}
changed = [n for n in old_nodes.keys() & new_nodes.keys() if old_nodes[n] != new_nodes[n]]
assert changed == ['value_transpose']
assert original['compile_regions'] == candidate['compile_regions']
# Compose independently defined inverse index maps in opposite order over
# all 3,985,408 output elements. No checkpoint, random tensors or torch.
def transpose_source(index, first, second):
    output = list(index)
    output[first], output[second] = output[second], output[first]
    return tuple(output)

def repeat_source(index, count):
    output = list(index)
    output[1] //= count
    return tuple(output)

count = 0
for batch in range(2):
    for h in range(28):
        for token in range(556):
            for channel in range(128):
                output = (batch,h,channel,token)
                repeat_then_transpose = repeat_source(transpose_source(output,2,3),7)
                transpose_then_repeat = transpose_source(repeat_source(output,7),2,3)
                assert repeat_then_transpose == transpose_then_repeat
                assert repeat_then_transpose == (batch,h//7,token,channel)
                count += 1
(folder / 'equivalence-receipt.json').write_text(json.dumps({
    'base': str(base),
    'candidate_canonical_sha256': hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'changed_existing_nodes': changed, 'removed_nodes': ['value_repeat'],
    'added_nodes': ['value_head_channels'], 'compile_regions_identical': True,
    'coordinate_equalities_checked_both_batches': count,
    'index_argument': 'For output[b,h,d,t], both paths select original value_heads[b,floor(h/7),t,d]. Transpose and repetition are exact copies/views and commute.',
    'arithmetic_preservation': 'All score/mask/softmax/probability nodes and original value-left matmul/weights/norm/MLP/residual stay unchanged. No dot-product partition or extra rounding. GEMM memory layout can alter kernel dispatch, so actual Dev required.',
    'logical_left_operand_shape': [2,28,128,556],
    'expected_eager_left_operand_strides_old': [1992704,71168,1,128],
    'expected_eager_left_operand_strides_new': [1992704,71168,556,1],
    'source_evidence_scope': 'Original own eager41 value_repeat has 28 copy kernels; both contraction transposes have no kernels. Does not profile selected compiled53.'
}, indent=2)+'\n')
