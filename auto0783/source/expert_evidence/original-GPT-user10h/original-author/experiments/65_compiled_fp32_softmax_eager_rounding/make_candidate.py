"""Separate dtype materialization from fixed compiled FP32 softmax.

Use the actual selected own graph, with RMS eager. The earlier failed softmax
method 55 compiled the output BF16 cast as well; this changes both boundaries.
"""
from pathlib import Path

source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 65: FP32-only compiled softmax with eager dtype conversion.')
anchor = "    image_prob_fp32=add('image_prob_fp32','softmax',[image_masked],dim=-1)"
replacement = """    # Native softmax(dtype=FP32) converts BF16 logits to FP32 internally.
    # Make this conversion and the original output BF16 cast eager boundaries.
    image_logits_fp32=add('image_logits_fp32','cast',[image_masked],dtype='fp32')
    image_prob_fp32=add('image_prob_fp32','softmax',[image_logits_fp32],dim=-1)"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = """            'compile_regions':[
                {'id':'image_rotary_compiled'"""
replacement = """            'compile_regions':[
                {'id':'image_softmax_fp32_compiled','nodes':['image_prob_fp32']},
                {'id':'image_rotary_compiled'"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
Path('method.py').write_text(source)
Path('experiments/65_compiled_fp32_softmax_eager_rounding/method.py').write_text(source)
