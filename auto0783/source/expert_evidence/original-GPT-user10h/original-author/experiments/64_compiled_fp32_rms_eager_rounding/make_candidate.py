"""RMS numerical-boundary diagnostic using own experiment 63 construction.

Explicit casts are the same FP32 accumulation/BF16 normalization boundaries
as published RMS, now separated by the legal compiled-region interface.
"""
from pathlib import Path

source = Path('experiments/63_compiled_unit_rms/method.py').read_text()
source = source.replace(
    'Experiment 63: isolate compiled unit RMS from eager BF16 weight multiplication.',
    'Experiment 64: FP32 compiled RMS with eager BF16 rounding and weighting.')
anchor = """    unit_bf16=add('unit_bf16','cast',[unit_fp32],dtype='bf16',lifetime='prefix')
    # RMS internally rounds normalized FP32 values to input BF16. Multiplying
    # by BF16 one is exact. Apply the original weight eagerly after the region.
    image_unit_norm=add('image_unit_norm','rms_norm',[image_state,unit_bf16])
    image_norm=add('image_input_norm','mul',['w:vision_branch_input_layernorm.weight',image_unit_norm])"""
replacement = """    # BF16 input -> FP32 normalization -> BF16 rounding remains unchanged.
    # Both conversions and final checkpoint-weight multiply execute eagerly.
    image_input_fp32=add('image_input_fp32','cast',[image_state],dtype='fp32')
    image_unit_norm=add('image_unit_norm','rms_norm',[image_input_fp32,unit_fp32])
    image_unit_bf16=add('image_unit_bf16','cast',[image_unit_norm],dtype='bf16')
    image_norm=add('image_input_norm','mul',['w:vision_branch_input_layernorm.weight',image_unit_bf16])"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = """    image_post_unit_norm=add('image_post_unit_norm','rms_norm',[image_residual,unit_bf16])
    image_post_norm=add('image_post_norm','mul',['w:vision_branch_post_attention_layernorm.weight',image_post_unit_norm])"""
replacement = """    image_post_fp32=add('image_post_fp32','cast',[image_residual],dtype='fp32')
    image_post_unit_norm=add('image_post_unit_norm','rms_norm',[image_post_fp32,unit_fp32])
    image_post_unit_bf16=add('image_post_unit_bf16','cast',[image_post_unit_norm],dtype='bf16')
    image_post_norm=add('image_post_norm','mul',['w:vision_branch_post_attention_layernorm.weight',image_post_unit_bf16])"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
Path('method.py').write_text(source)
Path('experiments/64_compiled_fp32_rms_eager_rounding/method.py').write_text(source)
