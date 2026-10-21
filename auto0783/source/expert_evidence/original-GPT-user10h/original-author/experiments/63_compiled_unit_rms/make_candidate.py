"""Generate a reproducible explicit builder from the selected own experiment.

The only new numerical decomposition is unit-weight RMS followed by the
original eager BF16 checkpoint-weight multiply. The RMS regions cannot fuse
that external multiply across their observable BF16 output boundary.
"""
from pathlib import Path

source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 63: isolate compiled unit RMS from eager BF16 weight multiplication.')
anchor = "    image_norm=add('image_input_norm','rms_norm',[image_state,'w:vision_branch_input_layernorm.weight'])"
replacement = """    # One-element softmax of a finite frozen weight is exactly FP32 one.
    # Its BF16 cast remains exactly one; numerical values reset per prompt.
    unit_source=add('unit_source','slice',['w:vision_branch_input_layernorm.weight'],dim=0,start=0,stop=1,lifetime='prefix')
    unit_fp32=add('unit_fp32','softmax',[unit_source],dim=0,lifetime='prefix')
    unit_bf16=add('unit_bf16','cast',[unit_fp32],dtype='bf16',lifetime='prefix')
    # RMS internally rounds normalized FP32 values to input BF16. Multiplying
    # by BF16 one is exact. Apply the original weight eagerly after the region.
    image_unit_norm=add('image_unit_norm','rms_norm',[image_state,unit_bf16])
    image_norm=add('image_input_norm','mul',['w:vision_branch_input_layernorm.weight',image_unit_norm])"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = """    image_post_norm=add('image_post_norm','rms_norm',[
        image_residual,'w:vision_branch_post_attention_layernorm.weight'])"""
replacement = """    image_post_unit_norm=add('image_post_unit_norm','rms_norm',[image_residual,unit_bf16])
    image_post_norm=add('image_post_norm','mul',['w:vision_branch_post_attention_layernorm.weight',image_post_unit_norm])"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = """            'compile_regions':[
                {'id':'image_rotary_compiled'"""
replacement = """            'compile_regions':[
                {'id':'image_unit_rms_compiled','nodes':['image_unit_norm']},
                {'id':'image_post_unit_rms_compiled','nodes':['image_post_unit_norm']},
                {'id':'image_rotary_compiled'"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
Path('method.py').write_text(source)
Path('experiments/63_compiled_unit_rms/method.py').write_text(source)
