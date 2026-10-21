"""Compile the original attention projection and its only residual consumer.

Keep the BF16 linear -> BF16 add node semantics; do not substitute addmm bias.
Move only dependency-independent cached text nodes before the stage pair.
"""
from pathlib import Path

source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 68: compile attention output projection with residual consumer.')
text_nodes = """    text_state=add('text_state_rows','slice',['state'],dim=1,start=0,stop=300)
    text_residual=add('text_residual_rows','add',[text_state,text_out_rows])
    text_post_norm=add('text_post_norm','rms_norm',[
        text_residual,'w:post_attention_layernorm.weight'],lifetime='prefix')
"""
assert source.count(text_nodes) == 1
source = source.replace(text_nodes, '')
anchor = "    image_out=add('image_attention_out','linear',[image_attended,'w:self_attn.vision_branch_o_proj.weight'])"
assert source.count(anchor) == 1
source = source.replace(anchor, text_nodes + anchor)
anchor = "            'compile_regions':[\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "                {'id':'image_projection_residual_compiled','nodes':['image_attention_out','image_residual_rows']},\n")
Path('method.py').write_text(source)
Path('experiments/68_compiled_projection_residual/method.py').write_text(source)
