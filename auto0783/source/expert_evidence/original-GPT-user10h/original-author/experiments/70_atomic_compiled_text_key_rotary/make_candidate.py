"""Use the catalog rotary atom so only its final prefix key is exposed.

The original raw text cos/sin are already constructed for eager text attention.
Multimodal channel assembly commutes with the text token-position slice.
"""
from pathlib import Path

source = Path('experiments/53_compiled_image_rotary/method.py').read_text()
source = source.replace(
    'Experiment 53: compile pointwise image rotary alongside image MLP.',
    'Experiment 70: atomic compiled prefix text-key rotary composition.')
anchor = """    text_key_cos=add('text_key_cos','slice',[full_cos],dim=2,start=0,stop=300)
    text_key_sin=add('text_key_sin','slice',[full_sin],dim=2,start=0,stop=300)
    text_key=apply_rotary('text_key_rotary',text_key_heads,text_key_cos,text_key_sin)"""
replacement = """    # Published rotary assembles the same channel sections internally. The
    # text position slice commutes with channel assembly; cache only its result.
    text_key=add('text_key_rotary','rotary',[text_key_heads,text_cos,text_sin])"""
assert source.count(anchor) == 1
source = source.replace(anchor, replacement)
anchor = "            'compile_regions':[\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "                {'id':'text_key_rotary_compiled','nodes':['text_key_rotary']},\n")
Path('method.py').write_text(source)
Path('experiments/70_atomic_compiled_text_key_rotary/method.py').write_text(source)
