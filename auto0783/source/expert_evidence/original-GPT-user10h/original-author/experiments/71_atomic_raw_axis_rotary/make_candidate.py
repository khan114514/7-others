"""Use atomic raw-axis rotary for both cached text keys and changing image Q/K.

This removes obsolete frequency-layout/sign helper nodes. The same published
multimodal channel sections and separately rounded BF16 products remain.
"""
from pathlib import Path

source = Path('experiments/70_atomic_compiled_text_key_rotary/method.py').read_text()
source = source.replace(
    'Experiment 70: atomic compiled prefix text-key rotary composition.',
    'Experiment 71: atomic raw-axis rotary with obsolete layout cache removal.')
start = source.index("    full_cos=rotary_layout('cos')")
end_marker = '    image_query=add('
stop = source.index(end_marker, start)
source = source[:start] + """    # Slice token positions before the catalog's internal static channel-axis
    # assembly. Unary negation stays inside the original rotary composition.
    image_raw_cos=add('image_raw_cos','slice',['cos'],dim=2,start=300,stop=556,lifetime='prefix')
    image_raw_sin=add('image_raw_sin','slice',['sin'],dim=2,start=300,stop=556,lifetime='prefix')
    image_qk_rotary=add('image_qk_rotary','rotary',[image_qk_heads,image_raw_cos,image_raw_sin])
""" + source[stop:]
# Neither explicit helper is needed after both call sites use catalog rotary.
start = source.index('    def rotary_layout(source):')
stop = source.index('    # RMS reduces only the hidden dimension;', start)
source = source[:start] + source[stop:]
anchor = """                {'id':'image_rotary_compiled','nodes':[
                    'image_qk_rotary_first','image_qk_rotary_second',
                    'image_qk_rotary_half','image_qk_rotary_direct',
                    'image_qk_rotary_rotated','image_qk_rotary']},"""
assert source.count(anchor) == 1
source = source.replace(anchor, "                {'id':'image_rotary_compiled','nodes':['image_qk_rotary']},")
Path('method.py').write_text(source)
Path('experiments/71_atomic_raw_axis_rotary/method.py').write_text(source)
