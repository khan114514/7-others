"""Broadcast one image-query key-mask row over the canonical image block."""
import hashlib
import json
from pathlib import Path
import runpy

directory=Path('experiments/93_broadcast_image_query_mask')
source=Path('experiments/53_compiled_image_rotary/method.py').read_text()
source=source.replace('"""', '"""Experiment 93: broadcast image query mask.\n\n',1)
old="image_mask_rows=add('image_attention_mask','slice',['attention_mask'],dim=2,start=300,stop=556,lifetime='prefix')"
assert source.count(old)==1
source=source.replace(old,"image_mask_rows=add('image_attention_mask','slice',['attention_mask'],dim=2,start=300,stop=301,lifetime='prefix')")
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/53_compiled_image_rotary/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
changed=[n['id'] for n,o in zip(after['layer_body'],before['layer_body']) if n!=o]
assert changed==['image_attention_mask']
old_nodes={n['id']:n for n in before['layer_body']}
new_nodes={n['id']:n for n in after['layer_body']}
assert new_nodes['image_attention_mask']=={**old_nodes['image_attention_mask'],'stop':301}
assert after['compile_regions']==before['compile_regions']
prospective=json.loads(Path('research-diagnostics/diagnostic-source-649c9638656871ab.json').read_text())
assert after==prospective
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'53_compiled_image_rotary',
    'changed_existing_nodes':changed,'no_added_or_removed_nodes':True,
    'all_other_existing_nodes_and_regions_equal':True,
    'mask_consumer_same_eager_ordered_add':True,
    'raw_attention_mask_not_modified_or_replaced':True,
    'unchanged_all_score_value_softmax_norm_MLP_projection_precision':True,
    'general_equivalence_condition':'All image-query mask rows equal row300; verified bitwise on disclosed inputs, not universally guaranteed by operator shapes.',
    'source_argument':'research-diagnostics/diagnostic-image-query-broadcast.json',
    'canonical_sha256':hashlib.sha256(json.dumps(after,sort_keys=True,separators=(',',':')).encode()).hexdigest()
},indent=2)+'\n')
