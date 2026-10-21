"""Use eager native attention with already-rotated Q/K and identity frequencies."""
import hashlib
import json
from pathlib import Path
import runpy

directory=Path('experiments/90_prerotated_opaque_text_attention')
source=Path('experiments/88_joint_eager_text_qk_rotary/method.py').read_text()
source=source.replace('Experiment 88: jointly rotate cached text Q/K with a contiguous query output.',
                      'Experiment 90: eager opaque text attention on pre-rotated Q/K and identity frequencies.')
start=source.index("    text_cached_key_repeat=add('text_cached_key_repeat'")
stop=source.index("    text_out_rows=add('text_attention_out'",start)
source=source[:start]+'''    # Pack the already-rotated heads for the unchanged native attention API.
    # Its extra identity rotation is real first-build work; scores/probabilities
    # remain internal temporaries rather than retained graph intermediates.
    text_prerotated_query_heads_to_rows=add('text_prerotated_query_heads_to_rows','transpose',[
        text_query_rotated],dim0=1,dim1=2)
    text_prerotated_query_rows=add('text_prerotated_query_rows','reshape',[
        text_prerotated_query_heads_to_rows],shape=[2,300,3584])
    text_prerotated_key_heads_to_rows=add('text_prerotated_key_heads_to_rows','transpose',[
        text_key],dim0=1,dim1=2)
    text_prerotated_key_rows=add('text_prerotated_key_rows','reshape',[
        text_prerotated_key_heads_to_rows],shape=[2,300,512])
    # Singleton softmax of a finite frozen scalar is exact FP32 one.
    # Constants and all views are counted prefix values, reset normally.
    text_identity_unit_source=add('text_identity_unit_source','slice',[
        'w:vision_branch_input_layernorm.weight'],dim=0,start=0,stop=1)
    text_identity_unit_fp32=add('text_identity_unit_fp32','softmax',[text_identity_unit_source],dim=0)
    text_identity_unit=add('text_identity_unit','cast',[text_identity_unit_fp32],dtype='bf16')
    text_identity_sin=add('text_identity_sin','scale',[text_cos],factor=0.0)
    text_identity_cos=add('text_identity_cos','add',[text_identity_sin,text_identity_unit])
    text_attended=add('text_attended','attention_eager',[
        text_prerotated_query_rows,text_prerotated_key_rows,'text_v_rows',
        text_mask,text_identity_cos,text_identity_sin])
'''+source[stop:]
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
program=runpy.run_path('method.py')['build_program']()
prospective=json.loads(Path('research-diagnostics/prerotated-opaque-prospective-program.json').read_text())
assert program==prospective
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'88_joint_eager_text_qk_rotary',
    'implemented_graph_equals_source_derived_prospective':True,
    'canonical_sha256':hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'original_image_regions_and_all_unaffected_nodes_equal':True,
    'eager_opaque_attention_and_singleton_unit_softmax_not_compiled':True,
    'constants_views_and_copies_all_prefix_counted_and_reconstructed':True,
    'extra_identity_work_not_removed_from_timing':True,
    'finiteness_precision_layout_memory_require_actual_Dev':True
},indent=2)+'\n')
