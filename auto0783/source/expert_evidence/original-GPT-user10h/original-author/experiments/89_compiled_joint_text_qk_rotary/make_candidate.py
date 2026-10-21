"""Add only a single prefix compiler region for joint text rotary on valid88."""
import hashlib
import json
from pathlib import Path
import runpy

directory=Path('experiments/89_compiled_joint_text_qk_rotary')
source=Path('experiments/88_joint_eager_text_qk_rotary/method.py').read_text()
source=source.replace('Experiment 88: jointly rotate cached text Q/K with a contiguous query output.',
                      'Experiment 89: compile joint cached text Q/K rotary with unchanged query copy.')
marker="            'compile_regions':[\n"
assert source.count(marker)==1
source=source.replace(marker,marker+"                {'id':'text_qk_rotary_compiled','nodes':['text_qk_rotary']},\n")
Path('method.py').write_text(source)
(directory/'method.py').write_text(source)
before=json.loads(Path('experiments/88_joint_eager_text_qk_rotary/program.json').read_text())
after=runpy.run_path('method.py')['build_program']()
assert before['layer_body']==after['layer_body']
assert before['layer_output']==after['layer_output']
assert before['compile_regions']==after['compile_regions'][1:]
argument=json.loads(Path('research-diagnostics/joint-text-rotary-compiler-argument.json').read_text())
# Region-list order is metadata order only; the public validator sorts by
# actual node start, so keep the builder's conventional declaration order.
comparison={**after,'compile_regions':after['compile_regions'][1:]+after['compile_regions'][:1]}
assert hashlib.sha256(json.dumps(comparison,sort_keys=True,separators=(',',':')).encode()).hexdigest()==argument['prospective_canonical_sha256']
(directory/'construction-receipt.json').write_text(json.dumps({
    'base':'88_joint_eager_text_qk_rotary',
    'whole_numerical_node_body_order_and_lifetimes_equal':True,
    'original_image_regions_equal':True,
    'sole_addition':after['compile_regions'][0],
    'uniform_prefix_single_output_boundary':True,
    'no_eager_norm_softmax_or_GEMM_compilation_introduced':True,
    'cache_budget_and_reset_semantics_unchanged':True,
    'region_declaration_order_compared_to_prospective':'Validated execution order is sorted by node start; prospective region appended, builder region declared first.'
},indent=2)+'\n')
