from pathlib import Path
import runpy,json,hashlib
base=Path('experiments/113_compiled_post_softmax_partition/method.py').read_text()
s=base.replace('Experiment 113: eager softmax followed by compiled BF16 cast/value preparation and contraction.','Experiment 115: complete post-softmax cast/value/output projection/residual execution partition.')
needle="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
add="""    # Complete post-softmax attention partition: hide canonical attended
    # rows behind their original outputprojection/residual consumer. No
    # normalization or probabilityreduction enters this region; all original
    # BF16 operator/cast boundaries and residual operand order remain intact.
    post_members=post_members+['image_attention_out','image_residual_rows']
    by_complete={n['id']:n for n in nodes if n['id'] in post_members}
    nodes=[n for n in nodes if n['id'] not in by_complete]
    at=next(i for i,n in enumerate(nodes) if n['id']=='text_state_rows')
    nodes[at:at]=[by_complete[name] for name in post_members]
"""
assert needle in s;s=s.replace(needle,add+needle)
s=s.replace("'id':'image_post_softmax_compiled'","'id':'image_post_softmax_output_compiled'")
folder=Path('experiments/115_compiled_complete_post_softmax');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
assert p==json.loads(Path('research-resume-current-gpu/prospective115-program.json').read_text())
(folder/'argument.json').write_bytes(Path('research-resume-current-gpu/post-softmax-output-boundary-argument.json').read_bytes())
h=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for q in Path('experiments').glob('*/program.json'):
    if q.parent!=folder:assert h!=hashlib.sha256(json.dumps(json.loads(q.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
print({'new_canonical':h,'matches_independently_validated_prospective':True})
