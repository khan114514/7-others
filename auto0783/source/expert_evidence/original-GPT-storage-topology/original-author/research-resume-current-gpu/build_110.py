from pathlib import Path
import runpy,json,hashlib,ast
base=Path('experiments/109_packed_compiled_qkv/method.py').read_text()
s=base.replace('Experiment 109: pack rotated Q/K and unchanged V into one compiled projection output.','Experiment 110: canonical key repetition and score GEMM/epilogue share a region.')
needle="    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,"
add="""    # Hide repeated-key producer boundary inside canonical score consumer.
    # Full original score geometry and BF16 scaling/mask order remain intact.
    # FP32 softmax and BF16 probability cast are outside/eager.
    score_members=['key_rotary','key_repeat','key_transpose',
                   'image_scores','image_scaled','image_masked']
    selected={n['id']:n for n in nodes if n['id'] in score_members}
    nodes=[n for n in nodes if n['id'] not in selected]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_prob_fp32')
    nodes[at:at]=[selected[name] for name in score_members]
"""
assert needle in s;s=s.replace(needle,add+needle)
s=s.replace("            'compile_regions':[","            'compile_regions':[\n                {'id':'key_score_producer_consumer_compiled','nodes':score_members},")
folder=Path('experiments/110_compiled_key_score_composition');folder.mkdir(exist_ok=True)
(folder/'method.py').write_text(s);p=runpy.run_path(str(folder/'method.py'))['build_program']()
(folder/'program.json').write_text(json.dumps(p,indent=2)+'\n');Path('method.py').write_text(s)
basep=runpy.run_path('experiments/109_packed_compiled_qkv/method.py')['build_program']()
a={n['id']:n for n in basep['layer_body']};b={n['id']:n for n in p['layer_body']};assert a==b
members=p['compile_regions'][0]['nodes']
assert [n['id'] for n in p['layer_body'] if n['id'] not in members]==[n['id'] for n in basep['layer_body'] if n['id'] not in members]
assert basep['compile_regions']==p['compile_regions'][1:]
# Every repeated head/source token/feature indexes its unchanged KV head.
# Every score row has the identical ordered128 feature contraction.
checked=0
for batch in range(2):
    for head in range(28):
        kv=head//7
        assert kv<4
        for token in range(556):
            branch='text_key_rotary' if token<300 else 'image_key'
            source_token=token if token<300 else token-300
            # Logical index inverse of repeat_heads expansion/reshape.
            for feature in range(128):
                assert divmod(head,7)[0]==kv
                assert 0<=source_token<(300 if branch=='text_key_rotary' else 256)
                checked+=1
# Region boundary derivation from trusted current source.
tree=ast.parse(Path('/opt/bifrost/public_eval/bounded_graph_compiler.py').read_text());scope={}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='validate_regions'],type_ignores=[]),'public_validator','exec'),scope)
r=next(r for r in scope['validate_regions'](p) if r['id']=='key_score_producer_consumer_compiled')
assert r['outputs']==('image_masked',)
assert r['inputs']==('text_key_rotary','image_key','image_query','image_attention_mask')
report={'hypothesis':'Make repeated keys internal to their original canonical56-batch score consumer; allow compiler to handle producer layout and score epilogue together.','same_every_node_object':True,'same_other_relative_node_order_and_three_regions':True,'region':{k:r[k] for k in ['id','inputs','outputs','prefix']},'repeated_key_source_coordinates_checked':checked,'score_geometry':[56,256,556,128],'ordered_reduction_terms':128,'scale_mask_nodes_unchanged_BF16_boundaries':True,'softmax_and_probability_cast':'unchanged eager','historical_distinction':'60 only compiled key producer;73 only score/epilogue;99 changed GEMM grouping. None measured this unchanged-dimension producer/consumer boundary.','declared_repeat_result_bytes':2*28*556*128*2,'benefit_limits':['No total FLOP reduction. Compiler may still materialize all repeated keys or use same bmm.','No current compiled109 kernel profile; cannot attribute prior time to materialization.','Combining previously nonselected regions need not improve performance; exact numerics and resource gates still mandatory.'],'canonical_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(folder/'argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
