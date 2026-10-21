"""Experiment 111: packed V restoration/repetition shares its original reversed contraction.

Emit an operator DAG. This builder executes only in the local candidate container;
the numerical worker accepts JSON, never imports this Python file.
"""
import json
from pathlib import Path

def build_program():
    nodes=[]
    def add(name,op,inputs,**attributes):
        nodes.append({'id':name,'op':op,'inputs':inputs,**attributes})
        return name
    def rotary_layout(source):
        axes=[]
        for axis in range(3):
            part=add(source+'_axis_'+str(axis),'slice',[source],
                     dim=0,start=axis,stop=axis+1,lifetime='prefix')
            axes.append(add(source+'_axis_shape_'+str(axis),'reshape',[part],
                            shape=[2,1,556,128],lifetime='prefix'))
        sections=[]
        start=0
        for index,width in enumerate([16,24,24,16,24,24]):
            sections.append(add(source+'_section_'+str(index),'slice',
                                [axes[index%3]],dim=3,start=start,
                                stop=start+width,lifetime='prefix'))
            start+=width
        return add(source+'_layout','concat',sections,dim=3,lifetime='prefix')
    def apply_rotary(name,value,cosine,sine,preflipped_sine=False):
        first=add(name+'_first','slice',[value],dim=3,start=0,stop=64)
        second=add(name+'_second','slice',[value],dim=3,start=64,stop=128)
        negative=second if preflipped_sine else add(name+'_negative','scale',[second],factor=-1.0)
        half_rotated=add(name+'_half','concat',[negative,first],dim=3)
        direct=add(name+'_direct','mul',[value,cosine])
        rotated=add(name+'_rotated','mul',[half_rotated,sine])
        return add(name,'add',[direct,rotated])
    # RMS reduces only the hidden dimension; construct just the text rows.
    # The input view is cheap to recreate; only its RMS result needs caching.
    text_input=add('input_text_rows','slice',['state'],dim=1,start=0,stop=300)
    text_norm=add('text_input_norm','rms_norm',[text_input,'w:input_layernorm.weight'])
    text_norm_rows=add('text_input_norm_rows','slice',[text_norm],dim=1,start=0,stop=300)
    image_state=add('image_input_rows','slice',['state'],dim=1,start=300,stop=556)
    image_norm=add('image_input_norm','rms_norm',[image_state,'w:vision_branch_input_layernorm.weight'])
    # Fuse the cached biased text projections over 600 contiguous rows.
    text_qkv_weight=add('text_qkv_weight','concat',[
        'w:self_attn.q_proj.weight','w:self_attn.k_proj.weight',
        'w:self_attn.v_proj.weight'],dim=0)
    text_qkv_bias=add('text_qkv_bias','concat',[
        'w:self_attn.q_proj.bias','w:self_attn.k_proj.bias',
        'w:self_attn.v_proj.bias'],dim=0)
    text_flat=add('text_input_flat','reshape',[text_norm_rows],shape=[600,3584])
    text_flat_qkv=add('text_flat_qkv','linear',[text_flat,text_qkv_weight,text_qkv_bias])
    text_qkv=add('text_qkv','reshape',[text_flat_qkv],shape=[2,300,4608])
    qkv_weight=add('vision_qkv_weight','concat',[
        'w:self_attn.vision_branch_q_proj.weight',
        'w:self_attn.vision_branch_k_proj.weight',
        'w:self_attn.vision_branch_v_proj.weight'],dim=0,lifetime='prefix')
    qkv_bias=add('vision_qkv_bias','concat',[
        'w:self_attn.vision_branch_q_proj.bias',
        'w:self_attn.vision_branch_k_proj.bias',
        'w:self_attn.vision_branch_v_proj.bias'],dim=0,lifetime='prefix')
    image_qkv=add('image_qkv','linear',[image_norm,qkv_weight,qkv_bias])
    projections=[]
    for kind in ['q','k','v']:
        start,stop={'q':(0,3584),'k':(3584,4096),'v':(4096,4608)}[kind]
        text_rows=add('text_'+kind+'_rows','slice',[text_qkv],dim=2,start=start,stop=stop)
        if kind=='v':
            image_rows=add('image_v','slice',[image_qkv],dim=2,start=start,stop=stop)
            projections.append(add(kind,'concat',[text_rows,image_rows],dim=1))
        else:
            projections.append(text_rows)
    # Preserve eager BF16 score scaling, mask addition, FP32 softmax and BF16
    # probabilities. Only query rows are partitioned; keys/values stay full.
    # Cached valid text queries cannot attend image keys. Padding query values
    # may differ, but padding key columns are masked from all valid queries.
    text_mask_rows=add('text_mask_rows','slice',['attention_mask'],dim=2,start=0,stop=300)
    text_mask=add('text_attention_mask','slice',[text_mask_rows],dim=3,start=0,stop=300)
    text_cos=add('text_cos','slice',['cos'],dim=2,start=0,stop=300)
    text_sin=add('text_sin','slice',['sin'],dim=2,start=0,stop=300)
    text_attended=add('text_attended','attention_eager',[
        projections[0],'text_k_rows','text_v_rows',text_mask,text_cos,text_sin])
    text_out_rows=add('text_attention_out','linear',[text_attended,'w:self_attn.o_proj.weight'])
    # Fused projection lays out 28 Q heads followed by 4 K heads; rotating
    # these 32 heads together avoids a second stage rotary composition.
    image_qk=add('image_qk','slice',[image_qkv],dim=2,start=0,stop=4096)
    image_qk_shape=add('image_qk_shape','reshape',[image_qk],shape=[2,256,32,128])
    image_qk_heads=add('image_qk_heads','transpose',[image_qk_shape],dim0=1,dim1=2)
    full_cos=rotary_layout('cos')
    full_sin=rotary_layout('sin')
    image_cos=add('image_cos','slice',[full_cos],dim=2,start=300,stop=556,lifetime='prefix')
    image_sin=add('image_sin','slice',[full_sin],dim=2,start=300,stop=556,lifetime='prefix')
    # Negation is exact in BF16. Move its first-half sign to the fixed sine
    # layout so the changing Q/K values only need swapping, not negation.
    image_sin_first=add('image_sin_first','slice',[image_sin],dim=3,start=0,stop=64,lifetime='prefix')
    image_sin_negative=add('image_sin_negative','scale',[image_sin_first],factor=-1.0,lifetime='prefix')
    image_sin_second=add('image_sin_second','slice',[image_sin],dim=3,start=64,stop=128,lifetime='prefix')
    image_signed_sin=add('image_signed_sin','concat',[image_sin_negative,image_sin_second],dim=3,lifetime='prefix')
    image_qk_rotary=apply_rotary('image_qk_rotary',image_qk_heads,image_cos,image_signed_sin,preflipped_sine=True)
    image_query=add('image_query','slice',[image_qk_rotary],dim=1,start=0,stop=28)
    image_key=add('image_key','slice',[image_qk_rotary],dim=1,start=28,stop=32)
    text_key_shape=add('text_key_shape','reshape',[projections[1]],shape=[2,300,4,128])
    text_key_heads=add('text_key_heads','transpose',[text_key_shape],dim0=1,dim1=2)
    text_key_cos=add('text_key_cos','slice',[full_cos],dim=2,start=0,stop=300)
    text_key_sin=add('text_key_sin','slice',[full_sin],dim=2,start=0,stop=300)
    text_key=apply_rotary('text_key_rotary',text_key_heads,text_key_cos,text_key_sin)
    key_rotary=add('key_rotary','concat',[text_key,image_key],dim=2)
    key_repeat=add('key_repeat','repeat_heads',[key_rotary],count=7)
    key_transpose=add('key_transpose','transpose',[key_repeat],dim0=2,dim1=3)
    value_shape=add('value_shape','reshape',[projections[2]],shape=[2,556,4,128])
    value_heads=add('value_heads','transpose',[value_shape],dim0=1,dim1=2)
    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)
    image_scores=add('image_scores','matmul',[image_query,key_transpose])
    image_scaled=add('image_scaled','scale',[image_scores],factor=128**-0.5)
    image_mask_rows=add('image_attention_mask','slice',['attention_mask'],dim=2,start=300,stop=556,lifetime='prefix')
    image_masked=add('image_masked','add',[image_scaled,image_mask_rows])
    image_prob_fp32=add('image_prob_fp32','softmax',[image_masked],dim=-1)
    image_prob=add('image_prob','cast',[image_prob_fp32],dtype='bf16')
    # Reverse only the probability/value contraction; keep the original
    # score, mask and softmax path. Restore axes before merging image heads.
    value_transpose=add('value_transpose','transpose',[value_repeat],dim0=2,dim1=3)
    probability_transpose=add('probability_transpose','transpose',[image_prob],dim0=2,dim1=3)
    channel_query_output=add('channel_query_output','matmul',[value_transpose,probability_transpose])
    image_output_heads=add('image_output_heads','transpose',[channel_query_output],dim0=2,dim1=3)
    image_output_rows=add('image_output_rows','transpose',[image_output_heads],dim0=1,dim1=2)
    image_attended=add('image_attended_rows','reshape',[image_output_rows],shape=[2,256,3584])
    image_out=add('image_attention_out','linear',[image_attended,'w:self_attn.vision_branch_o_proj.weight'])
    text_state=add('text_state_rows','slice',['state'],dim=1,start=0,stop=300)
    text_residual=add('text_residual_rows','add',[text_state,text_out_rows])
    text_post_norm=add('text_post_norm','rms_norm',[
        text_residual,'w:post_attention_layernorm.weight'],lifetime='prefix')
    image_residual=add('image_residual_rows','add',[image_state,image_out])
    image_post_norm=add('image_post_norm','rms_norm',[
        image_residual,'w:vision_branch_post_attention_layernorm.weight'])
    # The bias-free text MLP is row-local. Construct only the 300 text rows;
    # its previous image outputs were discarded before the residual addition.
    text_mlp=add('text_mlp','mlp',[text_post_norm,*[
        'w:mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])
    image_mlp=add('image_mlp','mlp',[image_post_norm,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])
    text_output=add('text_output','add',[text_residual,text_mlp])
    image_output=add('image_output','add',[image_residual,image_mlp])
    output=add('output','concat',[text_output,image_output],dim=1)
    # Valid causal text rows are image-independent. Padded text rows may vary,
    # but are masked from every valid query and never reach the image head.
    # Both projection branches consume only their respective token rows.
    for node in nodes:
        if node['id'].startswith('text_'):
            node['lifetime'] = 'prefix'
    # Retain full biased4608 projection geometry; replace the multi-output
    # projection/rotary boundary with one packed36-head output. V is only
    # reindexed/copied, never rotated or subjected to extra arithmetic.
    by_id={n['id']:n for n in nodes}
    region_members=['image_qkv','image_qk','image_qk_shape','image_qk_heads',
                    'image_qk_rotary_first','image_qk_rotary_second',
                    'image_qk_rotary_half','image_qk_rotary_direct',
                    'image_qk_rotary_rotated','image_qk_rotary']
    region_nodes=[by_id[name] for name in region_members]
    def make(name,op,inputs,**attrs):
        return {'id':name,'op':op,'inputs':inputs,**attrs}
    region_nodes.extend([
        make('image_v_raw','slice',['image_qkv'],dim=2,start=4096,stop=4608),
        make('image_v_raw_shape','reshape',['image_v_raw'],shape=[2,256,4,128]),
        make('image_v_raw_heads','transpose',['image_v_raw_shape'],dim0=1,dim1=2),
        make('image_packed_qkv','concat',['image_qk_rotary','image_v_raw_heads'],dim=1)])
    region_members=[n['id'] for n in region_nodes]
    outside=[
        make('image_query','slice',['image_packed_qkv'],dim=1,start=0,stop=28),
        make('image_key','slice',['image_packed_qkv'],dim=1,start=28,stop=32),
        make('image_v_packed_heads','slice',['image_packed_qkv'],dim=1,start=32,stop=36),
        make('image_v_packed_rows','transpose',['image_v_packed_heads'],dim0=1,dim1=2),
        make('image_v','reshape',['image_v_packed_rows'],shape=[2,256,512]),
        by_id['v']]
    remove=set(region_members)|{'image_query','image_key','image_v','v'}
    nodes=[n for n in nodes if n['id'] not in remove]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_signed_sin')+1
    nodes[at:at]=region_nodes+outside
    # Hide repeated-key producer boundary inside canonical score consumer.
    # Full original score geometry and BF16 scaling/mask order remain intact.
    # FP32 softmax and BF16 probability cast are outside/eager.
    score_members=['key_rotary','key_repeat','key_transpose',
                   'image_scores','image_scaled','image_masked']
    selected={n['id']:n for n in nodes if n['id'] in score_members}
    nodes=[n for n in nodes if n['id'] not in selected]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_prob_fp32')
    nodes[at:at]=[selected[name] for name in score_members]
    # Keep packed V restoration, text/image value concat, repetition and
    # unchanged reversed value GEMM in one region with a sole channel result.
    # BF16 probabilities stay an eager runtime input; full556-term reduction
    # and original56-batch contraction geometry are unchanged.
    value_members=['image_v_packed_heads','image_v_packed_rows','image_v','v',
                   'value_shape','value_heads','value_repeat','value_transpose',
                   'probability_transpose','channel_query_output']
    selected_values={n['id']:n for n in nodes if n['id'] in value_members}
    nodes=[n for n in nodes if n['id'] not in selected_values]
    at=next(i for i,n in enumerate(nodes) if n['id']=='image_output_heads')
    nodes[at:at]=[selected_values[name] for name in value_members]
    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,
            'compile_regions':[
                {'id':'packed_value_producer_consumer_compiled','nodes':value_members},
                {'id':'key_score_producer_consumer_compiled','nodes':score_members},
                {'id':'image_packed_projection_rotary_compiled','nodes':region_members},
                {'id':'image_mlp_compiled','nodes':['image_mlp']},
                {'id': 'frequency_prefix_compiled', 'nodes': ['cos_axis_0', 'cos_axis_shape_0', 'cos_axis_1', 'cos_axis_shape_1', 'cos_axis_2', 'cos_axis_shape_2', 'cos_section_0', 'cos_section_1', 'cos_section_2', 'cos_section_3', 'cos_section_4', 'cos_section_5', 'cos_layout', 'sin_axis_0', 'sin_axis_shape_0', 'sin_axis_1', 'sin_axis_shape_1', 'sin_axis_2', 'sin_axis_shape_2', 'sin_section_0', 'sin_section_1', 'sin_section_2', 'sin_section_3', 'sin_section_4', 'sin_section_5', 'sin_layout', 'image_cos', 'image_sin', 'image_sin_first', 'image_sin_negative', 'image_sin_second', 'image_signed_sin']}]}

if __name__=='__main__':
    path=Path('program.json')
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(build_program(),indent=2)+'\n')
    pending.replace(path)
