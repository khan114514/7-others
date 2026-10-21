"""Experiment 71: atomic raw-axis rotary with obsolete layout cache removal.

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
    # Slice token positions before the catalog's internal static channel-axis
    # assembly. Unary negation stays inside the original rotary composition.
    image_raw_cos=add('image_raw_cos','slice',['cos'],dim=2,start=300,stop=556,lifetime='prefix')
    image_raw_sin=add('image_raw_sin','slice',['sin'],dim=2,start=300,stop=556,lifetime='prefix')
    image_qk_rotary=add('image_qk_rotary','rotary',[image_qk_heads,image_raw_cos,image_raw_sin])
    image_query=add('image_query','slice',[image_qk_rotary],dim=1,start=0,stop=28)
    image_key=add('image_key','slice',[image_qk_rotary],dim=1,start=28,stop=32)
    text_key_shape=add('text_key_shape','reshape',[projections[1]],shape=[2,300,4,128])
    text_key_heads=add('text_key_heads','transpose',[text_key_shape],dim0=1,dim1=2)
    # Published rotary assembles the same channel sections internally. The
    # text position slice commutes with channel assembly; cache only its result.
    text_key=add('text_key_rotary','rotary',[text_key_heads,text_cos,text_sin])
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
    # Trusted fullgraph compilation may fuse SiLU and gating while retaining
    # checkpoint GEMMs and required precision casts. No tensor lifetime changes.
    return {'schema':'bifrost-tensor-dag/2','layer_body':nodes,'layer_output':output,
            'compile_regions':[
                {'id':'text_key_rotary_compiled','nodes':['text_key_rotary']},
                {'id':'image_rotary_compiled','nodes':['image_qk_rotary']},
                {'id':'image_mlp_compiled','nodes':['image_mlp']}]}

if __name__=='__main__':
    path=Path('program.json')
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(build_program(),indent=2)+'\n')
    pending.replace(path)
