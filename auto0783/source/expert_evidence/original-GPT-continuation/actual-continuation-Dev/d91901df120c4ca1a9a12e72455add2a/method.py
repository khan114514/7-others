"""Construct cached attention for text queries using shared full key/value heads.

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
    def apply_rotary(name,value,cosine,sine):
        first=add(name+'_first','slice',[value],dim=3,start=0,stop=64)
        second=add(name+'_second','slice',[value],dim=3,start=64,stop=128)
        negative=add(name+'_negative','scale',[second],factor=-1.0)
        half_rotated=add(name+'_half','concat',[negative,first],dim=3)
        direct=add(name+'_direct','mul',[value,cosine])
        rotated=add(name+'_rotated','mul',[half_rotated,sine])
        return add(name,'add',[direct,rotated])
    # RMS reduces only the hidden dimension; discarded image rows need not be
    # normalized during text-prefix construction. Full projection rows remain.
    # The input view is cheap to recreate; only its RMS result needs caching.
    text_input=add('input_text_rows','slice',['state'],dim=1,start=0,stop=300)
    text_norm=add('text_input_norm','rms_norm',[text_input,'w:input_layernorm.weight'])
    text_norm_rows=add('text_input_norm_rows','slice',[text_norm],dim=1,start=0,stop=300)
    image_state=add('image_input_rows','slice',['state'],dim=1,start=300,stop=556)
    image_norm=add('image_input_norm','rms_norm',[image_state,'w:vision_branch_input_layernorm.weight'])
    # Only prompt-cached full-shape text projections consume this tensor.
    normalized=add('normalized','concat',[text_norm_rows,image_norm],dim=1,lifetime='prefix')
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
        text=add('text_'+kind,'linear',[normalized,'w:self_attn.'+kind+'_proj.weight','w:self_attn.'+kind+'_proj.bias'])
        start,stop={'q':(0,3584),'k':(3584,4096),'v':(4096,4608)}[kind]
        image=add('image_'+kind,'slice',[image_qkv],dim=2,start=start,stop=stop)
        text_rows=add('text_'+kind+'_rows','slice',[text],dim=1,start=0,stop=300)
        image_rows=image
        projections.append(text_rows if kind=='q' else
                           add(kind,'concat',[text_rows,image_rows],dim=1))
    # Preserve eager BF16 score scaling, mask addition, FP32 softmax and BF16
    # probabilities. Only query rows are partitioned; keys/values stay full.
    # Text and image queries share the full stage key/value construction.
    image_q='image_q'
    image_q_shape=add('image_q_shape','reshape',[image_q],shape=[2,256,28,128])
    image_q_heads=add('image_q_heads','transpose',[image_q_shape],dim0=1,dim1=2)
    full_cos=rotary_layout('cos')
    full_sin=rotary_layout('sin')
    image_cos=add('image_cos','slice',[full_cos],dim=2,start=300,stop=556,lifetime='prefix')
    image_sin=add('image_sin','slice',[full_sin],dim=2,start=300,stop=556,lifetime='prefix')
    image_query=apply_rotary('image_query',image_q_heads,image_cos,image_sin)
    key_shape=add('key_shape','reshape',[projections[1]],shape=[2,556,4,128])
    key_heads=add('key_heads','transpose',[key_shape],dim0=1,dim1=2)
    key_rotary=apply_rotary('key_rotary',key_heads,full_cos,full_sin)
    key_repeat=add('key_repeat','repeat_heads',[key_rotary],count=7)
    key_transpose=add('key_transpose','transpose',[key_repeat],dim0=2,dim1=3)
    value_shape=add('value_shape','reshape',[projections[2]],shape=[2,556,4,128])
    value_heads=add('value_heads','transpose',[value_shape],dim0=1,dim1=2)
    value_repeat=add('value_repeat','repeat_heads',[value_heads],count=7)
    # Construct the cached text branch with only its 300 query rows. Full keys,
    # values, and text mask columns are preserved, including padded positions.
    text_q_shape=add('text_q_shape','reshape',[projections[0]],shape=[2,300,28,128])
    text_q_heads=add('text_q_heads','transpose',[text_q_shape],dim0=1,dim1=2)
    text_cos=add('text_cos','slice',[full_cos],dim=2,start=0,stop=300)
    text_sin=add('text_sin','slice',[full_sin],dim=2,start=0,stop=300)
    text_query=apply_rotary('text_query',text_q_heads,text_cos,text_sin)
    text_scores=add('text_scores','matmul',[text_query,key_transpose])
    text_scaled=add('text_scaled','scale',[text_scores],factor=128**-0.5)
    text_mask=add('text_attention_mask','slice',['attention_mask'],dim=2,start=0,stop=300)
    text_masked=add('text_masked','add',[text_scaled,text_mask])
    text_prob_fp32=add('text_prob_fp32','softmax',[text_masked],dim=-1)
    text_prob=add('text_prob','cast',[text_prob_fp32],dtype='bf16')
    text_output_heads=add('text_output_heads','matmul',[text_prob,value_repeat])
    text_attended_heads=add('text_attended_heads','transpose',[text_output_heads],dim0=1,dim1=2)
    text_attended=add('text_attended','reshape',[text_attended_heads],shape=[2,300,3584])
    text_out_rows=add('text_attention_out','linear',[text_attended,'w:self_attn.o_proj.weight'])
    image_scores=add('image_scores','matmul',[image_query,key_transpose])
    image_scaled=add('image_scaled','scale',[image_scores],factor=128**-0.5)
    image_mask_rows=add('image_attention_mask','slice',['attention_mask'],dim=2,start=300,stop=556,lifetime='prefix')
    image_masked=add('image_masked','add',[image_scaled,image_mask_rows])
    image_prob_fp32=add('image_prob_fp32','softmax',[image_masked],dim=-1)
    image_prob=add('image_prob','cast',[image_prob_fp32],dtype='bf16')
    image_output_heads=add('image_output_heads','matmul',[image_prob,value_repeat])
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
    # Text projection dimensions are retained; vision rows are partitioned.
    for node in nodes:
        if node['id'].startswith('text_'):
            node['lifetime'] = 'prefix'
    return {'schema':'bifrost-tensor-dag/1','layer_body':nodes,'layer_output':output}

if __name__=='__main__':
    path=Path('program.json')
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(build_program(),indent=2)+'\n')
    pending.replace(path)
