"""Isolate contiguous image MLP input while retaining stage concatenation.

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
    # Only selected text rows from this prompt-lifetime full-shape norm are
    # used. Image rows are normalized each stage, then recombined contiguously
    # before every original-shape biased projection.
    text_norm=add('text_input_norm','rms_norm',['state','w:input_layernorm.weight'])
    text_norm_rows=add('text_input_norm_rows','slice',[text_norm],dim=1,start=0,stop=300)
    image_state=add('image_input_rows','slice',['state'],dim=1,start=300,stop=556)
    image_norm=add('image_input_norm','rms_norm',[image_state,'w:vision_branch_input_layernorm.weight'])
    normalized=add('normalized','concat',[text_norm_rows,image_norm],dim=1)
    projections=[]
    for kind in ['q','k','v']:
        text=add('text_'+kind,'linear',[normalized,'w:self_attn.'+kind+'_proj.weight','w:self_attn.'+kind+'_proj.bias'])
        image=add('image_'+kind,'linear',[normalized,'w:self_attn.vision_branch_'+kind+'_proj.weight','w:self_attn.vision_branch_'+kind+'_proj.bias'])
        projections.append(add(kind,'mix',[text,image,'image_mask']))
    # Construct the original full-shape eager attention once for the text path.
    # Valid text rows cannot see image or padding columns. Cached padding and
    # image rows are irrelevant because only text projection rows are selected.
    attended=add('attended','attention_eager',[*projections,'attention_mask','cos','sin'],lifetime='prefix')
    text_out=add('text_attention_out','linear',[attended,'w:self_attn.o_proj.weight'])
    text_out_rows=add('text_attention_out_rows','slice',[text_out],dim=1,start=0,stop=300)
    # Preserve eager BF16 score scaling, mask addition, FP32 softmax and BF16
    # probabilities. Only query rows are partitioned; keys/values stay full.
    image_q=add('image_q_rows','slice',[projections[0]],dim=1,start=300,stop=556)
    image_q_shape=add('image_q_shape','reshape',[image_q],shape=[2,256,28,128])
    image_q_heads=add('image_q_heads','transpose',[image_q_shape],dim0=1,dim1=2)
    image_cos=add('image_cos','slice',['cos'],dim=2,start=300,stop=556,lifetime='prefix')
    image_sin=add('image_sin','slice',['sin'],dim=2,start=300,stop=556,lifetime='prefix')
    image_query=add('image_query','rotary',[image_q_heads,image_cos,image_sin])
    key_shape=add('key_shape','reshape',[projections[1]],shape=[2,556,4,128])
    key_heads=add('key_heads','transpose',[key_shape],dim0=1,dim1=2)
    key_rotary=add('key_rotary','rotary',[key_heads,'cos','sin'])
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
    normalized_post=add('normalized_post','concat',[text_post_norm,image_post_norm],dim=1)
    text_mlp=add('text_mlp','mlp',[normalized_post,*[
        'w:mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])
    text_mlp_rows=add('text_mlp_rows','slice',[text_mlp],dim=1,start=0,stop=300)
    image_post=add('image_post_rows','slice',[normalized_post],dim=1,start=300,stop=556)
    # Flattening across the sliced batch stride copies into contiguous rows.
    image_post_flat=add('image_post_flat','reshape',[image_post],shape=[512,3584])
    image_post_dense=add('image_post_dense','reshape',[image_post_flat],shape=[2,256,3584])
    image_mlp=add('image_mlp','mlp',[image_post_dense,*[
        'w:vision_branch_mlp.'+kind+'_proj.weight' for kind in ['gate','up','down']]])
    text_output=add('text_output','add',[text_residual,text_mlp_rows])
    image_output=add('image_output','add',[image_residual,image_mlp])
    output=add('output','concat',[text_output,image_output],dim=1)
    # Valid causal text rows are image-independent. Padded text rows may vary,
    # but are masked from every valid query and never reach the image head.
    # Biased Q/K/V projections retain the original 556-row GEMMs.
    for node in nodes:
        if node['id'].startswith('text_'):
            node['lifetime'] = 'prefix'
    return {'schema':'bifrost-tensor-dag/1','layer_body':nodes,'layer_output':output}

if __name__=='__main__':
    path=Path('program.json')
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(build_program(),indent=2)+'\n')
    pending.replace(path)
