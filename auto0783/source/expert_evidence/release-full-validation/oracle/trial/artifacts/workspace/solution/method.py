"""Author Reference: dense-shape tensor graph with per-prefix text reuse.

Standalone builder for the native Harbor Oracle; kept outside the Agent image.
"""
import json
from pathlib import Path

def build_program():
    nodes=[]
    def add(name,op,inputs,**attributes):
        nodes.append({'id':name,'op':op,'inputs':inputs,**attributes})
        return name
    text_norm=add('text_input_norm','rms_norm',['state','w:input_layernorm.weight'])
    image_norm=add('image_input_norm','rms_norm',['state','w:vision_branch_input_layernorm.weight'])
    normalized=add('normalized','mix',[text_norm,image_norm,'image_mask'])
    projections=[]
    for kind in ['q','k','v']:
        text=add('text_'+kind,'linear',[normalized,'w:self_attn.'+kind+'_proj.weight','w:self_attn.'+kind+'_proj.bias'])
        image=add('image_'+kind,'linear',[normalized,'w:self_attn.vision_branch_'+kind+'_proj.weight','w:self_attn.vision_branch_'+kind+'_proj.bias'])
        projections.append(add(kind,'mix',[text,image,'image_mask']))
    attended=add('attended','attention_eager',[*projections,'attention_mask','cos','sin'])
    text_out=add('text_attention_out','linear',[attended,'w:self_attn.o_proj.weight'])
    image_out=add('image_attention_out','linear',[attended,'w:self_attn.vision_branch_o_proj.weight'])
    attention_out=add('attention_out','mix',[text_out,image_out,'image_mask'])
    residual=add('attention_residual','add',['state',attention_out])
    text_post=add('text_post_norm','rms_norm',[residual,'w:post_attention_layernorm.weight'])
    image_post=add('image_post_norm','rms_norm',[residual,'w:vision_branch_post_attention_layernorm.weight'])
    normalized_post=add('normalized_post','mix',[text_post,image_post,'image_mask'])
    mlps=[]
    for branch,prefix in [('text','mlp.'),('image','vision_branch_mlp.')]:
        mlps.append(add(branch+'_mlp','mlp',[normalized_post,*['w:'+prefix+kind+'_proj.weight' for kind in ['gate','up','down']]]))
    mlp_out=add('mlp_out','mix',[*mlps,'image_mask'])
    output=add('output','add',[residual,mlp_out])
    invariant={'text_input_norm','text_q','text_k','text_v','text_attention_out','text_post_norm','text_mlp'}
    for node in nodes:
        if node['id'] in invariant:
            node['lifetime']='prefix'
    return {'schema':'bifrost-tensor-dag/1','layer_body':nodes,'layer_output':output}

if __name__=='__main__':
    path=Path('program.json')
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(build_program(),indent=2)+'\n')
    pending.replace(path)
