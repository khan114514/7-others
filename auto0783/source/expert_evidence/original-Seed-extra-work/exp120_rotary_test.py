"""Bifrost-1 Decoder Optimization — Exp 120

Test: Can rotary be safely compiled with Inductor?
The rotate function uses cat, split, unsqueeze, and bf16 elementwise ops.
No accumulation — should be exact.

If safe, we can include rotary in larger compiled regions,
which would let us consolidate into fewer, bigger regions.
"""
import json
from pathlib import Path

TEXT_LEN = 300
SEQ = 556
IMAGE_LEN = SEQ - TEXT_LEN
Q_DIM = 3584
KV_DIM = 512
QKV_DIM = Q_DIM + KV_DIM + KV_DIM

def build_program():
    prefix_nodes = []
    stage_nodes = []
    
    def add_prefix(name, op, inputs, **attrs):
        prefix_nodes.append({'id': name, 'op': op, 'inputs': inputs, 'lifetime': 'prefix', **attrs})
        return name
    
    def add_stage(name, op, inputs, **attrs):
        stage_nodes.append({'id': name, 'op': op, 'inputs': inputs, **attrs})
        return name

    # ===== PREFIX NODES =====
    text_state = add_prefix('text_state', 'slice', ['state'],
                     dim=1, start=0, stop=TEXT_LEN)
    text_norm = add_prefix('text_input_norm', 'rms_norm',
                    [text_state, 'w:input_layernorm.weight'])
    text_fused_qkv_w = add_prefix('text_fused_qkv_weight', 'concat',
        ['w:self_attn.q_proj.weight',
         'w:self_attn.k_proj.weight',
         'w:self_attn.v_proj.weight'], dim=0)
    text_fused_qkv_b = add_prefix('text_fused_qkv_bias', 'concat',
        ['w:self_attn.q_proj.bias',
         'w:self_attn.k_proj.bias',
         'w:self_attn.v_proj.bias'], dim=0)
    text_qkv = add_prefix('text_qkv', 'linear',
                   [text_norm, text_fused_qkv_w, text_fused_qkv_b])
    text_q = add_prefix('text_q_slice', 'slice', [text_qkv],
                 dim=2, start=0, stop=Q_DIM)
    text_k = add_prefix('text_k_slice', 'slice', [text_qkv],
                 dim=2, start=Q_DIM, stop=Q_DIM + KV_DIM)
    text_v = add_prefix('text_v_slice', 'slice', [text_qkv],
                 dim=2, start=Q_DIM + KV_DIM, stop=QKV_DIM)
    img_fused_qkv_w = add_prefix('fused_qkv_weight', 'concat',
        ['w:self_attn.vision_branch_q_proj.weight',
         'w:self_attn.vision_branch_k_proj.weight',
         'w:self_attn.vision_branch_v_proj.weight'], dim=0)
    img_fused_qkv_b = add_prefix('fused_qkv_bias', 'concat',
        ['w:self_attn.vision_branch_q_proj.bias',
         'w:self_attn.vision_branch_k_proj.bias',
         'w:self_attn.vision_branch_v_proj.bias'], dim=0)
    text_cos = add_prefix('text_cos', 'slice', ['cos'],
                   dim=2, start=0, stop=TEXT_LEN)
    text_sin = add_prefix('text_sin', 'slice', ['sin'],
                   dim=2, start=0, stop=TEXT_LEN)
    img_cos = add_prefix('image_cos', 'slice', ['cos'],
                  dim=2, start=TEXT_LEN, stop=SEQ)
    img_sin = add_prefix('image_sin', 'slice', ['sin'],
                  dim=2, start=TEXT_LEN, stop=SEQ)
    text_mask_2d = add_prefix('text_mask_2d', 'slice', ['attention_mask'],
                       dim=2, start=0, stop=TEXT_LEN)
    text_mask = add_prefix('text_mask', 'slice', [text_mask_2d],
                    dim=3, start=0, stop=TEXT_LEN)
    text_attn = add_prefix('text_attn_eager', 'attention_eager',
                    [text_q, text_k, text_v, text_mask, text_cos, text_sin])
    text_out = add_prefix('text_out', 'linear',
                   [text_attn, 'w:self_attn.o_proj.weight'])
    text_resid = add_prefix('text_residual', 'add',
                     [text_state, text_out])
    img_mask_rows = add_prefix('img_mask_rows', 'slice', ['attention_mask'],
                        dim=2, start=TEXT_LEN, stop=SEQ)
    text_post_norm = add_prefix('text_post_norm', 'rms_norm',
        [text_resid, 'w:post_attention_layernorm.weight'])
    text_mlp = add_prefix('text_mlp', 'mlp', [text_post_norm,
        'w:mlp.gate_proj.weight', 'w:mlp.up_proj.weight', 'w:mlp.down_proj.weight'])
    text_output = add_prefix('text_output', 'add',
                      [text_resid, text_mlp])

    # ===== PER-STAGE NODES =====
    img_state = add_stage('image_state', 'slice', ['state'],
                    dim=1, start=TEXT_LEN, stop=SEQ)
    img_norm = add_stage('image_input_norm', 'rms_norm',
                   [img_state, 'w:vision_branch_input_layernorm.weight'])
    img_qkv_in = add_stage('image_qkv_flat_in', 'reshape',
                     [img_norm], shape=[512, 'hidden'])
    img_qkv = add_stage('image_qkv_flat', 'linear',
                  [img_qkv_in, img_fused_qkv_w, img_fused_qkv_b])
    img_q_flat = add_stage('image_q_flat', 'slice', [img_qkv],
                     dim=1, start=0, stop=Q_DIM)
    img_k_flat = add_stage('image_k_flat', 'slice', [img_qkv],
                     dim=1, start=Q_DIM, stop=Q_DIM + KV_DIM)
    img_v_flat = add_stage('image_v_flat', 'slice', [img_qkv],
                     dim=1, start=Q_DIM + KV_DIM, stop=QKV_DIM)
    img_q = add_stage('image_q_reshaped', 'reshape', [img_q_flat],
                shape=['batch', 256, 'heads', 'head_dim'])
    img_k_3d = add_stage('image_k_3d', 'reshape', [img_k_flat],
                   shape=['batch', 256, KV_DIM])
    img_v_3d = add_stage('image_v_3d', 'reshape', [img_v_flat],
                   shape=['batch', 256, KV_DIM])
    full_k = add_stage('full_k', 'concat', [text_k, img_k_3d], dim=1)
    full_v = add_stage('full_v', 'concat', [text_v, img_v_3d], dim=1)
    img_q_t = add_stage('image_q_t', 'transpose', [img_q], dim0=1, dim1=2)
    
    # === COMPILED REGION: just image Q rotary (test safety) ===
    img_q_rot = add_stage('image_q_rot', 'rotary',
                    [img_q_t, img_cos, img_sin])
    # === END ===
    
    k_4d = add_stage('k_reshaped', 'reshape', [full_k],
               shape=['batch', 'sequence', 'kv_heads', 'head_dim'])
    k_t = add_stage('k_t', 'transpose', [k_4d], dim0=1, dim1=2)
    k_rot = add_stage('k_rot', 'rotary', [k_t, 'cos', 'sin'])
    k_rep = add_stage('k_rep', 'repeat_heads', [k_rot], count=7)
    v_4d = add_stage('v_reshaped', 'reshape', [full_v],
               shape=['batch', 'sequence', 'kv_heads', 'head_dim'])
    v_t = add_stage('v_t', 'transpose', [v_4d], dim0=1, dim1=2)
    v_rep = add_stage('v_rep', 'repeat_heads', [v_t], count=7)
    q_flat = add_stage('q_flat', 'reshape', [img_q_rot], shape=[56, 256, 128])
    k_rep_t = add_stage('k_rep_t', 'transpose', [k_rep], dim0=2, dim1=3)
    k_flat_t = add_stage('k_flat_t', 'reshape', [k_rep_t], shape=[56, 128, 'sequence'])
    scores_flat = add_stage('img_scores_flat', 'matmul', [q_flat, k_flat_t])
    scores = add_stage('img_scores', 'reshape', [scores_flat],
                 shape=['batch', 'heads', 256, 'sequence'])
    scaled = add_stage('img_scaled', 'scale', [scores], factor=0.08838834764831843)
    masked = add_stage('img_masked', 'add', [scaled, img_mask_rows])
    sm_fp32 = add_stage('img_softmax_fp32', 'softmax', [masked], dim=-1)
    softmax = add_stage('img_softmax', 'cast', [sm_fp32], dtype='bf16')
    attn_out = add_stage('img_attn_out', 'matmul', [softmax, v_rep])
    out_t = add_stage('img_out_t', 'transpose', [attn_out], dim0=1, dim1=2)
    o_flat_in = add_stage('img_attn_o_flat', 'reshape', [out_t], shape=[512, 'hidden'])
    img_out_flat = add_stage('image_out_flat', 'linear',
                       [o_flat_in, 'w:self_attn.vision_branch_o_proj.weight'])
    img_out = add_stage('image_out', 'reshape', [img_out_flat],
                  shape=['batch', 256, 'hidden'])
    img_resid = add_stage('image_residual', 'add', [img_state, img_out])
    img_post_norm = add_stage('image_post_norm', 'rms_norm',
        [img_resid, 'w:vision_branch_post_attention_layernorm.weight'])
    img_post_flat = add_stage('image_post_flat', 'reshape',
                        [img_post_norm], shape=[512, 'hidden'])
    img_mlp_flat = add_stage('image_mlp_flat', 'mlp', [img_post_flat,
        'w:vision_branch_mlp.gate_proj.weight',
        'w:vision_branch_mlp.up_proj.weight',
        'w:vision_branch_mlp.down_proj.weight'])
    img_mlp = add_stage('image_mlp', 'reshape', [img_mlp_flat],
                  shape=['batch', 256, 'hidden'])
    img_output = add_stage('image_output', 'add', [img_resid, img_mlp])
    output = add_stage('output', 'concat', [text_output, img_output], dim=1)

    all_nodes = prefix_nodes + stage_nodes
    
    compile_regions = [
        {'id': 'test_image_q_rotary', 'nodes': ['image_q_rot']},
    ]

    program = {
        'schema': 'bifrost-tensor-dag/2',
        'layer_body': all_nodes,
        'layer_output': 'output',
        'compile_regions': compile_regions,
    }
    return program

if __name__ == '__main__':
    path = Path('exp120_program.json')
    pending = path.with_suffix('.pending.json')
    program = build_program()
    pending.write_text(json.dumps(program, indent=2) + '\n')
    pending.replace(path)
    print(f"Generated: {len(program['layer_body'])} nodes, {len(program['compile_regions'])} region(s)")
