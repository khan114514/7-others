"""Experiment 91: Use `mlp` op on flattened 2D image input.

Building on exp89 (71.63% speedup, BEST).

Idea: Replace manual 5-node MLP (gate+up+silu+mul+down) with single `mlp` op.
Saves 4 per-stage nodes → less Python interpreter overhead.

The `mlp` op internally does: F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)
Same computation, fewer graph nodes.

Exp76 found "mlp op matches manual flattened speed" within noise.
Test with our current optimized setup to confirm.
"""
import json
from pathlib import Path

TEXT_LEN = 300
SEQ = 556
IMAGE_LEN = SEQ - TEXT_LEN
Q_DIM = 3584
KV_DIM = 512
QKV_DIM = Q_DIM + KV_DIM + KV_DIM  # 4608

def build_program():
    nodes = []
    def add(name, op, inputs, **attrs):
        nodes.append({'id': name, 'op': op, 'inputs': inputs, **attrs})
        return name

    # --- Input: slice text (prefix) and image ---
    text_state = add('text_state', 'slice', ['state'], dim=1, start=0, stop=TEXT_LEN, lifetime='prefix')
    image_state = add('image_state', 'slice', ['state'], dim=1, start=TEXT_LEN, stop=SEQ)

    # --- Input norm: slice-based + prefix cache text ---
    text_norm = add('text_input_norm', 'rms_norm', [text_state, 'w:input_layernorm.weight'], lifetime='prefix')
    image_norm = add('image_input_norm', 'rms_norm', [image_state, 'w:vision_branch_input_layernorm.weight'])

    # --- Text QKV: FUSED on text-only input (prefix) ---
    text_fused_qkv_weight = add('text_fused_qkv_weight', 'concat',
                                ['w:self_attn.q_proj.weight',
                                 'w:self_attn.k_proj.weight',
                                 'w:self_attn.v_proj.weight'],
                                dim=0, lifetime='prefix')
    text_fused_qkv_bias = add('text_fused_qkv_bias', 'concat',
                              ['w:self_attn.q_proj.bias',
                               'w:self_attn.k_proj.bias',
                               'w:self_attn.v_proj.bias'],
                              dim=0, lifetime='prefix')
    text_qkv = add('text_qkv', 'linear', [text_norm, text_fused_qkv_weight, text_fused_qkv_bias], lifetime='prefix')
    text_q_slice = add('text_q_slice', 'slice', [text_qkv], dim=2, start=0, stop=Q_DIM, lifetime='prefix')
    text_k_slice = add('text_k_slice', 'slice', [text_qkv], dim=2, start=Q_DIM, stop=Q_DIM+KV_DIM, lifetime='prefix')
    text_v_slice = add('text_v_slice', 'slice', [text_qkv], dim=2, start=Q_DIM+KV_DIM, stop=QKV_DIM, lifetime='prefix')

    # --- Image QKV: FUSED on image-only input (per-stage) ---
    fused_qkv_weight = add('fused_qkv_weight', 'concat',
                           ['w:self_attn.vision_branch_q_proj.weight',
                            'w:self_attn.vision_branch_k_proj.weight',
                            'w:self_attn.vision_branch_v_proj.weight'],
                           dim=0, lifetime='prefix')
    fused_qkv_bias = add('fused_qkv_bias', 'concat',
                         ['w:self_attn.vision_branch_q_proj.bias',
                          'w:self_attn.vision_branch_k_proj.bias',
                          'w:self_attn.vision_branch_v_proj.bias'],
                         dim=0, lifetime='prefix')
    image_qkv_flat_in = add('image_qkv_flat_in', 'reshape', [image_norm], shape=[512, 'hidden'])
    image_qkv_flat = add('image_qkv_flat', 'linear', [image_qkv_flat_in, fused_qkv_weight, fused_qkv_bias])
    image_q_flat = add('image_q_flat', 'slice', [image_qkv_flat], dim=1, start=0, stop=Q_DIM)
    image_k_flat = add('image_k_flat', 'slice', [image_qkv_flat], dim=1, start=Q_DIM, stop=Q_DIM+KV_DIM)
    image_v_flat = add('image_v_flat', 'slice', [image_qkv_flat], dim=1, start=Q_DIM+KV_DIM, stop=QKV_DIM)
    image_q_reshaped = add('image_q_reshaped', 'reshape', [image_q_flat], shape=['batch', 256, 'heads', 'head_dim'])
    image_k_3d = add('image_k_3d', 'reshape', [image_k_flat], shape=['batch', 256, KV_DIM])
    image_v_3d = add('image_v_3d', 'reshape', [image_v_flat], shape=['batch', 256, KV_DIM])

    # --- Slice cos/sin ---
    text_cos = add('text_cos', 'slice', ['cos'], dim=2, start=0, stop=TEXT_LEN, lifetime='prefix')
    text_sin = add('text_sin', 'slice', ['sin'], dim=2, start=0, stop=TEXT_LEN, lifetime='prefix')
    image_cos = add('image_cos', 'slice', ['cos'], dim=2, start=TEXT_LEN, stop=SEQ, lifetime='prefix')
    image_sin = add('image_sin', 'slice', ['sin'], dim=2, start=TEXT_LEN, stop=SEQ, lifetime='prefix')

    # --- Text-only mask ---
    text_mask_2d = add('text_mask_2d', 'slice', ['attention_mask'], dim=2, start=0, stop=TEXT_LEN, lifetime='prefix')
    text_mask = add('text_mask', 'slice', [text_mask_2d], dim=3, start=0, stop=TEXT_LEN, lifetime='prefix')

    # --- Text attention via attention_eager (cached) ---
    text_attn_eager = add('text_attn_eager', 'attention_eager',
                          ['text_q_slice', 'text_k_slice', 'text_v_slice',
                           text_mask, text_cos, text_sin],
                          lifetime='prefix')

    # --- Text O-projection + text residual (PREFIX CACHED) ---
    text_out = add('text_out', 'linear', [text_attn_eager, 'w:self_attn.o_proj.weight'], lifetime='prefix')
    text_residual = add('text_residual', 'add', [text_state, text_out], lifetime='prefix')

    # --- Full K, V for image attention ---
    full_k = add('full_k', 'concat', ['text_k_slice', image_k_3d], dim=1)
    full_v = add('full_v', 'concat', ['text_v_slice', image_v_3d], dim=1)

    # --- Image Q: transpose + rotary ---
    image_q_t = add('image_q_t', 'transpose', [image_q_reshaped], dim0=1, dim1=2)
    image_q_rot = add('image_q_rot', 'rotary', [image_q_t, image_cos, image_sin])

    # --- K: reshape + transpose + rotary + repeat ---
    k_reshaped = add('k_reshaped', 'reshape', [full_k], shape=['batch', 'sequence', 'kv_heads', 'head_dim'])
    k_t = add('k_t', 'transpose', [k_reshaped], dim0=1, dim1=2)
    k_rot = add('k_rot', 'rotary', [k_t, 'cos', 'sin'])
    k_rep = add('k_rep', 'repeat_heads', [k_rot], count=7)

    # --- V: reshape + transpose + repeat ---
    v_reshaped = add('v_reshaped', 'reshape', [full_v], shape=['batch', 'sequence', 'kv_heads', 'head_dim'])
    v_t = add('v_t', 'transpose', [v_reshaped], dim0=1, dim1=2)
    v_rep = add('v_rep', 'repeat_heads', [v_t], count=7)

    # --- Flatten batch*heads for Q@K^T scores matmul ---
    q_flat = add('q_flat', 'reshape', [image_q_rot], shape=[56, 256, 128])
    k_rep_t = add('k_rep_t', 'transpose', [k_rep], dim0=2, dim1=3)
    k_flat_t = add('k_flat_t', 'reshape', [k_rep_t], shape=[56, 128, 'sequence'])
    img_scores_flat = add('img_scores_flat', 'matmul', [q_flat, k_flat_t])
    img_scores = add('img_scores', 'reshape', [img_scores_flat], shape=['batch', 'heads', 256, 'sequence'])

    img_scaled = add('img_scaled', 'scale', [img_scores], factor=0.08838834764831843)

    img_mask_rows = add('img_mask_rows', 'slice', ['attention_mask'], dim=2, start=TEXT_LEN, stop=SEQ, lifetime='prefix')
    img_masked = add('img_masked', 'add', [img_scaled, img_mask_rows])
    img_softmax_fp32 = add('img_softmax_fp32', 'softmax', [img_masked], dim=-1)
    img_softmax = add('img_softmax', 'cast', [img_softmax_fp32], dtype='bf16')

    # --- V matmul + combined reshape to flat O-projection ---
    img_attn_out = add('img_attn_out', 'matmul', [img_softmax, v_rep])
    img_out_t = add('img_out_t', 'transpose', [img_attn_out], dim0=1, dim1=2)
    img_attn_o_flat = add('img_attn_o_flat', 'reshape', [img_out_t], shape=[512, 'hidden'])

    # --- Image O-projection (FLATTENED) + image residual ---
    image_out_flat = add('image_out_flat', 'linear', [img_attn_o_flat, 'w:self_attn.vision_branch_o_proj.weight'])
    image_out = add('image_out', 'reshape', [image_out_flat], shape=['batch', 256, 'hidden'])
    image_residual = add('image_residual', 'add', [image_state, image_out])

    # --- Post-attention norm: split text (prefix) + image (per stage) ---
    text_post_norm = add('text_post_norm', 'rms_norm', [text_residual, 'w:post_attention_layernorm.weight'], lifetime='prefix')
    image_post_norm = add('image_post_norm', 'rms_norm', [image_residual, 'w:vision_branch_post_attention_layernorm.weight'])

    # --- Text MLP: mlp op + prefix cache ---
    text_mlp = add('text_mlp', 'mlp', [text_post_norm,
                        'w:mlp.gate_proj.weight', 'w:mlp.up_proj.weight', 'w:mlp.down_proj.weight'],
                   lifetime='prefix')
    text_output = add('text_output', 'add', [text_residual, text_mlp], lifetime='prefix')

    # --- Image MLP: `mlp` op on flattened 2D input (1 node instead of 5) ---
    image_post_flat = add('image_post_flat', 'reshape', [image_post_norm], shape=[512, 'hidden'])
    image_mlp_flat = add('image_mlp_flat', 'mlp', [image_post_flat,
                         'w:vision_branch_mlp.gate_proj.weight',
                         'w:vision_branch_mlp.up_proj.weight',
                         'w:vision_branch_mlp.down_proj.weight'])
    image_mlp = add('image_mlp', 'reshape', [image_mlp_flat], shape=['batch', 256, 'hidden'])
    image_output = add('image_output', 'add', [image_residual, image_mlp])

    # --- Final output: concat text + image ---
    output = add('output', 'concat', [text_output, image_output], dim=1)

    program = {'schema': 'bifrost-tensor-dag/1', 'layer_body': nodes, 'layer_output': output}
    return program

if __name__ == '__main__':
    path = Path('program.json')
    pending = path.with_suffix('.pending.json')
    program = build_program()
    pending.write_text(json.dumps(program, indent=2) + '\n')
    pending.replace(path)
    print(f"Generated program with {len(program['layer_body'])} nodes")
