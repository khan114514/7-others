"""Initial condition model and training recipe."""
import random
import numpy as np
import torch
from conditional_backend.model import ConditionModel

MODEL_CONFIG = {
    'kl_scale':0.5, 'hidden_dim':64, 'enc_dim':64, 'expand_dim':512,
    'enc_num_heads':4, 'dec_num_heads':4, 'enc_depth':3, 'dec_depth':3,
    'enc_affine':True, 'enc_norm_type':'ln', 'dec_norm_init':True,
    'dec_blk_type':'adaLN', 'gene_block_size':400, 'latent_block_size':400,
    'is_flash_attn':False,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(train, device, seed):
    return ConditionModel(adata=train, device=device, is_initialized=True, seed=seed,
                          condition_cols=['condition', 'cell_type'], **MODEL_CONFIG)

def train_model(model, seed):
    model.train(batch_size=1024, is_grad_clip=True, lr=0.0001, epoch=200, seed=seed,
                iter=3200, weight_decay=0.000025, is_lr_scheduler=True,
                col_msk_threshold=0.6, max_sample_mask_prob=None,
                is_col_wise_mask=False, loss_monitor=None)

def decode(model, controls, target, inference_seed):
    set_seed(inference_seed)
    prediction = model.transfer_cond(
        input_adata=controls,
        input_cond={'condition':'control', 'cell_type':'<mask>'},
        target_cond={'condition':target, 'cell_type':'<mask>'},
        batch_size=128, is_output_attn_weights=True, verbose=False)
    return prediction[0].X

def predict(model, controls, scaler, inference_seed):
    return scaler.inverse_transform(decode(model, controls, 'stimulated', inference_seed))

# Expert-private conditional residual inference; independent training unchanged.
def predict(model, controls, scaler, inference_seed):
    raw = scaler.inverse_transform(controls.X)
    raw = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
    target = scaler.inverse_transform(decode(model, controls, 'stimulated', inference_seed))
    control = scaler.inverse_transform(decode(model, controls, 'control', inference_seed))
    return np.maximum(raw + target - control, 0)
