"""Real paired inference pilot using fixed model, images and compute caps.

Only expert-authored methods run here. This is not an isolation boundary for
untrusted candidate code and must not be reused as the formal agent harness.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from torchtune import training, utils
from torchtune.data import Message, padded_collate_tiled_images_and_mask, load_image
from torchtune.generation import sample
from torchtune.models.llama3_2_vision import lora_llama3_2_vision_11b, llama3_2_vision_transform
from torchtune.training import FullModelHFCheckpointer

from gpu_official_encoder_smoke import wait_for_idle
from mcq_metrics import parse_answer

from deployment_paths import PROJECT_ROOT as ROOT

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

class Encoder:
    def __init__(self):
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        base = ROOT / "assets/base-bf16"
        checkpointer = FullModelHFCheckpointer(checkpoint_dir=str(base), checkpoint_files=[f"model-{i:05d}-of-00009.safetensors" for i in range(1, 10)], output_dir=str(ROOT / "private/pilot-checkpointer"), model_type="LLAMA3_VISION")
        state = checkpointer.load_checkpoint()[training.MODEL_KEY]
        state.update(torch.load(ROOT / "assets/heima/encoder/main/adapter_0.pt", map_location="cpu", weights_only=True))
        with training.set_default_dtype(self.dtype), self.device:
            self.model = lora_llama3_2_vision_11b(decoder_trainable="lora", encoder_trainable="frozen", fusion_trainable="lora", lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"], apply_lora_to_mlp=True, apply_lora_to_output=True, lora_rank=16, lora_alpha=32, lora_dropout=0.0, image_size=560)
        self.model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = llama3_2_vision_transform(path=str(base / "original/tokenizer.model"), image_size=560, max_seq_len=2048)

    @torch.inference_mode()
    def generate(self, image, prompt_text, maximum, rng_seed, protocol):
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        messages = [Message(role="user", content=[{"type": "image", "content": load_image(str(image))}, {"type": "text", "content": prompt_text}]), Message(role="assistant", content="")]
        inputs = self.transform({"messages": messages}, inference=True)
        length = len(inputs["tokens"])
        if length > protocol["maximum_public_question_tokens"]:
            raise ValueError("Input token cap exceeded")
        total = length + maximum
        with self.device:
            self.model.setup_caches(batch_size=1, dtype=self.dtype, encoder_max_seq_len=self.transform.image_seq_len, decoder_max_seq_len=total)
        mask = torch.tril(torch.ones((total, total), dtype=torch.bool, device=self.device))
        position = torch.arange(total)
        batch = padded_collate_tiled_images_and_mask([inputs], pad_direction="left", pad_max_images=1, pad_max_tiles=self.transform.max_num_tiles)
        batch["encoder_mask"] = batch["encoder_mask"][:, :length]
        tokens_in = batch.pop("tokens").to(self.device)
        batch["mask"] = mask[None, :length]
        batch["input_pos"] = position[None, :length]
        utils.batch_to_device(batch, self.device)
        started = time.time()
        logits = self.model(tokens_in, **batch)[:, -1]
        sampled_token = sample(logits, temperature=protocol["temperature"], top_k=protocol["top_k"])
        generated = [sampled_token.item()]
        probabilities = [float(torch.log_softmax(logits.float(), dim=-1)[0, sampled_token.item()].item())]
        batch.pop("encoder_input")
        batch["encoder_mask"] = batch["encoder_mask"][:, -1:]
        for _ in range(maximum - 1):
            if sampled_token.item() in self.transform.stop_tokens:
                break
            batch["input_pos"] = position[None, length]
            batch["mask"] = mask[None, length, None, :]
            logits = self.model(sampled_token, **batch)[:, -1]
            sampled_token = sample(logits, temperature=protocol["temperature"], top_k=protocol["top_k"])
            generated.append(sampled_token.item())
            probabilities.append(float(torch.log_softmax(logits.float(), dim=-1)[0, sampled_token.item()].item()))
            length += 1
        torch.cuda.synchronize()
        output = self.transform.decode([x for x in generated if x <= 128015], skip_special_tokens=True)
        return {"output": output, "generated_tokens": len(generated), "mean_logprob": sum(probabilities) / len(probabilities), "generation_seconds": time.time() - started, "rng_seed": rng_seed}

