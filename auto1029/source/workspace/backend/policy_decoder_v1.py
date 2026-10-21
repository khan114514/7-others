"""Frozen target verifier with bounded IPC to an isolated draft policy.

Candidate code never runs in the GPU interpreter. All drafting, serialization,
sampling and discarded replay work is included in measured question latency.
This engine contains no canonical private draft policy.
"""
import time
import torch
from torchtune import utils
from torchtune.data import Message, padded_collate_tiled_images_and_mask, load_image
from torchtune.generation import sample
from run_inference_pilot import Encoder

class PolicyEncoder(Encoder):
    @torch.inference_mode()
    def generate_verified(self, image, prompt_text, rng_seed, use_draft, maximum=128, chunk=16):
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        messages = [Message(role="user", content=[{"type": "image", "content": load_image(str(image))}, {"type": "text", "content": prompt_text}]), Message(role="assistant", content="")]
        inputs = self.transform({"messages": messages}, inference=True)
        prompt_length = len(inputs["tokens"])
        if prompt_length > 1600:
            raise ValueError("Input token cap exceeded")
        total = prompt_length + maximum
        with self.device:
            self.model.setup_caches(batch_size=1, dtype=self.dtype, encoder_max_seq_len=self.transform.image_seq_len, decoder_max_seq_len=total)
        mask = torch.tril(torch.ones((total, total), dtype=torch.bool, device=self.device))
        position = torch.arange(total)
        batch = padded_collate_tiled_images_and_mask([inputs], pad_direction="left", pad_max_images=1, pad_max_tiles=self.transform.max_num_tiles)
        batch["encoder_mask"] = batch["encoder_mask"][:, :prompt_length]
        tokens_in = batch.pop("tokens").to(self.device)
        batch["mask"] = mask[None, :prompt_length]
        batch["input_pos"] = position[None, :prompt_length]
        utils.batch_to_device(batch, self.device)
        logits = self.model(tokens_in, **batch)[:, -1]
        batch.pop("encoder_input")
        image_mask = batch["encoder_mask"][:, -1:]
        # Identical distribution and random draw for each target position in both
        # paths; this avoids acceptance/rejection-dependent RNG consumption.
        noise = torch.empty((maximum, logits.shape[-1]), device=self.device, dtype=logits.dtype).exponential_(1)
        caches = [module for module in self.model.modules() if hasattr(module, "cache_pos") and hasattr(module, "k_cache")]
        generated = []
        guard_uncertain = False
        guard_positions = []
        forward_calls, drafted, accepted = 1, 0, 0
        policy_calls = 0
        policy_seconds = 0.0

        def draw(row, offset):
            nonlocal guard_uncertain
            value = int(sample(row, temperature=0.6, top_k=50, q=noise[offset:offset + 1]).item())
            if use_draft and drafted > 0:
                scaled = row / 0.6
                pivot = torch.topk(scaled, 50).values[:, -1:]
                scores = torch.where(scaled < pivot, -float("inf"), scaled) - torch.log(noise[offset:offset + 1])
                top = torch.topk(scores, 2).values[0]
                margin = float((top[0] - top[1]).item())
                if margin < 0.25:
                    guard_uncertain = True
                    guard_positions.append({"position": offset, "log_score_margin": margin})
            return value

        def forward(values, offset):
            nonlocal forward_calls
            start = prompt_length + offset
            batch["input_pos"] = position[None, start:start + len(values)]
            batch["mask"] = mask[None, start:start + len(values)]
            batch["encoder_mask"] = image_mask.expand(-1, len(values), -1)
            token_tensor = torch.tensor([values], dtype=torch.int32, device=self.device)
            forward_calls += 1
            return self.model(token_tensor, **batch)

        while len(generated) < maximum:
            if use_draft and guard_uncertain:
                break
            offset = len(generated)
            proposal = []
            if use_draft:
                cap = min(32, maximum - offset)
                public_state = {
                    "question": prompt_text,
                    "generated_token_ids": list(generated),
                    "decoded_text": self.transform.decode([v for v in generated if v <= 128015], skip_special_tokens=False),
                    "remaining_tokens": maximum - offset,
                    "maximum_draft_tokens": cap,
                }
                policy_started = time.perf_counter()
                raw_proposal = self.policy.draft(public_state)
                if isinstance(raw_proposal, dict):
                    proposal = self.transform.tokenizer.encode(raw_proposal['text'], add_bos=False, add_eos=False)[:cap]
                else:
                    proposal = raw_proposal
                policy_calls += 1
                policy_seconds += time.perf_counter() - policy_started
            if not proposal:
                value = draw(logits, offset)
                generated.append(value)
                if value in self.transform.stop_tokens or len(generated) == maximum:
                    break
                logits = forward([value], offset)[:, -1]
                continue

            old_positions = [cache.size for cache in caches]
            verified = forward(proposal, offset)
            drafted += len(proposal)
            matching = 0
            replacement = None
            terminal_accepted = False
            for index, value in enumerate(proposal):
                row = logits if index == 0 else verified[:, index - 1]
                target = draw(row, offset + index)
                if target != value:
                    replacement = target
                    break
                matching += 1
                if value in self.transform.stop_tokens:
                    terminal_accepted = True
                    break
            accepted += matching
            generated.extend(proposal[:matching])
            if matching == len(proposal) and not terminal_accepted:
                logits = verified[:, -1]
                continue

            # Only caches changed by this decoder pass are rewound; vision
            # cross-attention caches remain immutable after the image prefill.
            for cache, old in zip(caches, old_positions):
                advance = cache.size - old
                if advance:
                    if advance != len(proposal):
                        raise RuntimeError("Unexpected cache advancement")
                    cache.cache_pos -= len(proposal) - matching
            if terminal_accepted:
                break
            generated.append(replacement)
            if replacement in self.transform.stop_tokens or len(generated) == maximum:
                break
            logits = forward([replacement], offset + matching)[:, -1]
        torch.cuda.synchronize()
        text = self.transform.decode([token for token in generated if token <= 128015], skip_special_tokens=True)
        return {"output": text, "token_ids": generated, "generated_tokens": len(generated), "latency_seconds": time.perf_counter() - started, "target_forward_calls": forward_calls, "drafted_tokens": drafted, "accepted_draft_tokens": accepted, "rng_seed": rng_seed, "guard_uncertain": guard_uncertain, "guard_positions": guard_positions, "policy_calls": policy_calls, "policy_seconds": policy_seconds}



class GuardedPolicyEncoder(PolicyEncoder):
    @torch.inference_mode()
    def generate_verified(self, image, prompt_text, rng_seed, use_draft, maximum=128, chunk=16):
        started = time.perf_counter()
        first = super().generate_verified(image, prompt_text, rng_seed, use_draft, maximum, chunk)
        if use_draft and first["guard_uncertain"]:
            final = super().generate_verified(image, prompt_text, rng_seed, False, maximum, chunk)
            final["guard_replay"] = True
            final["guard_positions"] = first["guard_positions"]
            final["discarded_speculative_tokens"] = first["generated_tokens"]
            final["total_sampled_tokens"] = first["generated_tokens"] + final["generated_tokens"]
            final["target_forward_calls"] += first["target_forward_calls"]
            final["policy_calls"] += first["policy_calls"]
            final["policy_seconds"] += first["policy_seconds"]
            final["drafted_tokens"] = first["drafted_tokens"]
            final["accepted_draft_tokens"] = first["accepted_draft_tokens"]
        else:
            final = first
            final["guard_replay"] = False
            final["discarded_speculative_tokens"] = 0
            final["total_sampled_tokens"] = final["generated_tokens"]
        final["latency_seconds"] = time.perf_counter() - started
        if final["total_sampled_tokens"] > 256 or final["target_forward_calls"] > 256:
            raise ValueError("Shared total inference budget exceeded")
        return final
