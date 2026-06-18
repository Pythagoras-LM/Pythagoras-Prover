"""
Block diffusion-style sampler for LLaDA2.1-MoE.

Mirrors the iterative-refinement generate logic from
`dllm.pipelines.llada21.models.modeling_llada21_moe`, but as a standalone sampler that follows the
BaseSampler interface (similar to llada2/sampler.py).

Key differences from LLaDA2:
- Uses a `while True` loop with post-step counting instead of a fixed step schedule.
- Adds an editing phase: already-placed non-prompt tokens may be revised when the model's
  confidence in a new token exceeds `editing_threshold` and the token value has changed.
- Attention mask is log-transformed (additive 0/-inf bfloat16), matching the 2.1 forward pass.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput


def top_k_top_p(
    logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]
) -> torch.Tensor:
    """Filter logits with top-k / top-p; returns filtered logits."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(
            logits < min_values, torch.full_like(logits, float("-inf")), logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one token per position; returns sampled ids and their probabilities.
    """
    if temperature == 0.0:
        probs = F.softmax(logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        token_prob = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, token_prob

    logits = logits / temperature
    filtered = top_k_top_p(logits, top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
        *probs.shape[:-1]
    )
    token_prob = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, token_prob


@dataclass
class LLaDA21SamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    threshold: float = 0.95
    editing_threshold: float = 0.9
    max_post_steps: int = 16
    num_to_transfer: int = 1
    eos_early_stop: bool = False


@dataclass
class LLaDA21Sampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21SamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Block diffusion sampler with iterative editing that mirrors `LLaDA2MoeModelLM.generate`
        from the LLaDA2.1 model. Currently supports equal-length prompts.

        Because confidence-based sampling causes each sequence to unmask and edit at its own
        pace, different batch items diverge in progress per step, so batch size must be 1.

        Args:
            inputs: Length-1 list containing a single prompt tensor or list of token IDs.
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences and histories, or raw tensor if return_dict=False.
        """
        if config is None:
            config = LLaDA21SamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        threshold = kwargs.get("threshold", config.threshold)
        editing_threshold = kwargs.get("editing_threshold", config.editing_threshold)
        max_post_steps = kwargs.get("max_post_steps", config.max_post_steps)
        num_to_transfer = kwargs.get("num_to_transfer", config.num_to_transfer)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        return_dict = kwargs.get("return_dict", config.return_dict)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # Normalize inputs
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "LLaDA21Sampler expects all prompts to have the same length."
            )

        prompt_len = prompt_lens[0]
        B = len(inputs)
        if B != 1:
            raise ValueError(
                "LLaDA21Sampler requires batch_size=1 because confidence-based sampling "
                "causes sequences to diverge across steps."
            )

        if max_new_tokens:
            max_length = max_new_tokens + prompt_len
        else:
            max_new_tokens = max_length - prompt_len

        num_blocks = (max_length + block_size - 1) // block_size
        total_len = num_blocks * block_size

        # Block-causal, bidirectional-within-block attention mask.
        # Log-transformed to produce additive mask (0 for attended, -inf for masked).
        block_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=self.model.device)
        )
        block_attn = (
            (
                block_mask.repeat_interleave(block_size, dim=0)
                .repeat_interleave(block_size, dim=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .log()
            .to(torch.bfloat16)
        )

        position_ids = torch.arange(total_len, device=self.model.device).unsqueeze(0)

        # Canvas initialized with masks, prompts filled at the front
        x = torch.full(
            (B, total_len),
            mask_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for i, p in enumerate(inputs):
            x[i, :prompt_len] = p

        prompt_blocks = prompt_len // block_size
        histories = [x.clone()] if return_dict else None

        for blk in range(prompt_blocks, num_blocks):
            window_end = (blk + 1) * block_size
            # cur_x is a view into x; writes to cur_x propagate to x
            cur_x = x[:, :window_end]
            cur_attn = block_attn[:, :, :window_end, :window_end]
            cur_pos = position_ids[:, :window_end]

            block_start_pos = blk * block_size

            # Mark prompt positions within this block so they are never edited
            prompt_mask_in_block = torch.zeros(
                block_size, dtype=torch.bool, device=self.model.device
            )
            if block_start_pos < prompt_len:
                prompt_end_in_block = min(prompt_len - block_start_pos, block_size)
                prompt_mask_in_block[:prompt_end_in_block] = True

            post_steps = 0
            while True:
                old_block_tokens = cur_x[:, -block_size:].clone()
                active_mask = cur_x[:, -block_size:] == mask_id

                if not active_mask.any():
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                logits = self.model(
                    cur_x,
                    attention_mask=cur_attn,
                    position_ids=cur_pos,
                ).logits

                logits_block = logits[:, -block_size:, :]
                tokens, probs = sample_tokens(
                    logits_block, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # --- mask-token transfer ---
                transfer_index = torch.zeros_like(tokens, dtype=torch.bool)
                if active_mask.sum() > 0:
                    for b in range(B):
                        conf = torch.where(
                            active_mask[b],
                            probs[b],
                            torch.full_like(probs[b], -float("inf")),
                        )
                        high_conf = (conf > threshold) & active_mask[b]
                        if high_conf.sum().item() >= num_to_transfer:
                            transfer_index[b] = high_conf
                        else:
                            num_available = active_mask[b].sum().item()
                            if num_available > 0:
                                k = min(num_to_transfer, num_available)
                                _, idx = torch.topk(conf, k=k)
                                transfer_index[b, idx] = True

                # --- editing of already-placed non-prompt tokens ---
                editable_positions = (~active_mask) & (~prompt_mask_in_block[None, :])
                editing_conf = torch.where(
                    editable_positions,
                    probs,
                    torch.full_like(probs, -float("inf")),
                )
                high_conf_editing = (
                    editing_conf > editing_threshold
                ) & editable_positions
                token_changed = tokens != old_block_tokens
                editing_transfer_index = high_conf_editing & token_changed

                final_transfer_index = transfer_index | editing_transfer_index
                if final_transfer_index.any():
                    cur_x[:, -block_size:][final_transfer_index] = tokens[
                        final_transfer_index
                    ]

                if histories is not None:
                    histories.append(x.clone())

                if active_mask.sum() == 0 and not editing_transfer_index.any():
                    break

            if eos_early_stop and eos_id is not None:
                generated_part = x[0, prompt_len:window_end]
                if (generated_part == mask_id).sum() == 0:
                    eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        break

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21SamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
