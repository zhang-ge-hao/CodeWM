"""Audited ACW-s generation and detection primitives.

The implementation follows ``models/sweetcode.py`` and ``models/wllm.py``
from TimeLovercc/code-watermark commit 4291a8e07abda0bc20560626f98340a90059be67.
ACW-s uses the source model's logits for both entropy gating and pairwise
green/red partitioning.  The previous token and a secret key seed the PRNG.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import torch
from transformers import LogitsProcessor


def entropy_from_logits(scores: torch.Tensor) -> torch.Tensor:
    """Return categorical entropy for the last dimension of ``scores``."""

    probabilities = torch.softmax(scores, dim=-1)
    terms = torch.where(
        probabilities > 0,
        probabilities * probabilities.log(),
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=-1)


class ACWSBase:
    """Shared deterministic pairwise partition used at generation/detection."""

    def __init__(
        self,
        *,
        vocab_size: int,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        entropy_threshold: float = 1.2,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if not math.isclose(gamma, 0.5):
            raise ValueError(
                "ACW-s pairwise partitioning has an effective gamma of 0.5"
            )
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.entropy_threshold = entropy_threshold
        self._rng: torch.Generator | None = None
        self._rng_device: torch.device | None = None

    def _ensure_rng(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng_device = device
        return self._rng

    def _seed_rng(self, prefix_ids: torch.Tensor, device: torch.device) -> None:
        if prefix_ids.numel() < 1:
            raise ValueError("ACW-s requires at least one prefix token")
        previous_token = int(prefix_ids[-1].item())
        self._ensure_rng(device).manual_seed(self.hash_key * previous_token)

    def greenlist_ids(
        self,
        prefix_ids: torch.Tensor,
        source_scores: torch.Tensor,
    ) -> list[int]:
        """Select one token from each adjacent source-logit rank pair.

        The odd final vocabulary item follows the official implementation: a
        final PRNG bit decides whether it is included in the green list.
        """

        if source_scores.ndim != 1:
            raise ValueError("source_scores must be a one-dimensional vector")
        if source_scores.shape[0] != self.vocab_size:
            raise ValueError(
                f"expected {self.vocab_size} scores, got {source_scores.shape[0]}"
            )

        device = source_scores.device
        self._seed_rng(prefix_ids, device)
        sorted_indices = torch.argsort(source_scores, descending=True)
        pair_count = (self.vocab_size + 1) // 2
        choices = torch.randint(
            0,
            2,
            (pair_count,),
            generator=self._rng,
            device=device,
        )

        pair_starts = torch.arange(0, self.vocab_size - 1, 2, device=device)
        selected_ranks = pair_starts + choices[: pair_starts.numel()]
        if self.vocab_size % 2 == 1 and bool(choices[-1].item()):
            selected_ranks = torch.cat(
                [selected_ranks, selected_ranks.new_tensor([self.vocab_size - 1])]
            )
        return sorted_indices[selected_ranks].cpu().tolist()

    def _z_score(self, green_count: int, scored_count: int) -> float:
        if scored_count <= 0:
            return 0.0
        expected = self.gamma * scored_count
        denominator = math.sqrt(
            scored_count * self.gamma * (1.0 - self.gamma)
        )
        return (green_count - expected) / denominator


class ACWSLogitsProcessor(ACWSBase, LogitsProcessor):
    """Apply the ACW-s bias at source-model high-entropy positions."""

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must both be batched tensors")
        if input_ids.shape[0] != scores.shape[0]:
            raise ValueError("input_ids and scores must have the same batch size")

        high_entropy = entropy_from_logits(scores) > self.entropy_threshold
        for batch_index in range(input_ids.shape[0]):
            if not bool(high_entropy[batch_index].item()):
                continue
            green_ids = self.greenlist_ids(
                input_ids[batch_index], scores[batch_index]
            )
            scores[batch_index, green_ids] += self.delta
        return scores


class ACWSDetector(ACWSBase):
    """Reconstruct ACW-s partitions from source-model logits and score tokens."""

    def __init__(self, *, z_threshold: float = 4.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.z_threshold = z_threshold

    def detect(
        self,
        *,
        token_ids: torch.Tensor,
        prefix_len: int,
        entropies: torch.Tensor | list[float],
        source_scores: torch.Tensor,
    ) -> dict[str, Any]:
        """Score a full prompt-plus-completion token sequence.

        ``source_scores[i]`` and ``entropies[i]`` must describe the distribution
        that predicted ``token_ids[i]``.  The row for the first token is a dummy.
        """

        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if source_scores.ndim != 2:
            raise ValueError("source_scores must have shape [tokens, vocabulary]")
        if source_scores.shape != (token_ids.shape[0], self.vocab_size):
            raise ValueError("source_scores are not aligned with token_ids")

        entropy_tensor = torch.as_tensor(
            entropies, device=source_scores.device, dtype=torch.float32
        )
        if entropy_tensor.shape != token_ids.shape:
            raise ValueError("entropies are not aligned with token_ids")

        prefix_len = max(1, prefix_len)
        generated_count = token_ids.shape[0] - prefix_len
        if generated_count < 1:
            return {"invalid": True}

        scored_count = 0
        green_count = 0
        green_mask: list[bool] = []
        for position in range(prefix_len, token_ids.shape[0]):
            if float(entropy_tensor[position].item()) <= self.entropy_threshold:
                green_mask.append(False)
                continue
            scored_count += 1
            green_ids = self.greenlist_ids(
                token_ids[:position], source_scores[position]
            )
            is_green = int(token_ids[position].item()) in green_ids
            green_count += int(is_green)
            green_mask.append(is_green)

        z_score = self._z_score(green_count, scored_count)
        p_value = NormalDist().cdf(-z_score)
        return {
            "num_tokens_generated": generated_count,
            "num_tokens_scored": scored_count,
            "num_green_tokens": green_count,
            "watermarking_fraction": scored_count / generated_count,
            "green_fraction": green_count / scored_count if scored_count else 0.0,
            "z_score": z_score,
            "p_value": p_value,
            "prediction": z_score > self.z_threshold,
            "green_token_mask": green_mask,
        }
