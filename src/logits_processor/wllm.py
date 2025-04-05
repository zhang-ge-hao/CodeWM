from __future__ import annotations
import collections
from math import sqrt
import pdb
import scipy.stats
from typing import List

import torch
from torch import Tensor
from transformers import LogitsProcessor


class WLLMLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab: list[int], gamma: float, delta: float, eos_token_id):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self.ngram_len = 5
        self.green_size = int(self.vocab_size * self.gamma)
        self.eos_token_id = eos_token_id

    def hash_integer_array_to_int(self, arr):
        import hashlib
        arr_str = ','.join(map(str, arr))
        hash_obj = hashlib.sha3_512(arr_str.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int & 0xFFFFFFFFFFFFFFFF

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        assert input_ids.shape[-1] >= self.ngram_len - 1, f"Requires at least a {
            self.ngram_len - 1} token prefix sequence to seed rng"
        prev_tokens = [input_ids[-i].item() for i in range(1, self.ngram_len)]
        seed = self.hash_integer_array_to_int(prev_tokens)
        self.rng.manual_seed(seed)

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids, self.hash_key)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        if self.select_green_tokens: # directly
            greenlist_ids = vocab_permutation[:greenlist_size] # new
        else: # select green via red
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
        return greenlist_ids
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator()

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores[green_tokens_mask] = scores[green_tokens_mask] + self.delta
        return scores

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def compute_z_score(
        self,
        input_ids: Tensor,
        prefix_len: int,
        entropy: list[float]=None,
    ):
        assert len(input_ids.size()) == 1

        prefix_len = max(self.ngram_len - 1, prefix_len)

        num_tokens_scored = len(input_ids) - prefix_len
        if num_tokens_scored < 1:
            print(f"only {num_tokens_scored} scored : cannot score.")
            return 0
        
        # Standard method.
        # Since we generally need at least 1 token (for the simplest scheme)
        # we start the iteration over the token sequence with a minimum 
        # num tokens as the first prefix for the seeding scheme,
        # and at each step, compute the greenlist induced by the
        # current prefix and check if the current token falls in the greenlist.
        green_token_count, green_token_mask = 0, []
        for idx in range(prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)

        return self._compute_z_score(green_token_count, num_tokens_scored)
