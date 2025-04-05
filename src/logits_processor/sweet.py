import torch
from . import WLLMLogitsProcessor


class SweetLogitsProcessor(WLLMLogitsProcessor):
    def __init__(self, vocab: list[int], gamma: float, delta: float, entropy_threshold: float, eos_token_id):
        super().__init__(vocab=vocab, gamma=gamma, 
                         delta=delta, eos_token_id=eos_token_id)
        self.entropy_threshold = entropy_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator()

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        raw_probs = torch.softmax(scores, dim=-1)  # batch_size, vocab_size
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.entropy_threshold).view(-1, 1)
        
        green_tokens_mask = green_tokens_mask * entropy_mask

        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores
    
    def compute_z_score(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list[float],
    ):
        prefix_len = max(self.ngram_len - 1, prefix_len)

        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            print(f"only {num_tokens_generated} generated : cannot score.")
            return 0

        assert len(entropy) == len(input_ids)

        num_tokens_scored = num_tokens_generated - len(
            [e for e in entropy[prefix_len:] if e <= self.entropy_threshold]
        )

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

            if entropy[idx] > self.entropy_threshold:
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
            else:
                # when entropy is low; i.e., watermarking is not applied
                green_token_mask.append(False)

        return self._compute_z_score(green_token_count, num_tokens_scored)