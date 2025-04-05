import math
import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

class SynthIDLogitsProcessor(SynthIDTextWatermarkLogitsProcessor):
    def default_config_dict(self, ngram_len=5) -> dict:
        return {
            # This corresponds to H=4 context window size in the paper.
            "ngram_len": ngram_len,
            "keys": [673, 197, 281, 206, 634, 513, 697, 187, 876, 555, 
                     837, 271, 897, 455, 314, 494, 236, 539, 394, 414, 
                     531, 108, 285, 596, 820, 219, 312, 183, 392, 972],
            "sampling_table_size": 2**16,
            "sampling_table_seed": 0,
            "context_history_size": 1024,
        }

    def __init__(self, **kwargs):
        config_dict = self.default_config_dict()
        super().__init__(**kwargs, **config_dict)

    def compute_z_score(self, input_ids: torch.Tensor, prefix_len: int, 
                        entropy: list[float]=None,
                        coinflip_prob: float = 0.5) -> float:
        assert len(input_ids.size()) == 1
        output_ids = input_ids[prefix_len - (self.ngram_len - 1): ].unsqueeze(0)
        
        observed_g_tensor = self.compute_g_values(output_ids) 
        observed_g_tensor = observed_g_tensor.reshape(-1)

        x = observed_g_tensor.float().sum().item()
        n = observed_g_tensor.size(-1)
        p = coinflip_prob
        exp = n * p
        std = math.sqrt(n * p * (1 - p))
        z_score = (x - exp) / std
        return z_score
