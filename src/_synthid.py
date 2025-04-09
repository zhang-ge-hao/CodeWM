import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

class SynthIDLogitsProcessor_withTemperature(SynthIDTextWatermarkLogitsProcessor):
    def __init__(self, *args, temperature, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def update_scores(self, scores: torch.FloatTensor, g_values: torch.FloatTensor) -> torch.FloatTensor:
        _, _, depth = g_values.shape

        probs = torch.softmax(scores / self.temperature, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.finfo(log_probs.dtype).min)
        return log_probs
