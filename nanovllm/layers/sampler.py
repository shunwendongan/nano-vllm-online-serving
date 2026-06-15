import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        sample_mask = temperatures != 0
        if not sample_mask.any():
            return greedy_tokens
        scaled_logits = logits[sample_mask] / temperatures[sample_mask].unsqueeze(dim=1)
        probs = torch.softmax(scaled_logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(scaled_logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        output_tokens = greedy_tokens.clone()
        output_tokens[sample_mask] = sample_tokens
        return output_tokens
