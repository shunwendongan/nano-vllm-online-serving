import torch
from torch import nn
from nanovllm.layers.ops import silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
