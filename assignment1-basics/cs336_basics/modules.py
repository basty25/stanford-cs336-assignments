import torch
import math
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
            ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        mean = 0
        std = math.sqrt(2 / (self.in_features + self.out_features))
        self.weight = torch.empty(self.out_features, self.in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3, b=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# class Embedding: