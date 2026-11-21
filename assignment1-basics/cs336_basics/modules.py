import torch
import math
from torch import nn
from einops import rearrange, einsum, repeat


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
        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3*std, b=3*std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int, # vocab size
            embedding_dim: int, # d_model
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        empty_embed = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.embedding = nn.Parameter(
            torch.nn.init.trunc_normal_(empty_embed, mean=0, std=1, a=-3, b=3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5,
            device:  torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch sequence d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(1/self.d_model * (x**2).sum(-1, keepdim=True) + self.eps)
        result = x / RMS * self.gain
        return result.to(in_dtype)


class SiLU(nn.Module):
    def __init__(
            self, 
            device: torch.device | None,
            dtype: torch.dtype | None
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# unfinished
# class GLU(nn.Module):
#     def __init__(
#             self, 
#             d_model: int,
#             d_ff: int,
#             device: torch.device | None,
#             dtype: torch.dtype | None
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.device = device
#         self.dtype = dtype

#         empty_weight_1 = torch.empty((d_ff,  d_model), device=device, dtype=dtype)
#         empty_weight_2 = torch.empty((d_ff,  d_model), device=device, dtype=dtype)
#         self.weight_1 = nn.Parameter(empty_weight_1)
#         self.weight_2 = nn.Parameter(empty_weight_2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         w1_x = einsum(x, self.weight_1, "... d_model, d_ff d_model -> ... d_ff")
#         w2_x = 



class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):

        super().__init__()
        self.d_model = d_model
        self.d_ff = 64 * round(d_ff / 64) # ensure d_ff is rounded to nearest multiple of 64

        # use same init as in Linear because in the pdf nothing is mentioned
        mean = 0
        std = math.sqrt(2 / (self.d_model + self.d_ff))
        empty_w1 = torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype)
        empty_w2 = torch.empty((self.d_model, self.d_ff), device=device, dtype=dtype)
        empty_w3 = torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype)
        self.w1_weights = nn.Parameter(torch.nn.init.trunc_normal_(empty_w1, mean=mean, std=std, a=-3*std, b=3*std))
        self.w2_weights = nn.Parameter(torch.nn.init.trunc_normal_(empty_w2, mean=mean, std=std, a=-3*std, b=3*std))
        self.w3_weights = nn.Parameter(torch.nn.init.trunc_normal_(empty_w3, mean=mean, std=std, a=-3*std, b=3*std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # SiLU = x * torch.sigmoid(einsum(self.weight_1, x))
        w1_x = einsum(x, self.w1_weights, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.w3_weights, "... d_model, d_ff d_model -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)

        # res = self.weight_2 @ (silu_w1_x * w3_x)
        res = einsum(self.w2_weights, silu_w1_x * w3_x, "d_model d_ff, ... d_ff -> ... d_model")
        return res 


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        positions = torch.arange(0, max_seq_len, device=device)
        ks = torch.arange(1, (d_k // 2) + 1, device=device)
        denom = self.theta ** ((2 * ks - 2) / d_k)
        thetas = positions[:, None] / denom[None, :]

        sines = torch.sin(thetas).to(device) # (max_seq_len, d_k/2)
        cosines = torch.cos(thetas).to(device)

        self.register_buffer("sin", sines, persistent=False)
        self.register_buffer("cos", cosines, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]

        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        left = even * cos - odd * sin
        right = even * sin + odd * cos

        return rearrange([left, right], "two ... d -> ... (d two)")