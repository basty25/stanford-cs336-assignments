import torch
import math

from einops import reduce, einsum

from jaxtyping import Bool, Float, Int, jaxtyped


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    maximum, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - maximum
    return torch.exp(x) / torch.exp(x).sum(dim, keepdim=True)


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    ) -> Float[torch.Tensor, " ... queries d_v"]:

    d_k = Q.shape[-1]
    query_key = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    query_key = query_key / math.sqrt(d_k)

    if mask is not None:
        mask_to_add = torch.zeros_like(mask, dtype=torch.float)
        mask_to_add = torch.masked_fill(mask_to_add, ~mask, float('-inf'))
        query_key = query_key + mask_to_add

    soft = softmax(query_key, dim=-1)
    return einsum(soft, V, "... queries keys, ... keys d_v -> ... queries d_v")
