"""Building blocks of the modern decoder (Models 4 / 5): RMSNorm, SwiGLU MLP, pre-norm transformer block.

Torch basics you will need:
  nn.Parameter(torch.ones(dim))       -> a trainable tensor registered in the module
  x.pow(2).mean(-1, keepdim=True)     -> mean of squares over the last dim, keeping it as size 1 so it broadcasts
  torch.rsqrt(t)                      -> 1 / sqrt(t)
  x.float() / t.to(x.dtype)           -> convert to float32 / back to x's original dtype (matters with fp16 training)
  torch.nn.functional.silu(t)         -> SiLU activation: t * sigmoid(t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    """Zhang & Sennrich 2019 (arXiv 1910.07467): x / sqrt(mean(x^2) + eps) * weight.
    LayerNorm without the mean subtraction and without a bias: cheaper, works as well."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # Step 1. store eps; Step 2. self.weight = a trainable vector of ones, length dim.
        raise NotImplementedError

    def forward(self, x):
        """x: (..., dim) -> same shape. Compute the statistic in float32."""
        # Step 1. xf = x.float()
        # Step 2. normalised = xf * rsqrt(mean of xf squared over the last dim (keepdim) + eps)
        # Step 3. convert back to x.dtype, multiply by self.weight, return.
        raise NotImplementedError


class SwiGLU(nn.Module):
    """Shazeer 2020 (arXiv 2002.05202): down( silu(gate(x)) * up(x) ).
    Two parallel projections up to d_ff: one is passed through SiLU and acts as a GATE that scales the other."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Three nn.Linear(..., bias=False): self.gate and self.up (d_model -> d_ff), self.down (d_ff -> d_model).
        raise NotImplementedError

    def forward(self, x):
        # One line: the formula in the class docstring. (* between tensors is element-wise.)
        raise NotImplementedError


class ModernBlock(nn.Module):
    """Pre-norm block: normalise BEFORE each sub-layer, add the sub-layer's output to the residual stream x."""
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, max_seq_len):
        super().__init__()
        # Four sub-modules: self.attn_norm = RMSNorm(d_model), self.attn = GroupedQueryAttention(...),
        #                   self.mlp_norm = RMSNorm(d_model), self.mlp = SwiGLU(d_model, d_ff)
        raise NotImplementedError

    def forward(self, x, key_keep=None, causal=True):
        # Step 1. x = x + attention of the normalised x (pass key_keep and causal through)
        # Step 2. x = x + mlp of the normalised x (use mlp_norm here)
        # Step 3. return x
        raise NotImplementedError


if __name__ == "__main__":
    x = torch.randn(2, 5, 32)
    print(RMSNorm(32)(x).pow(2).mean(-1)[0])             # expect values close to 1
    print(ModernBlock(32, 4, 2, 64, 16)(x).shape)        # expect torch.Size([2, 5, 32])
