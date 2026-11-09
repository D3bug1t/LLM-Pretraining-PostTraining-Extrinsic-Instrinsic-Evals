"""LoRMA (Bihany, Patel & Modi 2025, arXiv 2506.07621): multiplicative low-rank update of a frozen Linear layer.

IDEA: LoRA ADDS a low-rank matrix:        W = W0 + B A.
      LoRMA MULTIPLIES W0 by one:          W = (I + (alpha / r) B A) W0       (the "+" variant, LoRMA+)
Here the low-rank matrix acts on the OUTPUT side of W0, so A is (r x out) and B is (out x r): BA is (out x out).
The identity I keeps the start at W0 when B = 0 (a pure product B A W0 would start at the zero matrix).

Compute it cheaply, never building the (out x out) matrix:
      z = W0 x                 (the frozen layer, without its bias)
      h = z + (alpha / r) * B (A z)
      then add the bias of the base layer, if it has one.

Torch basics: same as lora.py, plus
  torch.nn.functional.linear(x, W)    -> x @ W.t() (a Linear layer without bias)
  base.bias                           -> the bias tensor, or None if the layer has no bias
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRMALinear(nn.Module):
    """LoRMA+ variant: W = (I + (alpha / r) B A) W0."""
    def __init__(self, base, r, alpha):
        """base: the pretrained nn.Linear (stays frozen). r: rank. alpha: scaling.
        The wrapped layer must give exactly base(x) before any training step."""
        super().__init__()
        # Step 1. self.base = base, frozen.  Step 2. self.scale = alpha / r
        # Step 3. self.A: Parameter (r, OUT_features), random (kaiming_uniform_).   <- note: out, not in (unlike LoRA)
        # Step 4. self.B: Parameter (out_features, r), zeros.
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r,self.base.out_features))
        nn.init.kaiming_uniform_(self.A, a = math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.base.out_features,r))
        self.scale = alpha/r
        # raise NotImplementedError

    def forward(self, x):
        """x: (..., in_features) -> (..., out_features). Do not build the (out x out) matrix explicitly."""
        # Step 1. z = F.linear(x, self.base.weight)                    -> (..., out)   (no bias yet!)
        # Step 2. z = z + scale * ((z @ A.t()) @ B.t())                  -> (..., out)
        # Step 3. if self.base.bias is not None: add it. Return.
        # Why the bias waits: the paper multiplies the WEIGHT matrix W0, so the bias must not be scaled too.
        z = F.linear(x, self.base.weight)
        z = z + self.scale*(z @ self.A.t() @ self.B.t())
        if self.base.bias is not None:
            z = z + self.base.bias

        return z
        # raise NotImplementedError


if __name__ == "__main__":
    base = nn.Linear(16, 8)
    m = LoRMALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    print(torch.allclose(m(x), base(x)))                                  # expect True (B starts at 0)
    with torch.no_grad(): m.B.normal_()                                   # now check against the explicit formula
    W = (torch.eye(8) + m.scale * m.B @ m.A) @ base.weight
    print(torch.allclose(m(x), x @ W.t() + base.bias, atol=1e-5))         # expect True
