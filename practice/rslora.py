"""rsLoRA (Kalajdzievski 2023, arXiv 2312.03732): LoRA with a different scaling factor. Do lora.py first.

IDEA: LoRA scales the update by alpha / r. The paper shows that with this scaling the size of the update
B A x SHRINKS as the rank r grows (roughly like 1 / sqrt(r)), so a large rank learns slowly and gives no benefit.
Using  alpha / sqrt(r)  instead keeps the update the same size for every rank ("rank-stabilised").
That is the whole change: everything else (A random, B zero, frozen base) is exactly LoRA.

So this file is a 2-line exercise in INHERITANCE: reuse LoRALinear and only change self.scale.

Python basics you will need:
  class Child(Parent):                 -> Child gets all methods of Parent
  super().__init__(base, r, alpha)     -> run the Parent's __init__ first (builds base, A, B, scale)
  math.sqrt(r)                         -> square root
"""
import math
import torch
import torch.nn as nn
from lora import LoRALinear


class RSLoRALinear(LoRALinear):
    def __init__(self, base, r, alpha):
        """Same arguments as LoRALinear. Only the scale differs: alpha / sqrt(r) instead of alpha / r."""
        # Step 1. Call the parent's __init__ with the same three arguments (this builds self.base, self.A, self.B).
        # Step 2. Overwrite self.scale with alpha / math.sqrt(r).
        # (No forward needed: LoRALinear.forward already uses self.scale.)
        super().__init__(base,r,alpha)
        self.scale = alpha/math.sqrt(r)
        # raise NotImplementedError


if __name__ == "__main__":
    base = nn.Linear(64, 32)
    print(RSLoRALinear(base, r=4, alpha=8).scale, RSLoRALinear(base, r=16, alpha=8).scale)   # expect 4.0 2.0
    x = torch.randn(5, 64)
    print(torch.allclose(RSLoRALinear(base, 4, 8)(x), base(x)))                            # expect True (B starts at 0)

    # Why it matters: give B random values (as if training had moved it) and measure the update size at two ranks.
    torch.manual_seed(0)
    x = torch.randn(256, 512); big = nn.Linear(512, 512)
    def update_size(cls, r):
        m = cls(big, r, alpha=16)
        nn.init.normal_(m.B, std=0.02)
        return (m(x) - big(x)).norm().item()
    for cls in (LoRALinear, RSLoRALinear):
        print(cls.__name__, round(update_size(cls, 64) / update_size(cls, 4), 2))
    # expect roughly: LoRALinear 0.25 (update shrinks 4x when r grows 16x)   RSLoRALinear 1.0 (same size)
