"""LoRA (Hu et al. 2021, arXiv 2106.09685): additive low-rank update of a frozen Linear layer.

IDEA: the pretrained weight W0 (out x in) stays frozen. We learn a correction  delta_W = B @ A  where
A is (r x in) and B is (out x r) with a small rank r (e.g. 8): only r * (in + out) trainable numbers instead of
out * in. Output:  h = W0 x + (alpha / r) * B A x.
B starts at ZERO, so at step 0 the layer is exactly the pretrained one; A starts random so gradients can flow.

Torch basics you will need:
  nn.Parameter(torch.zeros(a, b))           -> trainable matrix of zeros
  torch.empty(a, b) + nn.init.kaiming_uniform_(t, a=math.sqrt(5))   -> random init (same as nn.Linear uses)
  base.in_features, base.out_features       -> sizes of an nn.Linear
  p.requires_grad_(False)                   -> freeze a parameter (or a whole module: module.requires_grad_(False))
  x @ M.t()                                 -> multiply the last dim of x by M transposed (that's what a Linear does)
  model.named_modules()                     -> yields (qualified_name, module), e.g. ("blocks.0.attn.q_proj", Linear)
  model.get_submodule("blocks.0.attn")      -> the module with that name
  setattr(parent, "q_proj", new_module)     -> replace a child module
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        """base: the pretrained nn.Linear (stays frozen). r: rank. alpha: scaling (update is scaled by alpha / r).
        The wrapped layer must give exactly base(x) before any training step."""
        super().__init__()
        # Step 1. Keep the base layer: self.base = base, and freeze it (requires_grad_(False)).
        # Step 2. self.scale = alpha / r
        # Step 3. self.A = Parameter of shape (r, in_features), random (kaiming_uniform_ on an empty tensor).
        # Step 4. self.B = Parameter of shape (out_features, r), zeros.
        self.base = base
        self.base.requires_grad_(False);
        self.scale = alpha/r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features,r))

        
        # raise NotImplementedError

    def forward(self, x):
        """x: (..., in_features) -> (..., out_features)."""
        # Step 1. frozen = self.base(x)
        # Step 2. low-rank path: first x @ A.t() -> (..., r), then that @ B.t() -> (..., out_features).
        #         Doing A first keeps it cheap: you never build the (out x in) matrix B @ A.
        # Step 3. return frozen + scale * low-rank path
        frozen = self.base(x)
        scale = self.scale
        low_rank_path = x @ self.A.t() @ self.B.t() # (batch, in) X (in, r) X (r, out)
        return frozen + scale*low_rank_path
        # raise NotImplementedError


def inject(model, cls, target_names, r, alpha):
    """Replace every nn.Linear whose qualified name (from model.named_modules()) is in target_names
    by cls(that_linear, r, alpha), in place. Returns the list of replaced names."""
    # Step 1. Collect the names first: [name for name, m in model.named_modules() if name in target_names and
    #         isinstance(m, nn.Linear)]. (Collect BEFORE replacing: changing modules while iterating breaks the loop.)
    # Step 2. For each name, e.g. "blocks.0.attn.q_proj":
    #         parent_name, child_name = name.rsplit(".", 1)      -> "blocks.0.attn", "q_proj"
    #         parent = model.get_submodule(parent_name); old = getattr(parent, child_name)
    #         setattr(parent, child_name, cls(old, r, alpha))
    #         (a top-level layer has no "." in its name: then the parent is the model itself)
    # Step 3. return the list of names
    names = [name for name, m in model.named_modules() if name in target_names and isinstance(m, nn.Linear)]
    for name in names:
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name)          # "" -> the model itself
        setattr(parent, child_name, cls(getattr(parent, child_name), r, alpha))
    return names


if __name__ == "__main__":
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    print(torch.allclose(lora(x), base(x)))                               # expect True (B starts at 0)
    print([n for n, p in lora.named_parameters() if p.requires_grad])     # expect ['A', 'B']
    lora(x).sum().backward()
    print(lora.B.grad.abs().sum().item() > 0)                             # expect True: B can learn (needs A random)

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    print(inject(model, LoRALinear, ["0", "2"], r=4, alpha=8), type(model[0]).__name__)   # expect ['0', '2'] LoRALinear
