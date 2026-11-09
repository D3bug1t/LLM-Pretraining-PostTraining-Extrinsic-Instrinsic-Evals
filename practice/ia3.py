"""IA3 (Liu et al. 2022, "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning",
arXiv 2205.05638): learn one SCALING VECTOR per adapted layer - the smallest of all the adapters here.

IDEA: instead of adding a low-rank matrix, IA3 multiplies some activations element-wise by a learned vector l
("Infused Adapter by Inhibiting and Amplifying Inner Activations"). l starts at all ONES, so the model starts as
the pretrained one. Three places are rescaled in every transformer block:
    keys:     k = l_k * (W_k x)            (vector of size out_features of k_proj)
    values:   v = l_v * (W_v x)            (size out_features of v_proj)
    FFN:      the hidden activation going INTO the down projection:  down(l_ff * hidden)   (size d_ff)
So the first two scale the OUTPUT of a Linear, the third scales the INPUT of a Linear.
Parameters per block: 2 * kv_dim + d_ff - a few thousand numbers, versus ~100k+ for LoRA.

In our Phase 1 models the layer names are:
    Models 4 / 5 (modern):  blocks.N.attn.k_proj, blocks.N.attn.v_proj, blocks.N.mlp.down
    Model 2 (A-3):          blocks.N.attn.k_proj, blocks.N.attn.v_proj, blocks.N.ffn.net.2   (net = Linear, GELU, Linear)
    Model 3 (GRU) has no attention or FFN Linear layers, so IA3 does not apply to it.

Torch basics: same as lora.py, plus
  x * l                 -> broadcasting: a vector l of size d multiplies the last dim of x (..., d)
  name.endswith(s)      -> string suffix test
"""
import torch
import torch.nn as nn

IA3_TARGETS = {"attn.k_proj": "output", "attn.v_proj": "output", "mlp.down": "input", "ffn.net.2": "input"}


class IA3Linear(nn.Module):
    def __init__(self, base, where):
        """where = "output": return base(x) * l     (l has out_features entries)
           where = "input":  return base(x * l)     (l has in_features entries)
        base is frozen; l starts at ones."""
        super().__init__()
        # Step 1. self.base = base, frozen;  self.where = where
        # Step 2. size = base.out_features if where == "output" else base.in_features
        #         self.l = Parameter of ones(size)
        raise NotImplementedError

    def forward(self, x):
        # Step 1. "input":  return self.base(x * self.l)
        # Step 2. "output": return self.base(x) * self.l
        raise NotImplementedError


def inject_ia3(model):
    """Wrap every nn.Linear whose name ends with a key of IA3_TARGETS, with the matching `where`.
    Returns the list of wrapped names. (Same replacing trick as inject() in lora.py.)"""
    # Step 1. Collect (name, where) for every name, module in model.named_modules() that is an nn.Linear and
    #         whose name ends with one of the IA3_TARGETS keys. Collect first, replace afterwards.
    # Step 2. For each: parent = model.get_submodule(name.rsplit(".", 1)[0]); child = name.rsplit(".", 1)[1]
    #         setattr(parent, child, IA3Linear(getattr(parent, child), where))
    # Step 3. return the names
    raise NotImplementedError


if __name__ == "__main__":
    # A toy model with the same names as one block of Model 4/5.
    blk = nn.Module(); blk.attn = nn.Module(); blk.mlp = nn.Module()
    blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj = nn.Linear(32, 32), nn.Linear(32, 16), nn.Linear(32, 16)
    blk.mlp.up, blk.mlp.down = nn.Linear(32, 64), nn.Linear(64, 32)
    model = nn.Module(); model.blocks = nn.ModuleList([blk])

    x, h = torch.randn(3, 32), torch.randn(3, 64)
    before = (model.blocks[0].attn.k_proj(x), model.blocks[0].mlp.down(h))
    for p in model.parameters(): p.requires_grad_(False)
    print(inject_ia3(model))              # expect ['blocks.0.attn.k_proj', 'blocks.0.attn.v_proj', 'blocks.0.mlp.down']
    after = (model.blocks[0].attn.k_proj(x), model.blocks[0].mlp.down(h))
    print(all(torch.allclose(a, b) for a, b in zip(before, after)))     # expect True (l starts at ones)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # expect 96 (= 16 + 16 + 64)
