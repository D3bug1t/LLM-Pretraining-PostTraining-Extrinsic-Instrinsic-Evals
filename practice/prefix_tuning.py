"""Prefix Tuning (Li & Liang 2021, arXiv 2101.00190): learned KEY/VALUE vectors prepended inside EVERY attention layer.

IDEA: prompt tuning adds learned vectors only at the input. Prefix tuning goes deeper: in each layer l, the
attention gets n extra key vectors and value vectors that every real token can attend to:
        K_l = [PK_l ; W_k x]      V_l = [PV_l ; W_v x]        (queries are unchanged: only real tokens ask)
The model is frozen; only the prefixes are trained. So the output still has T positions (unlike prompt tuning),
but each query now has n + T keys to look at.

Reparameterisation (from the paper - training the prefixes directly is unstable): do not learn PK, PV directly.
Learn a small matrix E (n x d_model) and an MLP, and compute all prefixes of all layers from it:
        MLP(E): (n, d_model) -> (n, n_layers * 2 * n_kv_heads * d_head),  then reshape into per-layer K and V.
After training, the MLP output can be computed once and stored; the MLP itself is thrown away.

Notes for our models:
  * The prefix keys get NO RoPE: they have no position, they are "global" memory slots.
  * GQA: keys/values have n_kv heads, queries n_heads; each kv head is shared by n_heads // n_kv query heads,
    so repeat k and v along the head dim (repeat_interleave) before the dot product.
  * Model 3 (GRU) has no attention, so prefix tuning does not apply to it.

Torch basics you will need:
  t.view(a, b, c, d, e) / t.permute(1, 2, 3, 0, 4)    -> reshape / reorder dims
  t[None].expand(B, -1, -1, -1)                       -> repeat along a new batch dim without copying
  torch.cat([a, b], dim=2)                            -> join along the key (sequence) dim of (B, H, T, d)
  t.repeat_interleave(g, dim=1)                       -> each head repeated g times: [h0,h0,h1,h1,...]
  q @ k.transpose(-1, -2)                             -> (B, H, Tq, Tk) attention scores
  s.masked_fill(~keep, float("-inf"))                 -> forbid some keys
  torch.softmax(s, dim=-1)
"""
import torch
import torch.nn as nn


class PrefixKV(nn.Module):
    def __init__(self, n_layers, n_prefix, n_kv, d_head, d_model, hidden=256):
        super().__init__()
        # Step 1. Store n_layers, n_prefix, n_kv, d_head as attributes.
        # Step 2. self.E = Parameter(torch.randn(n_prefix, d_model) * 0.02)
        # Step 3. self.mlp = nn.Sequential(Linear(d_model, hidden), Tanh(), Linear(hidden, n_layers * 2 * n_kv * d_head))
        raise NotImplementedError

    def forward(self, B):
        """Returns a list with one (pk, pv) per layer, each of shape (B, n_kv, n_prefix, d_head)."""
        # Step 1. out = self.mlp(self.E)                                  -> (n_prefix, n_layers * 2 * n_kv * d_head)
        # Step 2. out = out.view(n_prefix, n_layers, 2, n_kv, d_head)
        #         out = out.permute(1, 2, 3, 0, 4)                        -> (n_layers, 2, n_kv, n_prefix, d_head)
        # Step 3. return [(out[l, 0][None].expand(B, -1, -1, -1), out[l, 1][None].expand(B, -1, -1, -1))
        #                 for l in range(n_layers)]
        raise NotImplementedError


def attend_with_prefix(q, k, v, pk, pv, key_keep, scale):
    """q (B, H, T, dh); k, v (B, Hkv, T, dh); pk, pv (B, Hkv, P, dh); key_keep (B, T) bool or None.
    Bidirectional attention over the P prefix keys + T real keys. Returns (B, H, T, dh)."""
    # Step 1. k = cat([pk, k], dim=2);  v = cat([pv, v], dim=2)                -> (B, Hkv, P + T, dh)
    # Step 2. GQA: if q.size(1) != k.size(1): g = q.size(1) // k.size(1); repeat_interleave k and v by g on dim 1.
    # Step 3. s = (q @ k.transpose(-1, -2)) * scale                           -> (B, H, T, P + T)
    # Step 4. Padding: if key_keep is not None:
    #           keep = cat([ones (B, P) bool, key_keep], dim=1)                -> (B, P + T)  (prefix always visible)
    #           s = s.masked_fill(~keep[:, None, None, :], float("-inf"))
    #         (No row can be all -inf: every query sees the prefix, so no NaN when P > 0.)
    # Step 5. return torch.softmax(s, dim=-1) @ v
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, Hkv, T, dh, P = 2, 4, 2, 5, 8, 3
    pre = PrefixKV(n_layers=2, n_prefix=P, n_kv=Hkv, d_head=dh, d_model=16)
    kv = pre(B)
    print(len(kv), kv[0][0].shape, kv[1][1].shape)          # expect 2 torch.Size([2, 2, 3, 8]) torch.Size([2, 2, 3, 8])

    q, k, v = torch.randn(B, H, T, dh), torch.randn(B, Hkv, T, dh), torch.randn(B, Hkv, T, dh)
    # With an EMPTY prefix it must be plain attention:
    empty = torch.zeros(B, Hkv, 0, dh)
    ref = torch.softmax(q @ k.repeat_interleave(2, 1).transpose(-1, -2) * dh ** -0.5, -1) @ v.repeat_interleave(2, 1)
    print(torch.allclose(attend_with_prefix(q, k, v, empty, empty, None, dh ** -0.5), ref, atol=1e-6))   # expect True

    keep = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    out = attend_with_prefix(q, k, v, kv[0][0], kv[0][1], keep, dh ** -0.5)
    print(out.shape, torch.isfinite(out).all().item())     # expect torch.Size([2, 4, 5, 8]) True
    v2 = v.clone(); v2[1, :, 3:] = 100.0                    # change the PADDED values of item 1 ...
    out2 = attend_with_prefix(q, k, v2, kv[0][0], kv[0][1], keep, dh ** -0.5)
    print(torch.allclose(out, out2))                        # ... expect True: padding is never attended
    out.sum().backward()
    print(pre.E.grad is not None)                           # expect True: gradient reaches the prefix
