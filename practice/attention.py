"""Causal multi-head self-attention and grouped-query attention (Ainslie et al. 2023, arXiv 2305.13245),
structured as in Assignment 3 (Q1 CausalSelfAttention, Q2 MHA/GQA/MQA via n_kv_heads, Q3 RoPE).

IDEA: every token makes a query q ("what am I looking for"), a key k ("what do I contain") and a value v
("what do I hand over"). Token t's output = weighted average of all values, weights = softmax(q_t . k_s * scale).
Multi-head: do this H times in parallel with smaller vectors (d_head = d_model / H), each head can look for
something different, then glue the heads back together.
Causal: token t may only look at tokens 0..t (a language model must not see the future).
GQA: several query heads SHARE one key/value head -> fewer K/V parameters and a smaller KV cache.

Shapes convention: B = batch, T = tokens, H = heads, d = d_head.
Torch basics you will need:
  q @ k.transpose(-2, -1)      -> matrix product over the last two dims: (.., T, d) @ (.., d, T) = (.., T, T)
  t.masked_fill(mask, value)   -> put `value` wherever mask is True
  torch.softmax(t, dim=-1)     -> softmax over the last dimension
  torch.tril(torch.ones(T, T, dtype=torch.bool))   -> lower-triangular True matrix (position t sees 0..t)
  t.view(B, T, H, d)           -> reshape without copying;  t.transpose(1, 2) swaps dims 1 and 2
  t.reshape(B, T, H * d)       -> like view but works after a transpose (view may complain there)
  t.repeat_interleave(n, dim=1)  -> repeat every slice along dim 1 n times: [a, b] -> [a, a, b, b] for n=2
  self.register_buffer("name", tensor, persistent=False)   -> a tensor stored in the module, moved by .to(device),
                                                             not trained and not saved in the checkpoint
"""
import math
import torch
import torch.nn as nn
from rope import rope_cache, apply_rope


def scaled_dot_product(q, k, v, scale = None, mask=None):
    """q, k, v: (B, H, T, d). scale: number the scores are multiplied by.
    mask: (T, T) bool, True = allowed, or None = every query sees every key. The function does not build any mask:
    the caller (the attention class) decides what is allowed and passes it in. Returns (B, H, T, d)."""
    # Step 1. scores = (q @ k.transpose(-2, -1)) * scale                                    -> (B, H, T, T)
    # Step 2. If mask is not None: scores = scores.masked_fill(~mask, float("-inf"))
    #         (-inf before the softmax -> weight exactly 0; it broadcasts over B and H)
    # Step 3. weights = torch.softmax(scores, dim=-1)                                        -> (B, H, T, T)
    # Step 4. return weights @ v 
    scores = (q @ k.transpose(-2,-1)) * scale if scale else (q @ k.transpose(-2,-1))*1/math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ v

    # raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0, attention_scale=None, use_rope=False):
        """attention_scale: None -> the standard 1 / sqrt(d_head) (the Q1.4 hook).
        use_rope: rotate q and k with RoPE (Q3; needs rope.py done)."""
        super().__init__()
        # Step 1. assert d_model % n_heads == 0; store self.n_heads, self.d_head = d_model // n_heads, self.use_rope.
        # Step 2. self.attention_scale = 1 / math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        # Step 3. self.q_proj, self.k_proj, self.v_proj, self.out_proj = nn.Linear(d_model, d_model, bias=False) each.
        # Step 4. self.resid_dropout = nn.Dropout(dropout)
        # Step 5. Causal mask, built once: mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        #         (True = allowed) and self.register_buffer("causal_mask", mask, persistent=False).
        #         (Register it directly; do not first assign self.causal_mask = ..., register_buffer then complains.)
        # Step 6. If use_rope: cos, sin = rope_cache(max_seq_len, self.d_head); register both as buffers
        #         "rope_cos", "rope_sin" (persistent=False).
        assert d_model % n_heads ==0
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.use_rope = use_rope
        self.attention_scale = 1/math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias = False)
        self.resid_dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(max_seq_len,max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask",causal_mask,persistent=False)
        if use_rope==True:
            cos, sin = rope_cache(max_seq_len,self.d_head)
            self.register_buffer("rope_cos",cos, persistent=False)
            self.register_buffer("rope_sin",sin, persistent=False)
    


        # raise NotImplementedError

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        # Step 1. B, T, C = x.shape
        # Step 2. q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)               -> each (B, T, C)
        # Step 3. Split into heads: .view(B, T, self.n_heads, self.d_head).transpose(1, 2)  -> (B, H, T, d)
        # Step 4. If use_rope: cos = self.rope_cos[:T][None, None] (same for sin)          -> (1, 1, T, d)
        #         q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   (NOT v: values carry content, not position)
        # Step 5. The class builds the mask: mask = self.causal_mask[:T, :T]  (top-left T x T corner)
        # Step 6. out = scaled_dot_product(q, k, v, self.attention_scale, mask)            -> (B, H, T, d)
        # Step 7. Merge heads: .transpose(1, 2) -> (B, T, H, d), then .reshape(B, T, C)
        # Step 8. return self.resid_dropout(self.out_proj(out))
        # (A3 also had attention dropout on the softmax weights; with dropout=0 it changes nothing, so we skip it.)
        B,T,C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.d_head).transpose(1,2)

        if self.use_rope==True:
            q = apply_rope(q,self.rope_cos[:T][None,None],self.rope_sin[:T][None,None])
            k = apply_rope(k,self.rope_cos[:T][None,None],self.rope_sin[:T][None,None])

        mask = self.causal_mask[:T,:T]
        out = scaled_dot_product(q, k, v, self.attention_scale, mask)
        out = out.transpose(1,2).reshape(B,T,C)
        return self.resid_dropout(self.out_proj(out))
        # raise NotImplementedError


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, dropout=0.0, attention_scale=None, use_rope=False):
        """n_heads query heads share n_kv_heads key/value heads (n_heads % n_kv_heads == 0).
        n_kv_heads == n_heads is MHA; n_kv_heads == 1 is multi-query attention (MQA)."""
        super().__init__()
        # Same as CausalSelfAttention.__init__, with these differences:
        #   * assert n_heads % n_kv_heads == 0; store self.n_kv_heads and self.n_rep = n_heads // n_kv_heads
        #     (how many query heads share each kv head)
        #   * k_proj and v_proj are SMALLER: nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        #     (q_proj and out_proj stay d_model -> d_model). This is where the parameter / KV-cache saving comes from.
        assert n_heads % n_kv_heads == 0
        assert d_model % n_heads ==0
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.n_rep = n_heads//n_kv_heads
        self.d_head = d_model//n_heads
        self.use_rope = use_rope
        self.attention_scale = 1/math.sqrt(self.d_head) if attention_scale is None else float(attention_scale)
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.d_head, bias = False)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.d_head, bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias = False)
        self.resid_dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(max_seq_len,max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask",causal_mask,persistent=False)
        if use_rope==True:
            cos, sin = rope_cache(max_seq_len,self.d_head)
            self.register_buffer("rope_cos",cos, persistent=False)
            self.register_buffer("rope_sin",sin, persistent=False)

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        # Same as CausalSelfAttention.forward, except:
        #   * k and v are split into self.n_kv_heads heads (not n_heads)                  -> (B, H_kv, T, d)
        #   * after RoPE and BEFORE scaled_dot_product, repeat each kv head for its group of query heads:
        #     k = k.repeat_interleave(self.n_rep, dim=1), same for v                      -> (B, H, T, d)
        #     (query heads 0..n_rep-1 use kv head 0, the next n_rep use kv head 1, ...)
        B,T,C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = k.view(B,T,self.n_kv_heads,self.d_head).transpose(1,2)
        v = v.view(B,T,self.n_kv_heads,self.d_head).transpose(1,2)

        if self.use_rope==True:
            q = apply_rope(q,self.rope_cos[:T][None,None],self.rope_sin[:T][None,None])
            k = apply_rope(k,self.rope_cos[:T][None,None],self.rope_sin[:T][None,None])

        mask = self.causal_mask[:T,:T]
        k = k.repeat_interleave(self.n_rep, dim = 1)
        v = v.repeat_interleave(self.n_rep, dim = 1)
        out = scaled_dot_product(q, k, v, self.attention_scale, mask)
        out = out.transpose(1,2).reshape(B,T,C)
        return self.resid_dropout(self.out_proj(out))
        # raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    import torch.nn.functional as F
    q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
    tri = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    print(torch.allclose(scaled_dot_product(q, k, v, 8 ** -0.5), F.scaled_dot_product_attention(q, k, v), atol=1e-6),
          torch.allclose(scaled_dot_product(q, k, v, 8 ** -0.5, tri),
                         F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6))   # expect True True

    x = torch.randn(2, 5, 32)
    mha = CausalSelfAttention(32, 4, 16)
    y = mha(x)
    print(y.shape)                                             # expect torch.Size([2, 5, 32])

    # Causality: changing the LAST token must not change the outputs of the earlier tokens.
    x2 = x.clone(); x2[:, -1] = 100.0
    print(torch.allclose(mha(x2)[:, :-1], y[:, :-1], atol=1e-5))   # expect True

    # Same numbers as PyTorch's built-in attention:
    B, T, C = x.shape
    split = lambda t: t.view(B, T, 4, 8).transpose(1, 2)
    ref = F.scaled_dot_product_attention(split(mha.q_proj(x)), split(mha.k_proj(x)), split(mha.v_proj(x)), is_causal=True)
    print(torch.allclose(y, mha.out_proj(ref.transpose(1, 2).reshape(B, T, C)), atol=1e-5))   # expect True

    gqa = GroupedQueryAttention(32, 4, 2, 16)
    print(gqa(x).shape, gqa.k_proj.weight.shape)               # expect torch.Size([2, 5, 32]) torch.Size([16, 32])
    # With n_kv_heads == n_heads and the same weights, GQA must equal MHA:
    g4 = GroupedQueryAttention(32, 4, 4, 16); g4.load_state_dict(mha.state_dict())
    print(torch.allclose(g4(x), y, atol=1e-6))                 # expect True
