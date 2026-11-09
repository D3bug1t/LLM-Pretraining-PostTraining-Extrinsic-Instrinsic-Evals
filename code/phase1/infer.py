"""Inference-time decoding strategies for the Model_5 checkpoint (Option 1, 101M modern decoder).

Standalone re-implementation of the notebook's model with a KV cache (same parameter names, so
modern_model.pt loads directly), plus:
  greedy / sampling (temperature, top-p, top-k, min-p) / beam search / contrastive search,
  token n-gram blocking, CTRL repetition penalty, frequency penalty.

Usage:
  python infer.py --strategy greedy --split pseudo --n 200        # score on pseudo-validation
  python infer.py --strategy greedy --split test                  # full test set -> submission zip

Pseudo-validation: a test prompt's first 384 words are the prompt and words 384-512 the reference
(the model never saw the test set; with the 640-token prompt window both cases look the same to it).
"""
import os, sys, json, time, argparse, zipfile, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.environ.get("M4_DIR", os.path.join(ROOT, "Model_5", "results", "cs781_phase1", "legal-modern-dec-101m"))   # M4 checkpoint folder
TEST_FILE = os.environ.get("P1_TEST_FILE", os.path.join(HERE, "data", "test.json"))
OUT_DIR = os.path.join(HERE, "outputs")
sys.path.insert(0, os.path.join(ROOT, "Evaluation_Script")); sys.path.insert(0, HERE)   # official evaluate_phase1.py

# ----------------------------------------------------------------------------- tokenizer
_SHIFT = 20
_MASK = (1 << _SHIFT) - 1
_EOW = _MASK


class BPETokenizer:
    """Same encoding as LegalBPETokenizer.encode_fast / decode, without transformers."""
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.itos = d["itos"]
        self.sym2id = {tuple(s): i for i, s in enumerate(self.itos) if isinstance(s, list)}

        def to_id(t):
            t = tuple(t)
            return _EOW if t == (256,) else self.sym2id[t]
        self.ranks = {(to_id(l) << _SHIFT) | to_id(r): self.sym2id[tuple(l) + tuple(r)] for l, r in d["merges"]}
        self.eow_id = len(self.itos)
        self.vocab_size = len(self.itos) + 1
        self.id_bytes = [b""] * 4 + [bytes(32 if b == 256 else b for b in s) for s in self.itos[4:]] + [b" "]
        self._cache = {}

    def _word(self, word):
        out = self._cache.get(word)
        if out is not None:
            return out
        ranks = self.ranks
        w = [b + 4 for b in word.encode("utf-8")] + [_EOW]
        while len(w) > 1:
            best_new, best_p = None, None
            for x, y in zip(w, w[1:]):
                n = ranks.get((x << _SHIFT) | y)
                if n is not None and (best_new is None or n < best_new):
                    best_new, best_p = n, (x, y)
            if best_new is None:
                break
            a, b = best_p
            i, res, L = 0, [], len(w)
            while i < L:
                if i < L - 1 and w[i] == a and w[i + 1] == b:
                    res.append(best_new); i += 2
                else:
                    res.append(w[i]); i += 1
            w = res
        out = [self.eow_id if t == _EOW else t for t in w]
        self._cache[word] = out
        return out

    def encode(self, text):
        ids = []
        for word in text.split():
            ids.extend(self._word(word))
        return ids

    def decode(self, ids):
        return b"".join(self.id_bytes[i] for i in ids).decode("utf-8", errors="replace").strip()


# ----------------------------------------------------------------------------- model with KV cache
@dataclass
class ModernLMConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    prefix_lm: bool = False


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ---- From-scratch attention (no nn.MultiheadAttention / F.scaled_dot_product_attention) ----
# out = softmax(q k^T * scale, masked) v, computed in query blocks with the softmax statistics kept per row
# (log-sum-exp), so the full (T, S) score matrix is never stored. The backward pass is written by hand and
# recomputes each block's probabilities from the saved log-sum-exp:
#   P = exp(S - lse),  dV = P^T dO,  dP = dO V^T,  dS = P * (dP - rowsum(dO * O)),  dQ = dS K * scale,  dK = dS^T Q * scale


def _block_scores(q_blk, k, scale, attn_mask, is_causal, i, j, kend, offset):
    """Scaled, masked scores (B, H, j-i, kend) in fp32 for queries [i, j) against keys [0, kend)."""
    s = torch.matmul(q_blk, k[:, :, :kend].transpose(-2, -1)).float() * scale
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask[..., i:j, :kend], float("-inf"))
    if is_causal:                                           # query row r sits at key position offset + r
        rows = torch.arange(i, j, device=s.device)[:, None] + offset
        s = s.masked_fill(torch.arange(kend, device=s.device)[None, :] > rows, float("-inf"))
    return s


class BlockwiseAttention(torch.autograd.Function):
    """q: (B, H, T, d); k, v: (B, H, S, d); attn_mask: bool, broadcastable to (B, H, T, S), True = may attend.
    Every query row must be allowed at least one key."""

    @staticmethod
    def forward(ctx, q, k, v, attn_mask, is_causal, scale, block):
        T, S = q.size(-2), k.size(-2)
        offset = S - T                                      # causal: the queries are the last T positions
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        for i in range(0, T, block):
            j = min(i + block, T)
            kend = min(S, offset + j) if is_causal else S   # causal: later keys are masked anyway, skip them
            s = _block_scores(q[:, :, i:j], k, scale, attn_mask, is_causal, i, j, kend, offset)
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            l = p.sum(-1, keepdim=True)
            out[:, :, i:j] = torch.matmul((p / l).to(v.dtype), v[:, :, :kend])
            lse[:, :, i:j] = (m + l.log()).squeeze(-1)
        ctx.save_for_backward(q, k, v, out, lse, attn_mask)
        ctx.is_causal, ctx.scale, ctx.block = is_causal, scale, block
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, out, lse, attn_mask = ctx.saved_tensors
        is_causal, scale, block = ctx.is_causal, ctx.scale, ctx.block
        T, S = q.size(-2), k.size(-2)
        offset = S - T
        d_out = d_out.contiguous()
        dq = torch.empty_like(q)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        delta = (d_out.float() * out.float()).sum(-1, keepdim=True)        # rowsum(dO * O) = rowsum(P * dP)
        for i in range(0, T, block):
            j = min(i + block, T)
            kend = min(S, offset + j) if is_causal else S
            q_blk, do_blk = q[:, :, i:j], d_out[:, :, i:j]
            p = torch.exp(_block_scores(q_blk, k, scale, attn_mask, is_causal, i, j, kend, offset)
                          - lse[:, :, i:j, None])                           # masked entries -> exactly 0
            dv[:, :, :kend] += torch.matmul(p.to(v.dtype).transpose(-2, -1), do_blk).float()
            dp = torch.matmul(do_blk, v[:, :, :kend].transpose(-2, -1)).float()
            ds = (p * (dp - delta[:, :, i:j])).to(q.dtype)
            dq[:, :, i:j] = torch.matmul(ds, k[:, :, :kend]) * scale
            dk[:, :, :kend] += torch.matmul(ds.transpose(-2, -1), q_blk).float() * scale
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def scratch_attention(q, k, v, attn_mask=None, is_causal=False, scale=None, block=128):
    """Drop-in for the fused kernel: q (B, H, T, d), k/v (B, H_kv, S, d) with H_kv dividing H (GQA).
    attn_mask: bool (True = may attend), broadcastable to (B, 1, T, S)."""
    scale = q.size(-1) ** -0.5 if scale is None else scale
    B, H, T, d = q.shape
    n_rep = H // k.size(1)
    if n_rep > 1 and not is_causal:
        # GQA without copying K/V: the n_rep query heads of a group become extra query rows for that kv head
        q = q.reshape(B, k.size(1), n_rep * T, d)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, 1, n_rep, 1)
        out = BlockwiseAttention.apply(q, k, v, attn_mask, False, scale, block)
        return out.reshape(B, H, T, d)
    if n_rep > 1:
        k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
    return BlockwiseAttention.apply(q, k, v, attn_mask, is_causal, scale, block)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class KVCache:
    """Pre-allocated per-layer K/V (N, kv_heads, Tmax, d_head); all rows share the write position L
    (prompts are left-padded). keep[:, t] = key t is a real token."""

    def __init__(self, cfg, N, Tmax, device, dtype):
        dh = cfg.d_model // cfg.n_heads
        self.K = [torch.zeros(N, cfg.n_kv_heads, Tmax, dh, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.V = [torch.zeros_like(k) for k in self.K]
        self.keep = torch.zeros(N, Tmax, dtype=torch.bool, device=device)
        self.L = 0

    def reorder(self, idx):
        for l in range(len(self.K)):
            self.K[l] = self.K[l].index_select(0, idx)
            self.V[l] = self.V[l].index_select(0, idx)
        self.keep = self.keep.index_select(0, idx)

    def commit_choice(self, T, choice):
        """Candidate mode: T candidates were written at [L, L+T); keep the chosen one at L."""
        L = self.L
        ar = torch.arange(choice.size(0), device=choice.device)
        for l in range(len(self.K)):
            self.K[l][:, :, L] = self.K[l][ar, :, L + choice]
            self.V[l][:, :, L] = self.V[l][ar, :, L + choice]
        self.keep[:, L] = True
        self.L = L + 1


class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)
        inv = 1.0 / (cfg.rope_base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head))
        ang = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float32), inv)
        ang = torch.cat([ang, ang], dim=-1)
        self.register_buffer("rope_cos", ang.cos(), persistent=False)
        self.register_buffer("rope_sin", ang.sin(), persistent=False)

    def forward(self, x, pos, cache, li, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        cos = self.rope_cos[pos].unsqueeze(1).to(q.dtype)
        sin = self.rope_sin[pos].unsqueeze(1).to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        L = cache.L
        cache.K[li][:, :, L:L + T] = k
        cache.V[li][:, :, L:L + T] = v
        out = scratch_attention(q, cache.K[li][:, :, :L + T], cache.V[li][:, :, :L + T], attn_mask=mask)  # GQA
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, pos, cache, li, mask):
        x = x + self.attn(self.attn_norm(x), pos, cache, li, mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([ModernBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, idx, pos, cache, mask):
        """Returns final-norm hidden states (N, T, d); K/V written at cache[L:L+T] (caller advances L)."""
        x = self.tok_emb(idx)
        for li, block in enumerate(self.blocks):
            x = block(x, pos, cache, li, mask)
        return self.final_norm(x)

    def logits(self, h):
        return F.linear(h, self.tok_emb.weight).float()


def load_model(device):
    with open(os.path.join(MODEL_DIR, "modern_config.json")) as f:
        cfg = ModernLMConfig(**json.load(f))
    model = ModernDecoderLM(cfg)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "modern_model.pt"), map_location="cpu"))
    return model.to(device=device, dtype=torch.float16).eval(), cfg


# ----------------------------------------------------------------------------- decoding helpers
def ban_ngrams(logits, generated, n):
    """-inf for every token that would repeat a token n-gram already in that row's continuation."""
    if n <= 0:
        return
    for r, toks in enumerate(generated):
        if len(toks) < n:
            continue
        prefix = toks[len(toks) - n + 1:]
        first = prefix[0]
        banned = [toks[i + n - 1] for i in range(len(toks) - n + 1)
                  if toks[i] == first and toks[i:i + n - 1] == prefix]
        if banned:
            logits[r, banned] = float("-inf")


def penalize(logits, counts, rep_pen, freq_pen):
    """counts: (N, V) float, occurrences of each token in the continuation (EOW excluded)."""
    if rep_pen != 1.0:
        seen = counts > 0
        logits[:] = torch.where(seen & (logits > 0), logits / rep_pen,
                                torch.where(seen, logits * rep_pen, logits))
    if freq_pen:
        logits -= freq_pen * counts


def sample(logits, temperature, top_p, top_k, min_p):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, -1)
    if min_p:
        probs = probs.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, 0.0)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        sp[(cum - sp) > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1).squeeze(-1)


def prefill(model, tok, prompts, cfg, args, device, n_rep=1):
    max_prompt = cfg.max_seq_len - args.max_new_tokens
    seqs = [(tok.encode(p)[-max_prompt:] or [tok.EOS]) for p in prompts]
    P = max(len(s) for s in seqs)
    ids = torch.tensor([[tok.PAD] * (P - len(s)) + s for s in seqs], device=device)
    keep = torch.tensor([[0] * (P - len(s)) + [1] * len(s) for s in seqs], device=device).bool()
    pos = (keep.long().cumsum(-1) - 1).clamp(min=0)
    if n_rep > 1:
        ids, keep, pos = (t.repeat_interleave(n_rep, 0) for t in (ids, keep, pos))
    N = ids.size(0)
    cache = KVCache(cfg, N, P + args.max_new_tokens + max(args.top_k, 1) + 1, device, torch.float16)
    cache.keep[:, :P] = keep
    i = torch.arange(P, device=device)
    mask = (i[None, :] <= i[:, None])[None] & keep[:, None, :]
    mask = (mask | torch.eye(P, dtype=torch.bool, device=device)[None])[:, None]
    h = model(ids, pos, cache, mask)
    cache.L = P
    return h, cache, pos[:, -1] + 1, keep


def step_mask(cache, T=1, candidates=False):
    """(N, 1, T, L+T) visibility for T new queries at positions [L, L+T)."""
    L = cache.L
    old = cache.keep[:, None, None, :L].expand(-1, 1, T, L)
    if candidates or T == 1:
        new = torch.eye(T, dtype=torch.bool, device=cache.keep.device)[None, None].expand(old.size(0), 1, T, T)
    return torch.cat([old, new], dim=-1)


def words_done(tok, generated, target):
    return all(len(tok.decode(g).split()) > target for g in generated)


# ----------------------------------------------------------------------------- strategies
@torch.inference_mode()
def generate_standard(model, tok, prompts, cfg, args, device):
    """greedy / sampling / contrastive, with n-gram blocking and repetition penalties."""
    h, cache, next_pos, keep = prefill(model, tok, prompts, cfg, args, device)
    N = h.size(0)
    logits = model.logits(h[:, -1])
    special = [tok.PAD, tok.UNK, tok.BOS, tok.EOS]
    counts = torch.zeros(N, tok.vocab_size, device=device)
    generated = [[] for _ in range(N)]
    contrastive = args.strategy_type == "contrastive"
    if contrastive:   # normalized hidden states of the context (padding never matches)
        ctx = torch.zeros(N, cache.K[0].size(2), h.size(-1), device=device, dtype=torch.float16)
        ctx[:, :h.size(1)] = F.normalize(h.float(), dim=-1).half()
        ctx_keep = cache.keep.clone()
    ar = torch.arange(N, device=device)

    for step in range(args.max_new_tokens):
        logits[:, special] = float("-inf")
        penalize(logits, counts, args.rep_penalty, args.freq_penalty)
        ban_ngrams(logits, generated, args.no_repeat_ngram)

        if contrastive:
            k = args.top_k
            probs = torch.softmax(logits, -1)
            top_p_, top_i = probs.topk(k, dim=-1)                         # (N, k)
            pos = next_pos[:, None].expand(N, k)
            hc = model(top_i, pos, cache, step_mask(cache, k, candidates=True))   # (N, k, d)
            hcn = F.normalize(hc.float(), dim=-1)
            L = cache.L
            sim = torch.einsum("nkd,ntd->nkt", hcn, ctx[:, :L].float())
            sim = sim.masked_fill(~ctx_keep[:, None, :L], -1.0).max(-1).values   # (N, k)
            score = (1 - args.alpha) * top_p_ - args.alpha * sim
            score = score.masked_fill(top_p_ <= 0, float("-inf"))
            choice = score.argmax(-1)
            nxt = top_i[ar, choice]
            cache.commit_choice(k, choice)
            hn = hc[ar, choice]
            ctx[:, L] = hcn[ar, choice].half()
            ctx_keep[:, L] = True
            logits = model.logits(hn)
        else:
            nxt = sample(logits, args.temperature, args.top_p, args.top_k, args.min_p)
            cache.keep[:, cache.L] = True
            hn = model(nxt[:, None], next_pos[:, None], cache, step_mask(cache))
            cache.L += 1
            logits = model.logits(hn[:, -1])
        next_pos = next_pos + 1

        nl = nxt.tolist()
        for r, t in enumerate(nl):
            generated[r].append(t)
        counts[ar, nxt] += 1
        counts[:, tok.eow_id] = 0
        if (step + 1) % 16 == 0 and words_done(tok, generated, args.target_words):
            break
    return generated


@torch.inference_mode()
def generate_beam(model, tok, prompts, cfg, args, device):
    nb = args.num_beams
    B = len(prompts)
    h, cache, next_pos, keep = prefill(model, tok, prompts, cfg, args, device, n_rep=nb)
    N = B * nb
    logp = torch.log_softmax(model.logits(h[:, -1]), -1)
    special = [tok.PAD, tok.UNK, tok.BOS, tok.EOS]
    scores = torch.zeros(B, nb, device=device)
    scores[:, 1:] = float("-inf")                                       # all beams start identical
    generated = [[] for _ in range(N)]
    counts = torch.zeros(N, tok.vocab_size, device=device)
    V = logp.size(-1)

    for step in range(args.max_new_tokens):
        logp[:, special] = float("-inf")
        penalize(logp, counts, args.rep_penalty, args.freq_penalty)
        ban_ngrams(logp, generated, args.no_repeat_ngram)
        if args.rep_penalty != 1.0 or args.freq_penalty:
            logp = torch.log_softmax(logp, -1)
        cand = (scores.view(B, nb, 1) + logp.view(B, nb, V)).view(B, nb * V)
        top_s, top_i = cand.topk(nb, dim=-1)                            # sorted: beam 0 is best
        src = (top_i // V) + torch.arange(B, device=device)[:, None] * nb
        nxt = (top_i % V).view(-1)
        src = src.view(-1)
        scores = top_s
        cache.reorder(src)
        sl = src.tolist()
        generated = [generated[s] + [t] for s, t in zip(sl, nxt.tolist())]
        counts = counts.index_select(0, src)
        counts[torch.arange(N, device=device), nxt] += 1
        counts[:, tok.eow_id] = 0
        cache.keep[:, cache.L] = True
        hn = model(nxt[:, None], next_pos[:, None], cache, step_mask(cache))
        cache.L += 1
        next_pos = next_pos + 1
        logp = torch.log_softmax(model.logits(hn[:, -1]), -1)
        if (step + 1) % 16 == 0 and words_done(tok, generated[::nb], args.target_words):
            break
    return generated[::nb]


# ----------------------------------------------------------------------------- presets
STRATEGIES = {
    # name: (type, settings).  ~1.04 tokens per word, so a token n-gram is ~ a word n-gram.
    "baseline":        ("standard", dict(temperature=0.1, top_p=0.95, no_repeat_ngram=4)),   # notebook default
    "greedy":          ("standard", dict(temperature=0.0, no_repeat_ngram=0)),
    "greedy_ng3":      ("standard", dict(temperature=0.0, no_repeat_ngram=3)),
    "greedy_ng4":      ("standard", dict(temperature=0.0, no_repeat_ngram=4)),
    "greedy_ng6":      ("standard", dict(temperature=0.0, no_repeat_ngram=6)),
    "greedy_ng4_rp":   ("standard", dict(temperature=0.0, no_repeat_ngram=4, rep_penalty=1.2)),
    "greedy_ng4_rp11": ("standard", dict(temperature=0.0, no_repeat_ngram=4, rep_penalty=1.1)),
    "greedy_ng4_fp":   ("standard", dict(temperature=0.0, no_repeat_ngram=4, freq_penalty=0.3)),
    "topp_t05":        ("standard", dict(temperature=0.5, top_p=0.9, no_repeat_ngram=4)),
    "topp_t07":        ("standard", dict(temperature=0.7, top_p=0.9, no_repeat_ngram=4)),
    "topp_t09":        ("standard", dict(temperature=0.9, top_p=0.9, no_repeat_ngram=4)),
    "minp_t07":        ("standard", dict(temperature=0.7, min_p=0.1, no_repeat_ngram=4)),
    "contrastive":     ("contrastive", dict(top_k=4, alpha=0.6, no_repeat_ngram=0)),
    "contrastive_a04": ("contrastive", dict(top_k=4, alpha=0.4, no_repeat_ngram=0)),
    "contrastive_ng4": ("contrastive", dict(top_k=4, alpha=0.4, no_repeat_ngram=4)),
    "beam4_ng3":       ("beam", dict(num_beams=4, no_repeat_ngram=3)),
    "beam4_ng4":       ("beam", dict(num_beams=4, no_repeat_ngram=4)),
    "beam4_ng4_rp":    ("beam", dict(num_beams=4, no_repeat_ngram=4, rep_penalty=1.2)),
    "greedy_ng3_rp11": ("standard", dict(temperature=0.0, no_repeat_ngram=3, rep_penalty=1.1)),
    "greedy_ng4_rp115": ("standard", dict(temperature=0.0, no_repeat_ngram=4, rep_penalty=1.15)),
    "beam4_ng3_rp11":  ("beam", dict(num_beams=4, no_repeat_ngram=3, rep_penalty=1.1)),
    "beam2_ng3_rp11":  ("beam", dict(num_beams=2, no_repeat_ngram=3, rep_penalty=1.1)),
    "topp_t07_rp11":   ("standard", dict(temperature=0.7, top_p=0.9, no_repeat_ngram=3, rep_penalty=1.1)),
    "minp_t07_rp11":   ("standard", dict(temperature=0.7, min_p=0.1, no_repeat_ngram=3, rep_penalty=1.1)),
    "minp_t05_rp11":   ("standard", dict(temperature=0.5, min_p=0.1, no_repeat_ngram=3, rep_penalty=1.1)),
}
DEFAULTS = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, no_repeat_ngram=0, rep_penalty=1.0,
                freq_penalty=0.0, alpha=0.6, num_beams=1)


# ----------------------------------------------------------------------------- data / metrics
def load_split(split, n):
    with open(TEST_FILE, encoding="utf-8") as f:
        test = json.load(f)
    items = list(test.items())
    if split == "test":
        return [(k, v["text"], None) for k, v in items]
    rng = random.Random(781)
    rng.shuffle(items)
    out = []
    for k, v in items[:n]:
        w = v["text"].split()
        out.append((k, " ".join(w[:384]), " ".join(w[384:512])))
    return out


def diagnostics(texts):
    """No-reference quality signals: repeated word 4-grams and distinct bigrams."""
    rep4, dist2 = [], []
    for t in texts:
        w = t.split()
        g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        g2 = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
        rep4.append(1 - len(set(g4)) / max(len(g4), 1))
        dist2.append(len(set(g2)) / max(len(g2), 1))
    return sum(rep4) / len(rep4), sum(dist2) / len(dist2)


def score(preds, refs, bertscore):
    from evaluate_phase1 import rouge_l_f1, bleu4
    res = {"rougeL": sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / len(refs),
           "bleu4": bleu4([p.split() for p in preds], [r.split() for r in refs])}
    if bertscore:
        import bert_score
        _, _, f = bert_score.score(preds, refs, lang="en", batch_size=32, verbose=False,
                                   device="cuda" if torch.cuda.is_available() else "cpu")
        res["bertscore"] = float(f.mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, nargs="+", choices=list(STRATEGIES))
    ap.add_argument("--split", choices=["pseudo", "test"], default="pseudo")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--target_words", type=int, default=128)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    tok = BPETokenizer(os.path.join(MODEL_DIR, "bpe_tokenizer.json"))
    data = load_split(cli.split, cli.n)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={device} | split={cli.split} | examples={len(data)}", flush=True)

    for name in cli.strategy:
        stype, settings = STRATEGIES[name]
        args = argparse.Namespace(**{**DEFAULTS, **settings, **vars(cli)})
        args.strategy_type = stype
        torch.manual_seed(cli.seed); random.seed(cli.seed)
        bs = max(1, cli.batch_size // args.num_beams) if stype == "beam" else cli.batch_size
        gen_fn = generate_beam if stype == "beam" else generate_standard
        preds, t0 = {}, time.time()
        for s in range(0, len(data), bs):
            batch = data[s:s + bs]
            gens = gen_fn(model, tok, [p for _, p, _ in batch], cfg, args, device)
            for (k, _, _), g in zip(batch, gens):
                text = " ".join(tok.decode(g).split()[:args.target_words])
                preds[k] = {"text": text if text.strip() else "the"}
            print(f"  [{name}] {min(s + bs, len(data))}/{len(data)}  {time.time() - t0:.0f}s", flush=True)
        elapsed = time.time() - t0
        texts = [preds[k]["text"] for k, _, _ in data]
        rep4, dist2 = diagnostics(texts)
        row = {"strategy": name, "split": cli.split, "n": len(data), "settings": settings,
               "seconds": round(elapsed, 1), "rep4": round(rep4, 4), "distinct2": round(dist2, 4),
               "avg_words": sum(len(t.split()) for t in texts) / len(texts)}
        if cli.split == "pseudo":
            row.update(score(texts, [r for _, _, r in data], cli.bertscore))
            with open(os.path.join(OUT_DIR, f"pseudo_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)
        else:
            d = os.path.join(OUT_DIR, name)
            os.makedirs(d, exist_ok=True)
            pj = os.path.join(d, "prediction.json")
            with open(pj, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)
            with zipfile.ZipFile(os.path.join(HERE, f"submission_{name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
                z.write(pj, arcname="prediction.json")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        with open(os.path.join(OUT_DIR, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
