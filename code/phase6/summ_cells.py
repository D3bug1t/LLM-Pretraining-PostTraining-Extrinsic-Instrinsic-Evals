# %% [markdown]
# # Phase 6 (Abstractive Summarization) - Fine-tuning the Phase 1 models
#
# One notebook, several experiments run back to back (`RUNS` below), one run per GPU. Every run:
#
# 1. loads a Phase 1 checkpoint (Model 2 / 3 / 4 / 5) as a **generator**:
#    - decoders (m2, m4, m5): one sequence = [document tokens] `<s>` [summary tokens] `</s>`, next-token loss on the
#      summary only. m5 (prefix-LM) reads the document part bidirectionally, as in its pretraining; m2 / m4 are causal.
#      The output layer is the (tied) input embedding, as in Phase 1. RoPE is extended from 1024 to `ctx_len` positions.
#    - m3 (A-2 GRU encoder-decoder with Bahdanau attention): the encoder reads the document, the decoder writes the summary.
# 2. reads a long document as `doc_part` (`head`: its first `doc_tokens` tokens, `head_tail`: first half + last half),
# 3. adapts the model with the run's method (`lora`, `lorma`, `houlsby`, `full`, `kd`),
# 4. keeps the epoch with the lowest summary loss on a held-out part of train.json (no dev split is given),
# 5. generates summaries (greedy, KV cache, no repeated n-grams) for held-out documents (ROUGE here; BERTScore is
#    computed afterwards with `evaluate_phase6.py`) and for test.json -> `<run>/submission.zip` (prediction.json).
#
# **Inputs to add on Kaggle:** the HF secret `HF` and one dataset per Phase 1 model (as in Phases 4-5).
# **Outputs:** `/kaggle/working/phase6_log.txt` and `/kaggle/working/phase6_results_bundle.zip` (logs, metrics,
# held-out and test summaries, submissions, trained weights in bf16) - download this one.
#
# The hyperparameters below are starting values, not tuned ones.

# %%
# ============================ CONFIG ============================
MODEL_DIRS = {          # folder (or any file inside it) of each Phase 1 checkpoint; None = auto-discover
    "m2": None,         # Model 2: A-3 decoder, RoPE + GQA          (a3_config.json + a3_model.pt)
    "m3": None,         # Model 3: A-2 GRU encoder-decoder           (gru_config.json + gru_model.pt, or Trainer *.safetensors)
    "m4": "/kaggle/input/datasets/sanskaars22/model-5",   # Model 4: modern decoder (this upload is legal-modern-dec-101m)
    "m5": "/kaggle/input/datasets/sanskaars22/model-4-prefix-lm-labelled-as-m5-in-notebook",   # Model 5: prefix-LM (legal-prefix-lm-101m)
}
DISCOVER_KEYWORDS = {"m2": ["model-2", "model_2", "model2", "dec-rope"], "m3": ["model-3", "model_3", "model3", "gru"],
                     "m4": ["model-4", "model_4", "model4"], "m5": ["model-5", "model_5", "model5", "prefix"]}

COMMON = dict(
    doc_part="head_tail",          # which tokens of a long document are read: "head" (the first doc_tokens) or
                                   # "head_tail" (the first and the last doc_tokens / 2: facts + the court's decision)
    doc_tokens=1024,               # document tokens read (decoders: + the summary must fit in ctx_len)
    sum_tokens=1024,               # training targets: the first sum_tokens summary tokens (+ </s> if the summary fits)
    ctx_len=2048,                  # decoders: RoPE positions (Phase 1 trained 1024); doc_tokens + 1 + sum_tokens + 1 <= ctx_len + 2
    batch_tokens=4096,             # decoders: tokens per batch (~2 sequences); m3: gru_batch sequences per batch
    gru_batch=8,
    epochs=2, grad_accum=4,
    train_minutes=150,             # training time budget per run (Kaggle: 12 h per session): the learning-rate schedule
                                   # is fitted to the batches that fit in it (measured on the first 30), then training stops
    eval_doc_parts=[],             # also generate the held-out summaries reading these doc_parts (long-document comparison)
    lr_backbone=5e-5,              # pretrained weights (full / kd student)
    lr_adapter=3e-4,               # LoRA / LoRMA / Houlsby adapter parameters
    weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0,
    rank=8, alpha=16, adapter_targets="attn+mlp",   # LoRA / LoRMA
    adapter_dim=64,                # Houlsby: bottleneck size of the adapters after attention and after the MLP
    kd_temperature=2.0, kd_alpha=0.5,
    student_layers=None,           # keep only the lowest N layers of the checkpoint (KD student; None = all layers)
    val_docs=200,                  # held out from train.json (fixed, the same for every run): epoch selection (loss)
    gen_val_docs=100,              # of those, summaries generated and scored (ROUGE here, BERTScore offline)
    max_new_tokens=1024, min_new_tokens=64, no_repeat_ngram=4, gen_batch=16,
    save_weights=True,             # trainable parameters of the best epoch (bf16) -> bundle
    seed=42,
)

# Techniques (spec 12.2.1 list): LoRMA (compulsory), LoRA, Houlsby adapters, knowledge distillation.
# m3 (GRU) has no Linear layers inside its recurrent cells: it is fully fine-tuned ("full").
# A KD run is dict(name=..., model=..., method="kd", teacher=<a run in this session>, student_layers=N): it waits for its
# teacher, then trains the smaller student on the teacher's token distributions (T = kd_temperature) + the references.
# Queue order = list order (2 GPUs: m5_lorma | m4_lora, then m5_kd_student | m2_houlsby, then m3_full); ~7.5 h in all.
RUNS = [
    # long-document comparison at no training cost: the held-out summaries also generated reading the head only
    dict(name="m5_lorma",      model="m5", method="lorma", eval_doc_parts=["head"]),
    dict(name="m4_lora",       model="m4", method="lora"),
    # KD: 4-layer m5 student (lowest 4 of 10 layers, all its weights trained), teacher m5_lorma
    dict(name="m5_kd_student", model="m5", method="kd", teacher="m5_lorma", student_layers=4, train_minutes=120),
    dict(name="m2_houlsby",    model="m2", method="houlsby"),
    dict(name="m3_full",       model="m3", method="full", lr_backbone=1e-4, train_minutes=80),   # RNN: a higher starting lr
    # long-document strategy trained (optional, ~3.3 h more): the first 1024 document tokens only
    dict(name="m5_lorma_head", model="m5", method="lorma", doc_part="head"),
]
RUN_ONLY = ["m5_lorma", "m4_lora", "m5_kd_student", "m2_houlsby", "m3_full"]   # None = all of RUNS
SMOKE = False              # True: 40 training documents, 8 held-out, 1 epoch, 48 new tokens per selected run (checks
                           # loading, training, generating and zipping); outputs go to /kaggle/working/smoke/

# %%
import os, re, json, glob, math, time, random, zipfile, hashlib, gc, platform, threading
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("p6_work")
if SMOKE: WORK = os.path.join(WORK, "smoke")
os.makedirs(WORK, exist_ok=True)
LOG_PATH = os.path.join(WORK, "phase6_log.txt")
_log = open(LOG_PATH, "a", encoding="utf-8")

_log_lock = threading.Lock()
def log(*a):
    s = " ".join(str(x) for x in a)
    t = threading.current_thread().name
    if t.startswith("gpu"): s = "\n".join(f"[{t}] {l}" for l in s.split("\n"))   # parallel runs: which GPU worker
    with _log_lock:
        print(s, flush=True); _log.write(s + "\n"); _log.flush()

def section(t):
    log("\n" + "=" * 100); log(t); log("=" * 100)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fp16 (+ loss scaling) on GPUs without bf16 tensor cores (T4), as the Phase 1 models were trained; bf16 on Ampere and newer
AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(os.environ.get("P6_AMP"), torch.bfloat16
            if DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
section(f"Phase 6 fine-tuning - {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"torch {torch.__version__} | python {platform.python_version()} | device {DEVICE}"
    + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} GPU)" if DEVICE.type == "cuda" else "")
    + f" | amp {AMP_DTYPE}")
log("COMMON =", json.dumps(COMMON))

# %%
# ---- data: Phase_6_Summ from the official dataset ----
def load_phase6_files():
    local = os.environ.get("P6_LOCAL_DATA")                       # local testing only
    if local:
        return {f: os.path.join(r, f) for r, _, fs in os.walk(local) for f in fs}
    from huggingface_hub import HfApi, hf_hub_download
    token, why = os.environ.get("HF_TOKEN"), "HF_TOKEN env var not set"
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF")
        except Exception as e:
            why = f"Kaggle secret 'HF' not readable ({e!r})"
    assert token, (f"no Hugging Face token: {why}. In the Kaggle editor: Add-ons -> Secrets -> add/attach a secret "
                   "named exactly HF (value = your hf_... token) and make sure its checkbox is ON for this notebook.")
    try:
        log(f"HF login ok: {HfApi().whoami(token=token)['name']}")
    except Exception as e:
        raise RuntimeError(f"the HF token was found but is rejected by Hugging Face ({e!r}); "
                           "create a new read token at huggingface.co/settings/tokens") from None
    repo = "Exploration-Lab/CS-781-Capstone"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", token=token) if f.startswith("Phase_6_Summ/")]
    return {os.path.basename(f): hf_hub_download(repo, f, repo_type="dataset", token=token) for f in files}

P6 = load_phase6_files()
read = lambda n: json.load(open(P6[n], encoding="utf-8"))

DOCS = {}           # document id -> dict(text, summary (None for test), split)
TEST_IDS = []
for x in read("test.json"):
    d = str(x["id"]); assert d not in DOCS, f"duplicate test id {d}"
    DOCS[d] = dict(text=x["document"], summary=None, split="test"); TEST_IDS.append(d)
_test_texts = {DOCS[d]["text"] for d in TEST_IDS}

_st, TRAIN_ALL, _seen = Counter(), [], set()
for x in read("train.json"):
    d = str(x["id"])
    if x["document"] in _test_texts: _st["dropped (test document)"] += 1; continue
    if x["document"] in _seen: _st["dropped (duplicate document)"] += 1; continue
    if not x["summary"].strip() or not x["document"].strip(): _st["dropped (empty)"] += 1; continue
    assert d not in DOCS, f"duplicate train id {d}"
    _seen.add(x["document"])
    DOCS[d] = dict(text=x["document"], summary=x["summary"], split="train"); TRAIN_ALL.append(d)

# fixed held-out part of train.json (the same for every run): epoch selection; the first gen_val_docs are also generated
_ids = sorted(TRAIN_ALL); random.Random(0).shuffle(_ids)
VAL_IDS = _ids[:COMMON["val_docs"]]
GEN_VAL_IDS = VAL_IDS[:COMMON["gen_val_docs"]]
TRAIN_IDS = [d for d in TRAIN_ALL if d not in set(VAL_IDS)]
if SMOKE:
    TRAIN_IDS, VAL_IDS, GEN_VAL_IDS, TEST_IDS = TRAIN_IDS[:40], VAL_IDS[:8], VAL_IDS[:4], TEST_IDS[:4]

section("DATA")
log(f"test:  {len(TEST_IDS)} documents")
log(f"train.json: kept {len(TRAIN_ALL)} " + ", ".join(f"{v} {k}" for k, v in _st.items()))
log(f"-> train {len(TRAIN_IDS)}, held-out {len(VAL_IDS)} (generated: {len(GEN_VAL_IDS)})")
for name, ids, key in (("train document", TRAIN_IDS, "text"), ("train summary", TRAIN_IDS, "summary"),
                       ("test document", TEST_IDS, "text")):
    w = np.array([len(DOCS[d][key].split()) for d in ids])
    log(f"{name:15s} words: median {int(np.median(w))}, p10 {int(np.percentile(w, 10))}, p90 {int(np.percentile(w, 90))}, "
        f"max {w.max()}")

# %%
# ---- Phase 1 byte-level BPE (same algorithm and ids as LegalBPETokenizer), with character offsets ----
WORD_RE = re.compile(r"\S+")
_SHIFT = 20
_EOW = (1 << _SHIFT) - 1

class OffsetBPE:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.itos = d["itos"]
        sym2id = {tuple(s): i for i, s in enumerate(self.itos) if isinstance(s, list)}
        to_id = lambda t: _EOW if tuple(t) == (256,) else sym2id[tuple(t)]
        self.ranks = {(to_id(l) << _SHIFT) | to_id(r): sym2id[tuple(l) + tuple(r)] for l, r in d["merges"]}
        self.nbytes = [sum(b != 256 for b in s) if isinstance(s, list) else 0 for s in self.itos]
        self.eow_id = len(self.itos)               # LegalBPETokenizer: a bare end-of-word symbol gets id len(itos)
        self.vocab_size = len(self.itos) + 1
        self.cache = {}

    def tokenize_word(self, word):
        out = self.cache.get(word)
        if out is not None: return out
        w = [b + 4 for b in word.encode("utf-8")] + [_EOW]
        ranks = self.ranks
        while len(w) > 1:
            best = None
            for x, y in zip(w, w[1:]):
                n = ranks.get((x << _SHIFT) | y)
                if n is not None and (best is None or n < best[0]): best = (n, x, y)
            if best is None: break
            n, a, b = best; i, res = 0, []
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b: res.append(n); i += 2
                else: res.append(w[i]); i += 1
            w = res
        self.cache[word] = w
        return w

    def encode(self, text):
        """-> ids, char_start, char_end (np arrays). A bare end-of-word token has an empty span at the word end."""
        ids, cs, ce = [], [], []
        for m in WORD_RE.finditer(text):
            word, base = m.group(), m.start()
            b2c = []
            for ci, ch in enumerate(word): b2c.extend([ci] * len(ch.encode("utf-8")))
            p = 0
            for t in self.tokenize_word(word):
                nb = 0 if t == _EOW else self.nbytes[t]
                ids.append(self.eow_id if t == _EOW else t)
                if nb == 0:
                    cs.append(base + len(word)); ce.append(base + len(word))
                else:
                    cs.append(base + b2c[p]); ce.append(base + b2c[p + nb - 1] + 1); p += nb
        return np.array(ids, np.int64), np.array(cs, np.int64), np.array(ce, np.int64)

# ---- From-scratch attention (no nn.MultiheadAttention / F.scaled_dot_product_attention), as in Phase 1 ----
# out = softmax(q k^T * scale, masked) v, computed in query blocks with the softmax statistics kept per row
# (log-sum-exp), so the full (T, S) score matrix is never stored. The backward pass is written by hand and
# recomputes each block's probabilities from the saved log-sum-exp:
#   P = exp(S - lse),  dV = P^T dO,  dP = dO V^T,  dS = P * (dP - rowsum(dO * O)),  dQ = dS K * scale,  dK = dS^T Q * scale
def _block_scores(q_blk, k, scale, attn_mask, is_causal, i, j, kend, offset):
    s = torch.matmul(q_blk, k[:, :, :kend].transpose(-2, -1)).float() * scale
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask[..., i:j, :kend], float("-inf"))
    if is_causal:
        rows = torch.arange(i, j, device=s.device)[:, None] + offset
        s = s.masked_fill(torch.arange(kend, device=s.device)[None, :] > rows, float("-inf"))
    return s


class BlockwiseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask, is_causal, scale, block):
        T, S = q.size(-2), k.size(-2)
        offset = S - T
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        for i in range(0, T, block):
            j = min(i + block, T)
            kend = min(S, offset + j) if is_causal else S
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
        delta = (d_out.float() * out.float()).sum(-1, keepdim=True)
        for i in range(0, T, block):
            j = min(i + block, T)
            kend = min(S, offset + j) if is_causal else S
            q_blk, do_blk = q[:, :, i:j], d_out[:, :, i:j]
            p = torch.exp(_block_scores(q_blk, k, scale, attn_mask, is_causal, i, j, kend, offset) - lse[:, :, i:j, None])
            dv[:, :, :kend] += torch.matmul(p.to(v.dtype).transpose(-2, -1), do_blk).float()
            dp = torch.matmul(do_blk, v[:, :, :kend].transpose(-2, -1)).float()
            ds = (p * (dp - delta[:, :, i:j])).to(q.dtype)
            dq[:, :, i:j] = torch.matmul(ds, k[:, :, :kend]) * scale
            dk[:, :, :kend] += torch.matmul(ds.transpose(-2, -1), q_blk).float() * scale
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def scratch_attention(q, k, v, attn_mask=None, is_causal=False, scale=None, block=128):
    """q (B, H, T, d), k/v (B, H_kv, S, d) with H_kv dividing H (GQA). attn_mask: bool (True = may attend)."""
    scale = q.size(-1) ** -0.5 if scale is None else scale
    B, H, T, d = q.shape
    n_rep = H // k.size(1)
    if n_rep > 1 and not is_causal:
        q = q.reshape(B, k.size(1), n_rep * T, d)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, 1, n_rep, 1)
        out = BlockwiseAttention.apply(q, k, v, attn_mask, False, scale, block)
        return out.reshape(B, H, T, d)
    if n_rep > 1:
        k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
    return BlockwiseAttention.apply(q, k, v, attn_mask, is_causal, scale, block)


def attend(q, k, v, key_keep, causal, scale=None):
    """causal or bidirectional attention; key_keep (B, T) bool marks real tokens (None = no padding)."""
    if key_keep is None:
        return scratch_attention(q, k, v, is_causal=causal, scale=scale)
    T = q.size(-2)
    allowed = key_keep[:, None, None, :].expand(-1, 1, T, -1)
    if causal:
        allowed = allowed & torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    allowed = allowed | torch.eye(T, dtype=torch.bool, device=q.device)   # padding rows see themselves (no NaN)
    return scratch_attention(q, k, v, attn_mask=allowed, scale=scale)


def rope_cache(T, d_head, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    ang = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

# %%
# ---- Phase 1 backbones as encoders. Parameter names match the Phase 1 checkpoints exactly. ----
class A3Attention(nn.Module):              # Model 2 (A-3 CausalSelfAttention: MHA/GQA, RoPE or sinusoidal)
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv, self.d_head = c["n_heads"], c["n_kv_heads"], c["d_model"] // c["n_heads"]
        self.scale = c.get("attention_scale") or self.d_head ** -0.5
        self.q_proj = nn.Linear(c["d_model"], self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(c["d_model"], self.n_kv * self.d_head, bias=False)
        self.v_proj = nn.Linear(c["d_model"], self.n_kv * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, c["d_model"], bias=False)
        self.use_rope = c.get("pos_emb_type") == "rope"
        if self.use_rope:
            cos, sin = rope_cache(c["max_seq_len"], self.d_head, c.get("rope_base", 10000.0))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, key_keep, causal):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        if self.use_rope:
            cos, sin = self.rope_cos[:T][None, None].to(q.dtype), self.rope_sin[:T][None, None].to(q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = attend(q, k, v, key_keep, causal, self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))


class A3Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm_first = c.get("norm_first", False)
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.attn = A3Attention(c)
        self.ln2 = nn.LayerNorm(c["d_model"])
        self.ffn = nn.Module()
        self.ffn.net = nn.Sequential(nn.Linear(c["d_model"], c["d_ff"]), nn.GELU(), nn.Linear(c["d_ff"], c["d_model"]), nn.Dropout(0.0))

    def forward(self, x, key_keep, causal):
        if self.norm_first:
            x = x + self.attn(self.ln1(x), key_keep, causal)
            return x + self.ffn.net(self.ln2(x))
        x = self.ln1(x + self.attn(x, key_keep, causal))
        return self.ln2(x + self.ffn.net(x))


class A3Backbone(nn.Module):
    kind = "a3"
    def __init__(self, c):
        super().__init__()
        assert c.get("attention_type", "standard") == "standard", "only the standard (MHA/GQA) A-3 attention is supported"
        self.c, self.hidden_size = c, c["d_model"]
        self.tok_emb = nn.Embedding(c["vocab_size"], c["d_model"])
        if c.get("pos_emb_type") == "sinusoidal":
            pos = torch.arange(c["max_seq_len"], dtype=torch.float32)[:, None]
            div = torch.exp(torch.arange(0, c["d_model"], 2, dtype=torch.float32) * (-math.log(10000.0) / c["d_model"]))
            pe = torch.zeros(c["max_seq_len"], c["d_model"]); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pos_emb", pe, persistent=False)
        self.blocks = nn.ModuleList([A3Block(c) for _ in range(c["n_layers"])])
        self.ln_f = nn.LayerNorm(c["d_model"])

    def forward(self, ids, key_keep=None, causal=False):
        x = self.tok_emb(ids)
        if hasattr(self, "pos_emb"): x = x + self.pos_emb[:ids.size(1)][None]
        for b in self.blocks: x = b(x, key_keep, causal)
        return self.ln_f(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class ModernAttention(A3Attention):         # Models 4/5 GQAttention (always RoPE), same projections
    def __init__(self, c):
        super().__init__({**c, "pos_emb_type": "rope", "attention_scale": None})


class ModernBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c["d_model"], c.get("norm_eps", 1e-5))
        self.attn = ModernAttention(c)
        self.mlp_norm = RMSNorm(c["d_model"], c.get("norm_eps", 1e-5))
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(c["d_model"], c["d_ff"], bias=False)
        self.mlp.up = nn.Linear(c["d_model"], c["d_ff"], bias=False)
        self.mlp.down = nn.Linear(c["d_ff"], c["d_model"], bias=False)

    def forward(self, x, key_keep, causal):
        x = x + self.attn(self.attn_norm(x), key_keep, causal)
        h = self.mlp_norm(x)
        return x + self.mlp.down(F.silu(self.mlp.gate(h)) * self.mlp.up(h))


class ModernBackbone(nn.Module):
    kind = "modern"
    def __init__(self, c):
        super().__init__()
        self.c, self.hidden_size = c, c["d_model"]
        self.tok_emb = nn.Embedding(c["vocab_size"], c["d_model"])
        self.blocks = nn.ModuleList([ModernBlock(c) for _ in range(c["n_layers"])])
        self.final_norm = RMSNorm(c["d_model"], c.get("norm_eps", 1e-5))

    def forward(self, ids, key_keep=None, causal=False):
        x = self.tok_emb(ids)
        for b in self.blocks: x = b(x, key_keep, causal)
        return self.final_norm(x)


class GRUBackbone(nn.Module):               # Model 3: the bidirectional GRU encoder of the A-2 seq2seq model
    kind = "gru"
    def __init__(self, c):
        super().__init__()
        self.c, self.hidden_size = c, 2 * c["hidden"]
        self.embedding = nn.Embedding(c["vocab_size"], c["emb_dim"])
        self.enc_rnn = nn.GRU(c["emb_dim"], c["hidden"], num_layers=c["enc_layers"], batch_first=True, bidirectional=True)

    def forward(self, ids, key_keep=None, causal=False):
        self.enc_rnn.flatten_parameters()
        x = self.embedding(ids)
        if key_keep is None:
            return self.enc_rnn(x)[0]
        lengths = key_keep.sum(1).clamp(min=1).cpu()
        out, _ = self.enc_rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        return pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))[0]

# %%
# ---- finding and loading the checkpoints ----
def _ckpt_kind(folder):
    names = os.listdir(folder)
    if "a3_config.json" in names: return "a3"
    if "modern_config.json" in names: return "modern"
    if "gru_config.json" in names: return "gru"
    st = [n for n in names if n.endswith(".safetensors")]
    if st:
        from safetensors import safe_open
        with safe_open(os.path.join(folder, st[0]), "pt") as f:
            keys = list(f.keys())
        if any("enc_rnn" in k for k in keys): return "gru"
        if any("attn_norm" in k for k in keys): return "modern"
        if any(".ln1." in k for k in keys): return "a3"
    return None

WANT_KIND = {"m2": "a3", "m3": "gru", "m4": "modern", "m5": "modern"}

def find_checkpoint(key):
    p = MODEL_DIRS.get(key)
    if p:
        folder = p if os.path.isdir(p) else os.path.dirname(p)
        if _ckpt_kind(folder) is None:          # a dataset root: look one level deeper
            subs = [os.path.dirname(f) for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                    if f.endswith(("_config.json", ".safetensors"))]
            subs = [s for s in dict.fromkeys(subs) if "checkpoint-" not in s and _ckpt_kind(s)]
            assert subs, f"no checkpoint found under {p}"
            folder = subs[0]
        return folder
    cands = []
    for f in glob.glob("/kaggle/input/**/*", recursive=True):
        if f.endswith(("_config.json", ".safetensors")):
            folder = os.path.dirname(f)
            if "checkpoint-" in folder: continue
            if _ckpt_kind(folder) == WANT_KIND[key]: cands.append(folder)
    cands = list(dict.fromkeys(cands))
    hit = [c for c in cands if any(w in c.lower() for w in DISCOVER_KEYWORDS[key])]
    shared = sum(k == WANT_KIND[key] for k in WANT_KIND.values()) > 1   # m4/m5: same format, never guess
    assert hit or (len(cands) == 1 and not shared), \
        f"{key}: cannot pick a checkpoint automatically from {cands}; set MODEL_DIRS['{key}']"
    return (hit or cands)[0]

def _strip(sd):
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "lm.", "model."):
            if k.startswith(pre): k = k[len(pre):]
        out[k] = v
    return out

def load_backbone(key, n_layers=None):
    """Phase 1 checkpoint -> (backbone, info). n_layers < full depth keeps only the lowest layers (KD student)."""
    folder = find_checkpoint(key)
    kind = _ckpt_kind(folder)
    names = os.listdir(folder)
    pt = [n for n in names if n.endswith("_model.pt")]
    if pt:
        sd = torch.load(os.path.join(folder, pt[0]), map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(folder, [n for n in names if n.endswith(".safetensors")][0]))
    sd = _strip(sd)
    cfg_name = {"a3": "a3_config.json", "modern": "modern_config.json", "gru": "gru_config.json"}[kind]
    if cfg_name in names:
        c = json.load(open(os.path.join(folder, cfg_name)))
    elif kind == "gru":                          # Trainer checkpoint without config: read the sizes off the weights
        c = dict(vocab_size=sd["embedding.weight"].shape[0], emb_dim=sd["embedding.weight"].shape[1],
                 hidden=sd["enc_rnn.weight_hh_l0"].shape[1],
                 enc_layers=len([k for k in sd if re.fullmatch(r"enc_rnn\.weight_ih_l\d+", k)]))
    else:
        raise FileNotFoundError(f"{folder}: {cfg_name} is needed to know the number of heads")
    full_layers = c["enc_layers"] if kind == "gru" else c["n_layers"]
    if n_layers:
        assert n_layers <= full_layers
        if kind == "gru":
            c = {**c, "enc_layers": n_layers}
            sd = {k: v for k, v in sd.items() if not re.search(r"_l(\d+)", k) or int(re.search(r"_l(\d+)", k).group(1)) < n_layers}
        else:
            c = {**c, "n_layers": n_layers}
            sd = {k: v for k, v in sd.items() if not k.startswith("blocks.") or int(k.split(".")[1]) < n_layers}
    model = {"a3": A3Backbone, "modern": ModernBackbone, "gru": GRUBackbone}[kind](c)
    own = model.state_dict()
    used = {k: v for k, v in sd.items() if k in own}
    missing = [k for k in own if k not in used]
    assert not missing, f"weights missing from the checkpoint: {missing[:5]}"
    for k, v in used.items():
        assert v.shape == own[k].shape, f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}"
    model.load_state_dict(used)
    tok = [os.path.join(folder, n) for n in names if n.startswith("bpe_tokenizer") and n.endswith(".json")]
    for where in (os.path.join(os.path.dirname(folder), "**"), "/kaggle/input/**"):     # next to the weights, then anywhere
        tok = tok or sorted(glob.glob(os.path.join(where, "bpe_tokenizer*.json"), recursive=True))
    assert tok, f"no bpe_tokenizer*.json found for {key}: add the Phase 1 tokenizer as an input"
    info = dict(folder=folder, kind=kind, config=c, full_layers=full_layers, n_layers=n_layers or full_layers,
                unused_keys=len(sd) - len(used), tokenizer=tok[0], prefix_lm=c.get("prefix_lm"))
    return model, info

# %%
# ---- adaptation methods ----
class LoRALinear(nn.Module):
    """LoRA (Hu et al. 2021): h = W0 x + (alpha / r) B A x, A (r x in) random, B (out x r) = 0."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


class LoRMALinear(nn.Module):
    """LoRMA+ (Bihany, Patel & Modi 2025): multiplicative update h = (I + (alpha / r) B A) W0 x, with A (r x out) random,
    B (out x r) = 0 so training starts at W0. Computed as z = W0 x, then z + (alpha / r) B (A z): never forms the d x d matrix."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.out_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        z = F.linear(x, self.base.weight)
        z = z + (z @ self.A.t().to(z.dtype)) @ self.B.t().to(z.dtype) * self.scale
        return z if self.base.bias is None else z + self.base.bias



class RSLoRALinear(LoRALinear):
    """rsLoRA (Kalajdzievski 2023): LoRA with the update scaled by alpha / sqrt(r) instead of alpha / r, so the update
    does not shrink as the rank grows."""
    def __init__(self, base, r, alpha):
        super().__init__(base, r, alpha)
        self.scale = alpha / math.sqrt(r)


def adapter_target_names(backbone, targets):
    names = []
    for n, m in backbone.named_modules():
        if not isinstance(m, nn.Linear): continue
        is_attn = ".attn." in n
        is_mlp = ".ffn." in n or ".mlp." in n
        if (is_attn and "attn" in targets) or (is_mlp and "mlp" in targets): names.append(n)
    return names

def inject(backbone, cls, targets, r, alpha):
    names = adapter_target_names(backbone, targets)
    for n in names:
        parent, child = backbone.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
        setattr(parent, child, cls(getattr(parent, child), r, alpha))
    return names

# %%
# ---- documents / summaries -> token ids (Phase 1 tokenizer), token ids -> text ----
PAD, UNK, BOS, EOS = 0, 1, 2, 3                       # Phase 1 special tokens (<pad>, <unk>, <s>, </s>)
_TOKENS, _TOK_LOCK = {}, threading.Lock()
def get_tokens(tok_path):
    """-> (bpe, {doc id: (document ids, summary ids or None)}) as int32 arrays, cached per tokenizer file content."""
    with _TOK_LOCK:
        h = hashlib.md5(open(tok_path, "rb").read()).hexdigest()
        if h not in _TOKENS:
            t0 = time.time(); bpe = OffsetBPE(tok_path); cache = {}
            def enc(text):
                out = []
                for w in text.split():                   # = WORD_RE words
                    t = cache.get(w)
                    if t is None:
                        t = cache[w] = [bpe.eow_id if i == _EOW else i for i in bpe.tokenize_word(w)]
                    out.extend(t)
                return np.array(out or [bpe.eow_id], np.int32)
            toks = {d: (enc(x["text"]), None if x["summary"] is None else enc(x["summary"])) for d, x in DOCS.items()}
            bpe.cache.clear()
            _TOKENS[h] = (bpe, toks)
            n = np.array([len(toks[d][0]) for d in TRAIN_IDS]); s = np.array([len(toks[d][1]) for d in TRAIN_IDS])
            log(f"tokenizer {tok_path} (md5 {h[:8]}): {len(DOCS)} documents in {time.time() - t0:.0f}s; train document "
                f"tokens median {int(np.median(n))}, p90 {int(np.percentile(n, 90))}; summary tokens median {int(np.median(s))}, "
                f"p90 {int(np.percentile(s, 90))}, {100 * np.mean(s <= COMMON['sum_tokens']):.1f}% <= {COMMON['sum_tokens']}")
        return _TOKENS[h]

def detok(bpe, ids):
    """token ids -> text (inverse of the Phase 1 byte-level BPE: 256 / the bare end-of-word id = a space)."""
    out = bytearray()
    for i in ids:
        i = int(i)
        if i == bpe.eow_id: out += b" "
        elif i < 4 or i >= bpe.eow_id: continue
        else:
            for b in bpe.itos[i]: out += b" " if b == 256 else bytes([b])
    return " ".join(out.decode("utf-8", errors="replace").split())

def doc_part(t, part, n):
    """The document tokens the model reads: the first n ("head"), or the first n/2 + the last n/2 ("head_tail")."""
    if len(t) <= n: return t
    if part == "head": return t[:n]
    assert part == "head_tail", f"unknown doc_part {part}"
    return np.concatenate([t[:n // 2], t[len(t) - (n - n // 2):]])

def target(s, n):
    """Training target: the first n summary tokens, + </s> if the whole summary fits."""
    return np.concatenate([s, [EOS]]).astype(np.int32) if len(s) <= n else s[:n]

# %%
# ---- ROUGE as in evaluate_phase6.py (rouge-score tokenizer with stemming, ROUGE-1 F1, ROUGE-L F1 by LCS) ----
try:
    from rouge_score import tokenizers as _rtok
    _ROUGE_TOK = _rtok.DefaultTokenizer(use_stemmer=True).tokenize
    ROUGE_NOTE = "rouge-score tokenizer with stemming (= evaluate_phase6.py)"
except Exception:
    _ROUGE_TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    ROUGE_NOTE = "rouge-score not installed: lower-cased alphanumeric tokens without stemming (approximate)"

def _lcs(a, b):
    if len(a) > len(b): a, b = b, a
    masks = {}
    for i, t in enumerate(a): masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        m = masks.get(t, 0); x = m | s; y = (s << 1) | 1; s = x & ~(x - y)
    return bin(s).count("1")

def rouge(preds, refs):
    r1 = rl = 0.0
    for p, r in zip(preds, refs):
        pt, rt = _ROUGE_TOK(p), _ROUGE_TOK(r)
        if not pt or not rt: continue
        ov = sum((Counter(pt) & Counter(rt)).values())
        r1 += 0.0 if ov == 0 else 2 * ov / (len(pt) + len(rt))
        z = _lcs(pt, rt); rl += 0.0 if z == 0 else 2 * z / (len(pt) + len(rt))
    return r1 / len(preds), rl / len(preds)

def repetition(texts, n=4):
    """Share of repeated word n-grams in the generated text (0 = no repetition)."""
    tot = rep = 0
    for t in texts:
        w = t.split(); g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        tot += len(g); rep += len(g) - len(set(g))
    return rep / max(1, tot)

# %%
# ---- decoders (m2 / m4 / m5) as generators: masks, KV cache, tied output layer ----
class DecoderLM(nn.Module):
    """[document] <s> [summary] </s>. prefix_bidir (m5): the document + <s> attend to each other in both directions
    (prefix-LM), the summary is causal. The forward pass is written here (same weights, same block maths as the Phase 1
    backbones) so that it can take a RoPE position offset and a KV cache."""
    def __init__(self, bb, prefix_bidir, n_pos):
        super().__init__()
        self.bb, self.prefix_bidir, self.is_a3 = bb, prefix_bidir, bb.kind == "a3"
        self.use_amp = True
        for b in bb.blocks:
            a = b.attn
            assert a.use_rope, "only RoPE decoders can take a position offset"
            a.rope_cos, a.rope_sin = rope_cache(n_pos, a.d_head, bb.c.get("rope_base", 10000.0))   # extended context

    def _attn(self, a, x, mask, start, cache, li):
        B, T, _ = x.shape
        q = a.q_proj(x).view(B, T, a.n_heads, a.d_head).transpose(1, 2)
        k = a.k_proj(x).view(B, T, a.n_kv, a.d_head).transpose(1, 2)
        v = a.v_proj(x).view(B, T, a.n_kv, a.d_head).transpose(1, 2)
        cos, sin = a.rope_cos[start:start + T][None, None].to(q.dtype), a.rope_sin[start:start + T][None, None].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:                            # preallocated (B, n_kv, max_len, d_head) per layer
            if cache["kv"][li] is None:
                shape = (B, a.n_kv, cache["max_len"], a.d_head)
                cache["kv"][li] = (torch.zeros(shape, dtype=k.dtype, device=k.device), torch.zeros(shape, dtype=v.dtype, device=v.device))
            K, V = cache["kv"][li]
            K[:, :, start:start + T] = k; V[:, :, start:start + T] = v
            k, v = K[:, :, :start + T], V[:, :, :start + T]
        out = scratch_attention(q, k, v, attn_mask=mask, scale=a.scale)
        return a.out_proj(out.transpose(1, 2).reshape(B, T, -1))

    def hidden(self, ids, mask, start=0, cache=None):
        bb, x = self.bb, self.bb.tok_emb(ids)
        for li, b in enumerate(bb.blocks):
            if self.is_a3 and b.norm_first:
                x = x + self._attn(b.attn, b.ln1(x), mask, start, cache, li)
                x = x + b.ffn.net(b.ln2(x))
            elif self.is_a3:
                x = b.ln1(x + self._attn(b.attn, x, mask, start, cache, li))
                x = b.ln2(x + b.ffn.net(x))
            else:
                x = x + self._attn(b.attn, b.attn_norm(x), mask, start, cache, li)
                h = b.mlp_norm(x)
                x = x + b.mlp.down(F.silu(b.mlp.gate(h)) * b.mlp.up(h))
        return (bb.ln_f if self.is_a3 else bb.final_norm)(x)

    def logits(self, h):
        return F.linear(h, self.bb.tok_emb.weight)       # tied output layer, as in Phase 1

    def mask(self, keep, pref):
        """keep (B, T) real tokens, pref (B,) prefix lengths -> (B, 1, T, T) bool, True = may attend."""
        T = keep.size(1); i = torch.arange(T, device=keep.device)
        allowed = (i[None, :] <= i[:, None])[None]                                   # causal
        if self.prefix_bidir:
            allowed = allowed | (i[None, None, :] < pref[:, None, None])             # the prefix: in both directions
        allowed = (allowed & keep[:, None, :]) | torch.eye(T, dtype=torch.bool, device=keep.device)[None]
        return allowed[:, None]

    def target_logits(self, batch):
        h = self.hidden(batch["ids"], self.mask(batch["keep"], batch["pref"]))
        return self.logits(h[batch["tgt_pos"]])

    @torch.no_grad()
    def generate(self, prompts, cfg, ban):
        """prompts: list of int arrays (document + <s>) -> list of generated token lists (without </s>)."""
        B, Lp = len(prompts), max(len(p) for p in prompts)
        ids = torch.zeros(B, Lp, dtype=torch.long); keep = torch.zeros(B, Lp, dtype=torch.bool)
        for b, p in enumerate(prompts):                  # left padding: every prompt ends at the same position
            ids[b, Lp - len(p):] = torch.from_numpy(p.astype(np.int64)); keep[b, Lp - len(p):] = True
        ids, keep = ids.to(DEVICE), keep.to(DEVICE)
        max_new = cfg["max_new_tokens"]
        cache = dict(kv=[None] * len(self.bb.blocks), max_len=Lp + max_new)
        keep_all = torch.cat([keep, torch.ones(B, max_new, dtype=torch.bool, device=DEVICE)], 1)
        pref = torch.full((B,), Lp, device=DEVICE)       # the whole prompt is the prefix
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            lg = self.logits(self.hidden(ids, self.mask(keep, pref), 0, cache)[:, -1]).float()
            gens, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool, device=DEVICE)
            for step in range(max_new):
                tok = ban(lg, gens, done, step).argmax(-1)
                tok = torch.where(done, torch.full_like(tok, EOS), tok)
                tl = tok.tolist(); dl = done.tolist()
                for b in range(B):
                    if not dl[b] and tl[b] != EOS: gens[b].append(tl[b])
                done |= tok == EOS
                if bool(done.all()) or step == max_new - 1: break
                cur = Lp + step
                lg = self.logits(self.hidden(tok[:, None], keep_all[:, None, None, :cur + 1], cur, cache)[:, 0]).float()
        return gens

# %%
# ---- m3: the Phase 1 A-2 GRU encoder-decoder (biGRU encoder + bridge, Bahdanau attention, GRU decoder, tied output) ----
class GRUSeq2Seq(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c; H, E = c["hidden"], c["emb_dim"]
        self.kind = "gru"
        self.use_amp = False                             # float32 (GRU cells in bf16 are not safe on every GPU)
        self.embedding = nn.Embedding(c["vocab_size"], E)
        self.enc_rnn = nn.GRU(E, H, num_layers=c["enc_layers"], batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * H, H)
        self.attention = nn.Module()
        self.attention.W_a = nn.Linear(H, c["attn_dim"]); self.attention.U_a = nn.Linear(2 * H, c["attn_dim"])
        self.attention.v_a = nn.Linear(c["attn_dim"], 1)
        self.cells = nn.ModuleList([nn.GRUCell(E + 2 * H if l == 0 else H, H) for l in range(c["dec_layers"])])
        self.out_proj = nn.Linear(H + 2 * H, E)

    def encode(self, src, mask):
        self.enc_rnn.flatten_parameters()
        x = self.embedding(src)
        packed = pack_padded_sequence(x, mask.sum(1).clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.enc_rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h0 = torch.tanh(self.bridge(torch.cat([h[-2], h[-1]], dim=1)))
        return out, self.attention.U_a(out), [h0 for _ in self.cells]

    def step(self, x, states, enc_out, keys, mask):
        """One decoder step (x = embedded input token) -> (output feature (B, E), new states); logits = feature @ embedding^T.
        Training embeds all decoder inputs once (one embedding backward instead of one dense 50k x E gradient per step)."""
        a = self.attention
        sc = a.v_a(torch.tanh(a.W_a(states[-1]).unsqueeze(1) + keys)).squeeze(-1).masked_fill(~mask, -1e4)
        ctx = torch.bmm(torch.softmax(sc, 1).unsqueeze(1), enc_out).squeeze(1)
        inp, new = torch.cat([x, ctx], 1), []
        for l, cell in enumerate(self.cells):
            h = cell(inp, states[l]); new.append(h); inp = h
        return self.out_proj(torch.cat([new[-1], ctx], 1)), new

    def logits(self, f):
        return F.linear(f, self.embedding.weight)

    def target_logits(self, batch):
        enc_out, keys, states = self.encode(batch["src"], batch["src_mask"])
        emb, mask, feats = self.embedding(batch["dec_in"]), batch["src_mask"], []
        K = 64                                           # decoder steps per checkpointed segment (memory: states only)
        def seg(e, enc_out, keys, *st):
            st, out = list(st), []
            for t in range(e.size(1)):
                f, st = self.step(e[:, t], st, enc_out, keys, mask); out.append(f)
            return (torch.stack(out, 1), *st)
        for t0 in range(0, emb.size(1), K):
            e = emb[:, t0:t0 + K]
            if self.training:
                r = torch.utils.checkpoint.checkpoint(seg, e, enc_out, keys, *states, use_reentrant=False)
            else:
                r = seg(e, enc_out, keys, *states)
            feats.append(r[0]); states = list(r[1:])
        return self.logits(torch.cat(feats, 1)[batch["tgt_pos"]])

    @torch.no_grad()
    def generate(self, prompts, cfg, ban):
        B, L = len(prompts), max(len(p) for p in prompts)
        src = torch.zeros(B, L, dtype=torch.long); mask = torch.zeros(B, L, dtype=torch.bool)
        for b, p in enumerate(prompts):                  # right padding (the encoder packs the sequences)
            src[b, :len(p)] = torch.from_numpy(p.astype(np.int64)); mask[b, :len(p)] = True
        src, mask = src.to(DEVICE), mask.to(DEVICE)
        enc_out, keys, states = self.encode(src, mask)
        tok = torch.full((B,), BOS, dtype=torch.long, device=DEVICE)
        gens, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool, device=DEVICE)
        for step in range(cfg["max_new_tokens"]):
            f, states = self.step(self.embedding(tok), states, enc_out, keys, mask)
            tok = ban(self.logits(f).float(), gens, done, step).argmax(-1)
            tok = torch.where(done, torch.full_like(tok, EOS), tok)
            tl = tok.tolist(); dl = done.tolist()
            for b in range(B):
                if not dl[b] and tl[b] != EOS: gens[b].append(tl[b])
            done |= tok == EOS
            if bool(done.all()): break
        return gens

def load_gru_seq2seq(key):
    """The whole Phase 1 m3 checkpoint (encoder + decoder) -> (GRUSeq2Seq, info)."""
    folder = find_checkpoint(key)
    names = os.listdir(folder)
    pt = [n for n in names if n.endswith("_model.pt")]
    if pt:
        sd = torch.load(os.path.join(folder, pt[0]), map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(folder, [n for n in names if n.endswith(".safetensors")][0]))
    sd = _strip(sd)
    c = dict(vocab_size=sd["embedding.weight"].shape[0], emb_dim=sd["embedding.weight"].shape[1],
             hidden=sd["enc_rnn.weight_hh_l0"].shape[1],
             enc_layers=len([k for k in sd if re.fullmatch(r"enc_rnn\.weight_ih_l\d+", k)]),
             dec_layers=len([k for k in sd if re.fullmatch(r"cells\.\d+\.weight_ih", k)]),
             attn_dim=sd["attention.W_a.weight"].shape[0])
    model = GRUSeq2Seq(c)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"m3 weights missing from the checkpoint: {missing[:5]}"
    tok = [os.path.join(folder, n) for n in names if n.startswith("bpe_tokenizer") and n.endswith(".json")]
    for where in (os.path.join(os.path.dirname(folder), "**"), "/kaggle/input/**"):
        tok = tok or sorted(glob.glob(os.path.join(where, "bpe_tokenizer*.json"), recursive=True))
    assert tok, f"no bpe_tokenizer*.json found for {key}"
    return model, dict(folder=folder, kind="gru", config=c, full_layers=c["enc_layers"], n_layers=c["enc_layers"],
                       unused_keys=len(unexpected), tokenizer=tok[0], prefix_lm=None)

# %%
# ---- adaptation methods ----
class HoulsbyAdapter(nn.Module):
    """Houlsby et al. 2019: a bottleneck adapter on a sublayer's output, before the residual add:
    y = W x;  y + W_up gelu(W_down y), W_up = 0 at the start (the model starts as the pretrained one)."""
    def __init__(self, base, dim):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.out_features, dim); self.up = nn.Linear(dim, base.out_features)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))

def setup_method(model, cfg):
    """Freezes / wraps parameters for the run's method -> list of (param group name, params, lr)."""
    method = cfg["method"]
    bb = model if isinstance(model, GRUSeq2Seq) else model.bb
    for p in model.parameters(): p.requires_grad = method in ("full", "kd")
    adapters = []
    if method in ("lora", "lorma"):
        assert not isinstance(model, GRUSeq2Seq), f"{method} is implemented for the decoders' Linear layers"
        names = inject(bb, {"lora": LoRALinear, "lorma": LoRMALinear}[method], cfg["adapter_targets"], cfg["rank"], cfg["alpha"])
        adapters = [p for n, p in bb.named_parameters() if n.endswith((".A", ".B"))]
        log(f"{method}: rank {cfg['rank']}, alpha {cfg['alpha']}, {len(names)} adapted Linear layers, e.g. {names[:4]}")
    elif method == "houlsby":
        assert not isinstance(model, GRUSeq2Seq), "houlsby is implemented for the decoders"
        for i, b in enumerate(bb.blocks):                # after attention and after the MLP of every block
            b.attn.out_proj = HoulsbyAdapter(b.attn.out_proj, cfg["adapter_dim"])
            if bb.kind == "a3": b.ffn.net[2] = HoulsbyAdapter(b.ffn.net[2], cfg["adapter_dim"])
            else: b.mlp.down = HoulsbyAdapter(b.mlp.down, cfg["adapter_dim"])
        adapters = [p for m in bb.modules() if isinstance(m, HoulsbyAdapter) for p in (*m.down.parameters(), *m.up.parameters())]
        norms = [p for n, p in bb.named_parameters() if re.search(r"(ln1|ln2|ln_f|attn_norm|mlp_norm|final_norm)\.", n)]
        adapters += norms                                # Houlsby also trains the layer norms
        log(f"houlsby: bottleneck {cfg['adapter_dim']}, {2 * len(bb.blocks)} adapters, {len(norms)} norm tensors trainable")
    else:
        assert method in ("full", "kd"), f"unknown method {method}"
    for p in adapters: p.requires_grad = True
    ad = {id(p) for p in adapters}
    return [("adapter", adapters, cfg["lr_adapter"]),
            ("backbone", [p for p in model.parameters() if p.requires_grad and id(p) not in ad], cfg["lr_backbone"])]

def build_model(cfg, n_layers=None):
    """-> (generator model, info) for cfg["model"] (decoders wrapped in DecoderLM)."""
    if cfg["model"] == "m3":
        return load_gru_seq2seq("m3")
    bb, info = load_backbone(cfg["model"], n_layers)
    if cfg["model"] in ("m4", "m5"):                     # same file format: make sure it is the right checkpoint
        assert bool(info["prefix_lm"]) == (cfg["model"] == "m5"), (
            f"{cfg['model']} loaded a checkpoint with prefix_lm={info['prefix_lm']} - wrong MODEL_DIRS['{cfg['model']}']?")
    n_pos = max(cfg["ctx_len"], cfg["doc_tokens"] + 1 + cfg["max_new_tokens"]) + 8
    return DecoderLM(bb, prefix_bidir=bool(info["prefix_lm"]), n_pos=n_pos), info

# %%
# ---- batches ----
def examples(toks, ids, cfg, gru):
    """doc id -> (encoder / prompt ids, target ids)."""
    return {d: (doc_part(toks[d][0], cfg["doc_part"], cfg["doc_tokens"]), target(toks[d][1], cfg["sum_tokens"])) for d in ids}

def ex_len(ex, gru):
    s, t = ex
    return len(t) if gru else len(s) + 1 + len(t)

def make_batches(ids, ex, cfg, gru, rng=None):
    """Groups of similar length (pools of 64 sorted by length): decoders <= batch_tokens tokens, m3 gru_batch documents."""
    ids = list(ids)
    if rng: rng.shuffle(ids)
    out = []
    for i in range(0, len(ids), 64):
        pool, cur, mx = sorted(ids[i:i + 64], key=lambda d: ex_len(ex[d], gru)), [], 0
        for d in pool:
            L = ex_len(ex[d], gru)
            full = len(cur) >= cfg["gru_batch"] if gru else (len(cur) + 1) * max(mx, L) > cfg["batch_tokens"]
            if cur and full: out.append(cur); cur, mx = [], 0
            cur.append(d); mx = max(mx, L)
        if cur: out.append(cur)
    if rng: rng.shuffle(out)
    return out

def to_batch(grp, ex, gru):
    if gru:
        S = max(len(ex[d][0]) for d in grp); Td = max(len(ex[d][1]) for d in grp)
        src = torch.zeros(len(grp), S, dtype=torch.long); sm = torch.zeros(len(grp), S, dtype=torch.bool)
        di = torch.zeros(len(grp), Td, dtype=torch.long); tg = torch.full((len(grp), Td), -100, dtype=torch.long)
        for b, d in enumerate(grp):
            s, t = ex[d]; s = torch.from_numpy(s.astype(np.int64)); t = torch.from_numpy(t.astype(np.int64))
            src[b, :len(s)] = s; sm[b, :len(s)] = True
            di[b, 0] = BOS; di[b, 1:len(t)] = t[:-1]; tg[b, :len(t)] = t
        b = dict(src=src, src_mask=sm, dec_in=di, tgt_pos=tg != -100)
        b = {k: v.to(DEVICE) for k, v in b.items()}
        b["targets"] = tg.to(DEVICE)[b["tgt_pos"]]
        return b
    seqs = [np.concatenate([ex[d][0], [BOS], ex[d][1]]).astype(np.int64) for d in grp]
    T = max(len(s) for s in seqs)
    ids = torch.zeros(len(grp), T, dtype=torch.long); keep = torch.zeros(len(grp), T, dtype=torch.bool)
    pos = torch.zeros(len(grp), T, dtype=torch.bool); nxt = torch.zeros(len(grp), T, dtype=torch.long)
    pref = torch.tensor([len(ex[d][0]) + 1 for d in grp])
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.from_numpy(s); keep[b, :len(s)] = True
        pos[b, int(pref[b]) - 1:len(s) - 1] = True            # the position before each summary token predicts it
        nxt[b, :len(s) - 1] = torch.from_numpy(s[1:])
    b = dict(ids=ids, keep=keep, pref=pref, tgt_pos=pos)
    b = {k: v.to(DEVICE) for k, v in b.items()}
    b["targets"] = nxt.to(DEVICE)[b["tgt_pos"]]
    return b

# %%
# ---- generation: greedy, no repeated n-grams, minimum length; never <pad> / <unk> / <s> ----
def make_ban(cfg):
    n = cfg["no_repeat_ngram"]
    seen = None
    def ban(lg, gens, done, step):
        nonlocal seen
        if seen is None or step == 0: seen = [dict() for _ in gens]
        lg[:, [PAD, UNK, BOS]] = float("-inf")
        if step < cfg["min_new_tokens"]: lg[:, EOS] = float("-inf")
        if n:
            rows, cols = [], []
            for b, g in enumerate(gens):
                if len(g) >= n:                         # remember the n-gram that ended at the last step
                    seen[b].setdefault(tuple(g[-n:-1]), set()).add(g[-1])
                if len(g) >= n - 1:
                    for t in seen[b].get(tuple(g[len(g) - n + 1:]), ()): rows.append(b); cols.append(t)
            if rows: lg[rows, cols] = float("-inf")
        return lg
    return ban

def generate_texts(model, bpe, toks, ids, cfg, gru):
    """-> {doc id: summary text}, seconds, share of summaries that hit max_new_tokens."""
    model.eval(); t0 = time.time(); out, hit = {}, 0
    prompts = {d: doc_part(toks[d][0], cfg["doc_part"], cfg["doc_tokens"]) for d in ids}
    if not gru: prompts = {d: np.concatenate([p, [BOS]]).astype(np.int32) for d, p in prompts.items()}
    order = sorted(ids, key=lambda d: len(prompts[d]))
    for i in range(0, len(order), cfg["gen_batch"]):
        grp = order[i:i + cfg["gen_batch"]]
        gens = model.generate([prompts[d] for d in grp], cfg, make_ban(cfg))
        for d, g in zip(grp, gens):
            hit += len(g) >= cfg["max_new_tokens"]
            text = detok(bpe, g)
            if not text:                                 # the evaluator rejects empty summaries
                text = " ".join(DOCS[d]["text"].split()[:60]); log(f"  empty summary for {d}: the first 60 document words")
            out[d] = text
    return out, time.time() - t0, hit / max(1, len(ids))

# %%
# ---- one experiment ----
RESULTS, TEACHERS = [], {}
REF_PATH = os.path.join(WORK, "heldout_references.json")       # truth file for evaluate_phase6.py: {id: reference}
json.dump({d: DOCS[d]["summary"] for d in GEN_VAL_IDS}, open(REF_PATH, "w", encoding="utf-8"), ensure_ascii=False)

def count_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))

@torch.no_grad()
def eval_loss(model, ex, ids, cfg, gru):
    model.eval(); tot = n = 0.0
    for grp in make_batches(ids, ex, cfg, gru):
        b = to_batch(grp, ex, gru)
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda" and model.use_amp):
            lg = model.target_logits(b)
        tot += F.cross_entropy(lg.float(), b["targets"], reduction="sum").item(); n += len(b["targets"])
    return tot / max(1, n)

def load_teacher(name):
    t = TEACHERS.get(name)
    assert t, f"teacher run {name} has not finished in this session (it must be in RUNS / RUN_ONLY and must not fail)"
    model, _ = build_model(t["cfg"], t["cfg"]["student_layers"])
    setup_method(model, t["cfg"])
    model.load_state_dict(t["state"], strict=False)
    for p in model.parameters(): p.requires_grad = False
    return model.to(DEVICE).eval()

def run_experiment(run):
    cfg = {**COMMON, **run}
    if SMOKE: cfg.update(epochs=1, max_new_tokens=48, min_new_tokens=8)
    name = cfg["name"]
    out_dir = os.path.join(WORK, name); os.makedirs(out_dir, exist_ok=True)
    section(f"RUN {name}: model {cfg['model']}, method {cfg['method']}, reads {cfg['doc_part']} {cfg['doc_tokens']} tokens")
    log("config:", json.dumps({k: v for k, v in cfg.items() if k not in COMMON or COMMON[k] != v}))
    set_seed(cfg["seed"])
    if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
    gru = cfg["model"] == "m3"
    if not gru:
        assert cfg["doc_tokens"] + cfg["sum_tokens"] + 2 <= cfg["ctx_len"] + 2, "document + summary longer than ctx_len"

    model, info = build_model(cfg, cfg["student_layers"])
    log(f"checkpoint {info['folder']} (kind {info['kind']}, layers {info['n_layers']}/{info['full_layers']}, "
        f"prefix_lm={info['prefix_lm']}, unused checkpoint tensors {info['unused_keys']})")
    bpe, toks = get_tokens(info["tokenizer"])
    groups = setup_method(model, cfg)
    model.to(DEVICE)
    n_total, n_train = count_params(model), count_params(model, True)
    log(f"parameters: total {n_total / 1e6:.2f}M, trainable {n_train / 1e6:.3f}M ({100 * n_train / n_total:.2f}%)")

    teacher = None
    if cfg["method"] == "kd":
        teacher = load_teacher(cfg["teacher"])
        tb = TEACHERS[cfg["teacher"]]["cfg"]
        assert (tb["doc_part"], tb["doc_tokens"], tb["sum_tokens"]) == (cfg["doc_part"], cfg["doc_tokens"], cfg["sum_tokens"])
        log(f"kd: teacher {cfg['teacher']} ({count_params(teacher) / 1e6:.1f}M parameters), temperature "
            f"{cfg['kd_temperature']}, alpha {cfg['kd_alpha']} (loss = (1 - alpha) CE(reference) + alpha T^2 KL(teacher || student))")

    ex = examples(toks, TRAIN_IDS + VAL_IDS, cfg, gru)
    n_tok = sum(ex_len(ex[d], gru) for d in TRAIN_IDS)
    n_batches = len(make_batches(TRAIN_IDS, ex, cfg, gru))
    steps = math.ceil(n_batches / cfg["grad_accum"]) * cfg["epochs"]
    pg = [dict(params=[p for p in ps if p.ndim >= 2], lr=lr, weight_decay=cfg["weight_decay"]) for _, ps, lr in groups] + \
         [dict(params=[p for p in ps if p.ndim < 2], lr=lr, weight_decay=0.0) for _, ps, lr in groups]
    opt = torch.optim.AdamW([g for g in pg if g["params"]])
    warm = max(1, int(cfg["warmup_frac"] * steps))
    plan = dict(batches=n_batches * cfg["epochs"], steps=steps, warm=warm)    # refitted to the time budget at batch 30
    lr_at = lambda s: ((s + 1) / plan["warm"] if s < plan["warm"]                # warmup, linear decay to 0 at the planned end
                       else max(0.0, (plan["steps"] - s) / max(1, plan["steps"] - plan["warm"])))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = DEVICE.type == "cuda" and model.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and AMP_DTYPE == torch.float16))
    log(f"train: {len(TRAIN_IDS)} documents, {n_tok / 1e6:.2f}M {'target' if gru else 'sequence'} tokens/epoch, "
        f"~{n_batches} batches/epoch, optimizer steps ~{steps}, warmup {warm}; held-out {len(VAL_IDS)} documents")

    best, best_state, history = float("inf"), None, []
    rng = random.Random(cfg["seed"])
    t_train, done, stop, budget = time.time(), 0, False, 60 * cfg["train_minutes"]
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time()
        bl = make_batches(TRAIN_IDS, ex, cfg, gru, rng)
        tot = kdt = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for bi, grp in enumerate(bl):
            if done == 30 and not SMOKE:                 # fit the schedule to the batches that fit in the time budget
                fit = int(budget / ((time.time() - t_train) / 30))
                if fit < plan["batches"]:
                    plan.update(batches=fit, steps=math.ceil(fit / cfg["grad_accum"]))
                    plan["warm"] = max(1, int(cfg["warmup_frac"] * plan["steps"]))
                log(f"    {(time.time() - t_train) / 30:.2f} s/batch -> {plan['batches']} batches in the "
                    f"{cfg['train_minutes']} min budget ({plan['batches'] / n_batches:.2f} epochs), {plan['steps']} optimizer steps")
            if done >= plan["batches"] or time.time() - t_train > budget:
                stop = True; break
            done += 1
            b = to_batch(grp, ex, gru)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=amp):
                lg = model.target_logits(b)
            loss = F.cross_entropy(lg.float(), b["targets"])
            if teacher is not None:                      # word-level KD on the summary tokens
                T_ = cfg["kd_temperature"]
                with torch.no_grad(), torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=amp):
                    tl = teacher.target_logits(b)
                kd = F.kl_div(F.log_softmax(lg.float() / T_, -1), F.log_softmax(tl.float() / T_, -1),
                              log_target=True, reduction="batchmean") * T_ * T_
                kdt += kd.item()
                loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kd
            scaler.scale(loss / cfg["grad_accum"]).backward()
            nb += 1; tot += loss.item()
            if nb % cfg["grad_accum"] == 0 or bi == len(bl) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if bi % 100 == 0 and bi:
                el = time.time() - t0
                log(f"    batch {bi}/{len(bl)}: loss {tot / nb:.4f}" + (f" (kd {kdt / nb:.4f})" if teacher is not None else "")
                    + f", {n_tok * bi / len(bl) / el:.0f} tokens/s, epoch ETA {el / bi * (len(bl) - bi) / 60:.0f} min")
        t_ep = time.time() - t0
        if nb == 0: break
        vl = eval_loss(model, ex, VAL_IDS, cfg, gru)
        history.append(dict(epoch=ep + nb / len(bl), train_loss=tot / nb, val_loss=vl, val_ppl=math.exp(min(vl, 50)),
                            train_batches=nb, train_seconds=t_ep, seconds=time.time() - t0))
        log(f"  epoch {ep + nb / len(bl):.2f}/{cfg['epochs']}: train loss {tot / nb:.4f} | held-out summary loss {vl:.4f} "
            f"(ppl {math.exp(min(vl, 50)):.2f}) | train {t_ep:.0f}s ({n_tok * nb / len(bl) / t_ep:.0f} tokens/s), "
            f"eval {time.time() - t0 - t_ep:.0f}s")
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        if stop: break
    train_secs = time.time() - t_train
    model.load_state_dict(best_state, strict=False)
    best_ep = round(min(history, key=lambda h: h["val_loss"])["epoch"], 2)
    if any(r.get("teacher") == name for r in SELECTED):  # a KD run of this session uses this run as its teacher
        TEACHERS[name] = dict(cfg=cfg, state=best_state)
    del teacher; gc.collect()

    # held-out summaries (ROUGE here; BERTScore with evaluate_phase6.py on heldout_generations.json)
    gv, gv_secs, gv_hit = generate_texts(model, bpe, toks, GEN_VAL_IDS, cfg, gru)
    r1, rl = rouge([gv[d] for d in GEN_VAL_IDS], [DOCS[d]["summary"] for d in GEN_VAL_IDS])
    words = np.array([len(gv[d].split()) for d in GEN_VAL_IDS]); ref_w = np.array([len(DOCS[d]["summary"].split()) for d in GEN_VAL_IDS])
    rep = repetition(list(gv.values()))
    json.dump(gv, open(os.path.join(out_dir, "heldout_generations.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({d: DOCS[d]["summary"] for d in GEN_VAL_IDS}, open(os.path.join(out_dir, "heldout_references.json"), "w",
              encoding="utf-8"), ensure_ascii=False)   # truth file for evaluate_phase6.py
    log(f"best epoch {best_ep}: held-out ({len(GEN_VAL_IDS)}) ROUGE-1 {r1:.4f}, ROUGE-L {rl:.4f} [{ROUGE_NOTE}]; "
        f"{words.mean():.0f} words / summary (references {ref_w.mean():.0f}), {100 * gv_hit:.0f}% hit max_new_tokens, "
        f"repeated 4-grams {100 * rep:.1f}%; generation {gv_secs:.0f}s")
    log(f"  example {GEN_VAL_IDS[0]}: {gv[GEN_VAL_IDS[0]][:400]}")
    other_parts = {}
    for part in cfg["eval_doc_parts"]:                   # the same model reading another part of the document
        g2, s2, _ = generate_texts(model, bpe, toks, GEN_VAL_IDS, dict(cfg, doc_part=part), gru)
        p1, pl = rouge([g2[d] for d in GEN_VAL_IDS], [DOCS[d]["summary"] for d in GEN_VAL_IDS])
        other_parts[part] = dict(rouge1=p1, rougeL=pl, words=float(np.mean([len(g2[d].split()) for d in GEN_VAL_IDS])))
        json.dump(g2, open(os.path.join(out_dir, f"heldout_generations_{part}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"  held-out reading {part} {cfg['doc_tokens']} tokens instead: ROUGE-1 {p1:.4f}, ROUGE-L {pl:.4f}, "
            f"{other_parts[part]['words']:.0f} words / summary ({s2:.0f}s)")

    # test summaries -> submission
    gt, gt_secs, gt_hit = generate_texts(model, bpe, toks, TEST_IDS, cfg, gru)
    preds = [{"id": d, "summary": gt[d]} for d in TEST_IDS]
    assert len(preds) == len(TEST_IDS) and all(p["summary"].strip() for p in preds)
    with open(os.path.join(out_dir, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out_dir, "prediction.json"), "prediction.json")
    log(f"test: {len(TEST_IDS)} summaries, {np.mean([len(gt[d].split()) for d in TEST_IDS]):.0f} words each, "
        f"{100 * gt_hit:.0f}% hit max_new_tokens, generation {gt_secs:.0f}s")

    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0.0
    res = dict(name=name, model=cfg["model"], method=cfg["method"], doc_part=cfg["doc_part"], doc_tokens=cfg["doc_tokens"],
               layers=f"{info['n_layers']}/{info['full_layers']}", total_params=n_total, trainable_params=n_train,
               best_epoch=best_ep, val_loss=best, heldout_rouge1=r1, heldout_rougeL=rl, heldout_words=float(words.mean()),
               heldout_ref_words=float(ref_w.mean()), heldout_hit_max=gv_hit, heldout_repeated_4grams=rep, heldout_other_doc_parts=other_parts,
               train_minutes=train_secs / 60, heldout_gen_seconds=gv_secs, test_gen_seconds=gt_secs, peak_gpu_gb=peak,
               history=history, config=cfg, checkpoint=info["folder"])
    json.dump(res, open(os.path.join(out_dir, "val_metrics.json"), "w"), indent=1)
    if cfg["save_weights"]:
        torch.save(dict(trainable={k: v.to(torch.bfloat16) for k, v in best_state.items()}, config=cfg,
                        model_config=info["config"]), os.path.join(out_dir, "trainable_weights.pt"))
    log(f"train {train_secs / 60:.1f} min, peak GPU memory {peak:.2f} GB, outputs in {out_dir}")
    RESULTS.append(res)
    del model, opt; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return res

# %%
# ---- summary table + one zip to download (rewritten after every run, so a cut-off session keeps finished runs) ----
_bundle_lock = threading.Lock()
def write_summary():
    with _bundle_lock:
        section("SUMMARY")
        log(f"{'run':16s} {'model':5s} {'method':8s} {'input':15s} {'layers':6s} {'total':>8s} {'trainable':>10s} {'ep':>4s} "
            f"{'val loss':>8s} {'R-1':>6s} {'R-L':>6s} {'words':>5s} {'rep4':>5s} {'train min':>9s} {'gen s':>5s} {'peak GB':>7s}")
        for r in RESULTS:
            log(f"{r['name']:16s} {r['model']:5s} {r['method']:8s} {r['doc_part'] + ':' + str(r['doc_tokens']):15s} "
                f"{r['layers']:6s} {r['total_params'] / 1e6:7.1f}M {r['trainable_params'] / 1e6:9.3f}M {r['best_epoch']:4.2f} "
                f"{r['val_loss']:8.4f} {r['heldout_rouge1']:6.4f} {r['heldout_rougeL']:6.4f} {r['heldout_words']:5.0f} "
                f"{100 * r['heldout_repeated_4grams']:4.1f}% {r['train_minutes']:9.1f} {r['test_gen_seconds']:5.0f} {r['peak_gpu_gb']:7.2f}")
        json.dump(RESULTS, open(os.path.join(WORK, "phase6_results_summary.json"), "w"), indent=1)
        _log.flush()
        bundle = os.path.join(WORK, "phase6_results_bundle.zip")
        with zipfile.ZipFile(bundle + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
            for f in glob.glob(os.path.join(WORK, "**", "*"), recursive=True):
                if f.endswith((".txt", ".json", ".zip", "trainable_weights.pt")) and "phase6_results_bundle" not in f:
                    z.write(f, os.path.relpath(f, WORK))
        os.replace(bundle + ".tmp", bundle)
        log(f"bundle: {bundle} ({os.path.getsize(bundle) / 1e6:.1f} MB)")

# %%
# ---- run everything: one worker per GPU, each takes the next run from the queue (2 x T4 -> 2 runs at a time) ----
# torch.cuda.set_device is per thread, and DEVICE = "cuda" means "this thread's GPU", so every run stays on its GPU.
# A KD run waits until its teacher (a run of this session) has finished.
SELECTED = [r for r in RUNS if not RUN_ONLY or r["name"] in RUN_ONLY]
GPU_IDS = json.loads(os.environ.get("P6_GPU_IDS", "null")) or (list(range(torch.cuda.device_count())) if DEVICE.type == "cuda" else [None])
GPU_IDS = GPU_IDS[:max(1, len(SELECTED))]
_queue, _qlock, _finished = list(SELECTED), threading.Lock(), set()

def _worker(gpu):
    if gpu is not None: torch.cuda.set_device(gpu)
    while True:
        with _qlock:
            if not _queue: return
            ready = [r for r in _queue if not r.get("teacher") or r["teacher"] in _finished]
            run = ready[0] if ready else None
            if run: _queue.remove(run)
        if run is None:
            time.sleep(20); continue
        log(f"starting {run['name']}")
        try:
            run_experiment(run)
            write_summary()
        except Exception as e:                           # one failing run must not lose the others
            import traceback
            log(f"RUN {run['name']} FAILED: {e!r}\n{traceback.format_exc()}")
            gc.collect()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
        finally:
            with _qlock: _finished.add(run["name"])

section(f"RUNNING {[r['name'] for r in SELECTED]} on {len(GPU_IDS)} worker(s): GPUs {GPU_IDS}")
_threads = [threading.Thread(target=_worker, args=(g,), name=f"gpu{g if g is not None else 'cpu'}-{i}") for i, g in enumerate(GPU_IDS)]
for t in _threads: t.start()
for t in _threads: t.join()
write_summary()
