# Practice: small from-scratch implementations

Before wiring a technique into a phase notebook, I implemented its core idea in isolation, as a small PyTorch file.
Each file has the same structure:

- a docstring with the idea, the paper and the torch calls needed;
- the module or function, written step by step following numbered `# Step` comments;
- a self-test at the bottom that prints values next to an `# expect ...` comment.

Run a file directly, for example `python lorma.py`. CPU is enough. The files need only `torch` (and `numpy` in one
file). Some build on each other: `rslora.py` imports `lora.py`, `attention.py` imports `rope.py`, and
`modern_block.py` imports `attention.py`.

These are learning exercises, not the code behind any submission. The versions used in the competition are inside
the phase notebooks in [`code/`](../code).

## Files

| File | What it implements | Paper | Status |
|---|---|---|---|
| `attention.py` | causal multi-head self-attention, grouped-query attention, optional RoPE | Ainslie et al. 2023 | done, self-test passes |
| `lora.py` | LoRA: additive low-rank update `W0 + (α/r)·BA` on a frozen `nn.Linear`, plus layer injection | Hu et al. 2021 | done, self-test passes |
| `rslora.py` | rsLoRA: LoRA with the scale `α/√r` | Kalajdzievski 2023 | done, self-test passes |
| `lorma.py` | LoRMA: multiplicative update `(I + (α/r)·BA)·W0`, computed as `z + (α/r)·B(Az)` with `z = W0x` | Bihany et al. 2025 | done, self-test passes |
| `rope.py` | rotary position embeddings, rotate-half layout | Su et al. 2021 | skeleton |
| `modern_block.py` | RMSNorm, SwiGLU MLP, pre-norm transformer block (Models 4/5) | — | skeleton |
| `bitfit.py` | train only the bias terms | Ben Zaken et al. 2022 | skeleton |
| `ia3.py` | learned scaling vectors on keys, values and the FFN | Liu et al. 2022 | skeleton |
| `qlora.py` | LoRA on a 4-bit NF4-quantised frozen weight | Dettmers et al. 2023 | skeleton |
| `adalora.py` | SVD-shaped LoRA that prunes rank to a budget during training | Zhang et al. 2023 | skeleton |
| `prompt_tuning.py` | learned virtual-token embeddings before the input | Lester et al. 2021 | skeleton |
| `prefix_tuning.py` | learned key/value prefixes inside every attention layer | Li & Liang 2021 | skeleton |
| `kd.py` | distillation loss with temperature (teacher → student) | Hinton et al. 2015 | skeleton |
| `sentence_head.py` | sentence-level head: mean-pool tokens per sentence (Phase 3) | — | skeleton |
| `crf.py` | linear-chain CRF over BIO tags with transition constraints (Phases 2–3) | Lafferty et al. 2001 | skeleton |

"Skeleton" means the structure, docstring and self-test are in place but the body still raises
`NotImplementedError`.
