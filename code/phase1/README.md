# Phase 1: pretraining and decoding

| File | What it does |
|---|---|
| `model2_rope_gqa_decoder_kaggle.ipynb` | M2: 10-layer decoder with RoPE + grouped-query attention (101.4M) |
| `model3_gru_encoder_decoder_kaggle.ipynb` | M3: GRU encoder–decoder with Bahdanau attention (91.7M) |
| `model4_modern_decoder_kaggle.ipynb` | **M4 (best)**: RMSNorm, SwiGLU, RoPE, GQA, no biases, chunked cross-entropy (101.3M) |
| `model5_prefix_lm_kaggle.ipynb` | M5: the M4 network trained as a prefix-LM (bidirectional prompt, loss on the continuation) |
| `tokenizer_mixing_study_kaggle.ipynb` | side study: BPE tokenizer fertility for different domain mixes |
| `infer.py` | KV-cache decoder for M4 and the decoding strategies (greedy, n-gram block, repetition / frequency penalty, top-p, min-p, beam, contrastive) and the pseudo-validation set |
| `mbr.py` | minimum Bayes risk selection over a pool of candidate continuations (BERTScore / ROUGE-L utility) |

Kaggle: GPU T4 ×2, secret `HF`. Each pretraining notebook trains one model for one epoch on the Supreme Court file
and saves the checkpoint + `bpe_tokenizer.json`; these checkpoints (uploaded as Kaggle datasets) are the starting
point of every later phase.

Best submission: M4 with MBR decoding, `python mbr.py --split test --utility bert` (CodaBench 0.8407): 14 candidates
per prompt (six greedy/beam variants, eight seeded top-p/min-p samples), keeping the one with the highest mean
BERTScore against the others. Best single setting: `python infer.py`, greedy, 4-gram block, repetition penalty 1.2
(CodaBench 0.8363).
