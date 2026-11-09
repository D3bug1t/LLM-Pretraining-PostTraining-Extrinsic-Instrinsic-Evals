# CS781 Capstone: Small From-Scratch Language Models for Indian Legal Text

Sanskaar Srivastava (220967), IIT Kanpur, CS781.

Five language models (58–101M parameters) are pretrained from scratch on Indian Supreme Court judgments (Phase 1). The
four ~100M models are then adapted to six legal tasks from the IL-TUR benchmark (Phases 2–7). Every adaptation starts from
our own Phase 1 weights and uses only the official data: no pretrained checkpoints, no outside data, no LLM APIs. All
training ran on free Kaggle GPUs (2 × Tesla T4).

| Phase | Task | Metric | Best CodaBench | Best system |
|---|---|---|---|---|
| 1 | Continue a 512-word legal passage by 128 words | BERTScore F1 | **0.8407** | Model 4 (modern decoder), MBR over 14 decoded candidates (BERTScore utility) |
| 2 | Legal named entity recognition (12 types) | strict macro-F1 | **0.5122** | M5 + LoRMA + CRF + preamble context, vote + propagate |
| 3 | Rhetorical role of every sentence (13 roles) | macro-F1 | **0.5713** | ensemble of 4 M5 CRF runs, prior + Viterbi decoding |
| 4 | Court judgment prediction (binary) | macro-F1 | **0.7609** | ensemble of 4 M5 runs on all 31.7k cases, last 8×512 tokens, dev threshold |
| 5 | Statute identification (100 statutes, multi-label) | macro-F1 | **0.3664** | M4 full + LoRMA ensemble, per-statute thresholds |
| 6 | Abstractive summarization | BERTScore F1 | **0.8381** | M4 + LoRA, greedy with repetition penalty 1.15 |
| 7 | Prior case retrieval (7 per query) | Micro-F1@7 | **0.4318** | BM25 / bi-encoder ensemble + date rules + cross-query normalisation |

The full write-up (methods, all experiments, figures, tables) is in
[`Reports/CS781-CP-Sanskaar-Srivastava-220967-report.pdf`](Reports/CS781-CP-Sanskaar-Srivastava-220967-report.pdf).
Its appendix also explains why each architecture, task layer and adaptation technique was chosen, with their
advantages and drawbacks. The separate analysis is
[`Reports/analysis.pdf`](Reports/analysis.pdf).

## Repository layout

```
code/
  phase1/   pretraining notebooks of Models 2-5, tokenizer study, decoding (infer.py) and MBR (mbr.py)
  phase2/   NER fine-tuning notebook + document-level post-processing notebook
  phase3/   rhetorical-role notebook + decoding/ensembling script
  phase4/   judgment-prediction notebook + threshold/ensemble script
  phase5/   statute-identification notebook + threshold/ensemble script
  phase6/   summarization notebook + local re-generation (decoding variants) + MBR ensemble
  phase7/   retrieval notebook + BM25 fusion / normalisation / ensemble script
  evaluation/  the official evaluate_phase{1..7}.py scripts (course-provided, used for local scoring / format checks)
practice/   small from-scratch PyTorch exercises (LoRA, rsLoRA, LoRMA, attention, ...), see practice/README.md
Reports/
  CS781-CP-Sanskaar-Srivastava-220967-report.pdf
  analysis.pdf              LLM-written analysis (separate document requested by the course)
  phase1_results.png        Phase 1 Codabench submission history
  latex/    sources of both PDFs, figures, generated tables, make_figures.py / data_stats.py / llm_figures.py
```

Every phase notebook (Phases 2–7) is self-contained: configuration cell → data download (Hugging Face dataset
`Exploration-Lab/CS-781-Capstone`, token in the Kaggle secret `HF`) → model code → training → evaluation → submission zip →
results bundle. Its `RUNS` list holds **every experiment of the phase**; `RUN_ONLY` is set to the runs behind the best
submission, so running the notebook as is reproduces the final system (set `RUN_ONLY = None` for all experiments). Two
runs train in parallel, one per T4. Each phase README gives inputs, runtime and the post-processing command.

## The Phase 1 models (shared by every phase)

| Model | Type | Shape | Position | Norm / MLP | Attention | Params |
|---|---|---|---|---|---|---|
| M1 | GPT-2-style decoder (baseline) | 8L, d=560, 8 heads, ctx 2048 | sinusoidal | LayerNorm / GELU | causal MHA | 58.2M |
| M2 | decoder | 10L, d=768, GQA 12/4, ctx 1024 | RoPE | LayerNorm / GELU | causal GQA | 101.4M |
| M3 | GRU encoder–decoder | 2L bi-GRU enc. + 2L GRU dec., h=1024 | recurrent | – | Bahdanau cross-attention | 91.7M |
| M4 | modern decoder | 10L, d=768, GQA 12/4, ctx 1024 | RoPE | RMSNorm / SwiGLU, no biases | causal GQA | 101.3M |
| M5 | prefix-LM | as M4 | RoPE | RMSNorm / SwiGLU | bidirectional prompt, causal continuation | 101.3M |

- **Tokenizer:** a from-scratch byte-level BPE with vocabulary 50,001, trained on the Supreme Court file. Later
  phases re-implement it with character offsets (`OffsetBPE`), which gives the same ids plus exact spans.
- **Training:** Hugging Face `Trainer` on 2×T4 in fp16, AdamW with a cosine schedule, one epoch, 5% of documents held
  out.
- **M4 extras:** 65k-token batches, scaled residual initialisation and a chunked cross-entropy, so the
  batch × 1024 × 50k logits are never materialised.
- **Checkpoints on Kaggle:**

  | Model | Dataset |
  |---|---|
  | M2 | `sanskaars22/model-2-capstone` |
  | M3 | `sanskaars22/model-3-capstone` |
  | M4 | `sanskaars22/model-5` |
  | M5 | `sanskaars22/model-4-prefix-lm-labelled-as-m5-in-notebook` |

  The M4 and M5 dataset names are swapped, so the notebooks check each checkpoint's `prefix_lm` flag and refuse a
  mismatch.

## Adaptation techniques (implemented in the notebooks, no PEFT library)

| Technique | Implementation | Used in |
|---|---|---|
| Full fine-tuning | all weights | 2, 3, 4, 5, 7 (6: M3 only) |
| Partial | top half of the layers trainable | 5 |
| LoRA | `h = W0 x + (α/r) B A x`, rank 8, α 16, attention + MLP projections | 2, 3, 4, 6 |
| rsLoRA | LoRA scaled by α/√r | 3, 4 |
| **LoRMA** (compulsory) | multiplicative `W = (I + (α/r) B A) W0`, computed as `z = W0 x; z + (α/r) B (A z)` (same cost as LoRA) | 2–7 |
| IA³ | learned scaling of keys, values and the down-projection input | 7 |
| BitFit | biases only (M3) | 2 |
| Houlsby adapters | bottleneck MLP (d→64→d) after attention and after the MLP | 6 |
| Knowledge distillation | student = lowest 4 of the teacher's 10 layers (63.6M); `(1-λ) task + λ T² KL(teacher‖student)`, T=2, λ=0.5 | 2–7 |

- **M3's GRU:** its recurrent cells contain no `nn.Linear` layers, so M3 is fully or partially fine-tuned.
- **Decoders:** M2, M4 and M5 are fine-tuned **bidirectionally** (no causal mask) for tagging, classification and
  retrieval.
- **Shared settings:** these were the same for every run of a phase. They are starting values, not tuned:
  - learning rates of 3e-5 (pretrained weights), 3e-4 (adapters) and 1e-3 (new heads);
  - AdamW, 10% warm-up, then linear decay;
  - the best epoch chosen on validation data.

## Phases

### Phase 1: pretraining and decoding (`code/phase1`)
- **Loss and leaderboard:** going from the GPT-2-style baseline to the modern decoder lowered validation loss from 4.74
  to 2.99 nats/token, and raised BERTScore from 0.7830 to 0.8324.
- **Decoding study:** `infer.py` has a KV-cache decoder supporting greedy, n-gram blocking, repetition/frequency
  penalties, top-p/min-p, beam and contrastive search. Settings were scored on a pseudo-validation set cut from the
  test prompts (prompt words 1–384, reference 385–512), which matched the leaderboard to within 0.002.
- **Results:** the best single setting, greedy with a 4-gram block and repetition penalty 1.2, gave 0.8363 on the
  leaderboard. The best submission is `mbr.py` (minimum Bayes risk over 14 candidates, BERTScore utility): 0.8419 on
  pseudo-validation and **0.8407** on the leaderboard (ROUGE-L utility: 0.8393 / 0.8379).

### Phase 2: legal NER (`code/phase2`)
- **Tagging:** BIO tags (25) on 1024-token windows with stride 512; overlapping scores are averaged.
- **Span trimming:** decoded spans are trimmed with rules learned from the training gold (abbreviations, brackets,
  possessives). The gold round-trip ceiling is 0.979.
- **Output layers:** softmax, BIO-constrained Viterbi, or a linear-chain CRF. The CRF partition function is a
  parallel log-semiring scan, which took 3.8 s → 0.036 s per batch.
- **Preamble context:** the first 256 tokens of the document are prepended to every later window as context only.
- **Post-processing:** a document-level label vote and propagation of predicted strings.
- **Final run:** `m5_lorma_crf_pre`, then `phase2_postprocess_kaggle.ipynb` (vote + propagate).

### Phase 3: rhetorical roles (`code/phase3`)
- **Model:** sentences are packed into 1024-token windows that overlap by half. Each sentence's tokens are
  mean-pooled, then classified by a linear layer or by a CRF over the sentence sequence.
- **Decoding:** `postprocess_phase3.py` re-decodes using statistics counted on the training labels only: a role
  prior (logit adjustment), transition Viterbi, both together, or a position prior. It also builds per-model
  ensembles.
- **Final system:** the 4 M5 CRF runs (LoRA, rsLoRA, LoRMA, full), with emissions and transitions averaged and
  prior + Viterbi decoding.

### Phase 4: judgment prediction (`code/phase4`)
- **Data integrity:** the official `dev.json` contains the 1,517 test cases with labels. They are removed, and
  validation uses only the remaining 994 dev cases. Test labels are never read.
- **Model:** a case is read as K × 512-token chunks of a chosen section (head, middle, tail, head+tail or tail×K).
  Chunks are mean-pooled, then combined by attention pooling before a linear classifier.
- **Section analysis:** every trained model is also scored reading the other sections.
- **Post-processing:** `postprocess_phase4.py` applies argmax, a training-prior threshold, or a dev-fitted threshold
  (2-fold CV on dev for an honest estimate), across views and per-model ensembles.
- **Final system:** the 4 `m5_*_alldocs` runs (all 31.7k training cases), reading the last 8 × 512 tokens, with the
  dev threshold.

### Phase 5: statute identification (`code/phase5`)
- **Model:** the Phase 4 chunk reader (head, 4 × 512 tokens) with a 100-way sigmoid layer; its biases start at the
  training log-odds. Trained with binary cross-entropy.
- **KD:** the student learns from the teacher's tempered sigmoid outputs.
- **Thresholds:** `postprocess_phase5.py` compares one global threshold with per-statute thresholds (2-fold CV on
  dev), keeping at least one statute per document. It also builds ensembles.
- **Final system:** `m4_full` + `m4_lorma`, ensembled, with per-statute thresholds (CV dev macro-F1 0.3673; CodaBench **0.3664**).

### Phase 6: summarization (`code/phase6`)
- **Input format:** the decoders read `[document] <s> [summary] </s>`, with the loss on the summary only.
  - The document is cut to 1,024 tokens taken from its head and tail.
  - The target is cut to 1,024 tokens.
  - RoPE is extended to 2,048 positions.
- **M3:** used as a GRU sequence-to-sequence model.
- **Training:** fp16 with a gradient scaler, and a per-run time budget.
- **Validation:** 200 training documents are held out for epoch selection; 100 of them are generated and scored.
- **Decoding:** greedy with a KV cache and no repeated 4-grams.
- **Final system:** `m4_lora` (LoRA on M4), re-generated by `regen_phase6.py m4_lora rp115 both`, which adds
  repetition penalty 1.15 using the saved adapter weights.
- **MBR consensus:** `ensemble_phase6.py` builds one across runs.

### Phase 7: prior case retrieval (`code/phase7`)
- **Rules taken from the data:**
  - A query is never its own answer.
  - Candidates decided after the query are excluded. A candidate's earliest date is its header date, or else the
    latest year it mentions.
  - The header field `CITATOR INFO` lists the later cases citing a judgment, which is a label leak. It is stripped
    from all texts.
- **Bi-encoder:** each case is read as 2 × 512 head tokens, mean-pooled and L2-normalised, and scored by cosine.
- **Training:** InfoNCE (τ = 0.05) with in-batch negatives plus one BM25 hard negative per query. 16 queries per batch
  fit in memory through gradient caching, which was checked to match direct backpropagation.
- **Post-processing:** `postprocess_phase7.py` scores
  `F + λ·colz(F) − p·[candidate is a query]` with `F = rowz(w·rowz(BM25) + rowz(cos))`. Here `rowz` and `colz` are
  z-scores over a query's allowed candidates and over a candidate's column across all queries.
  - `(w, λ, p)` are chosen on dev, with a 2-fold CV estimate.
  - Every setting is also written with `p = 0`.
- **Final run:** `m5_lorma`, then the post-processing script.

## Reproducing

1. **Kaggle:** attach the four model datasets listed above, add the secret `HF`, choose GPU T4 ×2, and run the
   phase notebook. Download `phaseN_results_bundle.zip` from `/kaggle/working`.
2. **Locally:** unzip the bundle next to the phase's post-processing script, as each phase README describes. Put the
   official data under `data/`, run the script, and submit the zip it writes.

**Requirements:** Python 3.10+, PyTorch 2.x, numpy, scikit-learn, scipy, huggingface_hub, and `bert-score` and
`rouge-score` for Phase 6 scoring.

## Rules followed
- **Models and data:** only our own Phase 1 models and the official data.
- **Coverage:** all four architectures were used in Phases 2–6 (in Phase 7 the M2 runs did not complete), with ≥ 3 adaptation techniques per phase (LoRMA
  always included) and ≥ 1 distillation experiment.
- **Selection:** settings were chosen on validation data only. Leaderboard scores were never used to pick a setting.
- **LLM use:** declared in the report. `Reports/analysis.pdf` is the separate LLM-based analysis the course asked
  for, written by Claude Opus 5.5.
