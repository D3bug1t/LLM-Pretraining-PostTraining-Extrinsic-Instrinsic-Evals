# Phase 6: abstractive summarization

1. `phase6_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, M4 checkpoint attached): `RUN_ONLY = ["m4_lora"]` (LoRA on
   M4, 2 epochs within a 150-minute budget, then greedy generation of 100 held-out and 100 test summaries; ~2.5 h).
   `save_weights=True` stores the trained adapter weights in the bundle.
2. `python postprocess_phase6.py`: BERTScore / ROUGE of the held-out summaries of every run in `bundle/`.
3. Final decoding without retraining: `regen_phase6.py` re-generates from `bundle/<run>/trainable_weights.pt` with
   another decoding setting. It executes the notebook's code from `summ_cells.py` and reads the checkpoint folders
   from the environment variables `M2_DIR` … `M5_DIR` and the data from `data/Phase_6_Summ`:
   `M4_DIR=<path to the M4 checkpoint> python regen_phase6.py m4_lora rp115 both` (repetition penalty 1.15) →
   `pp_work/regen/m4_lora__rp115/submission.zip` (CodaBench 0.8381).
4. `ensemble_phase6.py`: MBR consensus (ROUGE-1/2 utility) over the held-out/test summaries of several runs.
