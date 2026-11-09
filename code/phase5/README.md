# Phase 5: legal statute identification

1. `phase5_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, M4 checkpoint attached): `RUN_ONLY = ["m4_full",
   "m4_lorma"]` (head 4×512 tokens, 15k training documents, 100-way sigmoid; ~4.5 h, the two in parallel).
2. Unzip the bundle here as `bundle_*/` and run `python postprocess_phase5.py`: one global threshold vs. per-statute
   thresholds (2-fold CV on dev), ensembles; writes `pp_work/submission_*.zip` and `pp_work/phase5_pp_log.txt`.
   Final system: `m4_ens_head4_perlabel` (best CV dev macro-F1 0.3673).
