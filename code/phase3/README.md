# Phase 3: rhetorical role prediction

1. `phase3_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, M5 checkpoint attached): `RUN_ONLY` = the four M5 CRF runs
   (LoRA, rsLoRA, LoRMA, full; ~45 min each, two in parallel → ~1.6 h). Each run saves dev/test per-sentence logits and
   its CRF transitions (`logits.pt`).
2. Unzip the bundle here as `phase3_results_bundle*/` and run `python postprocess_phase3.py` (needs the official
   `data/Phase_3_RR`): decoding with training-label statistics (prior, Viterbi, prior + Viterbi, position prior) and
   ensembles; writes `pp_work/submission_<base>_<strategy>.zip` and `pp_work/phase3_pp_log.txt`.
   Final submission: `m5_crf_ens` + `prior_viterbi` (CodaBench 0.5713).
