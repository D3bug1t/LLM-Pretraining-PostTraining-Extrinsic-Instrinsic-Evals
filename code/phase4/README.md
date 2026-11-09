# Phase 4: court judgment prediction

1. `phase4_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, M5 checkpoint attached): `RUN_ONLY` = the four
   `m5_*_alldocs` runs (LoRA, rsLoRA, LoRMA, full on all 31.7k deduplicated training cases; ~3.3 h each, two in parallel
   → ~7 h). The notebook removes the test cases that are present in `dev.json` and validates on the remaining 994
   cases; test labels are never read.
2. Unzip the bundle here as `phase4_results_bundle*/` and run `python postprocess_phase4.py`: views (tail, tail×8,
   both), ensembles, decisions (argmax / training prior / dev-fitted threshold with a 2-fold CV estimate); writes
   `pp_work/submission_<base>_<strategy>.zip` and `pp_work/phase4_pp_log.txt`.
   Final submission: `m5_alldocs_ens_tail8_devthr` (CodaBench 0.7609).
