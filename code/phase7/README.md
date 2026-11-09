# Phase 7: prior case retrieval

1. `phase7_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, M5 checkpoint attached): `RUN_ONLY = ["m5_lorma"]` (bi-encoder,
   head 2×512 tokens, InfoNCE with BM25 hard negatives and gradient caching, 60-minute budget; ~1 h with evaluation).
   It saves dev/test case embeddings (`emb.pt`) for several readings of each case (head, head:1, head:4, spread:4).
   `RUN_ONLY = None` runs all eight experiments (~5 h; the KD student waits for its teacher `m5_lorma`).
2. Unzip the bundle into `results/` (so that `results/<run>/emb.pt` exists), put the official data in
   `data/Phase_7_PCR`, and run `python postprocess_phase7.py`: BM25 fusion, date and self rules, cross-query (column)
   normalisation, optional query penalty, ensembles; settings chosen on dev with a 2-fold CV estimate; every zip is
   checked with the official `evaluate_phase7.py` format rules. Writes `pp_work/submission_<family>.zip` and
   `pp_work/pp_report.txt`.
