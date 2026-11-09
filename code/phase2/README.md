# Phase 2: legal named entity recognition

1. `phase2_finetune_kaggle.ipynb` (GPU T4 ×2, secret `HF`, the M5 checkpoint dataset attached): `RUN_ONLY =
   ["m5_lorma_crf_pre"]` = M5 + LoRMA + CRF output layer + 256-token preamble context (~30 min). Writes
   `phase2_results_bundle.zip` (per-run validation/test predictions, logs, metrics). All other experiments of the phase
   are in `RUNS` (`RUN_ONLY = None` runs them all).
2. `phase2_postprocess_kaggle.ipynb` (CPU; attach the bundle as an input): document-level vote and propagation,
   span ensembles; writes one submission zip per base × step set. The final submission is
   `m5_lorma_crf_pre` + vote + propagate (CodaBench 0.5122).
