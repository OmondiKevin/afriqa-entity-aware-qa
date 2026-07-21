# Claim to Evidence Map
- **Claim:** Direct QA beats translation when exposed sufficiently.
  - *Evidence:* ByT5 MV (EM 0.219) vs Translation (EM 0.116). See `results_master.csv`.
- **Claim:** mT5 requires NER supervision.
  - *Evidence:* Task Delta for mT5 is +0.013 EM. See Figure 3 (`fig3_delta_decomposition.pdf`).
- **Claim:** ByT5 is harmed by NER supervision.
  - *Evidence:* Task Delta for ByT5 is -0.027 EM. See Figure 3.
- **Claim:** LoRA fails for byte-level multitask.
  - *Evidence:* ByT5 LoRA EM drops to 0.045. See Figure 5 (`fig5_lora_comparison.pdf`).\n