# Experimental Setup
- **Architectures:** mT5-base (subword), ByT5-base (byte-level).
- **Optimizer:** AdamW.
- **Hardware:** NVIDIA A100-SXM4-80GB.
- **Precision:** bfloat16 (bf16).
- **Effective Batch Size:** 64 for all configurations (ensures rigorous scientific comparability).
- **Epochs:**
  - Baselines: 20 epochs (standard).
  - Matched-Volume & Multitask: Up to 200 epochs to accommodate the 20x upsampled volume while maintaining step parity.\n