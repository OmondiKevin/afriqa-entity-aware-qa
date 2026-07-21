# Training Configuration
- **Batch Size:** Physical batch size dynamically adjusted, but gradient accumulation enforces an Effective Batch Size of exactly 64 across all runs.
- **Learning Rate:** Standard Seq2Seq optimization rates used (5e-4 to 1e-3 ranges typically depending on adapter vs full fine-tuning).
- **Checkpoints:** Persistent auto-resume implemented to guard against Colab runtime disconnections.\n