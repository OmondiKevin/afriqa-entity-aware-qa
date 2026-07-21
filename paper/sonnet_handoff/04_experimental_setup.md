# Experimental Setup
**Hardware & Environment:**
- Compute: Single NVIDIA A100-SXM4-80GB GPU
- RAM: 167GB System RAM
- Environment: Google Colab Pro

**Hyperparameters & Training Configuration:**
- Precision: bfloat16 (bf16)
- Optimizer: AdamW
- Effective Batch Size: 64 (physical batch size dynamically adjusted with gradient accumulation to guarantee exact parity across all runs)
- Base Learning Rate: 5e-4 (standard seq2seq setup)
- Epochs: 
  - Standard (1x) baselines: 20 epochs
  - Matched-Volume (20x) and Multitask: Scaled up to 200 epochs to accommodate the upsampled data volume and ensure exact gradient step parity.
- Seeds: Random seed = 42 for all main experiments (Note: single-seed limitation acknowledged).\n