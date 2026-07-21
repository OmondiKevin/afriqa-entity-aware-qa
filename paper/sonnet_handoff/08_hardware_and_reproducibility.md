# Hardware & Reproducibility
- **Compute:** Single NVIDIA A100-SXM4-80GB GPU.
- **RAM:** 167GB System RAM.
- **Environment:** Google Colab Pro.
- **Reproducibility:** All metrics, predictions, and logs have been strictly version-controlled in the repository (`outputs/`). Code execution is deterministic for seed 42.
- **Limitation:** Due to compute constraints, multi-seed variance testing (e.g., seeds 42, 123, 456) was not performed. All reported metrics rely on a single run (seed 42), which should be clearly acknowledged as a threat to statistical significance.\n