# Appendix

## Appendix A: Complete Per-Language Results

**Table A1: Full per-language results for all configurations (EM / F1 / Semantic Similarity).**

| Configuration | SWA EM | SWA F1 | SWA Sem | HAU EM | HAU F1 | HAU Sem | YOR EM | YOR F1 | YOR Sem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1×) | 0.017 | 0.055 | 0.272 | 0.030 | 0.057 | 0.288 | 0.051 | 0.105 | 0.322 |
| Baseline ByT5 (1×) | 0.003 | 0.058 | 0.297 | 0.017 | 0.046 | 0.317 | 0.059 | 0.122 | 0.372 |
| Translation Pipeline | 0.122 | 0.202 | — | 0.157 | 0.234 | — | 0.059 | 0.122 | — |
| Matched-Vol mT5 (20×) | 0.176 | 0.214 | 0.423 | 0.227 | 0.261 | 0.468 | 0.186 | 0.233 | 0.425 |
| Matched-Vol ByT5 (20×) | 0.180 | 0.253 | 0.484 | **0.277** | **0.334** | **0.559** | 0.198 | 0.287 | 0.491 |
| Multitask mT5 | 0.176 | 0.224 | 0.426 | **0.277** | 0.306 | 0.513 | 0.170 | 0.255 | 0.467 |
| Multitask ByT5 | **0.190** | **0.266** | 0.455 | 0.227 | 0.293 | 0.515 | 0.154 | 0.254 | 0.455 |
| LoRA mT5 | 0.129 | 0.158 | 0.359 | 0.200 | 0.222 | 0.423 | 0.162 | 0.222 | 0.415 |
| LoRA ByT5 | 0.054 | 0.108 | 0.319 | 0.050 | 0.099 | 0.317 | 0.028 | 0.071 | 0.284 |

---

## Appendix B: Full Delta Decomposition Tables

**Table B1: Per-language delta decomposition for mT5 (all three metrics).**

| Language | Metric | Baseline | Matched-Vol | Multitask | Δ_exposure | Δ_task |
|---|---|---:|---:|---:|---:|---:|
| Overall | EM | 0.032 | 0.197 | 0.210 | +0.165 | +0.013 |
| Overall | F1 | 0.071 | 0.236 | 0.262 | +0.165 | +0.026 |
| Overall | Sem | 0.293 | 0.439 | 0.469 | +0.147 | +0.030 |
| Swahili | EM | 0.017 | 0.176 | 0.176 | +0.159 | 0.000 |
| Swahili | F1 | 0.055 | 0.214 | 0.224 | +0.158 | +0.011 |
| Swahili | Sem | 0.272 | 0.423 | 0.426 | +0.151 | +0.003 |
| Hausa | EM | 0.030 | 0.227 | 0.277 | +0.197 | +0.050 |
| Hausa | F1 | 0.057 | 0.261 | 0.306 | +0.203 | +0.045 |
| Hausa | Sem | 0.288 | 0.468 | 0.513 | +0.180 | +0.045 |
| Yoruba | EM | 0.051 | 0.186 | 0.170 | +0.134 | −0.016 |
| Yoruba | F1 | 0.105 | 0.233 | 0.255 | +0.128 | +0.022 |
| Yoruba | Sem | 0.322 | 0.425 | 0.467 | +0.104 | +0.042 |

**Table B2: Per-language delta decomposition for ByT5 (all three metrics).**

| Language | Metric | Baseline | Matched-Vol | Multitask | Δ_exposure | Δ_task |
|---|---|---:|---:|---:|---:|---:|
| Overall | EM | 0.025 | 0.219 | 0.192 | +0.195 | −0.027 |
| Overall | F1 | 0.073 | 0.292 | 0.272 | +0.219 | −0.020 |
| Overall | Sem | 0.327 | 0.513 | 0.476 | +0.186 | −0.037 |
| Swahili | EM | 0.003 | 0.180 | 0.190 | +0.177 | +0.010 |
| Swahili | F1 | 0.058 | 0.253 | 0.266 | +0.195 | +0.013 |
| Swahili | Sem | 0.297 | 0.484 | 0.455 | +0.188 | −0.030 |
| Hausa | EM | 0.017 | 0.277 | 0.227 | +0.260 | −0.050 |
| Hausa | F1 | 0.046 | 0.334 | 0.293 | +0.288 | −0.041 |
| Hausa | Sem | 0.317 | 0.559 | 0.515 | +0.242 | −0.044 |
| Yoruba | EM | 0.059 | 0.198 | 0.154 | +0.138 | −0.043 |
| Yoruba | F1 | 0.122 | 0.287 | 0.254 | +0.165 | −0.033 |
| Yoruba | Sem | 0.372 | 0.491 | 0.455 | +0.119 | −0.036 |

---

## Appendix C: NER Evaluation Details

**Table C1: NER evaluation results (MasakhaNER test set, n = 5,480).**

| Model | Parseable | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multitask mT5 | 409 (7.5%) | 0.840 | 0.036 | 0.070 | 368 | 70 | 9,766 |
| Multitask ByT5 | 41 (0.7%) | 0.122 | ~0.000 | 0.001 | 5 | 36 | 10,129 |

**Table C2: NER performance by entity type — Multitask mT5.**

| Type | Support | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PER | 3,095 | 94 | 9 | 3,001 | 0.913 | 0.030 | 0.059 |
| LOC | 3,215 | 84 | 4 | 3,131 | 0.955 | 0.026 | 0.051 |
| ORG | 2,174 | 94 | 30 | 2,080 | 0.758 | 0.043 | 0.082 |
| DATE | 1,650 | 96 | 27 | 1,554 | 0.780 | 0.058 | 0.108 |

---

## Appendix D: Qualitative Error Examples

The following examples are drawn from the automated error analysis pipeline (`outputs/analysis/error_analysis_report.md`) and represent real predictions from the Matched-Vol ByT5 model.

### Exact Match Examples

**Swahili:** Gold: "Oginga Odinga" → Predicted: "Oginga Odinga" *(F1 = 1.0)*  
**Hausa:** Gold: "1951" → Predicted: "1951" *(F1 = 1.0)*  
**Yoruba:** Gold: "Port-au-Prince" → Predicted: "Port-au-Prince" *(F1 = 1.0)*

### High-F1 Partial Match (Byte Truncation Pattern)

**Swahili:** Gold: "Thomas Alva Edison" → Predicted: "Thomas Alva Edis" *(F1 = 0.667)*  
**Hausa:** Gold: "James Francis Cameron" → Predicted: "James Francis Ca" *(F1 = 0.667)*  
**Hausa:** Gold: "Babagana Umara Zulum" → Predicted: "Babagana Umara Z" *(F1 = 0.667)*  
**Yoruba:** Gold: "John Stith Pemberton" → Predicted: "John Stith Pembe" *(F1 = 0.667)*

The consistent mid-name truncation (removing the final 2–4 characters) strongly suggests a generation length constraint interacting with byte-level encoding rather than semantic failure.

### Semantic Hallucination (Wrong but Plausible)

**Hausa:** Gold: "Doha" → Predicted: "Al Jazeera" *(F1 = 0.0)* — Alternative entity in the same Qatar domain.  
**Yoruba:** Gold: "Oluṣẹgun ọbasanjọ" → Predicted: "General Yakubu G..." *(F1 = 0.0)* — Alternative Nigerian head of state.  
**Swahili:** Gold: "Mwaka wa 1990" → Predicted: "National Party" *(F1 = 0.0)*

These errors suggest the model has acquired world-knowledge associations but cannot reliably select the specific referent demanded by the question.

---

## Appendix E: Training Configuration Files

Full YAML configuration for the ByT5 baseline experiment (`configs/baseline_byt5_1x.yaml`):

```yaml
project:
  name: afriqa-entity-aware-qa
  seed: 42

model:
  base: google/byt5-base
  prompt_prefix: "question: "
  max_source_length: 192
  max_target_length: 32

train:
  max_source_length: 256
  max_target_length: 128
  max_grad_norm: 1.0
  batch_size: 32
  grad_accum: 2
  lr: 1.0e-4
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 20
  early_stopping_patience: 3
  warmup_steps: 50
  fp16: false
  bf16: true
  num_workers: 4

lora:
  use_lora: false
  r: 16
  alpha: 32
  dropout: 0.05

eval:
  do_semantic: true
  labse_model: sentence-transformers/LaBSE
  generation_max_new_tokens: 16
  generation_min_new_tokens: 1
```

---

## Appendix F: Reproducibility Checklist

| Item | Status |
|---|---|
| Random seed reported | ✓ Seed = 42 |
| Hardware specified | ✓ A100-SXM4-80GB, 167GB RAM |
| Hyperparameters specified | ✓ See Appendix E |
| Model weights committed to repo | ✗ Excluded (model weights not committed) |
| Predictions committed | ✓ `outputs/predictions/` |
| Metrics committed | ✓ `outputs/metrics/` |
| Training logs committed | ✓ `outputs/logs/` |
| Evaluation scripts versioned | ✓ `scripts/04_eval_predictions.py` |
| Dataset sources cited | ✓ HuggingFace: masakhane/afriqa-gold-passages, masakhane/masakhaner2 |
| Multi-seed variance reported | ✗ Not yet conducted (acknowledged limitation) |
| Compute hours reported | ✗ Not precisely measured |

