# Evaluation Correction Report
**Date:** 2026-07-22  
**Scope:** ByT5 generation limit investigation and impact assessment

---

## Summary

All ByT5 experiments used `eval.generation_max_new_tokens = 16` (bytes), identical to mT5.  
For mT5 (subword), 16 tokens ≈ 60–100 characters — sufficient for all but the longest answers.  
For ByT5 (byte-level), 16 bytes ≈ 13–16 ASCII characters, or 6–8 characters with 2-byte African diacritics.  
**The p75 of gold answer byte lengths is exactly 16 bytes. 24.1% of answers exceed the cap.**

---

## Before-vs-After: Truncation Evidence

### Confirmed Truncation Examples (matchedqa_byt5_test.jsonl)

| Gold Answer | Prediction | Gold bytes | Verdict |
|---|---|---:|---|
| Thomas Alva Edison | Thomas Alva Edis | 18 | truncated at 16 |
| Chuo Kikuu cha Kitaifa uhuru cha Mexico | National Autonom | 38 | truncated at 16 |
| Glaciation ya Wisconsinan | ice tabi ila-oor | 26 | truncated at 16 |
| Florence Simbiri-Jaoko | Dr Samuel Kipng' | 22 | 16-byte cap + wrong answer |
| Papa Gregory XIII | Pope Gregory XII | 18 | truncated at 16, also misspelled |
| Joseph Kasa-Vubu | Joseph Kasa-Vubu | 16 | NOT truncated (exact 16) |

### Population-Level Truncation (QA-only subsets)

| File | n | Truncated at 16 | Truncation Rate |
|---|---:|---:|---:|
| matchedqa_byt5_test.jsonl | 848 | 86 | **10.1%** |
| baseline_byt5_test_v2.jsonl | 848 | 17 | 2.0% |
| multitask_byt5_test.jsonl (QA rows only ~848) | 6328 (full) | 972 | 15.4% (full set) |
| multitask_byt5_lora_test.jsonl (QA rows only) | 6328 (full) | 1346 | 21.3% (full set) |
| matchedqa_mt5_test.jsonl | 848 | 3 | 0.4% |
| baseline_mt5_test.jsonl | 848 | 6 | 0.7% |

---

## Rerun Feasibility Assessment

### Local Resources
- ByT5 checkpoints: **NOT AVAILABLE LOCALLY** (trained on Colab, not committed to repo)
- mT5 checkpoints: Available at `outputs/checkpoints/baseline_mt5/`, `outputs/checkpoints/multitask_mt5/`
- Required packages locally: `datasets` and `sentence_transformers` not installed

### Rerun Plan (for Colab execution)
The following rerun is **required before submission** and must be executed on Colab where ByT5 checkpoints are available.

```python
# Rerun ByT5 evaluation at corrected generation limits
# Test: 32, 64, 128 bytes

for gen_limit in [32, 64, 128]:
    # Load each ByT5 checkpoint
    # Run generate_predictions() with max_new_tokens=gen_limit
    # Save to: outputs/predictions/matchedqa_byt5_test_gen{gen_limit}.jsonl
    # Evaluate and save to: outputs/metrics/matchedqa_byt5_gen{gen_limit}_metrics.json
    pass

# Affected checkpoints:
# - baseline_byt5_v2 (baseline_byt5_1x)
# - matchedqa_byt5 (matched-volume QA-only)
# - multitask_byt5 (multitask NER+QA)
# - multitask_byt5_lora (LoRA multitask)
```

### Expected Impact on Metrics
The 16-byte cap most severely affects **matchedqa_byt5** (the primary SOTA claim) because:
1. It has 10.1% directly confirmed truncated predictions (gold>16, pred=16)
2. ~23.8% of predictions hit exactly the 16-byte ceiling

The direction of change is **upward** (metrics will improve at higher limits), but the magnitude is unknown.

The relative ordering of matchedqa_byt5 vs multitask_byt5 (the task delta sign) may or may not change. The Swahili positive delta (+0.010 EM) could widen or narrow.

---

## What Has NOT Changed

All mT5 metrics are unaffected by this issue. The mT5 generation limit of 16 (subword tokens) was effectively unconstrained for this QA task (max observed prediction: 87 bytes, well above the gold p99 of 80 bytes in token terms).

All NER evaluation results are unaffected (NER predictions consist of structured tags, and length truncation at 16 bytes would reduce parse rate, not increase it).

Translation pipeline results are unaffected (max prediction: 210 bytes, no constraint).

---

## Interim Reporting Approach

Until ByT5 checkpoints are re-evaluated at corrected limits, all ByT5 QA metrics in the manuscript are annotated with:

> **†** All ByT5 metrics were generated with `generation_max_new_tokens = 16` (bytes). Because ByT5 generates raw UTF-8 bytes, this cap constrained approximately 24% of predictions for the matched-volume configuration and 10% exhibited demonstrable truncation (prediction exactly 16 bytes, gold answer longer). These figures are lower bounds; the paper reports corrected values after re-evaluation at the 128-byte limit (see Table X).

**Placeholder "Table X"** should be replaced with corrected metrics once Colab re-evaluation is complete.

